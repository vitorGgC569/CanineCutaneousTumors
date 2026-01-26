import os
import argparse
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from models.clam import CLAM_SB
from sklearn.metrics import balanced_accuracy_score, classification_report
from tqdm import tqdm

class FeatureDataset(Dataset):
    def __init__(self, df, feature_dir, class_map):
        self.df = df
        self.feature_dir = feature_dir
        self.class_map = class_map
        self.data = []

        # Filter and prepare data
        print("Checking feature availability...")
        valid_count = 0
        for idx, row in df.iterrows():
            slide_name = row['Slide']
            feature_path = os.path.join(feature_dir, slide_name.replace('.svs', '.pt'))

            # Determine label from filename
            label_name = slide_name.split('_')[0]
            if label_name not in class_map:
                continue

            if os.path.exists(feature_path):
                self.data.append({
                    'slide': slide_name,
                    'path': feature_path,
                    'label': class_map[label_name]
                })
                valid_count += 1

        print(f"Found features for {valid_count} slides.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        features = torch.load(item['path'])
        return features, item['label']

def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []

    for features, label in tqdm(loader, desc="Training"):
        features = features.to(device) # (1, N, 768) if batch_size=1
        label = label.to(device)

        # CLAM expects (N, 768) input, so we usually run with batch_size=1 (1 bag)
        # Squeeze batch dim
        features = features.squeeze(0)

        optimizer.zero_grad()
        logits, Y_prob, Y_hat, _, _ = model(features)

        loss = criterion(logits, label)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        all_preds.extend(Y_hat.cpu().numpy())
        all_labels.extend(label.cpu().numpy())

    return total_loss / len(loader), balanced_accuracy_score(all_labels, all_preds)

def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for features, label in tqdm(loader, desc="Validation"):
            features = features.to(device).squeeze(0)
            label = label.to(device)

            logits, Y_prob, Y_hat, _, _ = model(features)
            loss = criterion(logits, label)

            total_loss += loss.item()
            all_preds.extend(Y_hat.cpu().numpy())
            all_labels.extend(label.cpu().numpy())

    return total_loss / len(loader), balanced_accuracy_score(all_labels, all_preds)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_csv', type=str, default='../datasets.csv')
    parser.add_argument('--feature_dir', type=str, default='features')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--input_dim', type=int, default=768) # 768 for Swin/ViT-Base
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Classes
    classes = ['Histiocytoma', 'MCT', 'Melanoma', 'Plasmacytoma', 'PNST', 'SCC', 'Trichoblastoma']
    class_map = {c: i for i, c in enumerate(classes)}
    print(f"Classes: {class_map}")

    # Load Data
    df = pd.read_csv(args.data_csv, sep=';')
    train_df = df[df['Dataset'] == 'train']
    val_df = df[df['Dataset'] == 'val']
    test_df = df[df['Dataset'] == 'test']

    train_ds = FeatureDataset(train_df, args.feature_dir, class_map)
    val_ds = FeatureDataset(val_df, args.feature_dir, class_map)

    # Batch size 1 for MIL (variable bag sizes)
    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=2)

    # Model
    model = CLAM_SB(input_dim=args.input_dim, n_classes=len(classes)).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(args.epochs):
        print(f"Epoch {epoch+1}/{args.epochs}")
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        print(f"Train Loss: {train_loss:.4f}, Train Balanced Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Balanced Acc: {val_acc:.4f}")

    # Save model
    torch.save(model.state_dict(), 'clam_model.pth')
    print("Model saved to clam_model.pth")

if __name__ == '__main__':
    main()
