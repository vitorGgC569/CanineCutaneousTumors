import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from models.clam import CLAM_SB
from sklearn.metrics import balanced_accuracy_score, roc_auc_score, confusion_matrix
from sklearn.model_selection import KFold
from tqdm import tqdm
import glob
import random

class FeatureBagDataset(Dataset):
    def __init__(self, file_list):
        self.file_list = file_list

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        path, label = self.file_list[idx]
        features = torch.load(path)
        return features, label

def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []

    # Use tqdm but careful not to spam logs
    for features, label in loader:
        features = features.to(device) # (1, N, Dim) -> but DataLoader usually gives (B, N, Dim) if collate is right
        # Since bag sizes differ, batch_size=1 is standard for simple MIL unless padded.
        # DataLoader batch_size=1 gives (1, N, Dim). Squeeze to (N, Dim)
        features = features.squeeze(0)
        label = label.to(device)

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
    all_probs = []

    with torch.no_grad():
        for features, label in loader:
            features = features.to(device).squeeze(0)
            label = label.to(device)

            logits, Y_prob, Y_hat, _, _ = model(features)
            loss = criterion(logits, label)

            total_loss += loss.item()
            all_preds.extend(Y_hat.cpu().numpy())
            all_labels.extend(label.cpu().numpy())
            all_probs.extend(Y_prob[:, 1].cpu().numpy()) # Prob of class 1 (Positive)

    try:
        auc = roc_auc_score(all_labels, all_probs)
    except:
        auc = 0.5 # If only one class present

    return total_loss / len(loader), balanced_accuracy_score(all_labels, all_preds), auc

def main():
    parser = argparse.ArgumentParser(description='Train MIL Model with K-Fold Cross Validation')
    parser.add_argument('--feature_dir', type=str, default='features', help='Directory containing "positivo" and "negativo" subdirs with .pt files')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--input_dim', type=int, default=768) # 768 for Swin/ViT-Base
    parser.add_argument('--k_folds', type=int, default=5, help='Number of folds for Cross Validation')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    # Set seed for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Gather Data
    positive_files = glob.glob(os.path.join(args.feature_dir, 'positivo', '*.pt'))
    negative_files = glob.glob(os.path.join(args.feature_dir, 'negativo', '*.pt'))

    print(f"Found {len(positive_files)} Positive samples")
    print(f"Found {len(negative_files)} Negative samples")

    if len(positive_files) == 0 and len(negative_files) == 0:
        print("No feature files found. Please run extract_features.py first.")
        return

    # Create dataset list: [(path, label), ...]
    # Label: 0 = Negative (Healthy), 1 = Positive (Cancer)
    all_data = []
    for p in positive_files:
        all_data.append((p, 1))
    for n in negative_files:
        all_data.append((n, 0))

    random.shuffle(all_data)

    # K-Fold CV
    kf = KFold(n_splits=args.k_folds, shuffle=True, random_state=args.seed)

    fold_results = []

    print(f"\nStarting {args.k_folds}-Fold Cross-Validation...")

    for fold, (train_idx, val_idx) in enumerate(kf.split(all_data)):
        print(f"\n=== Fold {fold+1}/{args.k_folds} ===")

        train_subset = [all_data[i] for i in train_idx]
        val_subset = [all_data[i] for i in val_idx]

        train_ds = FeatureBagDataset(train_subset)
        val_ds = FeatureBagDataset(val_subset)

        # Batch size 1 is standard for MIL (bags have different sizes)
        # Pin memory speeds up transfer to GPU
        train_loader = DataLoader(train_ds, batch_size=1, shuffle=True, num_workers=2, pin_memory=True)
        val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=2, pin_memory=True)

        # Initialize fresh model for each fold
        model = CLAM_SB(input_dim=args.input_dim, n_classes=2).to(device)
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
        criterion = nn.CrossEntropyLoss()

        best_val_acc = 0.0

        for epoch in range(args.epochs):
            train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
            val_loss, val_acc, val_auc = validate(model, val_loader, criterion, device)

            if (epoch + 1) % 5 == 0 or epoch == 0:
                print(f"Epoch {epoch+1:02d} | Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} AUC: {val_auc:.4f}")

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                # Save best model for this fold
                torch.save(model.state_dict(), f'clam_fold_{fold+1}.pth')

        fold_results.append(best_val_acc)
        print(f"Best Val Acc for Fold {fold+1}: {best_val_acc:.4f}")

    print("\n=== Cross-Validation Results ===")
    for i, res in enumerate(fold_results):
        print(f"Fold {i+1}: {res:.4f}")
    print(f"Average Accuracy: {np.mean(fold_results):.4f} (+/- {np.std(fold_results):.4f})")

if __name__ == '__main__':
    main()
