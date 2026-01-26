import os
import argparse
import pandas as pd
import numpy as np
import openslide
import torch
import torch.nn as nn
import timm
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import cv2
from tqdm import tqdm
from utils.preprocessing import MacenkoNormalizer

class CTransPath(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        # CTransPath uses a Swin Transformer backbone.
        # Using swin_tiny_patch4_window7_224 as a proxy for the architecture.
        # Ideally, load specific weights here.
        self.model = timm.create_model('swin_tiny_patch4_window7_224', pretrained=pretrained, num_classes=0)

    def forward(self, x):
        return self.model(x)

def get_patches(slide_path, patch_size=256, level=0):
    """
    Simple patching function.
    In a real scenario, use a tissue mask to filter background.
    """
    try:
        slide = openslide.OpenSlide(slide_path)
    except:
        print(f"Could not open {slide_path}")
        return []

    w, h = slide.level_dimensions[level]
    patches = []

    # Simple grid, stepping by patch_size (non-overlapping)
    for y in range(0, h, patch_size):
        for x in range(0, w, patch_size):
            # Check if patch is background (simple check)
            # In production, use a low-res tissue mask
            patches.append((x, y))

    return slide, patches

class PatchDataset(Dataset):
    def __init__(self, slide, patches, transform=None, normalizer=None, patch_size=256, level=0):
        self.slide = slide
        self.patches = patches
        self.transform = transform
        self.normalizer = normalizer
        self.patch_size = patch_size
        self.level = level

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx):
        x, y = self.patches[idx]
        patch = self.slide.read_region((x, y), self.level, (self.patch_size, self.patch_size))
        patch = patch.convert('RGB')
        patch_np = np.array(patch)

        # Normalize
        if self.normalizer:
            patch_np = self.normalizer.normalize(patch_np)

        patch = Image.fromarray(patch_np)

        if self.transform:
            patch = self.transform(patch)

        return patch

def main():
    parser = argparse.ArgumentParser(description='Extract Features using Foundation Model (CTransPath/Swin)')
    parser.add_argument('--data_csv', type=str, default='../datasets.csv', help='Path to datasets.csv')
    parser.add_argument('--data_dir', type=str, required=True, help='Directory containing .svs files')
    parser.add_argument('--output_dir', type=str, default='features', help='Directory to save features')
    parser.add_argument('--ref_image', type=str, default='../canine_cutaneous.JPG', help='Reference image for Macenko')
    parser.add_argument('--batch_size', type=int, default=32)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Initialize Model (Foundation Model)
    print("Loading Foundation Model (CTransPath proxy)...")
    model = CTransPath(pretrained=True)
    model.to(device)
    model.eval()

    # Initialize Normalizer
    print("Initializing Stain Normalizer...")
    normalizer = MacenkoNormalizer(target_path=args.ref_image)

    # Load CSV
    df = pd.read_csv(args.data_csv, sep=';')

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    transform = transforms.Compose([
        transforms.Resize(224), # Swin expects 224
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # ImageNet stats
    ])

    for idx, row in df.iterrows():
        slide_name = row['Slide']
        slide_path = os.path.join(args.data_dir, slide_name)

        print(f"Processing {slide_name}...")

        slide, patches = get_patches(slide_path)
        if not patches:
            continue

        # Filter patches? (Here we blindly take all, assuming tissue mask logic is added later)
        # For this example, let's just limit patches if too many to prevent OOM/Time on cpu for demo
        # patches = patches[:100] # DEBUG

        dataset = PatchDataset(slide, patches, transform=transform, normalizer=normalizer)
        loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=4)

        features_list = []
        with torch.no_grad():
            for batch in tqdm(loader):
                batch = batch.to(device)
                features = model(batch)
                features_list.append(features.cpu())

        if features_list:
            slide_features = torch.cat(features_list, dim=0)
            save_path = os.path.join(args.output_dir, slide_name.replace('.svs', '.pt'))
            torch.save(slide_features, save_path)
            print(f"Saved {slide_features.shape} to {save_path}")

if __name__ == '__main__':
    main()
