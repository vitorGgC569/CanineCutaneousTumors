import os
import argparse
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
import glob

class CTransPath(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        # CTransPath uses a Swin Transformer backbone.
        # Using swin_tiny_patch4_window7_224 as a proxy for the architecture.
        self.model = timm.create_model('swin_tiny_patch4_window7_224', pretrained=pretrained, num_classes=0)

    def forward(self, x):
        return self.model(x)

def get_patches(slide_path, patch_size=256, level=0):
    """
    Generate patches filtered by Tissue Segmentation (Otsu on HSV).
    """
    try:
        slide = openslide.OpenSlide(slide_path)
    except Exception as e:
        print(f"Could not open {slide_path}: {e}")
        return None, []

    w, h = slide.level_dimensions[level]
    
    # Generate tissue mask at a lower resolution (e.g. level 2 or 3)
    downsample_level = min(slide.level_count - 1, 3) if slide.level_count > 1 else 0
    mask_w, mask_h = slide.level_dimensions[downsample_level]
    downsample_factor = w / float(mask_w) if mask_w > 0 else 1.0
    
    thumbnail = slide.read_region((0, 0), downsample_level, (mask_w, mask_h)).convert('RGB')
    thumb_np = np.array(thumbnail)
    
    # Convert RGB to HSV and apply Otsu on Saturation channel
    hsv = cv2.cvtColor(thumb_np, cv2.COLOR_RGB2HSV)
    s_channel = hsv[:, :, 1]
    _, tissue_mask = cv2.threshold(s_channel, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    patches = []

    for y in range(0, h, patch_size):
        for x in range(0, w, patch_size):
            # Check mask overlapping
            mask_x = int(x / downsample_factor)
            mask_y = int(y / downsample_factor)
            mask_ps = max(1, int(patch_size / downsample_factor))
            
            patch_mask = tissue_mask[mask_y:mask_y+mask_ps, mask_x:mask_x+mask_ps]
            
            # If tissue covers at least 20% of the patch area (255 * 0.20 = 51)
            if np.mean(patch_mask) > 51.0:
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
        try:
            patch = self.slide.read_region((x, y), self.level, (self.patch_size, self.patch_size))
            patch = patch.convert('RGB')
            patch_np = np.array(patch)

            # Basic background check removed: robust tissue filtering is now performed by Otsu pre-segmentation in get_patches.

            # Normalize
            if self.normalizer:
                patch_np = self.normalizer.normalize(patch_np)

            patch = Image.fromarray(patch_np)

            if self.transform:
                patch = self.transform(patch)

            return patch
        except Exception as e:
            print(f"Error reading patch {x},{y}: {e}")
            return torch.zeros((3, 224, 224)) # Return dummy

def main():
    parser = argparse.ArgumentParser(description='Extract Features using Foundation Model (CTransPath/Swin)')
    parser.add_argument('--assets_dir', type=str, default='assets', help='Directory containing "positivo" and "negativo" folders')
    parser.add_argument('--output_dir', type=str, default='features', help='Directory to save features')
    parser.add_argument('--ref_image', type=str, default='../canine_cutaneous.JPG', help='Reference image for Macenko')
    parser.add_argument('--batch_size', type=int, default=64) # Increased batch size for GPU
    parser.add_argument('--num_workers', type=int, default=8) # Increased workers for CPU parallelism
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Initialize Model (Foundation Model)
    print("Loading Foundation Model (CTransPath proxy)...")
    model = CTransPath(pretrained=True)
    model.to(device)
    model.eval()

    # Initialize Normalizer
    print("Initializing Stain Normalizer...")
    if os.path.exists(args.ref_image):
        normalizer = MacenkoNormalizer(target_path=args.ref_image)
    else:
        print(f"Warning: Reference image {args.ref_image} not found. Skipping normalization.")
        normalizer = None

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    transform = transforms.Compose([
        transforms.Resize(224), # Swin expects 224
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # ImageNet stats
    ])

    # Find all slides in positive and negative folders
    # Supported extensions
    exts = ['*.svs', '*.tiff', '*.tif', '*.ndpi']
    slide_files = []

    for class_name in ['positivo', 'negativo']:
        class_dir = os.path.join(args.assets_dir, class_name)
        if not os.path.isdir(class_dir):
            print(f"Warning: Directory {class_dir} does not exist.")
            continue

        for ext in exts:
            found = glob.glob(os.path.join(class_dir, ext))
            for f in found:
                slide_files.append((f, class_name))

    print(f"Found {len(slide_files)} slides to process.")

    for slide_path, class_name in slide_files:
        slide_name = os.path.basename(slide_path)
        # Create output subdir for class to keep structure organized
        save_dir = os.path.join(args.output_dir, class_name)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        save_path = os.path.join(save_dir, os.path.splitext(slide_name)[0] + '.pt')

        if os.path.exists(save_path):
            print(f"Skipping {slide_name} (already processed).")
            continue

        print(f"Processing {class_name}/{slide_name}...")

        slide, patches = get_patches(slide_path)
        if slide is None or not patches:
            continue

        dataset = PatchDataset(slide, patches, transform=transform, normalizer=normalizer)

        # Optimize DataLoader
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=True if torch.cuda.is_available() else False
        )

        features_list = []
        with torch.no_grad():
            for batch in tqdm(loader, desc=f"Extracting {slide_name}"):
                batch = batch.to(device, non_blocking=True)
                features = model(batch)
                features_list.append(features.cpu())

        if features_list:
            slide_features = torch.cat(features_list, dim=0)
            torch.save(slide_features, save_path)
            print(f"Saved {slide_features.shape} to {save_path}")

if __name__ == '__main__':
    main()
