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
import h5py  # Para storage eficiente
import json
from typing import List, Tuple, Dict, Optional
import pandas as pd

class CTransPath(nn.Module):
    """Foundation Model com suporte a múltiplas resoluções."""
    def __init__(self, model_name='swin_tiny_patch4_window7_224', pretrained=True):
        super().__init__()
        self.model = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
        self.feature_dim = self.model.num_features  # 768 para Swin-Tiny
        
    def forward(self, x):
        return self.model(x)

class MultiResolutionFeatureExtractor(nn.Module):
    """Extrai features em múltiplas magnificações."""
    def __init__(self, model_name='swin_tiny_patch4_window7_224'):
        super().__init__()
        self.model_20x = CTransPath(model_name)  # Level 0
        self.model_10x = CTransPath(model_name)  # Level 1 (shared weights opcional)
        self.model_5x = CTransPath(model_name)   # Level 2
        
        # Projecção para dimensão comum (se modelos diferentes)
        self.proj = nn.Linear(768, 768)
        
    def forward(self, patches_20x, patches_10x=None, patches_5x=None):
        feats_20x = self.model_20x(patches_20x)
        
        feats = {"20x": feats_20x}
        if patches_10x is not None:
            feats["10x"] = self.proj(self.model_10x(patches_10x))
        if patches_5x is not None:
            feats["5x"] = self.proj(self.model_5x(patches_5x))
            
        return feats

class TissueQualityEstimator:
    """Estima qualidade do tecido para filtragem posterior."""
    def __init__(self):
        self.blur_threshold = 100  # Laplacian variance
        
    def assess(self, patch_np: np.ndarray) -> Dict[str, float]:
        gray = cv2.cvtColor(patch_np, cv2.COLOR_RGB2GRAY)
        
        # 1. Blur detection (Laplacian variance)
        lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        # 2. Contrast
        contrast = gray.std()
        
        # 3. Tissue coverage (Otsu já feito, mas refinar)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        tissue_ratio = np.mean(binary) / 255.0
        
        return {
            "sharpness": min(lap_var / self.blur_threshold, 1.0),
            "contrast": contrast / 50.0,  # Normalizado
            "tissue_ratio": tissue_ratio,
            "quality_score": (min(lap_var / self.blur_threshold, 1.0) * 0.4 + 
                            min(contrast / 50.0, 1.0) * 0.3 + 
                            tissue_ratio * 0.3)
        }

class AdvancedPatchDataset(Dataset):
    """Dataset com coordenadas espaciais e metadados."""
    def __init__(self, slide, patches: List[Tuple[int, int]], 
                 transform=None, normalizer=None, patch_size=256, level=0,
                 save_coords=True, quality_filter=0.3):
        self.slide = slide
        self.patches = patches  # Lista de (x, y)
        self.transform = transform
        self.normalizer = normalizer
        self.patch_size = patch_size
        self.level = level
        self.save_coords = save_coords
        self.quality_estimator = TissueQualityEstimator()
        self.quality_filter = quality_filter
        
        # Pré-filtrar por qualidade se necessário
        self.valid_indices = list(range(len(patches)))
        
    def __len__(self):
        return len(self.valid_indices)
    
    def __getitem__(self, idx):
        real_idx = self.valid_indices[idx]
        x, y = self.patches[real_idx]
        
        try:
            patch = self.slide.read_region((x, y), self.level, 
                                          (self.patch_size, self.patch_size))
            patch = patch.convert('RGB')
            patch_np = np.array(patch)
            
            # Quality assessment
            quality = self.quality_estimator.assess(patch_np)
            
            # Skip low quality (return None para filtragem posterior)
            if quality["quality_score"] < self.quality_filter:
                return None
            
            # Stain normalization
            if self.normalizer:
                patch_np = self.normalizer.normalize(patch_np)
                
            patch = Image.fromarray(patch_np)
            
            if self.transform:
                patch = self.transform(patch)
                
            result = {
                "patch": patch,
                "coords": torch.tensor([x, y], dtype=torch.float32),
                "quality": quality,
                "level": self.level
            }
            
            return result
            
        except Exception as e:
            print(f"Error reading patch {x},{y}: {e}")
            return None

def get_patches_multi_resolution(slide_path: str, patch_size=256, 
                                 tissue_threshold=0.2,
                                 levels=[0, 1, 2]) -> Dict[int, List[Tuple[int, int]]]:
    """
    Gera patches para múltiplos níveis de magnificação.
    
    Returns:
        Dict[level] -> lista de (x, y) coordenadas
    """
    try:
        slide = openslide.OpenSlide(slide_path)
    except Exception as e:
        print(f"Could not open {slide_path}: {e}")
        return {}
    
    all_patches = {}
    
    for level in levels:
        if level >= slide.level_count:
            continue
            
        w, h = slide.level_dimensions[level]
        
        # Usar thumbnail do nível mais baixo disponível para máscara
        mask_level = min(slide.level_count - 1, 3)
        mask_w, mask_h = slide.level_dimensions[mask_level]
        downsample_to_mask = slide.level_downsamples[mask_level]
        downsample_target = slide.level_downsamples[level]
        
        # Ler thumbnail para máscara de tecido
        thumbnail = slide.read_region((0, 0), mask_level, (mask_w, mask_h)).convert('RGB')
        thumb_np = np.array(thumbnail)
        
        # Otsu em HSV
        hsv = cv2.cvtColor(thumb_np, cv2.COLOR_RGB2HSV)
        s_channel = hsv[:, :, 1]
        _, tissue_mask = cv2.threshold(s_channel, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        patches = []
        step = patch_size
        
        for y in range(0, h, step):
            for x in range(0, w, step):
                # Mapear coordenadas para nível da máscara
                mask_x = int(x * downsample_target / downsample_to_mask)
                mask_y = int(y * downsample_target / downsample_to_mask)
                mask_ps = max(1, int(patch_size * downsample_target / downsample_to_mask))
                
                # Garantir bounds
                mask_x = min(mask_x, mask_w - 1)
                mask_y = min(mask_y, mask_h - 1)
                mask_ps = min(mask_ps, mask_w - mask_x, mask_h - mask_y)
                
                if mask_ps <= 0:
                    continue
                    
                patch_mask = tissue_mask[mask_y:mask_y+mask_ps, mask_x:mask_x+mask_ps]
                
                if patch_mask.size == 0:
                    continue
                
                # Threshold de tecido
                tissue_ratio = np.mean(patch_mask) / 255.0
                if tissue_ratio > tissue_threshold:
                    patches.append((x, y))
        
        all_patches[level] = patches
        print(f"Level {level}: {len(patches)} patches")
    
    slide.close()
    return all_patches

def extract_features_hierarchical(slide_path: str, model: nn.Module, 
                                  transform, normalizer, device,
                                  patch_size=256, levels=[0, 1],
                                  batch_size=64) -> Dict[str, torch.Tensor]:
    """
    Extrai features hierárquicas (multi-scale) de uma WSI.
    
    Returns:
        Dict com 'features', 'coords', 'qualities', 'level_map'
    """
    patches_by_level = get_patches_multi_resolution(slide_path, patch_size, levels=levels)
    
    if not patches_by_level:
        return {}
    
    all_features = []
    all_coords = []
    all_qualities = []
    level_map = []  # Qual level cada patch pertence
    
    slide = openslide.OpenSlide(slide_path)
    
    for level, patches in patches_by_level.items():
        if not patches:
            continue
            
        dataset = AdvancedPatchDataset(
            slide, patches, transform=transform, 
            normalizer=normalizer, patch_size=patch_size, level=level,
            quality_filter=0.3
        )
        
        # Filtrar Nones do dataset
        valid_samples = [s for s in dataset if s is not None]
        if not valid_samples:
            continue
            
        # Criar batch manualmente (mais controle)
        loader = DataLoader(
            dataset, batch_size=batch_size, num_workers=4,
            pin_memory=True, collate_fn=custom_collate
        )
        
        with torch.no_grad():
            for batch in tqdm(loader, desc=f"Level {level}"):
                if batch is None:
                    continue
                    
                patches_batch = batch["patch"].to(device, non_blocking=True)
                features = model(patches_batch)
                
                all_features.append(features.cpu())
                all_coords.append(batch["coords"])
                all_qualities.extend(batch["quality"])
                level_map.extend([level] * len(batch["quality"]))
    
    slide.close()
    
    if not all_features:
        return {}
    
    return {
        "features": torch.cat(all_features, dim=0),  # (N, 768)
        "coords": torch.cat(all_coords, dim=0),      # (N, 2)
        "qualities": all_qualities,                   # List[Dict]
        "level_map": torch.tensor(level_map),         # (N,)
        "slide_path": slide_path
    }

def custom_collate(batch):
    """Collation function que filtra amostras inválidas."""
    batch = [b for b in batch if b is not None]
    if not batch:
        return None
    
    return {
        "patch": torch.stack([b["patch"] for b in batch]),
        "coords": torch.stack([b["coords"] for b in batch]),
        "quality": [b["quality"] for b in batch],
        "level": [b["level"] for b in batch]
    }

def save_features_hdf5(features_dict: Dict, save_path: str, compression="gzip"):
    """
    Salva features em HDF5 com metadados (eficiente para grandes datasets).
    
    Estrutura:
        /features (N, D) - Features extraídas
        /coords (N, 2)   - Coordenadas (x, y)
        /level_map (N,)  - Nível de magnificação
        /qualities       - JSON com quality scores
        /metadata        - Informações do slide
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    with h5py.File(save_path, 'w') as f:
        # Features (compressão para economizar espaço)
        f.create_dataset('features', 
                        data=features_dict["features"].numpy(),
                        compression=compression,
                        chunks=True)
        
        # Coordenadas espaciais (essencial para MIL com posição)
        f.create_dataset('coords',
                        data=features_dict["coords"].numpy(),
                        compression=compression)
        
        # Level map
        f.create_dataset('level_map',
                        data=features_dict["level_map"].numpy(),
                        compression=compression)
        
        # Qualities como JSON
        qualities_json = json.dumps(features_dict["qualities"])
        f.create_dataset('qualities', data=qualities_json.encode('utf-8'))
        
        # Metadados
        metadata = {
            "slide_path": features_dict["slide_path"],
            "n_patches": len(features_dict["features"]),
            "feature_dim": features_dict["features"].shape[1],
            "levels": torch.unique(features_dict["level_map"]).tolist()
        }
        f.attrs['metadata'] = json.dumps(metadata)

def main():
    parser = argparse.ArgumentParser(
        description='Feature Extraction Otimizada para MIL (Multi-Scale + Coordenadas)'
    )
    parser.add_argument('--assets_dir', type=str, default='assets')
    parser.add_argument('--output_dir', type=str, default='features_v2')
    parser.add_argument('--ref_image', type=str, default='../canine_cutaneous.JPG')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--multi_resolution', action='store_true', 
                       help='Extrair em 20x, 10x e 5x')
    parser.add_argument('--quality_filter', type=float, default=0.3,
                       help='Threshold de qualidade (0-1)')
    parser.add_argument('--format', choices=['h5', 'pt'], default='h5',
                       help='Formato de saída (h5 recomendado para grandes datasets)')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Modelo
    print("Loading CTransPath (Swin Transformer)...")
    model = CTransPath()
    model.to(device)
    model.eval()
    
    # Multi-resolution se solicitado
    if args.multi_resolution:
        print("Modo Multi-Resolution ativado (20x, 10x, 5x)")
        model = MultiResolutionFeatureExtractor()
        model.to(device)
        model.eval()
        levels = [0, 1, 2]
    else:
        levels = [0]

    # Normalizer
    normalizer = None
    if os.path.exists(args.ref_image):
        print("Initializing Macenko Normalizer...")
        normalizer = MacenkoNormalizer(target_path=args.ref_image)

    # Transform
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])

    # Encontrar slides
    exts = ['*.svs', '*.tiff', '*.tif', '*.ndpi', '*.mrxs']
    slide_files = []
    
    for class_name in ['positivo', 'negativo']:
        class_dir = os.path.join(args.assets_dir, class_name)
        if not os.path.isdir(class_dir):
            continue
        for ext in exts:
            slide_files.extend([(f, class_name) for f in 
                              glob.glob(os.path.join(class_dir, ext))])

    print(f"Found {len(slide_files)} slides")
    
    # Processamento
    for slide_path, class_name in slide_files:
        slide_name = os.path.splitext(os.path.basename(slide_path))[0]
        save_dir = os.path.join(args.output_dir, class_name)
        os.makedirs(save_dir, exist_ok=True)
        
        ext = 'h5' if args.format == 'h5' else 'pt'
        save_path = os.path.join(save_dir, f"{slide_name}.{ext}")
        
        if os.path.exists(save_path):
            print(f"Skipping {slide_name}")
            continue

        print(f"\nProcessing {class_name}/{slide_name}...")
        
        try:
            if args.multi_resolution:
                features_dict = extract_features_hierarchical(
                    slide_path, model, transform, normalizer, device,
                    levels=levels, batch_size=args.batch_size
                )
            else:
                # Modo single-resolution (compatível com código anterior)
                patches = get_patches_multi_resolution(slide_path, levels=[0])[0]
                if not patches:
                    continue
                    
                dataset = AdvancedPatchDataset(
                    openslide.OpenSlide(slide_path), patches, 
                    transform, normalizer, quality_filter=args.quality_filter
                )
                
                loader = DataLoader(dataset, batch_size=args.batch_size,
                                  num_workers=args.num_workers, pin_memory=True,
                                  collate_fn=custom_collate)
                
                features_list, coords_list, qualities = [], [], []
                
                with torch.no_grad():
                    for batch in tqdm(loader, desc=slide_name):
                        if batch is None:
                            continue
                        feats = model(batch["patch"].to(device))
                        features_list.append(feats.cpu())
                        coords_list.append(batch["coords"])
                        qualities.extend(batch["quality"])
                
                features_dict = {
                    "features": torch.cat(features_list),
                    "coords": torch.cat(coords_list),
                    "qualities": qualities,
                    "level_map": torch.zeros(len(qualities)),
                    "slide_path": slide_path
                }
            
            # Salvar
            if args.format == 'h5':
                save_features_hdf5(features_dict, save_path)
            else:
                torch.save(features_dict, save_path)
                
            print(f"Saved {len(features_dict['features'])} patches to {save_path}")
            
        except Exception as e:
            print(f"Error processing {slide_name}: {e}")
            continue

if __name__ == '__main__':
    main()
