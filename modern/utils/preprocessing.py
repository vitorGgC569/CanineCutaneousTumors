import torch
import torchstain
import cv2
import numpy as np
import os

class MacenkoNormalizer:
    def __init__(self, target_path=None):
        """
        Initialize Macenko Normalizer.

        Args:
            target_path (str): Path to the target image to fit the normalizer to.
                               If None, it must be fit manually using fit().
        """
        self.normalizer = torchstain.normalizers.MacenkoNormalizer(backend='torch')
        self.is_fit = False

        if target_path and os.path.exists(target_path):
            target = cv2.imread(target_path)
            target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)
            self.fit(target)
            print(f"Macenko Normalizer fit to {target_path}")

    def fit(self, target_image):
        """
        Fit the normalizer to a target image.

        Args:
            target_image (np.ndarray): Target image in RGB format.
        """
        # torchstain expects CHW tensor
        t_tensor = torch.from_numpy(target_image).permute(2, 0, 1)
        self.normalizer.fit(t_tensor)
        self.is_fit = True

    def normalize(self, image):
        """
        Normalize an input image.

        Args:
            image (np.ndarray): Input image in RGB format.

        Returns:
            np.ndarray: Normalized image in RGB format.
        """
        if not self.is_fit:
            raise RuntimeError("Normalizer not fit. Please call fit() with a target image first.")

        img_t = torch.from_numpy(image).permute(2, 0, 1)

        try:
            # Normalize
            norm, H, E = self.normalizer.normalize(I=img_t, stains=True)

            # Convert back to numpy HWC
            # torchstain returns (H, W, C) even if input is (C, H, W)
            return norm.cpu().numpy().astype(np.uint8)
        except Exception as e:
            # If normalization fails (e.g. white background), return original
            return image
