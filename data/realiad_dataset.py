import torch
import numpy as np
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import pandas as pd


class RealIADTorchDataset(Dataset):
    """
    PyTorch Dataset wrapper around the realiad_utils dataframe.
    Used for custom training and inference loops that bypass
    Anomalib's internal datamodule — ensures identical data
    handling across all three models.

    Preprocessing matches published paper defaults:
    - Resize to 448x448
    - Centre crop to 392x392
    - Normalise with ImageNet mean/std
    """

    # Standard ImageNet normalisation — same for all three models
    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]

    def __init__(
        self,
        df: pd.DataFrame,
        input_size: int = 448,
        crop_size: int = 392,
        load_masks: bool = False
    ):
        """
        Args:
            df:           dataframe from load_realiad_category or load_realiad_all
            input_size:   resize images to this size before cropping
            crop_size:    centre crop to this size after resize
            load_masks:   if True also load GT masks for pixel-level evaluation
        """
        self.df = df.reset_index(drop=True)
        self.load_masks = load_masks

        # Image transform — matches all three model paper defaults
        self.image_transform = transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.CenterCrop(crop_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.MEAN, std=self.STD)
        ])

        # Mask transform — no normalisation, just resize
        self.mask_transform = transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.CenterCrop(crop_size),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Load and transform image
        image = Image.open(row['image_path']).convert('RGB')
        image = self.image_transform(image)

        # Build output dict
        item = {
            'image': image,
            'label': torch.tensor(row['label'], dtype=torch.long),
            'image_path': row['image_path'],
            'sample_id': row['sample_id'],
            'viewpoint': row['viewpoint'],
            'defect_type': row['defect_type'],
            'has_mask': row['has_mask'],
        }

        # Optionally load GT mask for pixel-level evaluation
        if self.load_masks and row['has_mask']:
            mask = Image.open(row['mask_path']).convert('L')
            mask = self.mask_transform(mask)
            mask = (mask > 0.5).float()
            item['mask'] = mask
        else:
            item['mask'] = torch.zeros(1, 392, 392)

        return item
    