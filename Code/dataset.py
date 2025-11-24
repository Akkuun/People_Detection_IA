# dataset.py
import random
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
import os

class UnpairedCelebA_ONOT(Dataset):
    """
    Dataset that yields (A_image, B_image) where:
      - A comes from CelebA dataset (PNG images)
      - B comes from ONOT/digital dataset (PNG images)
    Both datasets are unpaired for CycleGAN training.
    """
    def __init__(self, celeba_root='/home/paradox/Bureau/M2/ProjetImage/dataset/CelebA', 
                 onot_root='/home/paradox/Bureau/M2/ProjetImage/dataset/ONOT/digital',
                 image_size=256):
        
        # Load CelebA images (A domain)
        self.celeba_paths = list(Path(celeba_root).rglob('*.jpg')) + list(Path(celeba_root).rglob('*.png'))
        if len(self.celeba_paths) == 0:
            raise RuntimeError(f"No CelebA images found in {celeba_root}")
        print(f"✅ Loaded {len(self.celeba_paths)} CelebA images")
        
        # Load ONOT images (B domain)
        self.onot_paths = list(Path(onot_root).rglob('*.png')) + list(Path(onot_root).rglob('*.jpg'))
        if len(self.onot_paths) == 0:
            raise RuntimeError(f"No ONOT images found in {onot_root}")
        print(f"✅ Loaded {len(self.onot_paths)} ONOT images")
        
        self.size = image_size
        self.transform = T.Compose([
            T.Resize((image_size, image_size)),
            T.RandomHorizontalFlip(p=0.5),
            T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
            T.ToTensor(),
            T.Normalize([0.5]*3, [0.5]*3)
        ])
    
    def __len__(self):
        return max(len(self.celeba_paths), len(self.onot_paths))
    
    def __getitem__(self, idx):
        try:
            # Load CelebA image (domain A)
            a_path = self.celeba_paths[idx % len(self.celeba_paths)]
            a_img = Image.open(a_path).convert('RGB')
            
            # Load ONOT image (domain B)
            b_path = self.onot_paths[random.randrange(len(self.onot_paths))]
            b_img = Image.open(b_path).convert('RGB')
            
            return self.transform(a_img), self.transform(b_img)
        except Exception as e:
            print(f"⚠️  Error loading images: {e}")
            # Return black images as fallback
            black_img = Image.new('RGB', (self.size, self.size))
            return self.transform(black_img), self.transform(black_img)
