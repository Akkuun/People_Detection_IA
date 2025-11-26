"""
Face Frontalization Training - PyTorch Version avec Pose Detection
Utilise les landmarks CelebA pour séparer profils et faces frontales
"""
from __future__ import print_function
import time
import os
from os import listdir
from os.path import join
from PIL import Image

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as vutils
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader

from network import G, D, weights_init

np.random.seed(42)
torch.manual_seed(999)
torch.backends.cudnn.deterministic = True

# ========================
# POSE DETECTION FROM LANDMARKS
# ========================
def detect_pose_from_landmarks(landmarks):
    """
    Détecte si l'image est un profil ou une face frontale
    basé sur la position des yeux et du nez
    
    landmarks: [lefteye_x, lefteye_y, righteye_x, righteye_y, nose_x, nose_y, ...]
    """
    lefteye_x, lefteye_y = landmarks[0], landmarks[1]
    righteye_x, righteye_y = landmarks[2], landmarks[3]
    nose_x, nose_y = landmarks[4], landmarks[5]
    
    # Distance horizontale entre les yeux
    eye_distance = abs(righteye_x - lefteye_x)
    
    # Distance du nez au centre des yeux
    eye_center_x = (lefteye_x + righteye_x) / 2.0
    nose_offset = abs(nose_x - eye_center_x)
    
    # Ratio: si nez loin du centre, c'est un profil
    # Si ratio > 0.3, c'est un profil; sinon face frontale
    pose_ratio = nose_offset / (eye_distance + 1e-8)
    
    is_profile = pose_ratio > 0.3
    return is_profile, pose_ratio

# ========================
# LOAD LANDMARKS
# ========================
def load_landmarks(landmarks_file):
    """Charge les landmarks CelebA"""
    landmarks_dict = {}
    
    with open(landmarks_file, 'r') as f:
        # Skip header
        f.readline()
        
        for line in f:
            parts = line.strip().split()
            if len(parts) < 11:
                continue
            
            filename = parts[0]
            try:
                landmarks = list(map(int, parts[1:11]))
                landmarks_dict[filename] = landmarks
            except:
                continue
    
    return landmarks_dict

# ========================
# SIMPLE DATASET WITH POSE DETECTION
# ========================

class IdocMugshotDataset(Dataset):
    """
    Dataset pour le frontalization GAN basé sur iDoc Mugshots :
    - profils dans side/side
    - frontales dans front/front
    """

    def __init__(self, base_dir, image_size=128):
        self.base_dir = base_dir
        self.image_size = image_size

        # Charger listes de fichiers
        self.front_dir = os.path.join(base_dir, "front/front")
        self.side_dir  = os.path.join(base_dir, "side/side")

        self.front_files = sorted(os.listdir(self.front_dir))
        self.side_files  = sorted(os.listdir(self.side_dir))

        # On garde uniquement les IDs présents dans les deux dossiers
        self.common_ids = list(set(self.front_files).intersection(set(self.side_files)))

        if len(self.common_ids) == 0:
            raise RuntimeError("❌ Aucun ID présent à la fois dans front/front et side/side")

        print(f"📦 Dataset iDoc Mugshots chargé : {len(self.common_ids)} échantillons")

        # Transforms pour PyTorch
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
        ])

    def __len__(self):
        return len(self.common_ids)

    def __getitem__(self, idx):
        img_id = self.common_ids[idx]

        # Charger les images
        front_path = os.path.join(self.front_dir, img_id)
        side_path  = os.path.join(self.side_dir,  img_id)

        try:
            front_img = Image.open(front_path).convert("RGB")
            side_img  = Image.open(side_path).convert("RGB")
        except Exception as e:
            print("⚠️ Erreur sur", img_id, ":", e)
            # remplacer par un autre index
            return self.__getitem__((idx+1) % len(self))

        return self.transform(side_img), self.transform(front_img)
        #          profil -----------------------  face frontale

# ========================
# CONFIG
# ========================
datapath = 'training_set/archive'
gpu_id = 0
device = torch.device("cuda", gpu_id) if torch.cuda.is_available() else torch.device("cpu")
print(f"🚀 Using device: {device}")

# Dataset & DataLoader
dataset = IdocMugshotDataset(datapath, image_size=128)
train_loader = DataLoader(dataset, batch_size=30, shuffle=True, num_workers=4, pin_memory=True)
m_train = len(dataset)

# ========================
# MODELS
# ========================
netG = G().to(device)
netG.apply(weights_init)

netD = D().to(device)
netD.apply(weights_init)

print("✅ Models initialized")

# ========================
# LOSS & OPTIMIZERS
# ========================
L1_factor = 0
L2_factor = 1
GAN_factor = 0.0005

criterion = nn.BCELoss()

optimizerD = optim.Adam(netD.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=0.0002, betas=(0.5, 0.999), eps=1e-8)

# Create output directory
os.makedirs('output', exist_ok=True)

# ========================
# TRAINING LOOP
# ========================
start_time = time.time()

print(f"\n🎓 Starting training for 100 epochs...\n")

for epoch in range(100):
    
    loss_L1_total = 0
    loss_L2_total = 0
    loss_gan_total = 0
    batch_count = 0
    
    for batch_idx, (profile, frontal) in enumerate(train_loader):
        profile = profile.to(device)
        frontal = frontal.to(device)
        
        # ========================
        # TRAIN DISCRIMINATOR
        # ========================
        netD.zero_grad()
        
        # Real images
        target_real = torch.ones(frontal.size(0)).to(device)
        output_real = netD(frontal)
        errD_real = criterion(output_real, target_real)
        
        # Fake images
        generated = netG(profile)
        target_fake = torch.zeros(frontal.size(0)).to(device)
        output_fake = netD(generated.detach())
        errD_fake = criterion(output_fake, target_fake)
        
        errD = errD_real + errD_fake
        errD.backward()
        optimizerD.step()
        
        # ========================
        # TRAIN GENERATOR
        # ========================
        netG.zero_grad()
        
        target_gen = torch.ones(frontal.size(0)).to(device)
        output_gen = netD(generated)
        errG_GAN = criterion(output_gen, target_gen)
        
        errG_L1 = torch.mean(torch.abs(frontal - generated))
        errG_L2 = torch.mean(torch.pow(frontal - generated, 2))
        
        errG = GAN_factor * errG_GAN + L1_factor * errG_L1 + L2_factor * errG_L2
        
        loss_L1_total += errG_L1.item()
        loss_L2_total += errG_L2.item()
        loss_gan_total += errG_GAN.item()
        batch_count += 1
        
        errG.backward()
        optimizerG.step()
    
    # ========================
    # EPOCH SUMMARY
    # ========================
    if epoch == 0:
        elapsed = time.time() - start_time
        print(f'First training epoch completed in {elapsed:.1f} seconds\n')
    
    avg_L1 = loss_L1_total / batch_count
    avg_L2 = loss_L2_total / batch_count
    avg_gan = loss_gan_total / batch_count
    
    print(f'[{epoch+1:2d}/100] L1: {avg_L1:.7f} | L2: {avg_L2:.7f} | GAN: {avg_gan:.7f}')
    
    # ========================
    # SAVE OUTPUTS
    # ========================
    try:
        # Denormalize for visualization
        def denorm(x):
            return (x + 1.0) / 2.0
        
        vutils.save_image(denorm(profile[:8].data), f'output/{epoch:03d}_input.jpg')
        vutils.save_image(denorm(frontal[:8].data), f'output/{epoch:03d}_real.jpg')
        vutils.save_image(denorm(generated[:8].data), f'output/{epoch:03d}_generated.jpg')
        
        # Save model
        torch.save(netG.state_dict(), f'output/netG_{epoch:02d}.pt')
    except Exception as e:
        print(f"⚠️ Error saving outputs: {e}")

total_time = time.time() - start_time
print(f"\n✨ Training complete! Total time: {total_time/3600:.1f} hours")
