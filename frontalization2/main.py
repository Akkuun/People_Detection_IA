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
from torchvision.transforms import functional as TF

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as vutils
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

from network import G, D, weights_init, VAE, UNetGenerator
from torchvision import models
import argparse

np.random.seed(42)
torch.manual_seed(999)
torch.backends.cudnn.deterministic = True

# ========================
# SIMPLE DATASET WITH POSE DETECTION
# ========================

class IdocMugshotDataset(Dataset):
    """
    Dataset pour le frontalization GAN basé sur iDoc Mugshots :
    - profils dans side/side
    - frontales dans front/front
    """

    def __init__(self, base_dir, image_size=128, max_samples=None):
        self.base_dir = base_dir
        self.image_size = image_size
        self.max_samples = max_samples

        # Charger listes de fichiers
        self.front_dir = os.path.join(base_dir, "front/front")
        self.side_dir  = os.path.join(base_dir, "side/side")

        self.front_files = sorted(os.listdir(self.front_dir))
        self.side_files  = sorted(os.listdir(self.side_dir))

        # On garde uniquement les IDs présents dans les deux dossiers
        common = list(set(self.front_files).intersection(set(self.side_files)))

        # If a max_samples limit is requested, sample deterministically
        if isinstance(max_samples, int) and max_samples > 0 and max_samples < len(common):
            rng = np.random.RandomState(42)
            # choose without replacement and keep deterministic order
            chosen = list(rng.choice(common, size=max_samples, replace=False))
            self.common_ids = sorted(chosen)
        else:
            self.common_ids = sorted(common)

        if len(self.common_ids) == 0:
            raise RuntimeError("❌ Aucun ID présent à la fois dans front/front et side/side")

        print(f"📦 Dataset iDoc Mugshots chargé : {len(self.common_ids)} échantillons (max_samples={self.max_samples})")

        # Transforms de base : resize + to tensor + normalize
        self.resize_size = (image_size, image_size)
        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        # Color jitter parameters (we'll apply deterministically per pair)
        self.jitter_params = dict(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)

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

        # Resize both images deterministically
        front_img = TF.resize(front_img, self.resize_size)
        side_img = TF.resize(side_img, self.resize_size)

        # Apply the same random horizontal flip to both images
        if np.random.rand() < 0.5:
            front_img = TF.hflip(front_img)
            side_img = TF.hflip(side_img)

        # Apply the same color jitter parameters to both images
        # Sample random factors within the jitter ranges
        b = self.jitter_params['brightness']
        c = self.jitter_params['contrast']
        s = self.jitter_params['saturation']
        h = self.jitter_params['hue']

        if any([b, c, s, h]):
            # brightness factor
            bf = np.random.uniform(max(0, 1 - b), 1 + b) if b else 1.0
            cf = np.random.uniform(max(0, 1 - c), 1 + c) if c else 1.0
            sf = np.random.uniform(max(0, 1 - s), 1 + s) if s else 1.0
            hf = np.random.uniform(-h, h) if h else 0.0

            front_img = TF.adjust_brightness(front_img, bf)
            front_img = TF.adjust_contrast(front_img, cf)
            front_img = TF.adjust_saturation(front_img, sf)
            front_img = TF.adjust_hue(front_img, hf)

            side_img = TF.adjust_brightness(side_img, bf)
            side_img = TF.adjust_contrast(side_img, cf)
            side_img = TF.adjust_saturation(side_img, sf)
            side_img = TF.adjust_hue(side_img, hf)

        # To tensor and normalize (same for both)
        side_t = self.normalize(self.to_tensor(side_img))
        front_t = self.normalize(self.to_tensor(front_img))

        return side_t, front_t
        #          profil -----------------------  face frontale

# ========================
# CONFIG
# ========================
num_epochs = 100  # Global parameter for the number of epochs
save_every_epoch = True  # Set to True to save every epoch, False to save every 5 epochs

datapath = 'training_set/archive'
gpu_id = 0
device = torch.device("cuda", gpu_id) if torch.cuda.is_available() else torch.device("cpu")
print(f"🚀 Using device: {device}")

# Dataset & DataLoader
# Parse CLI arguments (allow using a dataset sample)
parser = argparse.ArgumentParser(description='Face frontalization training')
parser.add_argument('--max-samples', '--max_samples', dest='max_samples', type=int, default=None,
                    help='Limit dataset to this many samples (deterministic sampling)')
args = parser.parse_args()
max_samples = args.max_samples

m_train = None
dataset = IdocMugshotDataset(datapath, image_size=128, max_samples=max_samples)
# Reduce batch size to avoid memory issues and improve stability
train_loader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=2, pin_memory=True)
m_train = len(dataset)
m_train = len(dataset)

# ========================
# MODELS
# ========================
# Use U-Net generator (skip connections) to better preserve details and reduce blurriness
netG = UNetGenerator().to(device)
netG.apply(weights_init)

netD = D().to(device)
netD.apply(weights_init)

netVAE = VAE(latent_dim=128).to(device)  # Initialize the VAE
print("✅ Models initialized")

# ========================
# LOSS & OPTIMIZERS
# ========================
# Re-balance losses: reduce L1 to avoid over-smoothing and keep adversarial term
# Loss weights
L1_factor = 5.0
L2_factor = 0.0
GAN_factor = 1.0
# Perceptual loss weight (VGG features) - increase to give stronger perceptual guidance
perc_factor = 5.0

# Diagnostics / loss options
use_hinge = False  # set True to use hinge loss instead of BCEWithLogits
log_every = 50
# Number of generator steps per discriminator step (increase to give G more updates)
G_steps_per_D = 3

# Small Gaussian noise added to inputs passed to D (helps regularize D)
D_input_noise = 0.05
# Probability to randomly flip / corrupt labels (one-sided label noise)
label_noise = 0.05

# Use BCEWithLogitsLoss and label smoothing for stability
criterion = nn.BCEWithLogitsLoss()
real_label = 0.9

# Optimizers: set lr_D smaller than lr_G for Test A (reduce D lr slightly)
optimizerD = optim.Adam(netD.parameters(), lr=3e-5, betas=(0.5, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=2e-4, betas=(0.5, 0.999), eps=1e-8)
vae_criterion = nn.MSELoss()  # Loss for VAE
vae_optimizer = optim.Adam(netVAE.parameters(), lr=0.0001, betas=(0.5, 0.999))

# Create output directory
os.makedirs('output', exist_ok=True)

# ============ Perceptual VGG setup ===========
# Load VGG features for perceptual loss
vgg_features = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).features[:16].to(device)
for p in vgg_features.parameters():
    p.requires_grad = False

def vgg_preprocess(x):
    # x in [-1,1] -> [0,1]
    x01 = (x + 1.0) / 2.0
    mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1,3,1,1)
    std = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1,3,1,1)
    return (x01 - mean) / std

# function to compute perceptual loss between two image batches
def perceptual_loss(a, b):
    fa = vgg_features(vgg_preprocess(a))
    fb = vgg_features(vgg_preprocess(b))
    return torch.mean((fa - fb) ** 2)

# ==============================================

# ========================
# TRAINING LOOP
# ========================
start_time = time.time()

print()

# Initialize lists to store loss values
loss_L1_history = []
loss_L2_history = []
loss_gan_history = []
loss_D_real_history = []
loss_D_fake_history = []

for epoch in range(num_epochs):
    
    loss_L1_total = 0
    loss_L2_total = 0
    loss_gan_total = 0
    batch_count = 0
    
    for batch_idx, (profile, frontal) in enumerate(train_loader):
        profile = profile.to(device)
        frontal = frontal.to(device)

        # ========================
        # TRAIN VAE
        # ========================
        netVAE.zero_grad()
        recon_profile, mu, logvar = netVAE(profile)
        vae_loss_recon = vae_criterion(recon_profile, profile)
        vae_loss_kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / profile.size(0)
        vae_loss = vae_loss_recon + vae_loss_kl
        vae_loss.backward()
        vae_optimizer.step()

        # ========================
        # TRAIN DISCRIMINATOR
        # ========================
        netD.zero_grad()
        
        # Generate images
        generated = netG(profile)

        # Add small Gaussian noise to inputs fed to D (regularizes D)
        if D_input_noise and D_input_noise > 0.0:
            frontal_for_D = frontal + torch.randn_like(frontal) * D_input_noise
            fake_for_D = generated.detach() + torch.randn_like(generated.detach()) * D_input_noise
        else:
            frontal_for_D = frontal
            fake_for_D = generated.detach()

        # Evaluate D on (possibly) noisy inputs
        output_real = netD(frontal_for_D)
        output_fake = netD(fake_for_D)

        # Discriminator loss: hinge or BCEWithLogits with optional label noise
        if use_hinge:
            # hinge loss
            errD_real = torch.mean(torch.nn.functional.relu(1.0 - output_real))
            errD_fake = torch.mean(torch.nn.functional.relu(1.0 + output_fake))
            errD = errD_real + errD_fake
        else:
            target_real = torch.ones_like(output_real).to(device) * real_label
            target_fake = torch.zeros_like(output_fake).to(device)

            # apply simple label noise / flipping (one-sided makes D less overconfident)
            if label_noise and label_noise > 0.0:
                flip_real = (torch.rand_like(target_real) < label_noise)
                target_real[flip_real] = 0.0
                flip_fake = (torch.rand_like(target_fake) < (label_noise * 0.5))
                target_fake[flip_fake] = real_label

            errD_real = criterion(output_real, target_real)
            errD_fake = criterion(output_fake, target_fake)
            errD = errD_real + errD_fake
        errD.backward()
        optimizerD.step()
        
        # Log discriminator losses
        loss_D_real_history.append(errD_real.item())
        loss_D_fake_history.append(errD_fake.item())

        # ========================
        # TRAIN GENERATOR (possibly multiple steps per D)
        # ========================
        # We'll run G_steps_per_D times; accumulate metrics once per batch
        first_step = True
        for g_step in range(G_steps_per_D):
            netG.zero_grad()
            generated_g = netG(profile)
            output_gen = netD(generated_g)

            if use_hinge:
                errG_GAN = -torch.mean(output_gen)
            else:
                target_gen = torch.ones_like(output_gen).to(device) * real_label
                errG_GAN = criterion(output_gen, target_gen)

            errG_L1 = torch.mean(torch.abs(frontal - generated_g))
            errG_L2 = torch.mean(torch.pow(frontal - generated_g, 2))

            try:
                errG_perc = perceptual_loss(frontal, generated_g)
            except Exception:
                errG_perc = torch.tensor(0.0, device=device)

            errG = GAN_factor * errG_GAN + L1_factor * errG_L1 + L2_factor * errG_L2 + perc_factor * errG_perc

            # On first G step, accumulate values for logging/plots
            if first_step:
                loss_L1_total += errG_L1.item()
                loss_L2_total += errG_L2.item()
                loss_gan_total += errG_GAN.item()
                batch_count += 1
                first_step = False


            errG.backward()
            optimizerG.step()
    
    # Log generator losses
    loss_L1_history.append(loss_L1_total / batch_count)
    loss_L2_history.append(loss_L2_total / batch_count)
    loss_gan_history.append(loss_gan_total / batch_count)
    
    # ========================
    # EPOCH SUMMARY
    # ========================
    # no per-epoch extra messages; only the summary line below will be printed
    
    avg_L1 = loss_L1_total / batch_count
    avg_L2 = loss_L2_total / batch_count
    avg_gan = loss_gan_total / batch_count

    # Print only the concise epoch summary line requested
    print(f'[{epoch+1:2d}/{num_epochs}] L1: {avg_L1:.7f} | L2: {avg_L2:.7f} | GAN: {avg_gan:.7f}')
    
    # ========================
    # SAVE OUTPUTS
    # ========================
    if save_every_epoch or (epoch + 1) % 5 == 0:  # Save outputs based on the parameter
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

# ========================
# PLOT LOSS CURVES
# ========================
plt.figure(figsize=(10, 6))
plt.plot(loss_L1_history, label='L1 Loss')
plt.plot(loss_L2_history, label='L2 Loss')
plt.plot(loss_D_real_history, label='Discriminator Real Loss')
plt.plot(loss_D_fake_history, label='Discriminator Fake Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training Loss Curves')
plt.legend()
plt.grid()
plt.xlim([0, num_epochs])  # Set x-axis limit to the number of epochs
plt.savefig('output/loss_curves.png')
plt.show()

# Separate plot for GAN loss
plt.figure(figsize=(10, 6))
plt.plot(loss_gan_history, label='GAN Loss', color='orange')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('GAN Loss Curve')
plt.legend()
plt.grid()
plt.xlim([0, num_epochs])  # Set x-axis limit to the number of epochs
plt.ylim([0, 2]) 
plt.savefig('output/gan_loss_curve.png')
plt.show()

total_time = time.time() - start_time
print(f"\n✨ Training complete! Total time: {total_time/3600:.1f} hours")
