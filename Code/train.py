# train_optimized.py
import os
import time
import random
from pathlib import Path
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from model import ResnetGenerator, PatchDiscriminator, init_weights, ReplayBuffer
from dataset import UnpairedCelebA_ONOT

# -------------------------
# Config
# -------------------------
CONFIG = {
    'celeba_root': '/home/paradox/Bureau/M2/ProjetImage/dataset/CelebA',
    'onot_root': '/home/paradox/Bureau/M2/ProjetImage/dataset/ONOT/digital',
    'image_size': 128,      # réduite pour accélérer
    'batch_size': 4,        # très petit pour GPU limité
    'lr': 2e-4,
    'beta1': 0.5,
    'num_epochs': 20,       # moins d'epochs pour rapide
    'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    'save_dir': 'checkpoints_cyclegan',
    'lambda_cycle': 10.0,
    'lambda_id': 5.0,
    'display_freq': 200,
    'max_images': 10000     # échantillonner le dataset
}
os.makedirs(CONFIG['save_dir'], exist_ok=True)

device = CONFIG['device']
print(f"🚀 Using device: {device}")

# -------------------------
# Dataset
# -------------------------
dataset = UnpairedCelebA_ONOT(
    celeba_root=CONFIG['celeba_root'], 
    onot_root=CONFIG['onot_root'],
    image_size=CONFIG['image_size']
)

# Échantillonner si dataset énorme
if CONFIG['max_images'] and len(dataset) > CONFIG['max_images']:
    indices = random.sample(range(len(dataset)), CONFIG['max_images'])
    dataset = Subset(dataset, indices)

loader = DataLoader(dataset, batch_size=CONFIG['batch_size'], shuffle=True,
                    num_workers=8, pin_memory=True, drop_last=True)

print(f"✅ Dataset loaded. Total samples: {len(dataset)}")

# -------------------------
# Models
# -------------------------
G = ResnetGenerator().to(device)
F = ResnetGenerator().to(device)
DA = PatchDiscriminator().to(device)
DB = PatchDiscriminator().to(device)
init_weights(G); init_weights(F); init_weights(DA); init_weights(DB)
print("✅ Models initialized")

# -------------------------
# Losses
# -------------------------
criterion_GAN = nn.MSELoss()
criterion_cycle = nn.L1Loss()
criterion_id = nn.L1Loss()

# -------------------------
# Optimizers & Schedulers
# -------------------------
opt_G = torch.optim.Adam(list(G.parameters()) + list(F.parameters()), lr=CONFIG['lr'], betas=(CONFIG['beta1'], 0.999))
opt_DA = torch.optim.Adam(DA.parameters(), lr=CONFIG['lr'], betas=(CONFIG['beta1'], 0.999))
opt_DB = torch.optim.Adam(DB.parameters(), lr=CONFIG['lr'], betas=(CONFIG['beta1'], 0.999))

scheduler_G = torch.optim.lr_scheduler.LambdaLR(opt_G, lr_lambda=lambda epoch: 1.0 - max(0, epoch-25)/(CONFIG['num_epochs']-25))
scheduler_DA = torch.optim.lr_scheduler.LambdaLR(opt_DA, lr_lambda=lambda epoch: 1.0 - max(0, epoch-25)/(CONFIG['num_epochs']-25))
scheduler_DB = torch.optim.lr_scheduler.LambdaLR(opt_DB, lr_lambda=lambda epoch: 1.0 - max(0, epoch-25)/(CONFIG['num_epochs']-25))

fake_A_buffer = ReplayBuffer()
fake_B_buffer = ReplayBuffer()

# -------------------------
# Training Loop
# -------------------------
scaler = torch.cuda.amp.GradScaler()  # mixed precision

print(f"🎓 Starting training for {CONFIG['num_epochs']} epochs...\n")

for epoch in range(1, CONFIG['num_epochs'] + 1):
    epoch_start = time.time()
    pbar = tqdm(loader, desc=f"Epoch {epoch}/{CONFIG['num_epochs']}")
    for i, (real_A, real_B) in enumerate(pbar):
        real_A = real_A.to(device)
        real_B = real_B.to(device)

        # ------------------
        # Generators G and F
        # ------------------
        opt_G.zero_grad()
        with torch.cuda.amp.autocast():  # mixed precision
            # identity loss
            idt_B = G(real_B)
            loss_idt_B = criterion_id(idt_B, real_B) * CONFIG['lambda_id']
            idt_A = F(real_A)
            loss_idt_A = criterion_id(idt_A, real_A) * CONFIG['lambda_id']

            # GAN loss
            fake_B = G(real_A)
            pred_fake_B = DB(fake_B)
            loss_GAN_AB = criterion_GAN(pred_fake_B, torch.ones_like(pred_fake_B))

            fake_A = F(real_B)
            pred_fake_A = DA(fake_A)
            loss_GAN_BA = criterion_GAN(pred_fake_A, torch.ones_like(pred_fake_A))

            # cycle loss
            rec_A = F(fake_B)
            loss_cycle_A = criterion_cycle(rec_A, real_A) * CONFIG['lambda_cycle']
            rec_B = G(fake_A)
            loss_cycle_B = criterion_cycle(rec_B, real_B) * CONFIG['lambda_cycle']

            loss_G = loss_GAN_AB + loss_GAN_BA + loss_cycle_A + loss_cycle_B + loss_idt_A + loss_idt_B

        scaler.scale(loss_G).backward()
        scaler.step(opt_G)

        # ------------------
        # Discriminator DA
        # ------------------
        opt_DA.zero_grad()
        with torch.cuda.amp.autocast():
            pred_real_A = DA(real_A)
            loss_DA_real = criterion_GAN(pred_real_A, torch.ones_like(pred_real_A))

            fake_A_detach = fake_A_buffer.push_and_pop(fake_A.detach())
            pred_fake_A = DA(fake_A_detach)
            loss_DA_fake = criterion_GAN(pred_fake_A, torch.zeros_like(pred_fake_A))
            loss_DA = 0.5 * (loss_DA_real + loss_DA_fake)

        scaler.scale(loss_DA).backward()
        scaler.step(opt_DA)

        # ------------------
        # Discriminator DB
        # ------------------
        opt_DB.zero_grad()
        with torch.cuda.amp.autocast():
            pred_real_B = DB(real_B)
            loss_DB_real = criterion_GAN(pred_real_B, torch.ones_like(pred_real_B))

            fake_B_detach = fake_B_buffer.push_and_pop(fake_B.detach())
            pred_fake_B = DB(fake_B_detach)
            loss_DB_fake = criterion_GAN(pred_fake_B, torch.zeros_like(pred_fake_B))
            loss_DB = 0.5 * (loss_DB_real + loss_DB_fake)

        scaler.scale(loss_DB).backward()
        scaler.step(opt_DB)

        scaler.update()
        pbar.set_postfix({'loss_G': loss_G.item(), 'loss_DA': loss_DA.item(), 'loss_DB': loss_DB.item()})

    scheduler_G.step()
    scheduler_DA.step()
    scheduler_DB.step()

    # save checkpoint
    ckpt = {
        'G': G.state_dict(),
        'F': F.state_dict(),
        'DA': DA.state_dict(),
        'DB': DB.state_dict(),
        'opt_G': opt_G.state_dict(),
        'opt_DA': opt_DA.state_dict(),
        'opt_DB': opt_DB.state_dict(),
        'epoch': epoch
    }
    torch.save(ckpt, Path(CONFIG['save_dir']) / f'epoch_{epoch}.pth')
    print(f"\n✅ Epoch {epoch} finished in {time.time()-epoch_start:.1f}s -- checkpoint saved.")

print("\n✨ Training complete!")
