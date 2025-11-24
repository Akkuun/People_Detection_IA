# train.py (VAE-GAN version with AUTO-TUNING)
import os
import time
import random
from pathlib import Path
from tqdm import tqdm
from collections import deque
import json

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import copy
from torchvision import models, utils as vutils
import shutil

# modèles (assure-toi que model/vae_generator.py existe)
from vae_generator import VAEGenerator
from model import PatchDiscriminator, init_weights, ReplayBuffer
from dataset import UnpairedCelebA_ONOT

# -------------------------
# AUTO-TUNING SYSTEM (Classe légère intégrée)
# -------------------------
class WeightAutoTuner:
    """Auto-ajuste les poids en analysant l'historique de loss"""
    
    def __init__(self, initial_config, strategy='smart', history_size=10):
        self.strategy = strategy
        self.history = deque(maxlen=history_size)
        self.weights = {
            'recon_weight': initial_config.get('recon_weight', 10.0),
            'perceptual_weight': initial_config.get('perceptual_weight', 1.0),
            'adv_weight': initial_config.get('adv_weight', 0.25),
            'kl_weight': initial_config.get('kl_weight', 0.01),
        }
        # Bounds pour éviter les extrêmes
        self.bounds = {
            'recon_weight': (0.1, 50.0),
            'perceptual_weight': (0.01, 5.0),
            'adv_weight': (0.01, 2.0),
            'kl_weight': (0.0001, 0.1),
        }
    
    def record_loss(self, total_loss):
        """Enregistrer la loss"""
        self.history.append(float(total_loss))
    
    def suggest_weights(self, epoch):
        """Suggérer une modification des poids basée sur la loss history"""
        if len(self.history) < 2:
            return self.weights
        
        recent_losses = list(self.history)
        avg_loss = sum(recent_losses) / len(recent_losses)
        
        if self.strategy == 'smart':
            return self._smart_tuning(recent_losses, avg_loss)
        elif self.strategy == 'conservative':
            return self._conservative_tuning(recent_losses, avg_loss)
        elif self.strategy == 'aggressive':
            return self._aggressive_tuning(recent_losses, avg_loss)
        else:
            return self.weights
    
    def _smart_tuning(self, losses, avg_loss):
        """Ajustements intelligents basés sur tendance"""
        if len(losses) < 3:
            return self.weights
        
        # Analyser la tendance
        recent = losses[-3:]
        trend = (recent[-1] - recent[0]) / (avg_loss + 1e-8)
        volatility = (max(recent) - min(recent)) / (avg_loss + 1e-8)
        
        new_weights = self.weights.copy()
        
        # Si loss augmente = adversarial trop fort
        if trend > 0.02:  # Loss augmente
            new_weights['adv_weight'] *= 0.97
            new_weights['recon_weight'] *= 1.02
        
        # Si loss diminue bien = peut augmenter adversarial
        elif trend < -0.05:  # Loss diminue bien
            new_weights['adv_weight'] *= 1.03
            new_weights['perceptual_weight'] *= 1.02
        
        # Si très volatil = réduire adversarial
        if volatility > 0.1:
            new_weights['adv_weight'] *= 0.98
        
        # Apply bounds
        for key in new_weights:
            min_val, max_val = self.bounds[key]
            new_weights[key] = max(min_val, min(max_val, new_weights[key]))
        
        self.weights = new_weights
        return self.weights
    
    def _conservative_tuning(self, losses, avg_loss):
        """Très stable, petits changements (pour mugshots)"""
        if len(losses) < 3:
            return self.weights
        
        recent = losses[-3:]
        trend = (recent[-1] - recent[0]) / (avg_loss + 1e-8)
        
        new_weights = self.weights.copy()
        
        # Changements minimes
        if trend > 0.01:
            new_weights['adv_weight'] *= 0.99
        elif trend < -0.03:
            new_weights['adv_weight'] *= 1.01
        
        for key in new_weights:
            min_val, max_val = self.bounds[key]
            new_weights[key] = max(min_val, min(max_val, new_weights[key]))
        
        self.weights = new_weights
        return self.weights
    
    def _aggressive_tuning(self, losses, avg_loss):
        """Changements importants pour styles"""
        if len(losses) < 3:
            return self.weights
        
        recent = losses[-3:]
        trend = (recent[-1] - recent[0]) / (avg_loss + 1e-8)
        volatility = (max(recent) - min(recent)) / (avg_loss + 1e-8)
        
        new_weights = self.weights.copy()
        
        if trend > 0.03:
            new_weights['adv_weight'] *= 0.95
            new_weights['recon_weight'] *= 1.05
        elif trend < -0.08:
            new_weights['adv_weight'] *= 1.06
            new_weights['perceptual_weight'] *= 1.05
        
        for key in new_weights:
            min_val, max_val = self.bounds[key]
            new_weights[key] = max(min_val, min(max_val, new_weights[key]))
        
        self.weights = new_weights
        return self.weights

# -------------------------
# Config
# -------------------------
CONFIG = {
    'celeba_root': '/home/paradox/Bureau/M2/ProjetImage/dataset/CelebA',
    'onot_root': '/home/paradox/Bureau/M2/ProjetImage/dataset/ONOT/digital',
    'image_size': 128,
    'batch_size': 32,
    'lr_G': 2e-4,
    'lr_D': 5e-5,          # lr réduit pour DA/DB
    'beta1': 0.5,
    'num_epochs': 40,  # ↑ Plus long pour meilleure convergence
    'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    'save_dir': 'checkpoints_vaegan',
    # VAE-GAN specific weights (AUTO-TUNED!)
    'recon_weight': 12.0,  # ↓ Réduit (moins flou, plus de détails)
    'kl_weight': 0.01,     # Normal (bonne variété)
    'kl_warmup_epochs': 5,
    'perceptual_weight': 2.0,  # ↑ Augmenté (meilleure qualité visuelle)
    'ema_decay': 0.99999,
    'adv_weight': 0.2,     # ↑ Un peu plus élevé (GAN plus compétitif = plus net)
    # Auto-tuning parameters
    'auto_tune': True,           # ← Activer l'auto-tuning
    'auto_tune_start_epoch': 2,  # ← Commencer à epoch 2
    'auto_tune_strategy': 'smart',  # 'smart', 'conservative', 'aggressive'
    # (général)
    'display_freq': 200,
    'max_images': 10000
}
os.makedirs(CONFIG['save_dir'], exist_ok=True)
os.makedirs('samples_vaegan', exist_ok=True)

device = CONFIG['device']
print(f"🚀 Using device: {device}")

# ========================
# INITIALISER AUTO-TUNER
# ========================
auto_tuner = WeightAutoTuner(
    CONFIG,
    strategy=CONFIG.get('auto_tune_strategy', 'smart'),
    history_size=10
)
print(f"🤖 Auto-tuning: {CONFIG.get('auto_tune', True)} (strategy: {CONFIG.get('auto_tune_strategy', 'smart')})")


def _has_enough_space(path: Path, min_bytes: int = 200 * 1024 * 1024) -> bool:
    """Return True if the filesystem containing `path` has at least `min_bytes` free."""
    try:
        usage = shutil.disk_usage(path.parent)
        return usage.free >= min_bytes
    except Exception:
        return True


def safe_torch_save(obj, path: Path) -> None:
    """Save `obj` to `path` atomically; if full save fails, try a compact variant.

    Writes to a temporary file then replaces the target. On failure attempts
    to save a minimal checkpoint (state_dicts + val_recon + epoch) to reduce size.
    """
    tmp = path.with_suffix(path.suffix + '.tmp')
    try:
        # quick check for disk space
        if not _has_enough_space(path):
            raise RuntimeError(f"Not enough free disk space to save checkpoint: {path}")

        torch.save(obj, tmp)
        # ensure data is flushed to disk
        try:
            with open(tmp, 'rb') as f:
                os.fsync(f.fileno())
        except Exception:
            pass
        tmp.replace(path)
        return
    except Exception as e:
        # attempt compact save if obj is a dict
        try:
            if isinstance(obj, dict):
                mini = {}
                for k in ('G', 'F', 'ema_G', 'ema_F', 'val_recon', 'epoch'):
                    if k in obj:
                        mini[k] = obj[k]
                torch.save(mini, tmp)
                try:
                    with open(tmp, 'rb') as f:
                        os.fsync(f.fileno())
                except Exception:
                    pass
                tmp.replace(path)
                return
        except Exception:
            pass

        # re-raise original exception with context
        raise

# -------------------------
# Dataset
# -------------------------
dataset = UnpairedCelebA_ONOT(
    celeba_root=CONFIG['celeba_root'],
    onot_root=CONFIG['onot_root'],
    image_size=CONFIG['image_size']
)

if CONFIG['max_images'] and len(dataset) > CONFIG['max_images']:
    indices = random.sample(range(len(dataset)), CONFIG['max_images'])
    dataset = Subset(dataset, indices)

# Split a small validation subset for monitoring and early-save
total_samples = len(dataset)
val_count = max(50, int(0.05 * total_samples)) if total_samples > 200 else max(10, int(0.1 * total_samples))
train_count = total_samples - val_count
if train_count <= 0:
    train_dataset = dataset
    val_dataset = None
else:
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_count, val_count])

loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], shuffle=True,
                    num_workers=8, pin_memory=True, drop_last=True)
val_loader = None
if val_dataset is not None:
    val_loader = DataLoader(val_dataset, batch_size=CONFIG['batch_size'], shuffle=False,
                            num_workers=4, pin_memory=True, drop_last=False)

print(f"✅ Dataset loaded. Total samples: {len(dataset)}")

# -------------------------
# Models
# -------------------------
# G : A -> B  (VAEGenerator)
# F : B -> A  (VAEGenerator)
G = VAEGenerator().to(device)
F = VAEGenerator().to(device)
DA = PatchDiscriminator().to(device)
DB = PatchDiscriminator().to(device)

# initialisation poids (si tu as init_weights)
try:
    init_weights(G); init_weights(F); init_weights(DA); init_weights(DB)
except Exception:
    # si init_weights n'existe pas ou signature différente, on ignore
    pass

print("✅ Models initialized (VAE-GAN)")

# -------------------------
# Perceptual (VGG) feature extractor
# -------------------------
vgg = models.vgg19(pretrained=True).features.to(device).eval()
for p in vgg.parameters():
    p.requires_grad = False

def compute_perceptual_loss(x, y):
    # both x,y in [-1,1], convert to [0,1] then ImageNet normalize
    def to_imagenet(t):
        t = (t + 1.0) / 2.0
        mean = torch.tensor([0.485, 0.456, 0.406], device=t.device).view(1,3,1,1)
        std = torch.tensor([0.229, 0.224, 0.225], device=t.device).view(1,3,1,1)
        return (t - mean) / std
    fx = vgg(to_imagenet(x))
    fy = vgg(to_imagenet(y))
    return nn.functional.l1_loss(fx, fy)

# -------------------------
# Losses
# -------------------------
criterion_GAN = nn.MSELoss()      # LSGAN style
criterion_recon = nn.L1Loss()    # reconstruction pixel-wise

# label smoothing
real_label = 0.8
fake_label = 0.0

# -------------------------
# Optimizers & schedulers
# -------------------------
opt_G = torch.optim.Adam(
    list(G.parameters()) + list(F.parameters()),
    lr=CONFIG['lr_G'],
    betas=(CONFIG['beta1'], 0.999)
)
opt_DA = torch.optim.Adam(DA.parameters(), lr=CONFIG['lr_D'], betas=(CONFIG['beta1'], 0.999))
opt_DB = torch.optim.Adam(DB.parameters(), lr=CONFIG['lr_D'], betas=(CONFIG['beta1'], 0.999))

scheduler_G = torch.optim.lr_scheduler.LambdaLR(
    opt_G,
    lr_lambda=lambda epoch: 1.0 - max(0, epoch-25) / max(1, (CONFIG['num_epochs'] - 25))
)
scheduler_DA = torch.optim.lr_scheduler.LambdaLR(
    opt_DA,
    lr_lambda=lambda epoch: 1.0 - max(0, epoch-25) / max(1, (CONFIG['num_epochs'] - 25))
)
scheduler_DB = torch.optim.lr_scheduler.LambdaLR(
    opt_DB,
    lr_lambda=lambda epoch: 1.0 - max(0, epoch-25) / max(1, (CONFIG['num_epochs'] - 25))
)

# Replay buffers (comme avant)
fake_A_buffer = ReplayBuffer()
fake_B_buffer = ReplayBuffer()

# EMA copies of generators for smoother sampling
ema_G = copy.deepcopy(G).to(device)
ema_F = copy.deepcopy(F).to(device)
for p in ema_G.parameters():
    p.requires_grad = False
for p in ema_F.parameters():
    p.requires_grad = False

# -------------------------
# Training Loop (AMP)
# -------------------------
scaler = torch.cuda.amp.GradScaler()

print(f"🎓 Starting training for {CONFIG['num_epochs']} epochs...\n")

for epoch in range(1, CONFIG['num_epochs'] + 1):
    epoch_start = time.time()
    
    # ========================
    # APPLIQUER AUTO-TUNING (après epoch 2)
    # ========================
    if CONFIG.get('auto_tune', False) and epoch >= CONFIG.get('auto_tune_start_epoch', 2):
        auto_tuner.suggest_weights(epoch)
        CONFIG['recon_weight'] = auto_tuner.weights['recon_weight']
        CONFIG['perceptual_weight'] = auto_tuner.weights['perceptual_weight']
        CONFIG['adv_weight'] = auto_tuner.weights['adv_weight']
        CONFIG['kl_weight'] = auto_tuner.weights['kl_weight']
        print(f"\n🔧 Weights adjusted (epoch {epoch}):")
        print(f"   recon={CONFIG['recon_weight']:.3f}, perceptual={CONFIG['perceptual_weight']:.3f}, " +
              f"adv={CONFIG['adv_weight']:.3f}, kl={CONFIG['kl_weight']:.5f}")
    
    pbar = tqdm(loader, desc=f"Epoch {epoch}/{CONFIG['num_epochs']}")
    epoch_losses = []
    
    for i, (real_A, real_B) in enumerate(pbar):
        real_A = real_A.to(device)
        real_B = real_B.to(device)

        # ------------------
        # GENERATORS (VAE-GAN)
        #   - G: encode A -> z -> decode -> fake_B
        #   - F: encode B -> z -> decode -> fake_A
        # Loss: recon (L1) + KL + adversarial (MSE vs real_label)
        # ------------------
        opt_G.zero_grad()
        with torch.cuda.amp.autocast():
            # A -> fake_B
            fake_B, mu_A, logvar_A = G(real_A)  # fake_B in [-1,1] if decoder uses Tanh
            # reconstruction loss toward real_B (image-to-image supervision)
            loss_recon_AB = criterion_recon(fake_B, real_B) * CONFIG['recon_weight']

            # perceptual loss A->B
            loss_perc_AB = compute_perceptual_loss(fake_B, real_B) * CONFIG['perceptual_weight']

            # KL divergence for A encoder
            # KL per sample: -0.5 * sum(1 + logvar - mu^2 - exp(logvar))
            kl_A = -0.5 * torch.sum(1 + logvar_A - mu_A.pow(2) - logvar_A.exp(), dim=1)
            # KL warmup (increase weight over first epochs)
            kl_w = CONFIG.get('kl_weight', 0.01)
            warmup = CONFIG.get('kl_warmup_epochs', 5)
            kl_scale = min(1.0, epoch / max(1, warmup))
            loss_kl_A = kl_A.mean() * kl_w * kl_scale

            # adversarial on fake_B
            pred_fake_B = DB(fake_B)
            loss_adv_AB = criterion_GAN(pred_fake_B, torch.full_like(pred_fake_B, real_label)) * CONFIG['adv_weight']

            # B -> fake_A
            fake_A, mu_B, logvar_B = F(real_B)
            loss_recon_BA = criterion_recon(fake_A, real_A) * CONFIG['recon_weight']

            # perceptual loss B->A
            loss_perc_BA = compute_perceptual_loss(fake_A, real_A) * CONFIG['perceptual_weight']

            kl_B = -0.5 * torch.sum(1 + logvar_B - mu_B.pow(2) - logvar_B.exp(), dim=1)
            loss_kl_B = kl_B.mean() * kl_w * kl_scale

            pred_fake_A = DA(fake_A)
            loss_adv_BA = criterion_GAN(pred_fake_A, torch.full_like(pred_fake_A, real_label)) * CONFIG['adv_weight']

            # Total generator(VAE-GAN) loss
            # combine losses, include perceptual
            loss_G_recon = loss_recon_AB + loss_recon_BA
            loss_G_perc = loss_perc_AB + loss_perc_BA
            loss_G_kl = loss_kl_A + loss_kl_B
            loss_G_adv = loss_adv_AB + loss_adv_BA

            loss_G = loss_G_recon + loss_G_perc + loss_G_kl + loss_G_adv

            # OPTIONNEL: si tu veux simuler "2 updates G per D" (comme auparavant),
            # tu peux multiplier loss_G * 2 (Solution A). Je laisse ceci commenté
            # mais tu peux activer si tu veux accélérer l'apprentissage du G.
            # loss_G_simulated = loss_G * 2.0
            loss_G_simulated = loss_G * 2.0 # par défaut une update

        # backward + step pour G
        scaler.scale(loss_G_simulated).backward()
        scaler.step(opt_G)

        # ------------------
        # DISCRIMINATEUR DA
        # ------------------
        opt_DA.zero_grad()
        with torch.cuda.amp.autocast():
            pred_real_A = DA(real_A)
            loss_DA_real = criterion_GAN(pred_real_A, torch.full_like(pred_real_A, real_label))

            # utiliser buffer pour fake_A
            fake_A_detach = fake_A_buffer.push_and_pop(fake_A.detach())
            pred_fake_A = DA(fake_A_detach)
            loss_DA_fake = criterion_GAN(pred_fake_A, torch.full_like(pred_fake_A, fake_label))

            loss_DA = 0.5 * (loss_DA_real + loss_DA_fake)

        scaler.scale(loss_DA).backward()
        scaler.step(opt_DA)

        # ------------------
        # DISCRIMINATEUR DB
        # ------------------
        opt_DB.zero_grad()
        with torch.cuda.amp.autocast():
            pred_real_B = DB(real_B)
            loss_DB_real = criterion_GAN(pred_real_B, torch.full_like(pred_real_B, real_label))

            fake_B_detach = fake_B_buffer.push_and_pop(fake_B.detach())
            pred_fake_B = DB(fake_B_detach)
            loss_DB_fake = criterion_GAN(pred_fake_B, torch.full_like(pred_fake_B, fake_label))

            loss_DB = 0.5 * (loss_DB_real + loss_DB_fake)

        scaler.scale(loss_DB).backward()
        scaler.step(opt_DB)

        # ------------------
        # scaler.update() une seule fois par itération (nécessaire avec AMP)
        # ------------------
        scaler.update()

        # Update EMA weights
        with torch.no_grad():
            decay = CONFIG.get('ema_decay', 0.999)
            for p, ema_p in zip(G.parameters(), ema_G.parameters()):
                ema_p.data.mul_(decay).add_(p.data, alpha=1 - decay)
            for p, ema_p in zip(F.parameters(), ema_F.parameters()):
                ema_p.data.mul_(decay).add_(p.data, alpha=1 - decay)

        # ------------------
        # Logging (affiche loss brutes)
        # ------------------
        total_loss = float((loss_G + loss_DA + loss_DB).item())
        epoch_losses.append(total_loss)
        
        pbar.set_postfix({
            'loss_G_recon': float(loss_G_recon.item()),
            'loss_G_kl': float(loss_G_kl.item()),
            'loss_G_adv': float(loss_G_adv.item()),
            'loss_DA': float(loss_DA.item()),
            'loss_DB': float(loss_DB.item())
        })

    # Scheduler step
    scheduler_G.step()
    scheduler_DA.step()
    scheduler_DB.step()
    
    # ========================
    # ENREGISTRER LOSS MOYENNE POUR AUTO-TUNING
    # ========================
    if epoch_losses and CONFIG.get('auto_tune', False):
        avg_epoch_loss = sum(epoch_losses) / len(epoch_losses)
        auto_tuner.record_loss(avg_epoch_loss)

    # Save checkpoint
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
    # Save latest checkpoint
    checkpoint_path = Path(CONFIG['save_dir']) / 'vaegan_model.pth'
    ckpt['ema_G'] = ema_G.state_dict()
    ckpt['ema_F'] = ema_F.state_dict()
    try:
        safe_torch_save(ckpt, checkpoint_path)
    except Exception as e:
        print(f"❌ Failed to save checkpoint {checkpoint_path}: {e}")

    # Validation pass (compute mean reconstruction L1 on val set) and save sample images
    best_metric_path = Path(CONFIG['save_dir']) / 'best_vaegan_model.pth'
    val_recon = None
    if val_loader is not None:
        G.eval(); F.eval(); ema_G.eval(); ema_F.eval()
        running = []
        with torch.no_grad():
            for va, vb in val_loader:
                va = va.to(device); vb = vb.to(device)
                # use EMA generators for evaluation
                fake_vb, *_ = ema_G(va)
                fake_va, *_ = ema_F(vb)
                loss_va = nn.functional.l1_loss(fake_vb, vb, reduction='mean')
                loss_vb = nn.functional.l1_loss(fake_va, va, reduction='mean')
                running.append((loss_va.item() + loss_vb.item()) * 0.5)
        if len(running) > 0:
            val_recon = float(sum(running) / len(running))

        # save sample images from first val batch
        try:
            sample_batch = next(iter(val_loader))
            sa, sb = sample_batch[0].to(device), sample_batch[1].to(device)
            with torch.no_grad():
                fake_sb, *_ = ema_G(sa)
                fake_sa, *_ = ema_F(sb)
            # denormalize from [-1,1] to [0,1]
            def denorm(x):
                return (x + 1.0) / 2.0
            grid = vutils.make_grid(torch.cat([denorm(sa.cpu()), denorm(fake_sb.cpu()), denorm(sb.cpu())], dim=0), nrow=sa.size(0))
            sample_path = Path('samples_vaegan') / f'epoch_{epoch:03d}.png'
            vutils.save_image(grid, str(sample_path))
        except Exception:
            pass

        # if improvement, save best checkpoint
        if val_recon is not None:
                # load existing best metric if present (prefer weights_only when available)
                try:
                    try:
                        best_data = torch.load(best_metric_path, map_location='cpu', weights_only=True)
                    except TypeError:
                        # older torch versions don't support weights_only
                        best_data = torch.load(best_metric_path, map_location='cpu')
                    except Exception:
                        # if weights_only experimental not available or fails, fallback
                        best_data = torch.load(best_metric_path, map_location='cpu')
                    best_val = best_data.get('val_recon', float('inf'))
                except Exception:
                    best_val = float('inf')

                if val_recon < best_val:
                    # create a compact checkpoint for the best model (smaller payload)
                    best_ckpt = {
                        'G': G.state_dict(),
                        'F': F.state_dict(),
                        'ema_G': ema_G.state_dict(),
                        'ema_F': ema_F.state_dict(),
                        'val_recon': val_recon,
                        'epoch': epoch
                    }
                    try:
                        safe_torch_save(best_ckpt, best_metric_path)
                        print(f"\n✅ New best val_recon={val_recon:.6f} saved to {best_metric_path}")
                    except Exception as e:
                        print(f"\n❌ Failed to save best checkpoint: {e}")

        G.train(); F.train(); ema_G.train(); ema_F.train()

    print(f"\n✅ Epoch {epoch} finished in {time.time() - epoch_start:.1f}s -- checkpoint saved to {checkpoint_path} | val_recon={val_recon}")

print("\n✨ Training complete!")
