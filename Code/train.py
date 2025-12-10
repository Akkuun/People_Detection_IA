"""
Face Frontalization Training - Version Simple basée sur le README
Adapté pour le dataset iDoc Mugshots (front/front et side/side)
"""
from __future__ import print_function
import time
import os
from PIL import Image

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as vutils
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader

from network import G, D, weights_init
import argparse

np.random.seed(42)
torch.manual_seed(999)

# ========================
# DATASET
# ========================

class MugshotPairDataset(Dataset):
    """
    Dataset pour le frontalization GAN :
    Version 1 (archive/) : profils dans side/side, frontales dans front/front
    Version 2 (README) : ID.jpg (frontal) + ID/profile.jpg (profil)
    Auto-détecte la structure
    """

    def __init__(self, base_dir, image_size=128, max_samples=None):
        self.base_dir = base_dir
        self.image_size = image_size
        
        # Auto-détecter la structure
        if os.path.exists(os.path.join(base_dir, "front/front")):
            # Structure archive/
            self._load_archive_structure(max_samples)
        else:
            # Structure README
            self._load_readme_structure(max_samples)
        
        if len(self.pairs) == 0:
            raise RuntimeError("❌ Aucune paire d'images trouvée")
        
        print(f"📦 Dataset chargé : {len(self.pairs)} paires d'images")
        
        # Transformations : resize, to tensor, normalize [-1, 1]
        self.transform = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    
    def _load_archive_structure(self, max_samples):
        """Charge structure archive/front/front et side/side"""
        self.front_dir = os.path.join(self.base_dir, "front/front")
        self.side_dir  = os.path.join(self.base_dir, "side/side")
        
        front_files = set(os.listdir(self.front_dir))
        side_files = set(os.listdir(self.side_dir))
        common_ids = sorted(list(front_files.intersection(side_files)))
        
        if max_samples and max_samples < len(common_ids):
            rng = np.random.RandomState(42)
            common_ids = sorted(rng.choice(common_ids, size=max_samples, replace=False))
        
        self.pairs = [
            (os.path.join(self.side_dir, img_id), os.path.join(self.front_dir, img_id))
            for img_id in common_ids
        ]
        print(f"📂 Structure détectée: archive/ (front/front + side/side)")
    
    def _load_readme_structure(self, max_samples):
        """Charge structure README : ID.jpg + ID/profile.jpg"""
        # Lister tous les fichiers .jpg à la racine (frontaux) - inclut les liens symboliques
        all_items = os.listdir(self.base_dir)
        frontal_files = [f for f in all_items
                        if f.endswith('.jpg') and 
                        (os.path.isfile(os.path.join(self.base_dir, f)) or 
                         os.path.islink(os.path.join(self.base_dir, f)))]
        
        pairs = []
        for frontal_file in frontal_files:
            # ID = nom sans extension
            subject_id = frontal_file[:-4]  # Enlever .jpg
            subject_dir = os.path.join(self.base_dir, subject_id)
            
            # Vérifier que le dossier existe
            if os.path.isdir(subject_dir):
                # Trouver toutes les images de profil dans ce dossier
                profile_files = [f for f in os.listdir(subject_dir) if f.endswith('.jpg')]
                
                frontal_path = os.path.join(self.base_dir, frontal_file)
                for profile_file in profile_files:
                    profile_path = os.path.join(subject_dir, profile_file)
                    pairs.append((profile_path, frontal_path))
        
        if max_samples and max_samples < len(pairs):
            rng = np.random.RandomState(42)
            indices = rng.choice(len(pairs), size=max_samples, replace=False)
            pairs = [pairs[i] for i in indices]
        
        self.pairs = pairs
        print(f"📂 Structure détectée: README (ID.jpg + ID/profile.jpg)")
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        profile_path, frontal_path = self.pairs[idx]
        
        try:
            # Charger les images
            front_img = Image.open(frontal_path).convert('RGB')
            side_img = Image.open(profile_path).convert('RGB')
            
            # Appliquer les transformations
            front_tensor = self.transform(front_img)
            side_tensor = self.transform(side_img)
            
            return side_tensor, front_tensor
            
        except Exception as e:
            print(f"⚠️  Erreur sur paire {idx} ({profile_path}): {e}")
            # En cas d'erreur, retourner un autre élément
            return self.__getitem__((idx + 1) % len(self))


# ========================
# MAIN
# ========================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Face frontalization training')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size')
    parser.add_argument('--max-samples', type=int, default=None, help='Limit dataset size')
    parser.add_argument('--lr-g', type=float, default=0.0002, help='Learning rate for Generator')
    parser.add_argument('--lr-d', type=float, default=0.0002, help='Learning rate for Discriminator')
    parser.add_argument('--L1-factor', type=float, default=1.0, help='L1 loss weight')
    parser.add_argument('--L2-factor', type=float, default=1.0, help='L2 loss weight')
    parser.add_argument('--GAN-factor', type=float, default=0.001, help='GAN loss weight')
    parser.add_argument('--output-dir', type=str, default='output', help='Output directory')
    parser.add_argument('--datapath', type=str, default='training_set/archive', help='Dataset path')
    parser.add_argument('--load-checkpoint-g', type=str, default=None, help='Path to Generator checkpoint to resume from')
    parser.add_argument('--load-checkpoint-d', type=str, default=None, help='Path to Discriminator checkpoint to resume from')
    args = parser.parse_args()

    # Paramètres
    num_epochs = args.epochs
    batch_size = args.batch_size
    datapath = args.datapath

    # Device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Using device: {device}")

    # Dataset & DataLoader
    dataset = MugshotPairDataset(datapath, image_size=128, max_samples=args.max_samples)
    train_loader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=True,  # Shuffle pour un meilleur entraînement
        num_workers=2, 
        pin_memory=True if torch.cuda.is_available() else False
    )

    print(f"📊 Total batches per epoch: {len(train_loader)}")

    # ========================
    # MODELS
    # ========================
    # Générateur (architecture classique G du README)
    netG = G().to(device)
    if args.load_checkpoint_g:
        print(f"📥 Loading Generator from {args.load_checkpoint_g}")
        netG.load_state_dict(torch.load(args.load_checkpoint_g, map_location=device))
    else:
        netG.apply(weights_init)

    # Discriminateur
    netD = D().to(device)
    if args.load_checkpoint_d:
        print(f"📥 Loading Discriminator from {args.load_checkpoint_d}")
        netD.load_state_dict(torch.load(args.load_checkpoint_d, map_location=device))
    else:
        netD.apply(weights_init)

    print("✅ Models initialized")

    # ========================
    # LOSS & OPTIMIZERS
    # ========================
    # Poids des pertes (comme dans le README/pretrained)
    L1_factor = args.L1_factor
    L2_factor = args.L2_factor
    GAN_factor = args.GAN_factor

    print(f"📝 Loss weights: L1={L1_factor}, L2={L2_factor}, GAN={GAN_factor}")

    # Loss functions
    # Utiliser BCEWithLogitsLoss car D ne termine pas par Sigmoid
    criterion_BCE = nn.BCEWithLogitsLoss()
    criterion_L1 = nn.L1Loss()
    criterion_L2 = nn.MSELoss()

    # Optimizers (comme dans le README)
    optimizerD = optim.Adam(netD.parameters(), lr=args.lr_d, betas=(0.5, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=args.lr_g, betas=(0.5, 0.999))

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Labels pour le GAN
    real_label = 1
    fake_label = 0

    # ========================
    # TRAINING LOOP
    # ========================
    start_time = time.time()

    print("\n🎯 Starting training...")
    print("="*60)

    # Historique des pertes
    loss_history = {
        'D': [], 'G': [], 'G_L1': [], 'G_L2': [], 'G_GAN': []
    }

    for epoch in range(num_epochs):
        epoch_start = time.time()
        
        # Accumulateurs pour les pertes
        running_lossD = 0.0
        running_lossG = 0.0
        running_lossG_L1 = 0.0
        running_lossG_L2 = 0.0
        running_lossG_GAN = 0.0
        
        for i, (profiles, frontals) in enumerate(train_loader):
            batch_size_current = profiles.size(0)
            profiles = profiles.to(device)
            frontals = frontals.to(device)
            
            # Générer les images une seule fois
            generated = netG(profiles)
            
            # ========================
            # (1) Update D network (seulement si GAN_factor > 0)
            # ========================
            if GAN_factor > 0:
                netD.zero_grad()
                
                # Train with real images
                label_real = torch.full((batch_size_current,), real_label, dtype=torch.float, device=device)
                output_real = netD(frontals).view(-1)
                errD_real = criterion_BCE(output_real, label_real)
                errD_real.backward()
                
                # Train with fake images
                label_fake = torch.full((batch_size_current,), fake_label, dtype=torch.float, device=device)
                output_fake = netD(generated.detach()).view(-1)
                errD_fake = criterion_BCE(output_fake, label_fake)
                errD_fake.backward()
                
                errD = errD_real + errD_fake
                optimizerD.step()
            else:
                # Si GAN désactivé, mettre errD à 0
                errD = torch.tensor(0.0, device=device)
            
            # ========================
            # (2) Update G network
            # ========================
            netG.zero_grad()
            
            # GAN loss (seulement si GAN_factor > 0)
            if GAN_factor > 0:
                output_gen = netD(generated).view(-1)
                label_real_g = torch.full((batch_size_current,), real_label, dtype=torch.float, device=device)
                errG_GAN = criterion_BCE(output_gen, label_real_g)
            else:
                errG_GAN = torch.tensor(0.0, device=device)
            
            # L1 loss
            errG_L1 = criterion_L1(generated, frontals)
            
            # L2 loss
            errG_L2 = criterion_L2(generated, frontals)
            
            # Total generator loss
            errG = GAN_factor * errG_GAN + L1_factor * errG_L1 + L2_factor * errG_L2
            errG.backward()
            optimizerG.step()
            
            # Accumuler les pertes
            running_lossD += errD.item()
            running_lossG += errG.item()
            running_lossG_L1 += errG_L1.item()
            running_lossG_L2 += errG_L2.item()
            running_lossG_GAN += errG_GAN.item()
            
            # Log toutes les 50 batches
            if i % 50 == 0:
                print(f'[{epoch+1}/{num_epochs}][{i}/{len(train_loader)}] '
                      f'Loss_D: {errD.item():.4f} Loss_G: {errG.item():.4f} '
                      f'(L1: {errG_L1.item():.4f}, L2: {errG_L2.item():.4f}, GAN: {errG_GAN.item():.4f})')
        
        # Moyennes des pertes pour l'epoch
        n_batches = len(train_loader)
        avg_lossD = running_lossD / n_batches
        avg_lossG = running_lossG / n_batches
        avg_lossG_L1 = running_lossG_L1 / n_batches
        avg_lossG_L2 = running_lossG_L2 / n_batches
        avg_lossG_GAN = running_lossG_GAN / n_batches
        
        # Sauvegarder l'historique
        loss_history['D'].append(avg_lossD)
        loss_history['G'].append(avg_lossG)
        loss_history['G_L1'].append(avg_lossG_L1)
        loss_history['G_L2'].append(avg_lossG_L2)
        loss_history['G_GAN'].append(avg_lossG_GAN)
        
        epoch_time = time.time() - epoch_start
        
        # Résumé de l'epoch
        print(f'{"="*60}')
        print(f'Epoch [{epoch+1}/{num_epochs}] completed in {epoch_time:.1f}s')
        print(f'Average Loss_D: {avg_lossD:.4f} | Loss_G: {avg_lossG:.4f}')
        print(f'  ↳ L1: {avg_lossG_L1:.4f}, L2: {avg_lossG_L2:.4f}, GAN: {avg_lossG_GAN:.4f}')
        print(f'{"="*60}\n')
        
        # ========================
        # SAVE OUTPUTS
        # ========================
        # Sauvegarder à chaque epoch
        if True:  # Sauvegarde à chaque epoch
            # Denormaliser pour la visualisation
            def denorm(x):
                return (x + 1.0) / 2.0
            
            # Sauvegarder des exemples d'images
            with torch.no_grad():
                # Prendre le dernier batch pour visualisation
                sample_profiles = profiles[:8]
                sample_frontals = frontals[:8]
                sample_generated = netG(sample_profiles)
                
                vutils.save_image(denorm(sample_profiles), 
                                f'{args.output_dir}/epoch_{epoch+1:03d}_profiles.jpg', 
                                nrow=4, padding=2)
                vutils.save_image(denorm(sample_frontals), 
                                f'{args.output_dir}/epoch_{epoch+1:03d}_frontals.jpg', 
                                nrow=4, padding=2)
                vutils.save_image(denorm(sample_generated), 
                                f'{args.output_dir}/epoch_{epoch+1:03d}_generated.jpg', 
                                nrow=4, padding=2)
            
            # Sauvegarder le modèle
            torch.save(netG.state_dict(), f'{args.output_dir}/netG_epoch{epoch+1:03d}.pt')
            torch.save(netD.state_dict(), f'{args.output_dir}/netD_epoch{epoch+1:03d}.pt')
            print(f'💾 Saved model and images for epoch {epoch+1}')

    # ========================
    # FINAL SAVE
    # ========================
    # Sauvegarder le modèle final
    torch.save(netG.state_dict(), f'{args.output_dir}/netG_final.pt')
    torch.save(netD.state_dict(), f'{args.output_dir}/netD_final.pt')

    # Sauvegarder l'historique des pertes
    np.save(f'{args.output_dir}/loss_history.npy', loss_history)

    total_time = time.time() - start_time
    print(f"\n✨ Training complete! Total time: {total_time/3600:.2f} hours")
    print(f"📁 Models and outputs saved in '{args.output_dir}/'")

    # ========================
    # PLOT LOSS CURVES
    # ========================
    try:
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(12, 5))
        
        # Plot 1: Discriminator and Generator losses
        plt.subplot(1, 2, 1)
        plt.plot(loss_history['D'], label='Discriminator', linewidth=2)
        plt.plot(loss_history['G'], label='Generator', linewidth=2)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('D and G Losses')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot 2: Generator loss components
        plt.subplot(1, 2, 2)
        plt.plot(loss_history['G_L1'], label='L1', linewidth=2)
        plt.plot(loss_history['G_L2'], label='L2', linewidth=2)
        plt.plot(loss_history['G_GAN'], label='GAN', linewidth=2)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Generator Loss Components')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{args.output_dir}/training_curves.png', dpi=150)
        print(f"📊 Loss curves saved to '{args.output_dir}/training_curves.png'")
        
    except ImportError:
        print("⚠️  Matplotlib not available, skipping plots")
