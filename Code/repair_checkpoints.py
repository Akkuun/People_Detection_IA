#!/usr/bin/env python3
"""
Script de réparation des checkpoints corrompus
Extrait les poids du générateur et crée de nouveaux checkpoints
"""

import torch
import sys
from pathlib import Path

def repair_vaegan_checkpoint(checkpoint_path):
    """Réparer un checkpoint VAE-GAN en extrayant les poids G/ema_G"""
    print(f"\n🔧 Réparation {checkpoint_path}...")
    
    try:
        # Essayer de charger le checkpoint (peut être corrompu)
        try:
            ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        except Exception as e:
            print(f"   ⚠️  Checkpoint corrompu: {e}")
            print(f"   Impossible de le réparer directement")
            return False
        
        # Créer un nouveau checkpoint compact
        mini_ckpt = {}
        
        # Copier les poids du générateur
        if 'G' in ckpt:
            mini_ckpt['G'] = ckpt['G']
            print(f"   ✅ Poids G extraits")
        
        if 'ema_G' in ckpt:
            mini_ckpt['ema_G'] = ckpt['ema_G']
            print(f"   ✅ Poids ema_G extraits")
        
        if 'F' in ckpt:
            mini_ckpt['F'] = ckpt['F']
            print(f"   ✅ Poids F extraits")
        
        if 'ema_F' in ckpt:
            mini_ckpt['ema_F'] = ckpt['ema_F']
            print(f"   ✅ Poids ema_F extraits")
        
        # Copier les métadonnées
        if 'epoch' in ckpt:
            mini_ckpt['epoch'] = ckpt['epoch']
        if 'val_recon' in ckpt:
            mini_ckpt['val_recon'] = ckpt['val_recon']
        
        if not mini_ckpt:
            print(f"   ❌ Aucun poids trouvé dans le checkpoint")
            return False
        
        # Sauvegarder le nouveau checkpoint compact
        new_path = checkpoint_path.with_suffix(checkpoint_path.suffix + '.repaired')
        torch.save(mini_ckpt, str(new_path))
        
        # Remplacer l'ancien par le nouveau
        checkpoint_path.unlink()
        new_path.replace(checkpoint_path)
        
        print(f"   ✅ Checkpoint réparé et sauvegardé ({len(mini_ckpt)} clés)")
        return True
        
    except Exception as e:
        print(f"   ❌ Erreur réparation: {e}")
        import traceback
        traceback.print_exc()
        return False


def repair_cyclegan_checkpoint(checkpoint_path):
    """Réparer un checkpoint CycleGAN en extrayant les poids G/D"""
    print(f"\n🔧 Réparation {checkpoint_path}...")
    
    try:
        # Essayer de charger le checkpoint
        try:
            ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        except Exception as e:
            print(f"   ⚠️  Checkpoint corrompu: {e}")
            print(f"   Impossible de le réparer directement")
            return False
        
        # Créer un nouveau checkpoint compact
        mini_ckpt = {}
        
        # Copier les poids importants (seulement G pour inférence)
        if 'G' in ckpt:
            mini_ckpt['G'] = ckpt['G']
            print(f"   ✅ Poids G (générateur CelebA->ONOT) extraits")
        
        if 'F' in ckpt:
            mini_ckpt['F'] = ckpt['F']
            print(f"   ✅ Poids F (générateur ONOT->CelebA) extraits")
        
        # Copier les métadonnées
        if 'epoch' in ckpt:
            mini_ckpt['epoch'] = ckpt['epoch']
        
        if not mini_ckpt:
            print(f"   ❌ Aucun poids trouvé dans le checkpoint")
            return False
        
        # Sauvegarder le nouveau checkpoint compact
        new_path = checkpoint_path.with_suffix(checkpoint_path.suffix + '.repaired')
        torch.save(mini_ckpt, str(new_path))
        
        # Remplacer l'ancien par le nouveau
        checkpoint_path.unlink()
        new_path.replace(checkpoint_path)
        
        print(f"   ✅ Checkpoint réparé et sauvegardé ({len(mini_ckpt)} clés)")
        return True
        
    except Exception as e:
        print(f"   ❌ Erreur réparation: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Réparer tous les checkpoints disponibles"""
    
    print("=" * 60)
    print("🔧 Réparation des Checkpoints")
    print("=" * 60)
    
    code_dir = Path('/home/paradox/Bureau/M2/ProjetImage/Code')
    
    # Réparer VAE-GAN checkpoints
    vaegan_dir = code_dir / 'checkpoints_vaegan'
    if vaegan_dir.exists():
        print(f"\n📁 Dossier VAE-GAN: {vaegan_dir}")
        for ckpt_file in vaegan_dir.glob('*.pth'):
            repair_vaegan_checkpoint(ckpt_file)
    
    # Réparer CycleGAN checkpoints
    cyclegan_dir = code_dir / 'checkpoints_cyclegan'
    if cyclegan_dir.exists():
        print(f"\n📁 Dossier CycleGAN: {cyclegan_dir}")
        for ckpt_file in cyclegan_dir.glob('*.pth'):
            repair_cyclegan_checkpoint(ckpt_file)
    
    print("\n" + "=" * 60)
    print("✅ Réparation terminée!")
    print("=" * 60)


if __name__ == '__main__':
    main()
