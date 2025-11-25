"""
VAE-GAN Inference
Applique le style ONOT aux images CelebA capturées en utilisant VAE-GAN
"""

import os
import cv2
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from pathlib import Path

class VAEGANInference:
    """Classe pour appliquer le style transfer VAE-GAN"""
    
    def __init__(self, checkpoint_path=None, device=None, image_size=128):
        """
        Initialiser le modèle VAE-GAN
        
        Args:
            checkpoint_path: Chemin vers le checkpoint du modèle entraîné (optionnel, cherche automatiquement)
            device: GPU ou CPU
            image_size: Taille des images (128)
        """
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        
        self.checkpoint_path = checkpoint_path
        self.image_size = image_size
        self.encoder = None  # Encodeur VAE
        self.decoder = None  # Décodeur VAE
        self.transform = T.Compose([
            T.Resize((image_size, image_size)),
            T.ToTensor(),
            T.Normalize([0.5]*3, [0.5]*3)
        ])
        
        # Charger le modèle : d'abord le checkpoint fourni, sinon chercher le checkpoint
        if checkpoint_path and os.path.exists(checkpoint_path):
            self._load_checkpoint(checkpoint_path)
        else:
            # Chercher le checkpoint disponible
            if not self.find_latest_checkpoint('checkpoints_vaegan'):
                print(f"⚠️  Aucun checkpoint VAE-GAN trouvé. Entraînez le modèle avec: python3 train.py")
    
    def _load_checkpoint(self, checkpoint_path):
        """Charger l'encodeur et décodeur depuis un checkpoint"""
        try:
            from vae_generator import VAEGenerator
            
            # Charger le checkpoint
            try:
                checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
            except TypeError:
                # Fallback pour les versions anciennes de torch
                checkpoint = torch.load(checkpoint_path, map_location=self.device)
            
            # Initialiser le VAE-GAN
            self.vae_generator = VAEGenerator().to(self.device)
            
            # Charger les poids - supporter plusieurs formats
            loaded = False
            
            # Format 1: 'G' (depuis trainVAEGAN.py avec ema_G pour inférence)
            if 'G' in checkpoint:
                self.vae_generator.load_state_dict(checkpoint['G'])
                loaded = True
                print(f"✅ Poids G chargés depuis checkpoint")
            
            # Format 2: 'ema_G' (EMA weights for better quality)
            elif 'ema_G' in checkpoint:
                self.vae_generator.load_state_dict(checkpoint['ema_G'])
                loaded = True
                print(f"✅ Poids ema_G chargés depuis checkpoint (meilleure qualité)")
            
            # Format 3: 'generator' (ancien format)
            elif 'generator' in checkpoint:
                self.vae_generator.load_state_dict(checkpoint['generator'])
                loaded = True
                print(f"✅ Poids generator chargés depuis checkpoint")
            
            if not loaded:
                print(f"⚠️  Format de checkpoint non reconnu. Clés disponibles: {list(checkpoint.keys())}")
                self.vae_generator = None
                return
            
            self.vae_generator.eval()
            
            epoch = checkpoint.get('epoch', 'unknown')
            print(f"✅ Modèle VAE-GAN chargé depuis {checkpoint_path} (Epoch {epoch})")
            
        except Exception as e:
            print(f"❌ Erreur chargement checkpoint VAE-GAN: {e}")
            import traceback
            traceback.print_exc()
            self.vae_generator = None
    
    def find_latest_checkpoint(self, checkpoint_dir='checkpoints_vaegan'):
        """Chercher et charger le checkpoint VAE-GAN (préfère best si disponible)"""
        checkpoint_dir_path = Path(checkpoint_dir)
        
        # Chercher dans le répertoire courant et le répertoire Code
        possible_dirs = [
            checkpoint_dir_path,
            Path('/home/paradox/Bureau/M2/ProjetImage/Code') / checkpoint_dir,
            Path.cwd() / checkpoint_dir
        ]
        
        for ckpt_dir in possible_dirs:
            if not ckpt_dir.exists():
                continue
            
            # Préférer le meilleur checkpoint
            best_checkpoint_path = ckpt_dir / 'best_vaegan_model.pth'
            if best_checkpoint_path.exists():
                print(f"📂 Meilleur checkpoint VAE-GAN trouvé: {best_checkpoint_path}")
                self._load_checkpoint(str(best_checkpoint_path))
                return self.vae_generator is not None
            
            # Fallback sur le checkpoint courant
            checkpoint_path = ckpt_dir / 'vaegan_model.pth'
            if checkpoint_path.exists():
                print(f"📂 Checkpoint VAE-GAN trouvé: {checkpoint_path}")
                self._load_checkpoint(str(checkpoint_path))
                return self.vae_generator is not None
        
        print(f"❌ Aucun checkpoint VAE-GAN trouvé: checkpoints_vaegan/vaegan_model.pth")
        print(f"   Entraînez d'abord le modèle avec: python3 trainVAEGAN.py")
        return False
    
    @property
    def G(self):
        """Propriété pour compatibilité avec le code existant"""
        return self.vae_generator
    
    def apply_style(self, image, return_original=False):
        """
        Appliquer le style transfer VAE-GAN à une image
        
        Args:
            image: Image numpy (BGR) ou chemin vers l'image
            return_original: Si True, retourne aussi l'original
        
        Returns:
            Image stylisée (numpy, BGR) ou tuple (original, stylisé) si return_original=True
        """
        if self.vae_generator is None:
            print("❌ Modèle VAE-GAN non chargé")
            return None
        
        try:
            # Charger l'image si c'est un chemin
            if isinstance(image, str):
                img = cv2.imread(image)
                if img is None:
                    print(f"❌ Impossible de charger {image}")
                    return None
            else:
                img = image.copy()
            
            original_size = img.shape[:2]
            
            # Redimensionner
            img_resized = cv2.resize(img, (self.image_size, self.image_size))
            img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
            
            # Convertir en tensor
            img_pil = Image.fromarray(img_rgb.astype('uint8'))
            img_tensor = self.transform(img_pil).unsqueeze(0).to(self.device)
            
            # Inférence avec VAE-GAN
            with torch.no_grad():
                output = self.vae_generator(img_tensor)
            
            # Gérer le cas où le modèle retourne un tuple (mu, logvar, reconstruction)
            if isinstance(output, tuple):
                fake_B = output[0]  # Prendre la reconstruction (premier élément)
            else:
                fake_B = output
            
            # Dénormaliser et convertir en numpy
            fake_B = fake_B.squeeze(0).cpu()
            fake_B = (fake_B * 0.5 + 0.5).clamp(0, 1)
            fake_B_np = (fake_B.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            
            # Convertir RGB -> BGR
            fake_B_bgr = cv2.cvtColor(fake_B_np, cv2.COLOR_RGB2BGR)
            
            # Redimensionner à la taille originale
            result = cv2.resize(fake_B_bgr, (original_size[1], original_size[0]))
            
            if return_original:
                return img, result
            return result
        
        except Exception as e:
            print(f"❌ Erreur lors du style transfer VAE-GAN: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def apply_style_batch(self, image_list):
        """
        Appliquer le style à plusieurs images
        
        Args:
            image_list: Liste d'images ou de chemins
        
        Returns:
            Liste d'images stylisées
        """
        results = []
        for img in image_list:
            styled = self.apply_style(img)
            if styled is not None:
                results.append(styled)
        return results


# Fonction de démonstration
def demo():
    """Démonstration du VAE-GAN inference"""
    
    inference = VAEGANInference(image_size=128)
    
    # Essayer de charger le checkpoint
    if inference.G is None:
        print("❌ Pas de modèle VAE-GAN entraîné. Entraînez d'abord avec: python3 train.py")
        return
    
    # Tester sur quelques images CelebA
    celeba_dir = '/home/paradox/Bureau/M2/ProjetImage/dataset/CelebA'
    test_images = list(Path(celeba_dir).glob('**/*.jpg'))[:5]
    
    os.makedirs('styled_outputs', exist_ok=True)
    
    print(f"\n🎨 Stylisation VAE-GAN de {len(test_images)} images...")
    for img_path in test_images:
        print(f"Traitement {img_path.name}...")
        
        # Appliquer le style
        original, styled = inference.apply_style(str(img_path), return_original=True)
        
        if styled is not None:
            # Combiner original et stylisé
            combined = np.hstack([original, styled])
            
            # Sauvegarder
            output_path = f'styled_outputs/{img_path.stem}_vaegan_stylized.jpg'
            cv2.imwrite(output_path, combined)
            print(f"✅ Sauvegardé: {output_path}")


if __name__ == '__main__':
    demo()
