"""
CycleGAN Inference
Applique le style ONOT aux images CelebA capturées
"""

import os
import cv2
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from pathlib import Path

class CycleGANInference:
    """Classe pour appliquer le style transfer CycleGAN"""
    
    def __init__(self, checkpoint_path=None, device=None, image_size=256):
        """
        Initialiser le modèle CycleGAN
        
        Args:
            checkpoint_path: Chemin vers le checkpoint du modèle entraîné (optionnel, cherche automatiquement)
            device: GPU ou CPU
            image_size: Taille des images (256)
        """
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        
        self.checkpoint_path = checkpoint_path
        self.image_size = image_size
        self.G = None  # Générateur A->B (CelebA -> ONOT style)
        self.transform = T.Compose([
            T.Resize((image_size, image_size)),
            T.ToTensor(),
            T.Normalize([0.5]*3, [0.5]*3)
        ])
        self.inverse_transform = T.Compose([
            T.Normalize([-1]*3, [2]*3),  # Inverse la normalisation
        ])
        
        # Charger le modèle : d'abord le checkpoint fourni, sinon chercher le dernier
        if checkpoint_path and os.path.exists(checkpoint_path):
            self._load_checkpoint(checkpoint_path)
        else:
            # Chercher le dernier checkpoint disponible
            if not self.find_latest_checkpoint('checkpoints_cyclegan'):
                print(f"⚠️  Aucun checkpoint trouvé. Entraînez le modèle avec: python3 train.py")
    
    def _load_checkpoint(self, checkpoint_path):
        """Charger le générateur depuis un checkpoint"""
        try:
            from model import ResnetGenerator
            
            # Charger le checkpoint
            try:
                checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
            except TypeError:
                # Fallback pour les versions anciennes de torch
                checkpoint = torch.load(checkpoint_path, map_location=self.device)
            
            # Initialiser le générateur
            self.G = ResnetGenerator().to(self.device)
            
            # Charger les poids - supporter plusieurs formats
            loaded = False
            
            # Format 1: 'G' (depuis train.py)
            if 'G' in checkpoint:
                self.G.load_state_dict(checkpoint['G'])
                loaded = True
                print(f"✅ Poids G chargés depuis checkpoint")
            
            if not loaded:
                print(f"⚠️  Format de checkpoint non reconnu. Clés disponibles: {list(checkpoint.keys())}")
                self.G = None
                return
            
            self.G.eval()
            
            epoch = checkpoint.get('epoch', 'unknown')
            print(f"✅ Modèle CycleGAN chargé depuis {checkpoint_path} (Epoch {epoch})")
            
        except Exception as e:
            print(f"❌ Erreur chargement checkpoint: {e}")
            import traceback
            traceback.print_exc()
            self.G = None
    
    def find_latest_checkpoint(self, checkpoint_dir='checkpoints_cyclegan'):
        """Chercher et charger le checkpoint CycleGAN unique"""
        checkpoint_dir_path = Path(checkpoint_dir)
        
        # Chercher dans le répertoire courant et le répertoire Code
        possible_dirs = [
            checkpoint_dir_path,
            Path('/home/paradox/Bureau/M2/ProjetImage/Code') / checkpoint_dir,
            Path.cwd() / checkpoint_dir,
            Path(__file__).parent / checkpoint_dir  # Relatif au script
        ]
        
        for ckpt_dir in possible_dirs:
            ckpt_dir = ckpt_dir.resolve()  # Résoudre les chemins relatifs
            if not ckpt_dir.exists():
                continue
            
            checkpoint_path = ckpt_dir / 'cyclegan_model.pth'
            if checkpoint_path.exists():
                print(f"📂 Checkpoint trouvé: {checkpoint_path}")
                self._load_checkpoint(str(checkpoint_path))
                return self.G is not None
        
        print(f"❌ Aucun checkpoint trouvé: checkpoints_cyclegan/cyclegan_model.pth")
        print(f"   Entraînez d'abord le modèle avec: python3 train.py")
        return False
    
    def apply_style(self, image, return_original=False):
        """
        Appliquer le style transfer CycleGAN à une image
        
        Args:
            image: Image numpy (BGR) ou chemin vers l'image
            return_original: Si True, retourne aussi l'original
        
        Returns:
            Image stylisée (numpy, BGR) ou tuple (original, stylisé) si return_original=True
        """
        if self.G is None:
            print("❌ Modèle non chargé")
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
            
            # Inférence
            with torch.no_grad():
                fake_B = self.G(img_tensor)
            
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
            print(f"❌ Erreur lors du style transfer: {e}")
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
    """Démonstration du CycleGAN inference"""
    
    inference = CycleGANInference(image_size=256)
    
    # Essayer de charger le dernier checkpoint
    if not inference.find_latest_checkpoint('checkpoints_cyclegan'):
        print("❌ Pas de modèle entraîné. Entraînez d'abord avec: python3 train.py")
        return
    
    # Tester sur quelques images CelebA
    celeba_dir = '/home/paradox/Bureau/M2/ProjetImage/dataset/CelebA'
    test_images = list(Path(celeba_dir).glob('**/*.jpg'))[:5]
    
    os.makedirs('styled_outputs', exist_ok=True)
    
    print(f"\n🎨 Stylisation de {len(test_images)} images...")
    for img_path in test_images:
        print(f"Traitement {img_path.name}...")
        
        # Appliquer le style
        original, styled = inference.apply_style(str(img_path), return_original=True)
        
        if styled is not None:
            # Combiner original et stylisé
            combined = np.hstack([original, styled])
            
            # Sauvegarder
            output_path = f'styled_outputs/{img_path.stem}_stylized.jpg'
            cv2.imwrite(output_path, combined)
            print(f"✅ Sauvegardé: {output_path}")


if __name__ == '__main__':
    demo()
