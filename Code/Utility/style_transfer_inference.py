# style_transfer_inference.py
import torch
import torchvision.transforms as T
from PIL import Image
import numpy as np
import cv2
from pathlib import Path

# Essaie d'utiliser MTCNN pour un recadrage/alignement plus précis
try:
    from facenet_pytorch import MTCNN
    MTCNN_AVAILABLE = True
except Exception:
    MTCNN_AVAILABLE = False

# Importer ton générateur Resnet (fichier models.py fourni précédemment)
try:
    from models import ResnetGenerator
except Exception:
    # Si models.py n'est pas dans le même dossier, adapte le chemin d'import
    raise ImportError("Impossible d'importer ResnetGenerator depuis models.py. Vérifie le chemin.")

class StyleTransferInference:
    def __init__(self,
                 ckpt_path='checkpoints_cyclegan/cyclegan_model.pth',
                 device=None,
                 image_size=256,
                 mtcnn_threshold=0.90):
        """
        ckpt_path: chemin vers le checkpoint CycleGAN contenant 'G' state_dict
        device: 'cuda' ou 'cpu' (None autodétecte)
        image_size: taille carrée utilisée par le réseau (doit correspondre au training)
        """
        self.ckpt_path = Path(ckpt_path)
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.image_size = image_size
        self.mtcnn_threshold = mtcnn_threshold

        # Charger le modèle
        self._load_generator()
        # transforms pour l'entrée du réseau
        self.to_tensor = T.Compose([
            T.Resize((self.image_size, self.image_size)),
            T.ToTensor(),
            T.Normalize([0.5]*3, [0.5]*3)
        ])
        self.to_pil = T.ToPILImage()

        # Initialiser MTCNN si dispo (pour recadrage précis)
        if MTCNN_AVAILABLE:
            try:
                self.mtcnn = MTCNN(keep_all=False, device=self.device)
            except Exception:
                self.mtcnn = None
        else:
            self.mtcnn = None

    def _load_generator(self):
        # Crée le réseau et charge le checkpoint
        self.netG = ResnetGenerator().to(self.device)
        if not self.ckpt_path.exists():
            print(f"[StyleTransferInference] Warning: checkpoint {self.ckpt_path} non trouvé. Le modèle sera laissé en random init.")
            return
        ckpt = torch.load(str(self.ckpt_path), map_location=self.device)
        # ckpt peut contenir 'G' ou être directement le state_dict de G
        if isinstance(ckpt, dict) and 'G' in ckpt:
            state = ckpt['G']
        elif isinstance(ckpt, dict) and all(isinstance(v, torch.Tensor) for v in ckpt.values()):
            state = ckpt
        else:
            # cas divers
            state = ckpt.get('G', ckpt)
        try:
            self.netG.load_state_dict(state)
            print(f"[StyleTransferInference] Checkpoint chargé depuis {self.ckpt_path}")
        except Exception as e:
            print(f"[StyleTransferInference] Erreur chargement checkpoint: {e}")

        self.netG.eval()

    def _preprocess_pil(self, pil_img):
        """Transform PIL->tensor (normalisé)"""
        return self.to_tensor(pil_img).unsqueeze(0).to(self.device)

    def _denormalize_tensor_to_pil(self, tensor):
        """Tensor in [-1,1] -> PIL Image"""
        t = tensor.clone().cpu().squeeze(0)
        t = (t * 0.5) + 0.5
        t = t.clamp(0,1)
        return self.to_pil(t)

    def apply_style_preserve_size(self, cv2_img):
        """
        Prend une image OpenCV (BGR numpy array) d'un visage (n'importe quelle taille),
        applique le style et renvoie une image OpenCV (BGR numpy array) de la même taille.
        Retourne None en cas d'erreur.
        """
        try:
            # Convert BGR -> RGB PIL
            rgb = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)
            pil = Image.fromarray(rgb)

            # Prétraitement: si MTCNN disponible, on l'utilise pour recentrer l'image au besoin.
            # Ici on suppose que l'image fournie est déjà un crop visage (comme dans ton main).
            # Redimensionner/padder pour garder le ratio puis resize au réseau
            h0, w0 = cv2_img.shape[:2]

            # Préparer l'input réseau
            inp = self._preprocess_pil(pil)  # -> tensor 1x3xHxW (H==W==image_size)

            # Inference
            with torch.no_grad():
                out = self.netG(inp.to(self.device))
            pil_out = self._denormalize_tensor_to_pil(out)

            # Resize de la sortie au format d'entrée original
            pil_resized = pil_out.resize((w0, h0), resample=Image.LANCZOS)

            # Convert back to OpenCV BGR
            out_np = np.array(pil_resized)
            out_bgr = cv2.cvtColor(out_np, cv2.COLOR_RGB2BGR)
            return out_bgr

        except Exception as e:
            print(f"[StyleTransferInference] apply_style_preserve_size erreur: {e}")
            import traceback; traceback.print_exc()
            return None

    def apply_style_from_fullframe(self, fullframe_bgr, bbox, pad=0.15):
        """
        Alternative utile: donner une fullframe + bbox (x,y,w,h) pour extraire
        le visage, styliser et replacer dans la fullframe
        """
        try:
            x, y, w, h = bbox
            H, W = fullframe_bgr.shape[:2]
            # Expand bbox
            pad_px = int(max(w, h) * pad)
            x1 = max(0, x - pad_px); y1 = max(0, y - pad_px)
            x2 = min(W, x + w + pad_px); y2 = min(H, y + h + pad_px)
            crop = fullframe_bgr[y1:y2, x1:x2].copy()
            if crop.size == 0:
                return None
            styled = self.apply_style_preserve_size(crop)
            if styled is None:
                return None
            # Replace (use same region)
            out_frame = fullframe_bgr.copy()
            # Resize styled to fit region exactly
            styled_resized = cv2.resize(styled, (x2-x1, y2-y1), interpolation=cv2.INTER_LINEAR)
            out_frame[y1:y2, x1:x2] = styled_resized
            return out_frame
        except Exception as e:
            print(f"[StyleTransferInference] apply_style_from_fullframe erreur: {e}")
            return None
