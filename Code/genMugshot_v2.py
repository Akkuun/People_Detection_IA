"""
Module pour générer des images frontales à partir de profils
Compatible avec la nouvelle architecture ConditionalUNetGenerator
"""
import os
import sys
from PIL import Image
import torch
import torchvision.transforms as transforms
import cv2
import numpy as np

# Ajouter le chemin du dossier frontalization2 pour importer le nouveau network
script_dir = os.path.dirname(os.path.abspath(__file__))
frontalization_dir = os.path.join(script_dir, "..", "frontalization2")
sys.path.insert(0, frontalization_dir)

try:
    # Importer la NOUVELLE architecture depuis frontalization2
    from network import ConditionalUNetGenerator
    NEW_ARCHITECTURE_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import ConditionalUNetGenerator: {e}")
    NEW_ARCHITECTURE_AVAILABLE = False
    # Fallback vers l'ancienne architecture
    import network
    ConditionalUNetGenerator = None

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🔧 Using device: {device}")

# ========================
# CONFIGURATION DU MODÈLE
# ========================

# IMPORTANT : Modifier ce chemin pour pointer vers ton nouveau modèle entraîné
# Après entraînement, tu auras quelque chose comme :
# /home/mathis/Programming/People_Detection_IA/frontalization2/output/netG_99.pt
DEFAULT_MODEL_PATH = os.path.join(script_dir, "output", "netG_99.pt")

# Variable globale pour le modèle chargé
_loaded_model = None
_model_architecture = None  # 'conditional' ou 'legacy'


def load_model(model_path=None):
    """
    Charge le modèle de frontalization.
    
    Args:
        model_path: Chemin vers le fichier .pt du modèle
                   Si None, utilise DEFAULT_MODEL_PATH
    
    Returns:
        True si chargement réussi, False sinon
    """
    global _loaded_model, _model_architecture
    
    if model_path is None:
        model_path = DEFAULT_MODEL_PATH
    
    if not os.path.exists(model_path):
        print(f"❌ Model file not found: {model_path}")
        return False
    
    try:
        print(f"📦 Loading model from: {model_path}")
        
        # Charger le state_dict
        state_dict = torch.load(model_path, weights_only=False, map_location=device)
        
        # Détecter l'architecture en fonction des clés
        if NEW_ARCHITECTURE_AVAILABLE and 'identity_encoder.encoder.conv1.weight' in state_dict:
            # C'est le nouveau ConditionalUNetGenerator
            print("🆕 Detected ConditionalUNetGenerator (NEW ARCHITECTURE)")
            _loaded_model = ConditionalUNetGenerator()
            _model_architecture = 'conditional'
        elif 'enc1.0.weight' in state_dict:
            # C'est l'ancien UNetGenerator
            print("⚠️  Detected UNetGenerator (LEGACY)")
            if NEW_ARCHITECTURE_AVAILABLE:
                print("   Warning: Using legacy architecture. Consider retraining with new architecture.")
            _loaded_model = network.UNetGenerator()
            _model_architecture = 'legacy'
        else:
            # C'est l'ancien G
            print("⚠️  Detected G (LEGACY)")
            _loaded_model = network.G()
            _model_architecture = 'legacy'
        
        # Charger les poids
        _loaded_model.load_state_dict(state_dict)
        _loaded_model.to(device)
        _loaded_model.eval()
        
        print(f"✅ Model loaded successfully!")
        print(f"   Architecture: {_model_architecture}")
        print(f"   Device: {device}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        import traceback
        traceback.print_exc()
        _loaded_model = None
        _model_architecture = None
        return False


def generate_frontal_from_image(input_image, output_path=None):
    """
    Génère une image frontale à partir d'une image de profil.
    
    Args:
        input_image: Chemin vers l'image (str) ou numpy array (cv2 BGR)
        output_path: Chemin où sauvegarder l'image générée (optionnel)
    
    Returns:
        numpy array (BGR) de l'image générée, ou None si erreur
    """
    global _loaded_model
    
    # Charger le modèle si pas encore fait
    if _loaded_model is None:
        if not load_model():
            return None
    
    try:
        # ========================
        # 1. CHARGER L'IMAGE
        # ========================
        if isinstance(input_image, str):
            # Chemin vers fichier
            if not os.path.exists(input_image):
                print(f"❌ Image file not found: {input_image}")
                return None
            img = Image.open(input_image).convert('RGB')
        elif isinstance(input_image, np.ndarray):
            # Numpy array (OpenCV BGR)
            img = Image.fromarray(cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB))
        else:
            print("❌ Invalid input image format")
            return None
        
        # ========================
        # 2. PREPROCESSING
        # ========================
        transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        
        tensor_img = transform(img).unsqueeze(0).to(device)
        
        # ========================
        # 3. GÉNÉRATION
        # ========================
        with torch.no_grad():
            generated = _loaded_model(tensor_img)
        
        # ========================
        # 4. POSTPROCESSING
        # ========================
        # Convertir le tensor en image
        generated_img = generated.squeeze(0).cpu()
        
        # Dénormaliser de [-1, 1] vers [0, 1]
        generated_img = generated_img * 0.5 + 0.5
        generated_img = generated_img.clamp(0, 1)
        
        # Convertir en numpy array (RGB)
        generated_np = generated_img.permute(1, 2, 0).numpy()
        generated_np = (generated_np * 255).astype(np.uint8)
        
        # Convertir RGB vers BGR pour OpenCV
        generated_bgr = cv2.cvtColor(generated_np, cv2.COLOR_RGB2BGR)
        
        # ========================
        # 5. SAUVEGARDER (optionnel)
        # ========================
        if output_path:
            output_dir = os.path.dirname(output_path)
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
            cv2.imwrite(output_path, generated_bgr)
            print(f"💾 Generated image saved: {output_path}")
        
        return generated_bgr
        
    except Exception as e:
        print(f"❌ Error generating frontal image: {e}")
        import traceback
        traceback.print_exc()
        return None


def batch_generate_frontal(input_folder, output_folder):
    """
    Génère des images frontales pour tous les fichiers d'un dossier.
    
    Args:
        input_folder: Dossier contenant les images de profil
        output_folder: Dossier où sauvegarder les images frontales
    """
    if not os.path.exists(input_folder):
        print(f"❌ Input folder not found: {input_folder}")
        return
    
    os.makedirs(output_folder, exist_ok=True)
    
    # Extensions d'images supportées
    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
    
    # Lister tous les fichiers image
    image_files = [
        f for f in os.listdir(input_folder)
        if os.path.splitext(f.lower())[1] in valid_extensions
    ]
    
    if not image_files:
        print(f"❌ No image files found in {input_folder}")
        return
    
    print(f"📁 Found {len(image_files)} images to process")
    
    success_count = 0
    for i, filename in enumerate(image_files, 1):
        input_path = os.path.join(input_folder, filename)
        output_filename = f"frontal_{filename}"
        output_path = os.path.join(output_folder, output_filename)
        
        print(f"\n[{i}/{len(image_files)}] Processing: {filename}")
        
        result = generate_frontal_from_image(input_path, output_path)
        
        if result is not None:
            success_count += 1
    
    print(f"\n✅ Batch processing complete!")
    print(f"   Successfully generated: {success_count}/{len(image_files)}")


# ========================
# CHARGEMENT AUTOMATIQUE
# ========================
# Charger le modèle au démarrage du module
if os.path.exists(DEFAULT_MODEL_PATH):
    load_model(DEFAULT_MODEL_PATH)
else:
    print(f"⚠️  Default model not found: {DEFAULT_MODEL_PATH}")
    print(f"   Model will be loaded on first use or call load_model(path) manually")


# ========================
# SCRIPT CLI (optionnel)
# ========================
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate frontal face images')
    parser.add_argument('--input', '-i', required=True, help='Input image or folder')
    parser.add_argument('--output', '-o', help='Output path or folder')
    parser.add_argument('--model', '-m', help='Path to model checkpoint (.pt file)')
    parser.add_argument('--batch', action='store_true', help='Batch process folder')
    
    args = parser.parse_args()
    
    # Charger un modèle spécifique si demandé
    if args.model:
        if not load_model(args.model):
            sys.exit(1)
    
    # Mode batch ou single
    if args.batch:
        output_folder = args.output or 'output_frontal'
        batch_generate_frontal(args.input, output_folder)
    else:
        output_path = args.output
        result = generate_frontal_from_image(args.input, output_path)
        
        if result is not None:
            print("✅ Success!")
            if output_path:
                print(f"   Saved to: {output_path}")
        else:
            print("❌ Failed to generate frontal image")
            sys.exit(1)
