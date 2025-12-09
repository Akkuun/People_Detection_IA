import os
from PIL import Image
import torch
import torchvision.utils as vutils
import torchvision.transforms as transforms
from torch.autograd import Variable
import network
import cv2
import numpy as np

device = 'cpu'

datapath = 'test_set'

# Charger le modèle une seule fois au démarrage
torch.serialization.add_safe_globals([network.G, network.UNetGenerator])
# Utiliser le chemin absolu du script pour trouver le modèle
script_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(script_dir, "output2", "netG_epoch100.pt")
try:
    # Charger les poids
    state_dict = torch.load(model_path, weights_only=False, map_location=torch.device('cpu'))
    
    # Déterminer quelle architecture utiliser en fonction des clés du state_dict
    if 'enc1.0.weight' in state_dict:
        # C'est un UNetGenerator
        _loaded_model = network.UNetGenerator()
        print("Detected UNetGenerator architecture")
    else:
        # C'est un G classique
        _loaded_model = network.G()
        print("Detected G architecture")
    
    _loaded_model.load_state_dict(state_dict)
    _loaded_model.eval()
    print(f"Model loaded successfully from: {model_path}")
except Exception as e:
    print(f"Warning: Could not load model from {model_path}: {e}")
    import traceback
    traceback.print_exc()
    _loaded_model = None

def generate_frontal_from_image(input_image_path, output_path=None):
    """
    Génère une image frontale à partir d'une image de profil ou frontale.
    
    Args:
        input_image_path: Chemin vers l'image d'entrée (peut être cv2 array ou chemin)
        output_path: Chemin où sauvegarder l'image générée (optionnel)
    
    Returns:
        numpy array (BGR) de l'image générée, ou None si erreur
    """
    if _loaded_model is None:
        print("Error: Model not loaded")
        return None
    
    try:
        # Charger l'image
        if isinstance(input_image_path, str):
            img = Image.open(input_image_path).convert('RGB')
        elif isinstance(input_image_path, np.ndarray):
            # Convertir de BGR (OpenCV) à RGB (PIL)
            img = Image.fromarray(cv2.cvtColor(input_image_path, cv2.COLOR_BGR2RGB))
        else:
            print("Error: Invalid input image format")
            return None
        
        # Transformation de l'image
        transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        
        tensor_img = transform(img).unsqueeze(0).to(device)
        
        # Génération
        with torch.no_grad():
            generated = _loaded_model(tensor_img)
        
        # Convertir le tensor en image
        generated_img = generated.squeeze(0).cpu()
        # Dénormaliser
        generated_img = generated_img * 0.5 + 0.5
        generated_img = generated_img.clamp(0, 1)
        
        # Convertir en numpy array (RGB)
        generated_np = generated_img.permute(1, 2, 0).numpy()
        generated_np = (generated_np * 255).astype(np.uint8)
        
        # Convertir RGB vers BGR pour OpenCV
        generated_bgr = cv2.cvtColor(generated_np, cv2.COLOR_RGB2BGR)
        
        # Sauvegarder si un chemin est fourni
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            cv2.imwrite(output_path, generated_bgr)
            print(f"Image générée sauvegardée: {output_path}")
        
        return generated_bgr
    
    except Exception as e:
        print(f"Error generating frontal image: {e}")
        import traceback
        traceback.print_exc()
        return None

# Generate frontal images from the test set
def frontalize(model, datapath, mtest):
    # Liste des fichiers images dans le dossier test_set  ewewdd dw
    profile_files = [f for f in os.listdir(datapath) if f.endswith('.jpg') or f.endswith('.png')]
    profile_files = profile_files[:mtest]
    
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor()
    ])
    profiles = []
    generated_imgs = []
    for fname in profile_files:
        img = Image.open(os.path.join(datapath, fname)).convert('RGB')
        tensor_img = transform(img).unsqueeze(0).to(device)
        profiles.append(tensor_img)
        with torch.no_grad():
            gen = model(tensor_img)
            generated_imgs.append(gen)
    # Concaténer les images pour l'affichage
    profiles_cat = torch.cat(profiles)
    generated_cat = torch.cat(generated_imgs)
    # Si tu as les images frontales réelles, charge-les ici, sinon crée un tensor vide
    # frontals_cat = ...
    # Pour l'instant, on sauvegarde juste profils et générées
    os.makedirs('output', exist_ok=True)
    vutils.save_image(torch.cat((profiles_cat, generated_cat)), 'output/test.jpg', nrow=mtest, padding=2, normalize=True)
    return

# Test si exécuté directement
if __name__ == "__main__":
    import sys
    if len(sys.argv) > 2 and sys.argv[1] == '--image':
        # Appelé depuis l'interface avec une image spécifique
        image_path = sys.argv[2]
        output_path = image_path.replace('.jpg', '_frontal.jpg')
        generate_frontal_from_image(image_path, output_path)
    elif _loaded_model is not None:
        # Test avec le dossier test_set
        frontalize(_loaded_model, datapath, 3)