import os
from PIL import Image
import torch
import torchvision.utils as vutils
import torchvision.transforms as transforms
import network

device = torch.device('cpu')

def denorm_tensor(t):
    # suppose que t est dans [-1,1], ramène dans [0,1]
    return (t + 1.0) / 2.0

# Generate frontal images from the test set
def frontalize(model, datapath, mtest):
    profile_files = [f for f in os.listdir(datapath) if f.lower().endswith('.jpg') or f.lower().endswith('.png')]
    profile_files = profile_files[:mtest]

    # transform pour afficher l'image source (0..1)
    to01 = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor()
    ])

    # transform utilisée pour fournir l'entrée au modèle (par ex. si le modèle attend [-1,1])
    model_input = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # -> range [-1,1]
    ])

    originals = []
    generated_imgs = []
    model.eval()
    with torch.no_grad():
        for fname in profile_files:
            img = Image.open(os.path.join(datapath, fname)).convert('RGB')

            # garder l'original pour l'affichage
            orig_t = to01(img).unsqueeze(0).to(device)  # range [0,1]
            originals.append(orig_t)

            # entrée du modèle (normalisée)
            inp_t = model_input(img).unsqueeze(0).to(device)
            # forward
            gen = model(inp_t)  # souvent range [-1,1] selon training
            # diagnostique rapide (min/max)
            print(f"{fname} -> input min/max: {inp_t.min().item():.3f}/{inp_t.max().item():.3f} | gen min/max: {gen.min().item():.3f}/{gen.max().item():.3f}")

            # si le modèle sort dans [-1,1], on le remet dans [0,1]
            gen01 = denorm_tensor(gen).clamp(0, 1)
            generated_imgs.append(gen01)

    originals_cat = torch.cat(originals, dim=0)  # shape (m,3,H,W) in [0,1]
    generated_cat = torch.cat(generated_imgs, dim=0)  # shape (m,3,H,W) in [0,1]

    # sauvegarde : first row = sources, second row = générées
    os.makedirs('output', exist_ok=True)
    # on n'utilise PAS normalize=True pour éviter recolorations inattendues
    vutils.save_image(torch.cat((originals_cat, generated_cat)), 'output/test_fixed.jpg', nrow=mtest, padding=2, normalize=False)
    print("Saved output/test_fixed.jpg")
    return

# Load model (adapter si nécessaire)
# Add UNetGenerator to safe globals in case the checkpoint contains references
try:
    torch.serialization.add_safe_globals([network.UNetGenerator])
except Exception:
    # older torch versions may not have this API or may not need it
    pass

# Instantiate the correct generator matching the checkpoint saved during training
model = network.UNetGenerator().to(device)

# Try to load the state_dict using weights_only=True when available (safer),
# otherwise fall back to the standard call.
checkpoint_path = "output/netG_99.pt"
try:
    state = torch.load(checkpoint_path, map_location=device, weights_only=True)
except TypeError:
    # weights_only arg not supported in this torch version
    state = torch.load(checkpoint_path, map_location=device)

model.load_state_dict(state)
model.eval()
frontalize(model, 'test_set', 3)