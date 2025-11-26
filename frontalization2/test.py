import os
from PIL import Image
import torch
import torchvision.utils as vutils
import torchvision.transforms as transforms
from torch.autograd import Variable
import network

device = 'cpu'

datapath = 'test_set'

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

# Load a pre-trained Pytorch model (mettre ici le model entrainé ) depuis le dossier "pretrained"
torch.serialization.add_safe_globals([network.G])
saved_model = torch.load("output/netG_99.pt", weights_only=False, map_location=torch.device('cpu'))

frontalize(saved_model, datapath, 3)

