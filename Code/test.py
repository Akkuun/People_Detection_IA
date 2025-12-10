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

# Instantiate the generator matching the OPetrova-style checkpoint
def load_model_from_checkpoint(model_cls, checkpoint_path: str, device=torch.device('cpu')):
    """Load a model instance from checkpoint robustly.

    - accepts checkpoints that are plain state_dict or dict containing keys like
      'state_dict', 'model_state_dict', 'G', 'generator', etc.
    - strips common prefixes ('module.', 'netG.')
    - filters tensors by matching shapes when needed
    """
    model = model_cls().to(device)
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    loaded = torch.load(checkpoint_path, map_location=device)

    # unwrap common containers
    state = None
    if isinstance(loaded, dict):
        # try common keys
        for key in ('state_dict', 'model_state_dict', 'G', 'generator', 'netG', 'model'):
            if key in loaded:
                state = loaded[key]
                break
        if state is None:
            # maybe the dict is already the state_dict
            # heuristic: check if keys look like parameter names (contain '.')
            if any('.' in k for k in loaded.keys()):
                state = loaded
            else:
                # try to find a nested dict value that is a state_dict
                for v in loaded.values():
                    if isinstance(v, dict) and any('.' in k for k in v.keys()):
                        state = v
                        break
    else:
        # loaded could be a nn.Module instance saved directly
        # or another object that exposes state_dict()
        if hasattr(loaded, 'state_dict') and callable(getattr(loaded, 'state_dict')):
            try:
                state = loaded.state_dict()
            except Exception:
                # fallback: wrap as dict if possible
                state = None
        else:
            state = None

    if state is None:
        raise RuntimeError('Could not find state_dict inside checkpoint')

    # normalize keys: remove common prefixes
    new_state = {}
    for k, v in state.items():
        new_k = k
        if new_k.startswith('module.'):
            new_k = new_k[len('module.'):]
        if new_k.startswith('netG.'):
            new_k = new_k[len('netG.'):]
        new_state[new_k] = v

    # attempt to load full state_dict
    try:
        model.load_state_dict(new_state)
    except Exception as e:
        print('Warning: full load failed:', e)
        # filter compatible tensors
        model_dict = model.state_dict()
        filtered = {}
        for k, v in new_state.items():
            if k in model_dict and v.size() == model_dict[k].size():
                filtered[k] = v
        missing = [k for k in model_dict.keys() if k not in filtered]
        unexpected = [k for k in new_state.keys() if k not in model_dict]
        print(f"Filtered {len(filtered)} tensor(s) that match model. Missing keys: {len(missing)}. Unexpected keys: {len(unexpected)}")
        model_dict.update(filtered)
        model.load_state_dict(model_dict)

    return model


checkpoint_path = os.path.join(os.path.dirname(__file__), 'output2', 'netG_epoch100.pt')
model = load_model_from_checkpoint(network.G, checkpoint_path, device=device)
model.eval()
frontalize(model, 'test_set', 10)