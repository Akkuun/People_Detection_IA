#!/usr/bin/env python3
"""
Evaluate generated outputs: compute LPIPS and identity distance (FaceNet/ArcFace),
and optionally FID (if torchmetrics available).

Usage example:
  python3 scripts/evaluate_outputs.py \
    --profiles_dir test_set \
    --fronts_dir training_set/front/front \
    --checkpoint output/netG_99.pt \
    --out_dir output/eval_gen --num_samples 50
"""
import os
import argparse
from PIL import Image
import torch
import torchvision.transforms as transforms
import torchvision.utils as vutils
import network
import csv

def build_transforms(image_size):
    model_input = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    to01 = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor()
    ])
    return model_input, to01

def load_model(checkpoint_path, device):
    model = network.UNetGenerator().to(device)
    try:
        state = torch.load(checkpoint_path, map_location=device, weights_only=True)
    except TypeError:
        state = torch.load(checkpoint_path, map_location=device)
    try:
        model.load_state_dict(state)
    except Exception:
        # Maybe checkpoint contains dict with 'state_dict'
        if isinstance(state, dict) and 'state_dict' in state:
            model.load_state_dict(state['state_dict'])
        else:
            raise
    model.eval()
    return model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--profiles_dir', required=True)
    parser.add_argument('--fronts_dir', default=None, help='Directory with ground-truth frontal images (optional)')
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--out_dir', default='output/eval_gen')
    parser.add_argument('--num_samples', type=int, default=50)
    parser.add_argument('--image_size', type=int, default=128)
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    model = load_model(args.checkpoint, device)

    model_input, to01 = build_transforms(args.image_size)

    files = [f for f in os.listdir(args.profiles_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
    files = sorted(files)[:args.num_samples]
    os.makedirs(args.out_dir, exist_ok=True)

    # Optional imports that may be missing
    try:
        import lpips
        lpips_alex = lpips.LPIPS(net='alex').to(device)
    except Exception:
        lpips = None
        lpips_alex = None

    try:
        from facenet_pytorch import InceptionResnetV1
        id_model = InceptionResnetV1(pretrained='vggface2').eval().to(device)
    except Exception:
        id_model = None

    # Try torchmetrics FID if available
    try:
        from torchmetrics.image.fid import FrechetInceptionDistance
        fid_metric = FrechetInceptionDistance(feature=2048).to(device)
    except Exception:
        fid_metric = None

    results = []

    for fname in files:
        p = os.path.join(args.profiles_dir, fname)
        img = Image.open(p).convert('RGB')
        inp = model_input(img).unsqueeze(0).to(device)
        with torch.no_grad():
            gen = model(inp)

        # save generated image
        gen01 = (gen + 1.0) / 2.0
        out_path = os.path.join(args.out_dir, fname)
        vutils.save_image(gen01.clamp(0,1), out_path)

        lpips_val = None
        id_dist = None

        # If ground-truth frontal exists, compute LPIPS and ID distance
        if args.fronts_dir:
            gt_path = os.path.join(args.fronts_dir, fname)
            if os.path.exists(gt_path):
                gt = Image.open(gt_path).convert('RGB')
                gt_t = to01(gt).unsqueeze(0).to(device)
                gen_t = gen01.clamp(0,1)

                if lpips_alex is not None:
                    # lpips expects input in [-1,1]
                    gt_lp = gt_t * 2.0 - 1.0
                    lp = lpips_alex(gen, gt_lp).item()
                    lpips_val = float(lp)

                if id_model is not None:
                    # facenet expects preprocessed tensors in [0,1] with normalization inside model
                    # We compute embeddings on resized tensors
                    gt_emb = id_model(gt_t.to(device))
                    gen_emb = id_model(gen01.to(device))
                    # cosine distance
                    cos = torch.nn.functional.cosine_similarity(gt_emb, gen_emb).item()
                    id_dist = float(1.0 - cos)

                # update fid metric if available
                if fid_metric is not None:
                    # FrechetInceptionDistance expects images in range [0,1]
                    fid_metric.update(gt_t, real=True)
                    fid_metric.update(gen01.clamp(0,1), real=False)

        results.append({'file': fname, 'lpips': lpips_val, 'id_dist': id_dist})

    # Compute FID if metric object available
    fid_val = None
    if fid_metric is not None:
        try:
            fid_val = float(fid_metric.compute().cpu().item())
        except Exception:
            fid_val = None

    # Save CSV
    csv_path = os.path.join(args.out_dir, 'evaluation.csv')
    with open(csv_path, 'w', newline='') as csvfile:
        fieldnames = ['file', 'lpips', 'id_dist']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow(r)

    print('Evaluation finished')
    if fid_val is not None:
        print(f'FID: {fid_val:.4f}')
    # Print averages
    lpips_vals = [r['lpips'] for r in results if r['lpips'] is not None]
    id_vals = [r['id_dist'] for r in results if r['id_dist'] is not None]
    if lpips_vals:
        print(f'Mean LPIPS: {sum(lpips_vals)/len(lpips_vals):.4f}')
    if id_vals:
        print(f'Mean ID distance (1-cos): {sum(id_vals)/len(id_vals):.4f}')

if __name__ == '__main__':
    main()
