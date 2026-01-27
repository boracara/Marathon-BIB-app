import os
import argparse
import pandas as pd
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from strhub.models.utils import load_from_checkpoint

class ImageDataset(Dataset):
    """Custom Dataset for efficient batch loading and preprocessing."""
    def __init__(self, image_paths, indices, img_h, img_w):
        self.image_paths = image_paths
        self.indices = indices
        self.img_h = img_h
        self.img_w = img_w

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        orig_idx = self.indices[idx]
        
        # Load and Preprocess
        im = Image.open(path).convert("RGB")
        im = im.resize((self.img_w, self.img_h), Image.BICUBIC)
        arr = np.asarray(im).astype(np.float32) / 255.0
        tensor = torch.from_numpy(arr).permute(2, 0, 1)
        
        return tensor, orig_idx

def unwrap_logits(out):
    """Handles various model return formats to extract logits."""
    if isinstance(out, dict):
        return out.get("logits", next(iter(v for v in out.values() if torch.is_tensor(v))))
    if isinstance(out, (tuple, list)):
        return out[0]
    return out

def get_mean_conf(pj):
    """Safely calculates confidence from a variety of tensor shapes."""
    if torch.is_tensor(pj):
        pj = pj.detach().cpu()
        return float(pj.mean().item()) if pj.numel() > 1 else float(pj.item())
    return float(pj)

@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, help="Path to .ckpt")
    ap.add_argument("--csv", required=True, help="Input CSV with image_path")
    ap.add_argument("--out_csv", required=True, help="Output CSV path")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--img_h", type=int, default=32)
    ap.add_argument("--img_w", type=int, default=128)
    ap.add_argument("--batch_size", type=int, default=64)
    args = ap.parse_args()

    # Initialization
    device = torch.device(args.device)
    df = pd.read_csv(args.csv)
    if "image_path" not in df.columns:
        raise ValueError("CSV must contain 'image_path' column.")

    # Model Loading
    print(f"Loading model from {args.checkpoint}...")
    model = load_from_checkpoint(args.checkpoint).to(device).eval()

    # Filter Valid Paths
    valid_paths = []
    valid_indices = []
    for i, p in enumerate(df["image_path"].astype(str)):
        p = p.strip()
        if os.path.isfile(p):
            valid_paths.append(p)
            valid_indices.append(i)

    print(f"Total rows: {len(df)} | Valid images: {len(valid_paths)}")

    # Data Loader
    dataset = ImageDataset(valid_paths, valid_indices, args.img_h, args.img_w)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    preds = [""] * len(df)
    confs = [0.0] * len(df)

    # Inference Loop
    for batch_idx, (images, original_idxs) in enumerate(loader):
        images = images.to(device)
        
        # PARSeq/STRHub models usually take 'images' as kwarg
        out = model(images=images)
        logits = unwrap_logits(out)
        
        # Decode
        texts, probs = model.tokenizer.decode(logits)

        # Record Results
        for i, row_idx in enumerate(original_idxs):
            row_idx = row_idx.item() # Convert tensor to int
            preds[row_idx] = texts[i]
            confs[row_idx] = get_mean_conf(probs[i])

        if (batch_idx + 1) % 10 == 0:
            print(f"Batch {batch_idx + 1} processed...")

    # Output
    df["pred_label"] = preds
    df["pred_conf"] = confs
    df.to_csv(args.out_csv, index=False)
    print(f"Success! Results saved to: {args.out_csv}")

if __name__ == "__main__":
    main()