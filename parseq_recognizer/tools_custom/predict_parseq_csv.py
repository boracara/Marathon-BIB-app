import os
import argparse
import pandas as pd
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from strhub.models.utils import load_from_checkpoint


class ImageDataset(Dataset):
    """Loads and preprocesses images for PARSeq inference."""
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

        try:
            im = Image.open(path).convert("RGB")
            im = im.resize((self.img_w, self.img_h), Image.BICUBIC)
            arr = np.asarray(im).astype(np.float32) / 255.0  # HWC, 0..1
            tensor = torch.from_numpy(arr).permute(2, 0, 1)  # CHW
            return tensor, orig_idx
        except Exception:
            # Return a blank image tensor if something is wrong with this file
            blank = torch.zeros(3, self.img_h, self.img_w, dtype=torch.float32)
            return blank, orig_idx


def unwrap_logits(out):
    """Extract logits from different forward() return formats."""
    if isinstance(out, dict):
        if "logits" in out and torch.is_tensor(out["logits"]):
            return out["logits"]
        for v in out.values():
            if torch.is_tensor(v):
                return v
        raise TypeError("Forward returned dict but no tensor logits found.")
    if isinstance(out, (tuple, list)):
        return out[0]
    return out


def mean_conf(pj) -> float:
    """Return scalar confidence from tensor or float."""
    if torch.is_tensor(pj):
        pj = pj.detach().cpu()
        if pj.numel() == 1:
            return float(pj.item())
        return float(pj.float().mean().item())
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
    ap.add_argument("--num_workers", type=int, default=0, help="Use 0 on Windows for stability")

    # NEW: post-processing controls
    ap.add_argument(
        "--min_conf",
        type=float,
        default=4.8,
        help="Flag predictions below this confidence into needs_review (set -1 to disable)."
    )
    ap.add_argument(
        "--max_digits",
        type=int,
        default=0,
        help="If >0, truncate pred_label to this many characters (e.g., 4 for bibs)."
    )

    args = ap.parse_args()
    device = torch.device(args.device)

    df = pd.read_csv(args.csv)
    if "image_path" not in df.columns:
        raise ValueError("CSV must contain 'image_path' column.")

    print(f"Loading model from: {args.checkpoint}")
    model = load_from_checkpoint(args.checkpoint).to(device).eval()

    # Build list of valid paths
    valid_paths, valid_indices = [], []
    for i, p in enumerate(df["image_path"].astype(str).tolist()):
        p = p.strip()
        if os.path.isfile(p):
            valid_paths.append(p)
            valid_indices.append(i)

    print(f"Total rows: {len(df)} | Valid images: {len(valid_paths)}")

    dataset = ImageDataset(valid_paths, valid_indices, args.img_h, args.img_w)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    preds = [""] * len(df)
    confs = [0.0] * len(df)

    for batch_idx, (images, original_idxs) in enumerate(loader):
        images = images.to(device)

        # PARSeq expects keyword 'images'
        out = model(images=images)
        logits = unwrap_logits(out)

        texts, probs = model.tokenizer.decode(logits)

        for j in range(len(original_idxs)):
            row_idx = int(original_idxs[j].item())
            preds[row_idx] = str(texts[j])
            confs[row_idx] = mean_conf(probs[j])

        if (batch_idx + 1) % 10 == 0:
            done = min((batch_idx + 1) * args.batch_size, len(valid_paths))
            print(f"Processed {done}/{len(valid_paths)}")

    # =========================
    # NEW: post-processing HERE
    # =========================
    df["pred_label"] = preds
    df["pred_conf"] = confs

    # Optional: truncate predictions to max_digits (helps when model repeats/extends)
    if args.max_digits and args.max_digits > 0:
        df["pred_label"] = df["pred_label"].astype(str).str.slice(0, args.max_digits)

    # Optional: flag low-confidence predictions
    if args.min_conf is not None and args.min_conf >= 0:
        df["needs_review"] = df["pred_conf"] < float(args.min_conf)

    # Write results
    df.to_csv(args.out_csv, index=False)
    print(f"Done. Saved: {args.out_csv}")


if __name__ == "__main__":
    main()
