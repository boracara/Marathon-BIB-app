import os
import re
import random
import argparse
import pandas as pd

DIGITS_RE = re.compile(r"^\d+$")


def clean_label(x):
    """Return cleaned digit-only label or None if invalid."""
    if x is None:
        return None
    s = str(x).strip()

    # Fix common case when CSV was created via Excel: "1152.0"
    if s.endswith(".0") and s[:-2].isdigit():
        s = s[:-2]

    if not DIGITS_RE.match(s):
        return None
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Input CSV with columns: path,label")
    ap.add_argument("--out_dir", required=True, help="Output directory to write gt_*.txt and *_meta.csv")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--train", type=float, default=0.8)
    ap.add_argument("--val", type=float, default=0.1)
    ap.add_argument("--test", type=float, default=0.1)
    ap.add_argument("--min_len", type=int, default=2)
    ap.add_argument("--max_len", type=int, default=6)
    args = ap.parse_args()

    if abs((args.train + args.val + args.test) - 1.0) > 1e-6:
        raise ValueError("train + val + test must equal 1.0")

    os.makedirs(args.out_dir, exist_ok=True)

    df = pd.read_csv(args.csv)

    required = ["path", "label"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}")

    # Normalize paths + clean labels
    df["path"] = df["path"].astype(str).str.strip().apply(os.path.abspath)
    df["label"] = df["label"].apply(clean_label)

    # Drop invalid labels
    df = df.dropna(subset=["label"]).copy()

    # Length filter
    df["lab_len"] = df["label"].str.len()
    df = df[(df["lab_len"] >= args.min_len) & (df["lab_len"] <= args.max_len)].copy()

    # Keep only rows whose image exists
    df = df[df["path"].apply(os.path.isfile)].copy()

    if len(df) < 50:
        raise RuntimeError(f"Too few valid samples after cleaning. Remaining: {len(df)}")

    # Shuffle + split
    random.seed(args.seed)
    idx = list(df.index)
    random.shuffle(idx)
    n = len(idx)
    n_train = int(n * args.train)
    n_val = int(n * args.val)

    train_idx = idx[:n_train]
    val_idx = idx[n_train:n_train + n_val]
    test_idx = idx[n_train + n_val:]

    train_df = df.loc[train_idx].copy()
    val_df = df.loc[val_idx].copy()
    test_df = df.loc[test_idx].copy()

    def write_gt(sub_df: pd.DataFrame, out_path: str):
        # Format: abs_path<TAB>label
        with open(out_path, "w", encoding="utf-8") as f:
            for _, r in sub_df.iterrows():
                f.write(f"{r['path']}\t{r['label']}\n")

    def write_meta(sub_df: pd.DataFrame, out_path: str):
        sub_df[["path", "label"]].to_csv(out_path, index=False)

    write_gt(train_df, os.path.join(args.out_dir, "gt_train.txt"))
    write_gt(val_df, os.path.join(args.out_dir, "gt_val.txt"))
    write_gt(test_df, os.path.join(args.out_dir, "gt_test.txt"))

    write_meta(train_df, os.path.join(args.out_dir, "train_meta.csv"))
    write_meta(val_df, os.path.join(args.out_dir, "val_meta.csv"))
    write_meta(test_df, os.path.join(args.out_dir, "test_meta.csv"))

    print("Done.")
    print(f"Total valid samples: {len(df)}")
    print(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")
    print(f"Output dir: {os.path.abspath(args.out_dir)}")


if __name__ == "__main__":
    main()
