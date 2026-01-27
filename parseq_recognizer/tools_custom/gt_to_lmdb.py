import os
import io
import argparse
import lmdb
from PIL import Image


def image_to_png_bytes(path: str) -> bytes:
    """Read image and store it as PNG bytes for LMDB."""
    with Image.open(path) as im:
        im = im.convert("RGB")
        buf = io.BytesIO()
        im.save(buf, format="PNG")
        return buf.getvalue()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", required=True, help="GT file with lines: abs_image_path<TAB>label")
    ap.add_argument("--out", required=True, help="Output LMDB directory (will be created)")
    ap.add_argument("--map_size_gb", type=float, default=2.0, help="LMDB map size in GB")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    with open(args.gt, "r", encoding="utf-8") as f:
        lines = [ln.strip() for ln in f.readlines() if ln.strip()]

    samples = []
    for ln in lines:
        if "\t" not in ln:
            continue
        img_path, label = ln.split("\t", 1)
        img_path = img_path.strip()
        label = label.strip()
        if os.path.isfile(img_path) and label:
            samples.append((img_path, label))

    if not samples:
        raise RuntimeError(f"No valid samples found in GT file: {args.gt}")

    map_size = int(args.map_size_gb * (1024**3))
    env = lmdb.open(
        args.out,
        map_size=map_size,
        subdir=True,
        lock=True,
        readahead=False,
        meminit=False,
    )

    with env.begin(write=True) as txn:
        for i, (img_path, label) in enumerate(samples, start=1):
            txn.put(f"image-{i:09d}".encode("utf-8"), image_to_png_bytes(img_path))
            txn.put(f"label-{i:09d}".encode("utf-8"), label.encode("utf-8"))
        txn.put(b"num-samples", str(len(samples)).encode("utf-8"))

    env.close()
    print(f"LMDB created: {os.path.abspath(args.out)}")
    print(f"num-samples: {len(samples)}")


if __name__ == "__main__":
    main()
