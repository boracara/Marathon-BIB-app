import csv
from pathlib import Path
from typing import Optional, Tuple
import cv2

IMG_EXTS = (".jpg", ".jpeg", ".png", ".webp")

def find_original_image(images_dir: Path, stem: str) -> Optional[Path]:
    for ext in IMG_EXTS:
        p = images_dir / f"{stem}{ext}"
        if p.exists():
            return p
    matches = []
    for ext in IMG_EXTS:
        matches.extend(images_dir.glob(f"{stem}*{ext}"))
    return sorted(matches)[0] if matches else None

def yolo_norm_to_xyxy(xc, yc, w, h, W, H) -> Tuple[int,int,int,int]:
    x1 = int((xc - w/2) * W)
    y1 = int((yc - h/2) * H)
    x2 = int((xc + w/2) * W)
    y2 = int((yc + h/2) * H)
    x1 = max(0, min(x1, W-1))
    y1 = max(0, min(y1, H-1))
    x2 = max(0, min(x2, W))
    y2 = max(0, min(y2, H))
    if x2 <= x1: x2 = min(W, x1 + 1)
    if y2 <= y1: y2 = min(H, y1 + 1)
    return x1, y1, x2, y2

def main():
    # Set these paths to YOUR run:
    PREDICT_DIR = Path(r"C:\Users\User\Desktop\marathon-bib-app\runs\detect\predict14")
    IMAGES_DIR  = Path(r"C:\Users\User\Desktop\marathon-bib-app\data\bib_dataset\images\test3")

    OUT_CROPS = Path(r"C:\Users\User\Desktop\marathon-bib-app\runs\detect\bib_crops_unique1")
    OUT_CSV   = Path(r"C:\Users\User\Desktop\marathon-bib-app\bib_detections2.csv")

    LABELS_DIR = PREDICT_DIR / "labels"
    if not LABELS_DIR.exists():
        raise FileNotFoundError(f"Missing labels folder: {LABELS_DIR}")

    OUT_CROPS.mkdir(parents=True, exist_ok=True)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "image_path",              # bib crop path (PARSeq reads this)
        "label",                   # unused placeholder
        "original_key",
        "bib_index",
        "bib_class",
        "bib_det_conf",
        "original_image_path",
        "bx1","by1","bx2","by2"    # bib bbox in original image pixels
    ]

    rows = []

    for label_file in sorted(LABELS_DIR.glob("*.txt")):
        stem = label_file.stem
        orig_path = find_original_image(IMAGES_DIR, stem)
        if not orig_path:
            continue

        img = cv2.imread(str(orig_path))
        if img is None:
            continue
        H, W = img.shape[:2]

        lines = [ln.strip() for ln in label_file.read_text(encoding="utf-8").splitlines() if ln.strip()]

        for bib_idx, ln in enumerate(lines):
            parts = ln.split()
            if len(parts) < 5:
                continue

            cls = int(float(parts[0]))
            xc  = float(parts[1]); yc = float(parts[2])
            bw  = float(parts[3]); bh = float(parts[4])
            det_conf = float(parts[5]) if len(parts) >= 6 else ""

            bx1, by1, bx2, by2 = yolo_norm_to_xyxy(xc, yc, bw, bh, W, H)

            crop = img[by1:by2, bx1:bx2]
            if crop.size == 0:
                continue

            crop_name = f"{stem}_bib_{bib_idx:03d}.jpg"
            crop_path = OUT_CROPS / crop_name
            cv2.imwrite(str(crop_path), crop)

            rows.append({
                "image_path": str(crop_path),
                "label": "",
                "original_key": stem,
                "bib_index": bib_idx,
                "bib_class": cls,
                "bib_det_conf": det_conf,
                "original_image_path": str(orig_path),
                "bx1": bx1, "by1": by1, "bx2": bx2, "by2": by2,
            })

    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    print(f"Created bib CSV: {OUT_CSV}")
    print(f"Saved bib crops to: {OUT_CROPS}")
    print(f"Rows: {len(rows)}")

if __name__ == "__main__":
    main()
