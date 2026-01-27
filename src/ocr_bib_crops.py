import easyocr
import cv2
import glob
import os
import re
import csv

CROPS_DIR = r"C:\Users\User\Desktop\marathon-bib-app\runs\detect\predict6\crops\bib"
STATIC_IMAGES_DIR = r"C:\Users\User\Desktop\marathon-bib-app\static\images"
OUT_CSV = r"C:\Users\User\Desktop\marathon-bib-app\FINAL_bib_ocr_results2.csv"

MIN_DIGITS = 2
MAX_DIGITS = 6
MIN_CONF = 0.55
GPU = False


def extract_digits(text):
    return re.findall(r"\d+", text or "")


def preprocess(img):
    """
    Return 2 variants:
      - gray: often best for clean printed digits
      - thr : helps when lighting is uneven
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    h, w = gray.shape[:2]
    if max(h, w) < 280:
        gray = cv2.resize(gray, None, fx=2.5, fy=2.5, interpolation=cv2.INTER_CUBIC)

    clahe = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    thr = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 31, 7
    )

    # gentle stroke repair
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    thr = cv2.morphologyEx(thr, cv2.MORPH_CLOSE, kernel, iterations=1)

    return gray, thr


def candidates_from_ocr(ocr_result):
    """
    Build candidates from EasyOCR output.
    Key improvement: if OCR splits a bib into multiple boxes ("328" + "6"),
    we join them left-to-right into one candidate ("3286").
    """
    parts = []
    singles = []

    for (bbox, text, conf) in ocr_result:
        digs = "".join(extract_digits(text))
        if not digs:
            continue

        # bbox is 4 points: [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]
        xs = [p[0] for p in bbox]
        x_center = sum(xs) / 4.0

        parts.append((x_center, digs, float(conf)))

        # also keep single detection as candidate
        if MIN_DIGITS <= len(digs) <= MAX_DIGITS:
            singles.append((digs, float(conf)))

    cands = list(singles)

    # Join all digit parts left->right (fixes 328 + 6 => 3286)
    if parts:
        parts.sort(key=lambda x: x[0])
        joined = "".join(p[1] for p in parts)

        if MIN_DIGITS <= len(joined) <= MAX_DIGITS:
            # conservative confidence: use min conf among parts
            joined_conf = min(p[2] for p in parts)
            cands.append((joined, joined_conf))

    return cands


# ----------------------------
# ORIGINAL IMAGE MAPPING
# ----------------------------
def build_original_index(images_dir):
    index = {}
    for f in os.listdir(images_dir):
        if f.lower().endswith((".jpg", ".jpeg", ".png")):
            stem = os.path.splitext(f)[0].lower()
            index[stem] = f
    return index


def map_crop_to_original(crop_filename, original_index):
    crop_stem = os.path.splitext(crop_filename)[0].lower()

    best = None  # (length, orig_file)
    for orig_stem, orig_file in original_index.items():
        if crop_stem.startswith(orig_stem):
            cand = (len(orig_stem), orig_file)
            if best is None or cand[0] > best[0]:
                best = cand

    return best[1] if best else None


def pick_best_candidate(candidates):
    if not candidates:
        return None

    # Deduplicate: keep max confidence per digits string
    dedup = {}
    for bib, conf in candidates:
        if bib not in dedup or conf > dedup[bib]:
            dedup[bib] = conf

    filtered = [(bib, conf) for bib, conf in dedup.items()
                if MIN_DIGITS <= len(bib) <= MAX_DIGITS and conf >= MIN_CONF]

    if not filtered:
        return None

    # Prefer longest valid length (prevents picking "328" when "3286" exists)
    max_len = max(len(b) for b, _ in filtered)
    longest = [(b, c) for b, c in filtered if len(b) == max_len]

    return max(longest, key=lambda x: x[1])


def main():
    if not os.path.isdir(CROPS_DIR):
        print(f"[ERROR] CROPS_DIR not found: {CROPS_DIR}")
        return

    if not os.path.isdir(STATIC_IMAGES_DIR):
        print(f"[ERROR] STATIC_IMAGES_DIR not found: {STATIC_IMAGES_DIR}")
        return

    crop_paths = []
    for ext in ("*.jpg", "*.jpeg", "*.png"):
        crop_paths.extend(glob.glob(os.path.join(CROPS_DIR, ext)))

    print(f"[INFO] Found {len(crop_paths)} crops in: {CROPS_DIR}")
    if not crop_paths:
        return

    original_index = build_original_index(STATIC_IMAGES_DIR)
    print(f"[INFO] Indexed {len(original_index)} original images in: {STATIC_IMAGES_DIR}")

    reader = easyocr.Reader(['en'], gpu=GPU)

    rows = []
    miss_ocr = 0
    miss_map = 0

    for crop_path in crop_paths:
        img = cv2.imread(crop_path)
        if img is None:
            print(f"[WARN] Could not read: {crop_path}")
            continue

        proc_gray, proc_thr = preprocess(img)

        candidates = []

        # OCR on grayscale first (often best on clean bib digits)
        ocr1 = reader.readtext(
            proc_gray,
            detail=1,
            allowlist="0123456789",
            paragraph=False,
            decoder="greedy",
            text_threshold=0.6,
            low_text=0.3
        )
        candidates.extend(candidates_from_ocr(ocr1))

        # OCR on thresholded as fallback/support
        ocr2 = reader.readtext(
            proc_thr,
            detail=1,
            allowlist="0123456789",
            paragraph=False,
            decoder="beamsearch",
            text_threshold=0.6,
            low_text=0.3
        )
        candidates.extend(candidates_from_ocr(ocr2))

        best = pick_best_candidate(candidates)
        if best is None:
            miss_ocr += 1
            continue

        bib_text, ocr_conf = best

        crop_file = os.path.basename(crop_path)
        orig_image = map_crop_to_original(crop_file, original_index)

        if orig_image is None:
            orig_image = ""
            miss_map += 1

        bib_instance_id = f"{os.path.splitext(crop_file)[0]}__{bib_text}"
        rows.append([crop_path, orig_image, bib_text, f"{ocr_conf:.6f}", bib_instance_id])

    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["crop_path", "orig_image", "bib_text", "ocr_conf", "bib_instance_id"])
        w.writerows(rows)

    print(f"[DONE] Wrote {len(rows)} rows → {OUT_CSV}")
    print(f"[STATS] OCR misses (no valid {MIN_DIGITS}–{MAX_DIGITS} digits >= {MIN_CONF}): {miss_ocr}")
    print(f"[STATS] Mapping misses (orig_image blank): {miss_map}")
    print("[NOTE] If orig_image is blank, the crop stem didn't match any file in static/images.")


if __name__ == "__main__":
    main()
