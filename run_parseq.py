import os
import tempfile
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import psycopg2
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader

from storage import get_s3, S3_BUCKET
from strhub.models.utils import load_from_checkpoint  # uses same import as your CSV script


DB_CONFIG = {
    "host": "127.0.0.1",
    "port": 5433,
    "dbname": "marathon_db",
    "user": "marathon",
    "password": "123456",
}

CHECKPOINT = r"C:\Users\User\Desktop\marathon-bib-app\parseq_recognizer\parseq\outputs\parseq-tiny\2025-12-30_20-50-15\checkpoints\last.ckpt"


class ImageDataset(Dataset):
    """Loads and preprocesses images for PARSeq inference."""
    def __init__(self, items: List[Tuple[int, str]], img_h: int, img_w: int):
        """
        items: list of (detection_id, local_image_path)
        """
        self.items = items
        self.img_h = img_h
        self.img_w = img_w

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        det_id, path = self.items[idx]
        try:
            im = Image.open(path).convert("RGB")
            im = im.resize((self.img_w, self.img_h), Image.BICUBIC)
            arr = np.asarray(im).astype(np.float32) / 255.0  # HWC, 0..1
            tensor = torch.from_numpy(arr).permute(2, 0, 1)  # CHW
            return tensor, det_id
        except Exception:
            blank = torch.zeros(3, self.img_h, self.img_w, dtype=torch.float32)
            return blank, det_id


def unwrap_logits(out):
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
    if torch.is_tensor(pj):
        pj = pj.detach().cpu()
        if pj.numel() == 1:
            return float(pj.item())
        return float(pj.float().mean().item())
    return float(pj)


def needs_review_rule(pred: str, conf: float, min_conf: float) -> bool:
    if min_conf >= 0 and conf < min_conf:
        return True
    p = (pred or "").strip()
    if len(p) < 2:
        return True
    return False


def fetch_pending_detections(cur, event_id: int, limit: Optional[int] = None):
    sql = """
        SELECT d.id, d.crop_path
        FROM marathon.detections d
        JOIN marathon.photos p ON p.id = d.photo_id
        LEFT JOIN marathon.ocr_results o ON o.detection_id = d.id
        WHERE p.event_id = %s AND o.id IS NULL
        ORDER BY d.id
    """
    params = [event_id]
    if limit is not None:
        sql += " LIMIT %s"
        params.append(limit)
    cur.execute(sql, tuple(params))
    return cur.fetchall()


@torch.no_grad()
def run_parseq_db(
    event_id: int,
    device: str = "cpu",
    img_h: int = 32,
    img_w: int = 128,
    batch_size: int = 64,
    num_workers: int = 0,
    min_conf: float = 4.8,   # matches your script default
    max_digits: int = 0,     # if you want bib length cap e.g. 4
    limit: Optional[int] = None,
):
    device_t = torch.device(device)

    if not Path(CHECKPOINT).exists():
        raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT}")

    print(f"Loading model from: {CHECKPOINT}")
    model = load_from_checkpoint(CHECKPOINT, weights_only=False).to(device_t).eval()


    s3 = get_s3()
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()

    rows = fetch_pending_detections(cur, event_id, limit=limit)
    print(f"CROPS TO OCR: {len(rows)}")
    if not rows:
        cur.close()
        conn.close()
        return

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        crops_dir = tmpdir / "crops"
        crops_dir.mkdir(parents=True, exist_ok=True)

        # Download all pending crops locally
        items: List[Tuple[int, str]] = []
        for det_id, crop_key in rows:
            local_path = crops_dir / f"{int(det_id)}.jpg"
            s3.download_file(S3_BUCKET, crop_key, str(local_path))
            items.append((int(det_id), str(local_path)))

        dataset = ImageDataset(items, img_h=img_h, img_w=img_w)
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,          # keep 0 on Windows
            pin_memory=(device_t.type == "cuda"),
        )

        processed = 0
        for batch_idx, (images, det_ids) in enumerate(loader):
            images = images.to(device_t)

            out = model(images=images)  # PARSeq expects keyword 'images'
            logits = unwrap_logits(out)

            texts, probs = model.tokenizer.decode(logits)

            for j in range(len(det_ids)):
                detection_id = int(det_ids[j].item())
                pred = str(texts[j]).strip()
                conf = mean_conf(probs[j])

                if max_digits and max_digits > 0:
                    pred = pred[:max_digits]

                nr = needs_review_rule(pred, conf, min_conf=min_conf)

                cur.execute("""
                    INSERT INTO marathon.ocr_results (detection_id, bib_pred, ocr_conf, needs_review)
                    VALUES (%s, %s, %s, %s)
                    ON CONFLICT (detection_id) DO NOTHING
                """, (detection_id, pred, float(conf), bool(nr)))

            processed += len(det_ids)
            if (batch_idx + 1) % 10 == 0:
                conn.commit()
                print(f"Processed {processed}/{len(items)}")

        conn.commit()

    cur.close()
    conn.close()
    print("DONE: inserted OCR results into DB.")


if __name__ == "__main__":
    run_parseq_db(
        event_id=1,
        device="cpu",
        batch_size=64,
        num_workers=0,
        min_conf=4.8,
        max_digits=0,   # set to 4 if your bibs are exactly 4 digits
    )
