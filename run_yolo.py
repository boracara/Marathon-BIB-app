# run_yolo.py
import os
import tempfile
from typing import Tuple

import cv2
import numpy as np
import psycopg2
from ultralytics import YOLO

from storage import get_s3, S3_BUCKET

DB_CONFIG = {
    "host": "127.0.0.1",
    "port": 5433,
    "dbname": "marathon_db",
    "user": "marathon",
    "password": "123456",
}

def download_s3_to_tmp(s3, key: str) -> str:
    fd, tmp_path = tempfile.mkstemp(suffix=os.path.splitext(key)[1] or ".jpg")
    os.close(fd)
    s3.download_file(S3_BUCKET, key, tmp_path)
    return tmp_path

def upload_bytes_to_s3(s3, key: str, data: bytes, content_type="image/jpeg"):
    s3.put_object(Bucket=S3_BUCKET, Key=key, Body=data, ContentType=content_type)

def crop_and_encode(img_bgr: np.ndarray, xyxy: Tuple[int, int, int, int]) -> bytes:
    x1, y1, x2, y2 = xyxy
    h, w = img_bgr.shape[:2]
    x1 = max(0, min(x1, w - 1))
    x2 = max(0, min(x2, w))
    y1 = max(0, min(y1, h - 1))
    y2 = max(0, min(y2, h))
    crop = img_bgr[y1:y2, x1:x2]
    ok, buf = cv2.imencode(".jpg", crop)
    if not ok:
        raise RuntimeError("Failed to encode crop to JPG")
    return buf.tobytes()

def run_yolo_for_event(event_id: int, yolo_weights_path: str, conf: float = 0.5, iou: float = 0.5, imgsz: int = 640):
    s3 = get_s3()
    model = YOLO(yolo_weights_path)

    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()

    # Select photos that do NOT yet have any detections
    cur.execute("""
        SELECT p.id, p.file_path, p.file_hash
        FROM marathon.photos p
        LEFT JOIN marathon.detections d ON d.photo_id = p.id
        WHERE p.event_id = %s
        GROUP BY p.id
        HAVING COUNT(d.id) = 0
        ORDER BY p.id
    """, (event_id,))
    photos = cur.fetchall()

    print(f"PHOTOS TO PROCESS: {len(photos)}")

    for photo_id, file_key, file_hash in photos:
        tmp_img = download_s3_to_tmp(s3, file_key)
        img = cv2.imread(tmp_img)
        os.remove(tmp_img)

        if img is None:
            print(f"SKIP unreadable image: photo_id={photo_id} key={file_key}")
            continue

        # Run YOLO
        results = model.predict(source=img, conf=conf, iou=iou, imgsz=imgsz, verbose=False)
        r = results[0]

        if r.boxes is None or len(r.boxes) == 0:
            print(f"NO DETECTIONS: photo_id={photo_id}")
            continue

        # Insert detections first to get detection_id, then upload crop using detection_id in the key
        for box in r.boxes:
            xyxy = box.xyxy[0].cpu().numpy().astype(int).tolist()
            x1, y1, x2, y2 = xyxy
            yolo_conf = float(box.conf[0].cpu().numpy())

            # placeholder crop path; we’ll update after we know detection_id
            cur.execute("""
                INSERT INTO marathon.detections (photo_id, x1, y1, x2, y2, yolo_conf, crop_path)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                RETURNING id
            """, (photo_id, float(x1), float(y1), float(x2), float(y2), yolo_conf, ""))
            detection_id = cur.fetchone()[0]

            crop_bytes = crop_and_encode(img, (x1, y1, x2, y2))
            crop_key = f"events/{event_id}/crops/{file_hash}/{detection_id}.jpg"
            upload_bytes_to_s3(s3, crop_key, crop_bytes, content_type="image/jpeg")

            cur.execute("""
                UPDATE marathon.detections
                SET crop_path = %s
                WHERE id = %s
            """, (crop_key, detection_id))

        conn.commit()
        print(f"DONE: photo_id={photo_id} detections={len(r.boxes)}")

    cur.close()
    conn.close()

if __name__ == "__main__":
   
    run_yolo_for_event(
        event_id=1,
        yolo_weights_path=r"runs\detect\train5\weights\best.pt",
        conf=0.5,
        iou=0.5,
        imgsz=640,
    )
