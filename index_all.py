import os
import sys
import pandas as pd

BASE = r"C:\Users\User\Desktop\marathon-bib-app"

# ====== INPUTS ======
# Your bib OCR output (PARSeq)
PRED_CSV     = os.path.join(BASE, "predict14_preds.csv")          # <-- update if needed

# Bib->person match output (your matching script output)
BIB_MATCHES  = os.path.join(BASE, "bib_person_matches2.csv")      # <-- update if needed

# Face crops + clusters
FACE_CROPS   = os.path.join(BASE, r"runs\detect\face_crops.csv")
CLUSTERS     = os.path.join(BASE, "faces_clustered.csv")

# ====== OUTPUTS ======
OUT_BASELINE = os.path.join(BASE, "bib_originals_index.csv")
OUT_MASTER   = os.path.join(BASE, "master_face_bib_index.csv")


# ----------------------------
# Normalizers
# ----------------------------
def norm_bib(x: str) -> str:
    x = "" if x is None else str(x)
    return "".join(ch for ch in x.strip() if ch.isdigit())


def norm_path(p: str) -> str:
    if p is None:
        return ""
    p = str(p).strip().strip('"').strip("'")
    if not p:
        return ""
    try:
        p = os.path.normpath(p)
        p = os.path.abspath(p)
        p = p.lower()
    except Exception:
        p = p.replace("/", "\\").lower()
    return p


def require_cols(df: pd.DataFrame, cols, name: str):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{name} missing columns: {missing}. Found: {list(df.columns)}")


# ----------------------------
# Build baseline: bib -> original_image_path (OCR only)
# ----------------------------
def build_bib_originals_index() -> pd.DataFrame:
    if not os.path.exists(PRED_CSV):
        raise FileNotFoundError(f"Missing: {PRED_CSV}")

    dfp = pd.read_csv(PRED_CSV, dtype=str)
    require_cols(dfp, ["pred_label", "original_image_path"], "predict*.csv")

    # Normalize bib from pred_label (this is what you should trust for matching)
    dfp["bib_norm"] = dfp["pred_label"].apply(norm_bib)
    dfp["original_image_path"] = dfp["original_image_path"].fillna("").astype(str).str.strip()

    # pred_conf numeric (optional)
    if "pred_conf" in dfp.columns:
        dfp["pred_conf_num"] = pd.to_numeric(dfp["pred_conf"], errors="coerce").fillna(0.0)
    else:
        dfp["pred_conf"] = ""
        dfp["pred_conf_num"] = 0.0

    # Keep only valid bib + path
    dfp = dfp[(dfp["bib_norm"] != "") & (dfp["original_image_path"] != "")].copy()

    # Dedup: keep highest pred_conf per (bib_norm, original_image_path)
    dfp = dfp.sort_values(["bib_norm", "original_image_path", "pred_conf_num"], ascending=[True, True, False])
    dfp = dfp.drop_duplicates(subset=["bib_norm", "original_image_path"], keep="first")

    baseline = dfp[["bib_norm", "pred_label", "pred_conf", "pred_conf_num", "original_image_path"]].copy()

    baseline.to_csv(OUT_BASELINE, index=False)
    print("Saved baseline:", OUT_BASELINE)
    print("Baseline rows:", len(baseline))
    print("Baseline unique bibs:", baseline["bib_norm"].nunique())

    return baseline


# ----------------------------
# Build enrichment: bib_person_matches -> face_crop -> cluster
# ----------------------------
def build_enrichment_index() -> pd.DataFrame:
    for p in [BIB_MATCHES, FACE_CROPS, CLUSTERS]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing: {p}")

    df_bib = pd.read_csv(BIB_MATCHES, dtype=str)
    df_fc  = pd.read_csv(FACE_CROPS, dtype=str)
    df_cl  = pd.read_csv(CLUSTERS, dtype=str)

    require_cols(df_bib, ["pred_label", "pred_conf", "original_image_path", "person_crop_path"], os.path.basename(BIB_MATCHES))
    require_cols(df_fc,  ["face_crop_path", "source_person_crop"], "face_crops.csv")
    require_cols(df_cl,  ["path", "face_cluster_id"], "faces_clustered.csv")

    # IMPORTANT: bib_norm must be from pred_label (NOT your old bib column)
    df_bib["bib_norm"] = df_bib["pred_label"].apply(norm_bib)

    # merge keys
    df_bib["person_crop_key"] = df_bib["person_crop_path"].apply(norm_path)
    df_fc["person_crop_key"]  = df_fc["source_person_crop"].apply(norm_path)

    df_fc["face_crop_key"] = df_fc["face_crop_path"].apply(norm_path)
    df_cl["face_crop_key"] = df_cl["path"].apply(norm_path)

    # pred_conf numeric
    df_bib["pred_conf_num"] = pd.to_numeric(df_bib["pred_conf"], errors="coerce").fillna(0.0)

    # Dedup face crops: one face per person crop (keep sharpest if present)
    if "sharpness" in df_fc.columns:
        df_fc["sharpness_num"] = pd.to_numeric(df_fc["sharpness"], errors="coerce").fillna(0.0)
        df_fc = df_fc.sort_values(["person_crop_key", "sharpness_num"], ascending=[True, False])
    else:
        df_fc = df_fc.sort_values(["person_crop_key"])

    df_fc = df_fc.drop_duplicates(subset=["person_crop_key"], keep="first")
    df_cl = df_cl.drop_duplicates(subset=["face_crop_key"], keep="first")

    # join bib->face
    df = df_bib.merge(df_fc, on="person_crop_key", how="left", suffixes=("", "_fc"))

    # join face->cluster
    df = df.merge(df_cl[["face_crop_key", "face_cluster_id"]], on="face_crop_key", how="left")

    df["face_cluster_id"] = pd.to_numeric(df["face_cluster_id"], errors="coerce").fillna(-9999).astype(int)

    # Keep only essentials (+ optional debug columns if present)
    keep = [
        "bib_norm",
        "pred_label",
        "pred_conf",
        "pred_conf_num",
        "original_image_path",
        "person_crop_path",
        "face_crop_path",
        "face_cluster_id",
    ]
    for c in ["match_iou", "person_index", "x0", "y0", "x1", "y1", "detector", "sharpness", "skin_frac"]:
        if c in df.columns and c not in keep:
            keep.append(c)

    out = df[keep].copy()

    # Dedup enrichment at the photo level too (avoid multiple persons mapping to same original for same bib)
    out["orig_key"] = out["original_image_path"].apply(norm_path)
    out = out.sort_values(["bib_norm", "orig_key", "pred_conf_num"], ascending=[True, True, False])
    out = out.drop_duplicates(subset=["bib_norm", "orig_key"], keep="first")
    out = out.drop(columns=["orig_key"])

    print("Enrichment rows:", len(out))
    print("Enrichment unique bibs:", out["bib_norm"].replace("", pd.NA).dropna().nunique())

    return out


# ----------------------------
# Union baseline + enrichment
# ----------------------------
def main():
    baseline = build_bib_originals_index()
    enrich = build_enrichment_index()

    # Create a key for union
    baseline["orig_key"] = baseline["original_image_path"].apply(norm_path)
    enrich["orig_key"] = enrich["original_image_path"].apply(norm_path)

    # Mark source and union
    baseline["source"] = "bib_ocr"
    enrich["source"] = "bib_person_face"

    # Align columns (baseline lacks face columns)
    for col in ["person_crop_path", "face_crop_path", "face_cluster_id", "match_iou", "person_index", "x0", "y0", "x1", "y1", "detector", "sharpness", "skin_frac"]:
        if col not in baseline.columns:
            baseline[col] = "" if col != "face_cluster_id" else -9999

    # Combine
    combined = pd.concat([baseline, enrich], ignore_index=True, sort=False)

    # Prefer enrichment over baseline for same (bib_norm, orig_key)
    combined["face_present"] = combined["face_crop_path"].fillna("").astype(str).str.strip().ne("")
    combined["cluster_present"] = pd.to_numeric(combined["face_cluster_id"], errors="coerce").fillna(-9999).astype(int) != -9999
    combined["pred_conf_num"] = pd.to_numeric(combined["pred_conf_num"], errors="coerce").fillna(0.0)

    combined = combined.sort_values(
        ["bib_norm", "orig_key", "cluster_present", "face_present", "pred_conf_num"],
        ascending=[True, True, False, False, False]
    )
    combined = combined.drop_duplicates(subset=["bib_norm", "orig_key"], keep="first")

    # Final output
    final_cols = [
        "bib_norm",
        "pred_label",
        "pred_conf",
        "pred_conf_num",
        "original_image_path",
        "person_crop_path",
        "face_crop_path",
        "face_cluster_id",
        "source",
    ]
    # add debug cols if exist
    for c in ["match_iou", "person_index", "x0", "y0", "x1", "y1", "detector", "sharpness", "skin_frac"]:
        if c in combined.columns and c not in final_cols:
            final_cols.append(c)

    combined[final_cols].to_csv(OUT_MASTER, index=False)

    # Diagnostics
    n = len(combined)
    has_face = combined["face_crop_path"].fillna("").astype(str).str.strip().ne("")
    has_cluster = pd.to_numeric(combined["face_cluster_id"], errors="coerce").fillna(-9999).astype(int) != -9999

    print("Saved master:", OUT_MASTER)
    print("Master rows:", n)
    print("Master unique bibs:", combined["bib_norm"].replace("", pd.NA).dropna().nunique())
    print("Has face_crop_path:", float(has_face.mean()))
    print("Has cluster_id:", float(has_cluster.mean()))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
