#!/usr/bin/env python3
"""
saliency_gt_overlay.py
======================
Render interpretability figures in which the *ground-truth* pathology region is
circled in red on top of the patches the IPS / MS-IPS / H-MS-IPS model actually
selected. The point of the figure is to let a reader see, at a glance, that the
selected patches coincide with the clinician-defined lesion, i.e. that patch
selection is an intrinsic, verifiable localisation.

WHY THIS IS A SEPARATE SCRIPT
-----------------------------
The raw images and the ground-truth annotations live on Kaggle (they are not
committed to the thesis repo), so this figure has to be produced in the same
environment the notebooks run in. Point --data-root at the Kaggle dataset and
--saliency-csv at the CSV your run already exported, and it will emit one PNG
per example into --out-dir. Copy the PNGs into
    own-work/master2-thesis/figures/results/saliency/
overwriting the old 3-panel exports; the thesis captions already describe the
red ground-truth outline.

GROUND-TRUTH SOURCES
--------------------
RSNA : stage2_train_metadata.csv  (columns patientId,x,y,width,height,Target)
       boxes are in the 1024x1024 processed-image frame, i.e. the same frame the
       saliency fine_row/fine_col coordinates use -> no rescaling needed.
DDR  : lesion_segmentation/<split>/label/<img>.tif  (binary lesion masks);
       we draw a red circle around each connected lesion component.

SALIENCY CSV FORMAT (already exported by the notebooks)
------------------------------------------------------
image_id,img_rank,npy_file,patch_rank,fine_row,fine_col,score,pred_label,
true_label,correct
fine_row/fine_col are the top-left pixel of each selected patch in the model's
working-resolution frame (1024 for RSNA/NODE21, 896 for DDR).

USAGE
-----
RSNA:
  python saliency_gt_overlay.py --dataset rsna \
      --saliency-csv results/ips-self-attention/exp_001/runs/ips_rsna_v1/saliency/ips_rsna_v1_saliency.csv \
      --data-root /kaggle/input/datasets/iamtapendu/rsna-pneumonia-processed-dataset \
      --patch-size 128 --img-size 1024 --only-correct-positives --out-dir ./saliency_gt

DDR:
  python saliency_gt_overlay.py --dataset ddr \
      --saliency-csv results/ips-self-attention/exp_001/runs/ips_ddr_v1/saliency/ips_ddr_v1_saliency.csv \
      --data-root /kaggle/input/datasets/samriddhibagchi/ddr-dataset-credits-to-authors/DDR-dataset \
      --patch-size 112 --img-size 896 --out-dir ./saliency_gt
"""
import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
from PIL import Image

RED = "#e6194B"
CYAN = "#19E6C8"


# ----------------------------------------------------------------------------- data loaders
def load_saliency(csv_path):
    """Return {image_id: {'meta': (pred,true,correct), 'patches': [(row,col,score,rank)]}}."""
    out = defaultdict(lambda: {"meta": None, "patches": []})
    with open(csv_path) as fh:
        for r in csv.DictReader(fh):
            iid = r["image_id"]
            out[iid]["meta"] = (int(float(r["pred_label"])),
                                int(float(r["true_label"])),
                                int(float(r["correct"])))
            out[iid]["patches"].append((int(float(r["fine_row"])),
                                        int(float(r["fine_col"])),
                                        float(r["score"]),
                                        int(float(r["patch_rank"]))))
    return out


def rsna_image_and_boxes(data_root, image_id, img_size):
    """Load an RSNA image (resized to img_size) and its GT pneumonia boxes."""
    import pandas as pd
    data_root = Path(data_root)
    meta = pd.read_csv(data_root / "stage2_train_metadata.csv")
    key = "patientId" if "patientId" in meta.columns else meta.columns[0]
    rows = meta[meta[key].astype(str) == str(image_id)]

    # Try a few plausible image locations for the processed dataset.
    candidates = [
        data_root / "stage2_train" / f"{image_id}.png",
        data_root / "train" / f"{image_id}.png",
        data_root / "images" / f"{image_id}.png",
    ]
    img_path = next((p for p in candidates if p.exists()), None)
    if img_path is None:
        hit = list(data_root.rglob(f"{image_id}.png")) + list(data_root.rglob(f"{image_id}.jpg"))
        img_path = hit[0] if hit else None
    if img_path is None:
        raise FileNotFoundError(f"image for {image_id} not found under {data_root}")

    img = Image.open(img_path).convert("L")
    sx, sy = img_size / img.width, img_size / img.height
    img = img.resize((img_size, img_size))

    boxes = []
    for _, row in rows.iterrows():
        if "x" in rows.columns and not np.isnan(row.get("x", np.nan)):
            boxes.append((row["x"] * sx, row["y"] * sy,
                          row["width"] * sx, row["height"] * sy))
    return np.asarray(img), boxes


def ddr_image_and_boxes(data_root, image_id, img_size):
    """Load a DDR fundus image (resized) and derive GT circles from lesion masks."""
    from scipy import ndimage
    data_root = Path(data_root)
    img_hits = list(data_root.rglob(f"{image_id}.jpg")) + list(data_root.rglob(f"{image_id}.png"))
    if not img_hits:
        raise FileNotFoundError(f"DDR image {image_id} not found under {data_root}")
    img = Image.open(img_hits[0]).convert("RGB")
    sx, sy = img_size / img.width, img_size / img.height
    img = img.resize((img_size, img_size))

    boxes = []
    mask_hits = list(data_root.rglob(f"lesion_segmentation/**/{image_id}.tif"))
    if mask_hits:
        m = np.asarray(Image.open(mask_hits[0]).convert("L").resize((img_size, img_size))) > 0
        lbl, n = ndimage.label(m)
        for i in range(1, n + 1):
            ys, xs = np.where(lbl == i)
            if len(xs) < 20:
                continue
            boxes.append((xs.min(), ys.min(), xs.max() - xs.min(), ys.max() - ys.min()))
    return np.asarray(img), boxes


# ----------------------------------------------------------------------------- rendering
def render(image, boxes, patches, patch_size, meta, title, out_png):
    pred, true, correct = meta
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(image, cmap="gray" if image.ndim == 2 else None)

    # selected patches (model evidence) in cyan, ranked
    smax = max((s for *_, s, _ in [(0, 0, p[2], 0) for p in patches]), default=1) or 1
    for row, col, score, rank in patches:
        ax.add_patch(Rectangle((col, row), patch_size, patch_size,
                               fill=False, edgecolor=CYAN, lw=1.4, alpha=0.9))
        if rank == 0:
            ax.text(col + 3, row + 14, "top-1", color=CYAN, fontsize=8, weight="bold")

    # ground-truth pathology region circled in red
    for (x, y, w, h) in boxes:
        cx, cy, rad = x + w / 2, y + h / 2, 0.62 * max(w, h)
        ax.add_patch(Circle((cx, cy), rad, fill=False, edgecolor=RED, lw=2.6))
    if boxes:
        ax.plot([], [], color=RED, lw=2.6, label="ground-truth lesion")
    ax.plot([], [], color=CYAN, lw=1.4, label="selected patch")
    ax.legend(loc="lower right", fontsize=8, framealpha=0.85)

    ax.set_title(f"{title}\npred={pred} true={true} "
                 f"({'correct' if correct else 'error'})", fontsize=10)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_png}  (GT boxes: {len(boxes)}, patches: {len(patches)})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["rsna", "ddr"], required=True)
    ap.add_argument("--saliency-csv", required=True)
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--patch-size", type=int, required=True)
    ap.add_argument("--img-size", type=int, required=True)
    ap.add_argument("--image-id", default=None, help="render only this id")
    ap.add_argument("--only-correct-positives", action="store_true",
                    help="keep only true=1 & correct=1 examples (the ones worth showing)")
    ap.add_argument("--max-examples", type=int, default=5)
    ap.add_argument("--out-dir", default="./saliency_gt")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    sal = load_saliency(args.saliency_csv)

    ids = [args.image_id] if args.image_id else list(sal.keys())
    if args.only_correct_positives:
        ids = [i for i in ids if sal[i]["meta"] and sal[i]["meta"][1] == 1 and sal[i]["meta"][2] == 1]
    ids = ids[: args.max_examples]
    if not ids:
        print("No matching examples. Re-export saliency on correct positive cases "
              "(true=1, correct=1) so the GT overlay is meaningful.")
        return

    loader = rsna_image_and_boxes if args.dataset == "rsna" else ddr_image_and_boxes
    for iid in ids:
        img, boxes = loader(args.data_root, iid, args.img_size)
        title = f"{args.dataset.upper()} img #{iid}"
        render(img, boxes, sal[iid]["patches"], args.patch_size,
               sal[iid]["meta"], title, out_dir / f"gt_overlay_{args.dataset}_{iid}.png")


if __name__ == "__main__":
    main()
