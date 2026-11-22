"""MTA-TransUNet: full experimental pipeline (internal CRIC experiments and CRIC overlap audit).

Generated from notebooks/Cervical_MTA_TransUNet.ipynb; each "# %%" block is one notebook cell.
Run from the repository root:

    CERVIX_ROOT=data/cric CERVIX_CONFIG=configs/default.json python src/run_pipeline.py

The external Mendeley LBC evaluation is in src/mendeley_external.py and src/reconcile_mendeley_roboflow.py.
"""
import os
import sys
import matplotlib
if "ipykernel" not in sys.modules:
    matplotlib.use(os.environ.get("MPLBACKEND", "Agg"))   # headless figure output when run as a script

# %% [cell 0]
# ---------------------------------------------------------------------------
# 0. Environment
# ---------------------------------------------------------------------------
import sys, subprocess, importlib

IN_COLAB = "google.colab" in sys.modules


def _ensure(module, pip_name=None):
    try:
        importlib.import_module(module)
    except ImportError:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", pip_name or module], check=True)


for _m, _p in [("segmentation_models_pytorch", "segmentation-models-pytorch"),
               ("albumentations", "albumentations"),
               ("skimage", "scikit-image"),
               ("pywt", "PyWavelets")]:
    _ensure(_m, _p)

import os, json, glob, time, math, random, warnings
from collections import defaultdict

import numpy as np
import pandas as pd
import cv2
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.colors import ListedColormap

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp

from scipy import stats
from scipy.ndimage import binary_erosion, distance_transform_edt
from sklearn.metrics import roc_curve, precision_recall_curve, auc, average_precision_score
from skimage.restoration import denoise_nl_means, estimate_sigma
from tqdm.auto import tqdm

warnings.filterwarnings("ignore", category=UserWarning)
print(f"torch {torch.__version__} | smp {smp.__version__} | albumentations {A.__version__}")


# %% [cell 1]
# ---------------------------------------------------------------------------
# 1. Configuration and reproducibility
# ---------------------------------------------------------------------------
QUICK = os.environ.get("CERVIX_QUICK", "0") == "1"   # smoke test only

CFG = dict(
    DATA_ROOT=os.environ.get("CERVIX_ROOT", "/content/gdrive/MyDrive/MTA-TransUNet/cervical" if IN_COLAB else "data/cric"),
    SPLITS=("train", "valid", "test"),
    LONG_SIDE=512,          # network input long side; short side follows the median aspect ratio
    BATCH_SIZE=8,
    EPOCHS=40,
    LR=3e-4,
    WEIGHT_DECAY=1e-4,
    WARMUP_EPOCHS=3,
    PATIENCE=10,            # early stopping on validation mDice
    SEED=42,
    NUM_WORKERS=2,
    USE_CLAHE=True,         # CLAHE on the L channel before training and inference
    TTA=True,               # flip test-time augmentation
    ENCODER_WEIGHTS="imagenet",
    HED_AUG=True,            # stain augmentation in HED space (all models)
    W_AUX=0.2,               # weight of the image-level auxiliary loss (proposed model)
    RUN_ABLATION=True,
    RUN_ROBUSTNESS=True,
    W_CE=1.0, W_DICE=1.0,
    N_BOOT=2000,
    PIXELS_PER_IMAGE=20000,  # random pixels per test image for ROC/PR curves
    RESUME=True,             # skip training if a checkpoint already exists
)
if QUICK:
    CFG.update(LONG_SIDE=128, BATCH_SIZE=4, EPOCHS=2, WARMUP_EPOCHS=1, PATIENCE=5,
               NUM_WORKERS=0, ENCODER_WEIGHTS=None, N_BOOT=200, PIXELS_PER_IMAGE=2000, RESUME=False)

# Optional overrides from a JSON file, e.g. CERVIX_CONFIG=configs/default.json
if os.environ.get("CERVIX_CONFIG"):
    with open(os.environ["CERVIX_CONFIG"]) as _f:
        CFG.update({k: v for k, v in json.load(_f).items() if not k.startswith("_")})
    CFG["SPLITS"] = tuple(CFG["SPLITS"])
CFG.setdefault("RUN_LBC_ORIGINAL", False)   # optional image-level analysis of the original Mendeley release

CFG["OUT_DIR"] = os.path.join(CFG["DATA_ROOT"], "results_q1_proposed")
DIRS = {k: os.path.join(CFG["OUT_DIR"], k) for k in ["figures", "tables", "checkpoints", "masks", "logs"]}

if IN_COLAB and CFG["DATA_ROOT"].startswith("/content/gdrive"):
    from google.colab import drive
    drive.mount("/content/gdrive")

for d in DIRS.values():
    os.makedirs(d, exist_ok=True)


def seed_everything(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = False   # True is slower; set True for bit-exact reruns
    torch.backends.cudnn.benchmark = True


def seed_worker(worker_id):
    s = torch.initial_seed() % 2**32
    np.random.seed(s); random.seed(s)


seed_everything(CFG["SEED"])
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_AMP = DEVICE.type == "cuda"
print("Device:", DEVICE, "| AMP:", USE_AMP)
if DEVICE.type == "cuda":
    print(torch.cuda.get_device_name(0))
with open(os.path.join(DIRS["logs"], "config.json"), "w") as f:
    json.dump({k: v for k, v in CFG.items()}, f, indent=2, default=str)


# %% [cell 2]
# ---------------------------------------------------------------------------
# 2. Publication figure style (Elsevier artwork guidelines)
#    single column 90 mm, 1.5 column 140 mm, double column 190 mm
#    fonts 7-8 pt, embedded TrueType, 600 dpi raster, vector PDF
# ---------------------------------------------------------------------------
MM = 1 / 25.4
FIG_W = {"single": 90 * MM, "onehalf": 140 * MM, "double": 190 * MM}

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"],
    "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "lines.linewidth": 1.2, "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False, "figure.dpi": 150, "savefig.dpi": 600,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
    "pdf.fonttype": 42, "ps.fonttype": 42, "image.interpolation": "none",
})

# Okabe-Ito colour-blind-safe palette
OI = dict(blue="#0072B2", orange="#E69F00", green="#009E73", vermillion="#D55E00",
          purple="#CC79A7", sky="#56B4E9", yellow="#F0E442", black="#000000", grey="#7F7F7F")
MODEL_COLORS = [OI["grey"], OI["sky"], OI["blue"], OI["orange"], OI["purple"], OI["green"]]
SHORT = {"U-Net": "U-Net", "Attention U-Net": "Att. U-Net", "U-Net (ResNet-34)": "U-Net-R34",
         "U-Net++ (ResNet-34)": "U-Net++-R34", "DeepLabV3+ (ResNet-34)": "DLv3+-R34"}
SPLIT_COLORS = {"train": OI["blue"], "valid": OI["orange"], "test": OI["grey"]}


def save_figure(fig, name):
    """Write PDF (vector), 600 dpi LZW TIFF (submission) and 300 dpi PNG (preview)."""
    base = os.path.join(DIRS["figures"], name)
    fig.savefig(base + ".pdf")
    fig.savefig(base + ".tiff", dpi=600, pil_kwargs={"compression": "tiff_lzw"})
    fig.savefig(base + ".png", dpi=300)
    print("saved", base + ".{pdf,tiff,png}")


def panel_label(ax, letter, x=-0.02, y=1.02):
    ax.text(x, y, f"({letter})", transform=ax.transAxes, fontsize=8,
            fontweight="bold", va="bottom", ha="right" if x < 0 else "left")


def image_axis(ax):
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


# %% [cell 3]
IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


def find_annotation_file(split_dir):
    for name in ["_combined_annotations.json", "_annotations.coco.json"]:
        p = os.path.join(split_dir, name)
        if os.path.exists(p):
            return p
    js = sorted(glob.glob(os.path.join(split_dir, "*.json")))
    return js[0] if js else None


def make_unique_ids(imgs, anns, split):
    """Give every image a unique key `uid` and link annotations to it.

    Merged ("combined") COCO files often repeat image ids, because each source file
    restarts its numbering. Such files are concatenations of blocks, so a new block
    starts wherever the image id (or annotation image_id) sequence drops.
    """
    imgs = imgs.drop_duplicates(subset=["id", "file_name"]).reset_index(drop=True)
    if not imgs["id"].duplicated().any():
        imgs["uid"] = imgs["id"]; anns["uid"] = anns["image_id"]
        return _merge_same_file(imgs, anns, split)
    n_dup = int(imgs["id"].duplicated().sum())
    img_block = (imgs["id"].diff() <= 0).cumsum()
    ann_block = (anns["image_id"].diff() < 0).cumsum()
    if img_block.max() != ann_block.max():
        raise ValueError(
            f"[{split}] {n_dup} repeated image ids, and the file is not a simple concatenation "
            f"({img_block.max() + 1} image blocks vs {ann_block.max() + 1} annotation blocks). "
            "Point this split at the original Roboflow '_annotations.coco.json' instead.")
    K = 10 ** 7
    imgs["uid"] = img_block * K + imgs["id"]
    anns["uid"] = ann_block * K + anns["image_id"]
    # Sanity check: every annotation must find its image and its box must lie inside it
    dims = imgs.set_index("uid")[["width", "height"]]
    ok = anns["uid"].isin(dims.index)
    j = anns[ok].join(dims, on="uid")
    b = np.stack(j["bbox"].values)
    inside = ((b[:, 0] + b[:, 2] <= j["width"] + 2) & (b[:, 1] + b[:, 3] <= j["height"] + 2)).mean()
    print(f"[{split}] {n_dup} repeated image ids resolved into {img_block.max() + 1} blocks | "
          f"annotations matched {ok.mean():.1%} | boxes inside image {inside:.1%}")
    if ok.mean() < 0.99 or inside < 0.99:
        raise ValueError(f"[{split}] could not safely re-link annotations to images. "
                         "Use the original per-split COCO file.")
    return _merge_same_file(imgs, anns, split)


def _merge_same_file(imgs, anns, split):
    """One record per image file; exact duplicate annotations are removed."""
    first = imgs.groupby("file_name")["uid"].transform("first")
    remap = dict(zip(imgs["uid"], first))
    n_rep = int(imgs["file_name"].duplicated().sum())
    imgs = imgs.drop_duplicates("file_name").reset_index(drop=True)
    anns = anns.copy(); anns["uid"] = anns["uid"].map(remap)
    key = anns["bbox"].map(lambda b: tuple(np.round(b, 1)))
    n0 = len(anns)
    anns = anns[~pd.DataFrame({"u": anns["uid"], "c": anns["category_id"], "k": key}).duplicated()].reset_index(drop=True)
    if n_rep or n0 - len(anns):
        print(f"[{split}] merged {n_rep} repeated file entries, removed {n0 - len(anns)} duplicate annotations")
    return imgs, anns


def load_coco(split):
    split_dir = os.path.join(CFG["DATA_ROOT"], split)
    ann_file = find_annotation_file(split_dir)
    if ann_file is None:
        raise FileNotFoundError(f"No COCO json in {split_dir}")
    with open(ann_file) as f:
        coco = json.load(f)
    imgs = pd.DataFrame(coco["images"])
    anns = pd.DataFrame(coco["annotations"])
    imgs, anns = make_unique_ids(imgs, anns, split)
    imgs["path"] = imgs["file_name"].map(lambda n: os.path.join(split_dir, os.path.basename(n)))
    missing = (~imgs["path"].map(os.path.exists)).sum()
    imgs = imgs[imgs["path"].map(os.path.exists)].reset_index(drop=True)
    cats = {c["id"]: c["name"] for c in coco["categories"]}
    anns["category"] = anns["category_id"].map(cats)
    anns = anns[anns["uid"].isin(imgs["uid"])].reset_index(drop=True)
    on_disk = len([p for p in os.listdir(split_dir) if p.lower().endswith(IMG_EXT)])
    print(f"{split:5s}: {os.path.basename(ann_file)} | {len(imgs)} images in json "
          f"({missing} missing on disk, {on_disk} image files in folder) | {len(anns)} annotations")
    return imgs, anns


coco = {s: load_coco(s) for s in CFG["SPLITS"]}

# Global class list: every category that owns at least one annotation (Roboflow adds an empty super-category)
_names = sorted(set().union(*[set(a["category"].dropna()) for _, a in coco.values()]))
FG_CLASSES = sorted(_names, key=lambda n: ("abnormal" in n.lower(), n.lower()))  # Abnormal drawn last
CLASS_NAMES = ["Background"] + FG_CLASSES
NC = len(CLASS_NAMES)
CLS_IDX = {n: i for i, n in enumerate(CLASS_NAMES)}
_default_cols = [OI["green"], OI["vermillion"], OI["blue"], OI["purple"], OI["yellow"]]
CLASS_COLORS = {"Background": "#000000"}
for i, n in enumerate(FG_CLASSES):
    CLASS_COLORS[n] = OI["vermillion"] if "abnormal" in n.lower() else (
        OI["green"] if "normal" in n.lower() else _default_cols[i % len(_default_cols)])
print("Classes:", CLASS_NAMES)


def _polygons(ann):
    seg = ann.get("segmentation", None)
    polys = []
    if isinstance(seg, list) and len(seg) > 0:
        if isinstance(seg[0], (int, float)):
            seg = [seg]
        for p in seg:
            if len(p) >= 6:
                polys.append(np.round(np.asarray(p, dtype=np.float64).reshape(-1, 2)).astype(np.int32))
    if not polys:  # RLE or empty: fall back to the bounding box
        x, y, w, h = ann["bbox"]
        polys = [np.round(np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]])).astype(np.int32)]
    return polys


def rasterize(h, w, img_anns):
    mask = np.zeros((h, w), np.uint8)
    for c in range(1, NC):                       # later classes overwrite earlier ones
        for ann in img_anns[img_anns["category"] == CLASS_NAMES[c]].to_dict("records"):
            cv2.fillPoly(mask, _polygons(ann), int(c))
    return mask


RECORDS = {}
for split, (imgs, anns) in coco.items():
    os.makedirs(os.path.join(DIRS["masks"], split), exist_ok=True)
    by_img = dict(tuple(anns.groupby("uid")))
    recs = []
    for row in tqdm(imgs.to_dict("records"), desc=f"masks {split}", leave=False):
        if "height" in row and not pd.isna(row.get("height")):
            h, w = int(row["height"]), int(row["width"])
        else:
            h, w = cv2.imread(row["path"]).shape[:2]
        stem = os.path.splitext(os.path.basename(row["path"]))[0]
        mpath = os.path.join(DIRS["masks"], split, stem + ".png")
        ia = by_img.get(row["uid"], anns.iloc[0:0])
        cv2.imwrite(mpath, rasterize(h, w, ia))   # always rebuilt (fast), so masks never go stale
        recs.append(dict(split=split, image_id=row["uid"], stem=stem, img_path=row["path"],
                         mask_path=mpath, h=h, w=w, n_ann=len(ia)))
    RECORDS[split] = recs

# Network input size: keep the median aspect ratio, both sides divisible by 32
_ar = np.median([r["w"] / r["h"] for s in RECORDS for r in RECORDS[s]])
if _ar >= 1:
    IMG_W, IMG_H = CFG["LONG_SIDE"], int(round(CFG["LONG_SIDE"] / _ar / 32) * 32)
else:
    IMG_H, IMG_W = CFG["LONG_SIDE"], int(round(CFG["LONG_SIDE"] * _ar / 32) * 32)
print(f"Median aspect ratio {_ar:.3f} -> network input {IMG_H} x {IMG_W}")


# %% [cell 4]
inst_rows, img_rows = [], []
for split, (imgs, anns) in coco.items():
    size = imgs.set_index("uid")[["width", "height"]] if "width" in imgs else None
    for a in anns.to_dict("records"):
        if a["category"] not in FG_CLASSES:
            continue
        W, H = (size.at[a["uid"], "width"], size.at[a["uid"], "height"]) if size is not None else (np.nan, np.nan)
        bw, bh = a["bbox"][2], a["bbox"][3]
        inst_rows.append(dict(split=split, cls=a["category"], side_px=math.sqrt(bw * bh),
                              rel_area_pct=100 * bw * bh / (W * H)))
    for r in RECORDS[split]:
        m = cv2.imread(r["mask_path"], cv2.IMREAD_UNCHANGED)
        cnt = np.bincount(m.ravel(), minlength=NC)
        row = dict(split=split, stem=r["stem"], fg_pct=100 * cnt[1:].sum() / m.size)
        ia = anns[anns["uid"] == r["image_id"]]
        for c in FG_CLASSES:
            row[f"n_{c}"] = int((ia["category"] == c).sum())
            row[f"px_{c}"] = int(cnt[CLS_IDX[c]])
        row["px_Background"] = int(cnt[0])
        img_rows.append(row)
INST = pd.DataFrame(inst_rows)
IMGS = pd.DataFrame(img_rows)

# Table 1
t1 = []
for split in CFG["SPLITS"]:
    d, di = IMGS[IMGS.split == split], INST[INST.split == split]
    row = {"Split": split.capitalize(), "Images": len(d)}
    for c in FG_CLASSES:
        row[f"{c} instances"] = int(d[f"n_{c}"].sum())
    row["Instances / image"] = f"{d[[f'n_{c}' for c in FG_CLASSES]].sum(1).mean():.1f} ± {d[[f'n_{c}' for c in FG_CLASSES]].sum(1).std():.1f}"
    row["Object size, px (median)"] = f"{di.side_px.median():.0f}"
    row["Foreground, % (mean ± SD)"] = f"{d.fg_pct.mean():.2f} ± {d.fg_pct.std():.2f}"
    t1.append(row)
TABLE1 = pd.DataFrame(t1)


def to_latex_booktabs(df, caption, label, path):
    cols = list(df.columns)
    esc = lambda s: str(s).replace("%", r"\%").replace("_", r"\_").replace("±", r"$\pm$").replace("&", r"\&")
    lines = [r"\begin{table}[t]", r"\centering", rf"\caption{{{caption}}}", rf"\label{{{label}}}",
             r"\small", r"\begin{tabular}{" + "l" + "c" * (len(cols) - 1) + "}", r"\toprule",
             " & ".join(esc(c) for c in cols) + r" \\", r"\midrule"]
    lines += [" & ".join(esc(v) for v in r) + r" \\" for r in df.astype(str).values]
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    with open(path, "w") as f:
        f.write("\n".join(lines))


TABLE1.to_csv(os.path.join(DIRS["tables"], "table1_dataset.csv"), index=False)
to_latex_booktabs(TABLE1, "Composition of the Pap smear dataset.", "tab:dataset",
                  os.path.join(DIRS["tables"], "table1_dataset.tex"))
display(TABLE1) if "display" in globals() else print(TABLE1.to_string(index=False))

# Pixel-level class frequency in the training split (used for loss weights)
PIX_FREQ = np.array([IMGS[IMGS.split == "train"][f"px_{c}"].sum() for c in CLASS_NAMES], dtype=float)
PIX_FREQ /= PIX_FREQ.sum()
print("Training pixel frequency:", dict(zip(CLASS_NAMES, np.round(PIX_FREQ, 4))))


# %% [cell 5]
# Fig. 1  Dataset overview
fig, axs = plt.subplots(2, 2, figsize=(FIG_W["double"], 105 * MM), constrained_layout=True)

ax = axs[0, 0]
xs = np.arange(len(CFG["SPLITS"])); bw = 0.8 / len(FG_CLASSES)
for k, c in enumerate(FG_CLASSES):
    vals = [IMGS[IMGS.split == s][f"n_{c}"].sum() for s in CFG["SPLITS"]]
    bars = ax.bar(xs + (k - (len(FG_CLASSES) - 1) / 2) * bw, vals, bw, color=CLASS_COLORS[c], label=c, edgecolor="none")
    ax.bar_label(bars, fontsize=6, padding=1)
ax.set_xticks(xs, [s.capitalize() for s in CFG["SPLITS"]])
ax.set_ylabel("Annotated instances"); ax.legend(loc="upper right")
ax.margins(y=0.15); panel_label(ax, "a")

ax = axs[0, 1]
data = [IMGS[f"n_{c}"].values for c in FG_CLASSES]
parts = ax.violinplot(data, showextrema=False, widths=0.7)
for b, c in zip(parts["bodies"], FG_CLASSES):
    b.set_facecolor(CLASS_COLORS[c]); b.set_alpha(0.35); b.set_edgecolor("none")
ax.boxplot(data, widths=0.15, showfliers=False, medianprops=dict(color="k", lw=1),
           boxprops=dict(lw=0.6), whiskerprops=dict(lw=0.6), capprops=dict(lw=0.6))
ax.set_xticks(range(1, len(FG_CLASSES) + 1), FG_CLASSES)
ax.set_ylabel("Instances per image"); panel_label(ax, "b")

ax = axs[1, 0]
bins = np.logspace(np.log10(max(INST.side_px.min(), 1)), np.log10(INST.side_px.max() + 1), 35)
for c in FG_CLASSES:
    ax.hist(INST[INST.cls == c].side_px, bins=bins, histtype="stepfilled", alpha=0.35, color=CLASS_COLORS[c], label=c)
    ax.hist(INST[INST.cls == c].side_px, bins=bins, histtype="step", color=CLASS_COLORS[c], lw=0.9)
    ax.axvline(INST[INST.cls == c].side_px.median(), color=CLASS_COLORS[c], ls="--", lw=0.8)
ax.set_xscale("log")
ax.xaxis.set_major_formatter(mpl.ticker.ScalarFormatter()); ax.xaxis.set_minor_formatter(mpl.ticker.NullFormatter())
ax.set_xlabel(r"Object size, $\sqrt{\mathrm{box\ area}}$ (px)"); ax.set_ylabel("Count")
ax.legend(); panel_label(ax, "c")

ax = axs[1, 1]
data = [IMGS[IMGS.split == s].fg_pct.values for s in CFG["SPLITS"]]
ax.boxplot(data, widths=0.5, showfliers=False, medianprops=dict(color="k", lw=1),
           boxprops=dict(lw=0.6), whiskerprops=dict(lw=0.6), capprops=dict(lw=0.6))
rng = np.random.default_rng(0)
for i, (d, s) in enumerate(zip(data, CFG["SPLITS"])):
    ax.scatter(i + 1 + rng.uniform(-0.15, 0.15, len(d)), d, s=4, color=SPLIT_COLORS[s], alpha=0.6, lw=0)
ax.set_xticks(range(1, len(CFG["SPLITS"]) + 1), [s.capitalize() for s in CFG["SPLITS"]])
ax.set_ylabel("Foreground pixels per image (%)"); panel_label(ax, "d")
save_figure(fig, "Fig1_dataset_overview"); plt.show()


# %% [cell 6]
# Fig. 2  Representative annotated samples (training split)


def read_rgb(path):
    return cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)


def overlay_mask(ax, img, mask, alpha=0.40, contour=True, lw=0.6):
    ax.imshow(img)
    rgba = np.zeros((*mask.shape, 4))
    for c in range(1, NC):
        col = mpl.colors.to_rgb(CLASS_COLORS[CLASS_NAMES[c]])
        rgba[mask == c] = (*col, alpha)
    ax.imshow(rgba)
    if contour:
        for c in range(1, NC):
            if (mask == c).any():
                ax.contour(mask == c, levels=[0.5], colors=[CLASS_COLORS[CLASS_NAMES[c]]], linewidths=lw)
    image_axis(ax)


def class_legend(fig, extra=None, y=-0.01, ncol=None):
    handles = [Patch(facecolor=CLASS_COLORS[c], edgecolor=CLASS_COLORS[c], alpha=0.8, label=c) for c in FG_CLASSES]
    handles += extra or []
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, y), ncol=ncol or len(handles))


# Fig. 2 layout: true-colour images, class outlines only (fill would hide the stain), 8 samples from sparse to dense
import matplotlib.patheffects as pe
VIS_CLASS = {c: ("#E6007E" if "abnormal" in c.lower() else "#00B0FF" if "normal" in c.lower() else "#FFD600")
             for c in FG_CLASSES}

def display_size(h, w, long=512):
    s_ = long / max(h, w)
    return int(round(w * s_)), int(round(h * s_))


tr = IMGS[IMGS.split == "train"].copy()
tr["both"] = (tr[[f"n_{c}" for c in FG_CLASSES]] > 0).all(1)
pool = tr[tr.both] if tr.both.sum() >= 8 else tr
pool = pool.sort_values("fg_pct").reset_index(drop=True)            # sparse -> dense
pick = pool.iloc[np.linspace(0, len(pool) - 1, 8).round().astype(int)].stem.tolist()
rec_by_stem = {r["stem"]: r for r in RECORDS["train"]}

n_r, n_c = 2, 4
aspect = np.median([rec_by_stem[s]["h"] / rec_by_stem[s]["w"] for s in pick])
Wf, left, right, wsp, hsp = FIG_W["double"], 0.005, 0.995, 0.02, 0.03
axw = (right - left) * Wf / (n_c + (n_c - 1) * wsp); axh = axw * aspect
top_in, bot_in = 0.05, 0.46
Hf = top_in + bot_in + n_r * axh + (n_r - 1) * hsp * axh
fig, axs = plt.subplots(n_r, n_c, figsize=(Wf, Hf), squeeze=False)
fig.subplots_adjust(left=left, right=right, top=1 - top_in / Hf, bottom=bot_in / Hf, wspace=wsp, hspace=hsp)

for k, (ax, stem) in enumerate(zip(axs.ravel(), pick)):
    r = rec_by_stem[stem]; dsz = display_size(r["h"], r["w"], long=1024)
    img = cv2.resize(read_rgb(r["img_path"]), dsz, interpolation=cv2.INTER_AREA)
    m = cv2.resize(cv2.imread(r["mask_path"], cv2.IMREAD_UNCHANGED), dsz, interpolation=cv2.INTER_NEAREST)
    ax.imshow(img)
    for c in FG_CLASSES:
        mc = m == CLS_IDX[c]
        if mc.any():
            cs = ax.contour(mc, levels=[0.5], colors=[VIS_CLASS[c]], linewidths=0.9)
            cs.set(path_effects=[pe.Stroke(linewidth=2.0, foreground="black"), pe.Normal()])
    image_axis(ax)
    row = tr.set_index("stem").loc[stem]
    counts = "   ".join(f"{c[0].upper()}: {int(row[f'n_{c}'])}" for c in FG_CLASSES)
    ax.text(0.98, 0.03, counts, transform=ax.transAxes, ha="right", va="bottom", fontsize=6, color="white",
            bbox=dict(boxstyle="round,pad=0.25", fc="black", ec="none", alpha=0.65))
    ax.text(0.02, 0.97, f"({'abcdefgh'[k]})", transform=ax.transAxes, ha="left", va="top", fontsize=7,
            fontweight="bold", color="black", bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.8))

handles = [Line2D([0], [0], color=VIS_CLASS[c], lw=1.6, path_effects=[pe.Stroke(linewidth=3, foreground="black"), pe.Normal()],
                  label=f"{c.capitalize()} ({c[0].upper()})") for c in FG_CLASSES]
fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.0), ncol=len(handles), handlelength=2.2, columnspacing=2.0,
           title="Annotated cells (count per image in each panel)", title_fontsize=7)
save_figure(fig, "Fig2_annotated_samples"); plt.show()


# %% [cell 7]
def apply_clahe(rgb, clip=2.0, tiles=8):
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    lab[..., 0] = cv2.createCLAHE(clipLimit=clip, tileGridSize=(tiles, tiles)).apply(lab[..., 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


r = rec_by_stem[pick[0]]
img0 = read_rgb(r["img_path"]); mask0 = cv2.imread(r["mask_path"], cv2.IMREAD_UNCHANGED)
img_c = apply_clahe(img0)
gray = cv2.cvtColor(img_c, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255
sigma = float(np.mean(estimate_sigma(gray)))
den = denoise_nl_means(gray, h=1.15 * sigma, sigma=sigma, fast_mode=True, patch_size=5, patch_distance=6)
gx = cv2.Scharr(den.astype(np.float32), cv2.CV_32F, 1, 0); gy = cv2.Scharr(den.astype(np.float32), cv2.CV_32F, 0, 1)
grad = np.hypot(gx, gy); grad /= np.percentile(grad, 99.5) + 1e-8

fig = plt.figure(figsize=(FIG_W["double"], 100 * MM), constrained_layout=True)
gs = fig.add_gridspec(2, 3)
ax = fig.add_subplot(gs[0, 0]); ax.imshow(img0); image_axis(ax); ax.set_title("Original"); panel_label(ax, "a", x=0)
ax = fig.add_subplot(gs[0, 1]); ax.imshow(img_c); image_axis(ax); ax.set_title("CLAHE (L channel)"); panel_label(ax, "b", x=0)
ax = fig.add_subplot(gs[0, 2]); ax.imshow(den, cmap="gray"); image_axis(ax)
ax.set_title(f"NLM denoised ($\\hat\\sigma$ = {sigma:.3f})"); panel_label(ax, "c", x=0)
ax = fig.add_subplot(gs[1, 0]); im = ax.imshow(np.clip(grad, 0, 1), cmap="magma"); image_axis(ax)
ax.set_title("Scharr gradient magnitude"); panel_label(ax, "d", x=0)
fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02, label="Normalised magnitude")
ax = fig.add_subplot(gs[1, 1])
L0 = cv2.cvtColor(img0, cv2.COLOR_RGB2LAB)[..., 0].ravel(); L1 = cv2.cvtColor(img_c, cv2.COLOR_RGB2LAB)[..., 0].ravel()
ax.hist(L0, bins=128, range=(0, 255), histtype="step", color=OI["grey"], label="Original", density=True)
ax.hist(L1, bins=128, range=(0, 255), histtype="step", color=OI["blue"], label="CLAHE", density=True)
ax.set_xlabel("L* intensity (8-bit)"); ax.set_ylabel("Density"); ax.legend(); ax.set_box_aspect(0.75)
panel_label(ax, "e")
ax = fig.add_subplot(gs[1, 2]); overlay_mask(ax, img_c, mask0, alpha=0.3); ax.set_title("Ground truth"); panel_label(ax, "f", x=0)
save_figure(fig, "Fig3_preprocessing"); plt.show()


# %% [cell 8]
# Images are resized once when cached (see PapSegDataset), so no Resize step here
train_tf = A.Compose([
    A.HorizontalFlip(p=0.5), A.VerticalFlip(p=0.5),
    A.Affine(scale=(0.9, 1.1), translate_percent=(-0.05, 0.05), rotate=(-20, 20), p=0.5),
    A.HueSaturationValue(hue_shift_limit=8, sat_shift_limit=15, val_shift_limit=10, p=0.5),
    A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.5),
    A.GaussianBlur(blur_limit=(3, 5), p=0.1),
    A.Normalize(), ToTensorV2(),
])
eval_tf = A.Compose([A.Normalize(), ToTensorV2()])


from concurrent.futures import ThreadPoolExecutor

_CACHE = {}


def _load_resized(r, clahe):
    """Read once from Drive, CLAHE at full resolution, resize to network size."""
    img = read_rgb(r["img_path"])
    if clahe:
        img = apply_clahe(img)
    img = cv2.resize(img, (IMG_W, IMG_H), interpolation=cv2.INTER_AREA)
    mask = cv2.resize(cv2.imread(r["mask_path"], cv2.IMREAD_UNCHANGED), (IMG_W, IMG_H), interpolation=cv2.INTER_NEAREST)
    return img, mask


from skimage.color import rgb2hed, hed2rgb


def hed_jitter(img, sigma=0.05, bias=0.05):
    """Stain augmentation (Tellez et al., 2018): random scale and shift of the H, E, D channels."""
    hed = rgb2hed(img)
    hed = hed * np.random.uniform(1 - sigma, 1 + sigma, 3) + np.random.uniform(-bias, bias, 3) * hed.std((0, 1))
    return (np.clip(hed2rgb(hed), 0, 1) * 255).astype(np.uint8)


class PapSegDataset(Dataset):
    """Images and masks are pre-processed once and held in RAM (about 0.6 MB per image at 384 x 512)."""
    def __init__(self, records, transform, clahe=CFG["USE_CLAHE"], train=False):
        self.records, self.transform, self.train = records, transform, train
        todo = [r for r in records if (r["img_path"], clahe) not in _CACHE]
        if todo:
            with ThreadPoolExecutor(8) as ex:
                for r, out in zip(todo, tqdm(ex.map(lambda r: _load_resized(r, clahe), todo), total=len(todo), desc="caching", leave=False)):
                    _CACHE[(r["img_path"], clahe)] = out
        self.items = [_CACHE[(r["img_path"], clahe)] for r in records]

    def __len__(self):
        return len(self.records)

    def __getitem__(self, i):
        img, mask = self.items[i]
        if self.train and CFG["HED_AUG"] and np.random.rand() < 0.5:
            img = hed_jitter(img)
        out = self.transform(image=img, mask=mask)
        return out["image"], out["mask"].long(), i


def make_loader(split, train):
    ds = PapSegDataset(RECORDS[split], train_tf if train else eval_tf, train=train)
    g = torch.Generator(); g.manual_seed(CFG["SEED"])
    return DataLoader(ds, batch_size=CFG["BATCH_SIZE"], shuffle=train, drop_last=train and len(ds) > CFG["BATCH_SIZE"],
                      num_workers=CFG["NUM_WORKERS"], pin_memory=DEVICE.type == "cuda",
                      worker_init_fn=seed_worker, generator=g, persistent_workers=CFG["NUM_WORKERS"] > 0)


loaders = {"train": make_loader("train", True), "valid": make_loader("valid", False)}
x, y, _ = next(iter(loaders["train"]))
print("batch", tuple(x.shape), tuple(y.shape), "labels present:", torch.unique(y).tolist())


# %% [cell 9]
# ---------------------------------------------------------------------------
# 7. Models: proposed MTA-TransUNet, baselines and ablations
# ---------------------------------------------------------------------------
PROPOSED = "MTA-TransUNet"
BASELINES = ["U-Net (R34)", "U-Net++ (R34)", "DeepLabV3+ (R34)", "MA-Net (R34)", "SegFormer (MiT-B2)", "UPerNet (ConvNeXt-T)"]
ABLATIONS = ["+ scSE attention", "− auxiliary head", "− transformer encoder"]
# Final design chosen by validation mDice: MiT-B2 encoder + U-Net decoder + image-level auxiliary head.
# Reuse checkpoints already trained under earlier names (no retraining needed for these two).
CKPT_ALIAS = {PROPOSED: "w_o_scSE_attention", "+ scSE attention": "MTA_TransUNet"}
MAIN_MODELS = BASELINES + [PROPOSED]
CFG["MODELS"] = MAIN_MODELS + (ABLATIONS if CFG["RUN_ABLATION"] else [])
ABN_IDX = next((CLS_IDX[c] for c in FG_CLASSES if "abnormal" in c.lower()), NC - 1)

MODEL_COLOR = {"U-Net (R34)": "#7F7F7F", "U-Net++ (R34)": "#56B4E9", "DeepLabV3+ (R34)": "#0072B2",
               "MA-Net (R34)": "#009E73", "SegFormer (MiT-B2)": "#E69F00", "UPerNet (ConvNeXt-T)": "#CC79A7",
               PROPOSED: "#C0392B", "+ scSE attention": "#F1948A", "− auxiliary head": "#E59866",
               "− transformer encoder": "#A9A9A9"}
SHORT = {m: m for m in CFG["MODELS"]}
SHORT.update({"U-Net (R34)": "U-Net", "U-Net++ (R34)": "U-Net++", "DeepLabV3+ (R34)": "DeepLabV3+",
              "MA-Net (R34)": "MA-Net", "SegFormer (MiT-B2)": "SegFormer", "UPerNet (ConvNeXt-T)": "UPerNet",
              PROPOSED: "MTA-TransUNet (proposed)"})


class MultiTaskSegNet(nn.Module):
    """U-Net decoder on any encoder, optional scSE attention and optional image-level head.
    Training returns (mask logits, image logit); evaluation returns mask logits only."""
    def __init__(self, encoder, attention=True, aux=True, weights=None):
        super().__init__()
        self.aux = aux
        self.net = smp.Unet(encoder, encoder_weights=weights, in_channels=3, classes=NC,
                            decoder_attention_type="scse" if attention else None,
                            aux_params=dict(classes=1, dropout=0.2) if aux else None)
        self.segmentation_head = self.net.segmentation_head

    def forward(self, x):
        out = self.net(x)
        if self.aux:
            return out if self.training else out[0]
        return out


def build_model(name):
    w = CFG["ENCODER_WEIGHTS"]
    if name == PROPOSED:
        return MultiTaskSegNet("mit_b2", attention=False, aux=True, weights=w)
    if name == "+ scSE attention":
        return MultiTaskSegNet("mit_b2", attention=True, aux=True, weights=w)
    if name == "− auxiliary head":
        return MultiTaskSegNet("mit_b2", attention=False, aux=False, weights=w)
    if name == "− transformer encoder":
        return MultiTaskSegNet("resnet34", attention=False, aux=True, weights=w)
    if name == "U-Net (R34)":
        return smp.Unet("resnet34", encoder_weights=w, classes=NC)
    if name == "U-Net++ (R34)":
        return smp.UnetPlusPlus("resnet34", encoder_weights=w, classes=NC)
    if name == "DeepLabV3+ (R34)":
        return smp.DeepLabV3Plus("resnet34", encoder_weights=w, classes=NC)
    if name == "MA-Net (R34)":
        return smp.MAnet("resnet34", encoder_weights=w, classes=NC)
    if name == "SegFormer (MiT-B2)":
        return smp.Segformer("mit_b2", encoder_weights=w, classes=NC)
    if name == "UPerNet (ConvNeXt-T)":
        return smp.UPerNet("tu-convnext_tiny", encoder_weights=w, classes=NC)
    raise ValueError(name)


def count_params(m):
    return sum(p.numel() for p in m.parameters()) / 1e6


for n in CFG["MODELS"]:
    m = build_model(n).eval()
    with torch.no_grad():
        o = m(torch.zeros(1, 3, IMG_H, IMG_W))
    print(f"{n:26s} {count_params(m):6.1f} M  {tuple(o.shape)}")
del m


# %% [cell 10]
# ---------------------------------------------------------------------------
# 8. Loss, metrics, resumable training (identical protocol for every model)
# ---------------------------------------------------------------------------
CE_W = 1 / np.sqrt(PIX_FREQ + 1e-8); CE_W = np.minimum(CE_W / CE_W.mean(), 10.0)


class DiceCELoss(nn.Module):
    def __init__(self, n_classes, ce_weight, w_ce=1.0, w_dice=1.0, smooth=1.0):
        super().__init__()
        self.n, self.w_ce, self.w_dice, self.smooth = n_classes, w_ce, w_dice, smooth
        self.register_buffer("ce_weight", torch.tensor(ce_weight, dtype=torch.float32))

    def forward(self, logits, target):
        logits = logits.float()
        ce = F.cross_entropy(logits, target, weight=self.ce_weight)
        prob = logits.softmax(1)
        oh = F.one_hot(target, self.n).permute(0, 3, 1, 2).float()
        inter = (prob * oh).sum((0, 2, 3)); card = prob.sum((0, 2, 3)) + oh.sum((0, 2, 3))
        dice = (2 * inter + self.smooth) / (card + self.smooth)
        return self.w_ce * ce + self.w_dice * (1 - dice[1:].mean())


def total_loss(out, y, criterion):
    if isinstance(out, (tuple, list)):
        seg, cls = out
        tgt = (y == ABN_IDX).flatten(1).any(1).float()[:, None]
        return criterion(seg, y) + CFG["W_AUX"] * F.binary_cross_entropy_with_logits(cls.float(), tgt)
    return criterion(out, y)


def confusion(pred, gt, n=NC):
    return np.bincount(n * gt.ravel().astype(np.int64) + pred.ravel(), minlength=n * n).reshape(n, n)


def _safe(a, b):
    return a / b if b > 0 else np.nan


def metrics_from_cm(cm, c):
    tp = cm[c, c]; fp = cm[:, c].sum() - tp; fn = cm[c, :].sum() - tp; tn = cm.sum() - tp - fp - fn
    return dict(Dice=_safe(2 * tp, 2 * tp + fp + fn), IoU=_safe(tp, tp + fp + fn), Precision=_safe(tp, tp + fp),
                Recall=_safe(tp, tp + fn), Specificity=_safe(tn, tn + fp))


def binary_metrics(pred, gt):
    tp = np.sum(pred & gt); fp = np.sum(pred & ~gt); fn = np.sum(~pred & gt); tn = np.sum(~pred & ~gt)
    return dict(Dice=_safe(2 * tp, 2 * tp + fp + fn), IoU=_safe(tp, tp + fp + fn), Precision=_safe(tp, tp + fp),
                Recall=_safe(tp, tp + fn), Specificity=_safe(tn, tn + fp), Accuracy=(tp + tn) / pred.size)


def hd95(pred, gt):
    if not pred.any() and not gt.any():
        return 0.0
    if not pred.any() or not gt.any():
        return np.nan
    pb = pred ^ binary_erosion(pred); gb = gt ^ binary_erosion(gt)
    d = np.concatenate([distance_transform_edt(~gb)[pb], distance_transform_edt(~pb)[gb]])
    return float(np.percentile(d, 95))


def lr_lambda(ep):
    if ep < CFG["WARMUP_EPOCHS"]:
        return (ep + 1) / CFG["WARMUP_EPOCHS"]
    t = (ep - CFG["WARMUP_EPOCHS"]) / max(1, CFG["EPOCHS"] - CFG["WARMUP_EPOCHS"])
    return 0.01 + 0.99 * 0.5 * (1 + math.cos(math.pi * t))


@torch.no_grad()
def validate(model, loader, criterion):
    model.eval(); cm = np.zeros((NC, NC), np.int64); loss = 0.0
    for x, y, _ in loader:
        x = x.to(DEVICE, non_blocking=True, memory_format=torch.channels_last); y = y.to(DEVICE, non_blocking=True)
        with torch.autocast(DEVICE.type, dtype=torch.float16, enabled=USE_AMP):
            logits = model(x)
        loss += criterion(logits, y).item() * x.size(0)
        cm += confusion(logits.argmax(1).cpu().numpy(), y.cpu().numpy())
    dices = [metrics_from_cm(cm, c)["Dice"] for c in range(1, NC)]
    return loss / len(loader.dataset), float(np.nanmean(dices)), dices


def slug(name):
    if name in CKPT_ALIAS:
        return CKPT_ALIAS[name]
    return "".join(ch if ch.isalnum() else "_" for ch in name.replace("−", "minus")).strip("_")


def _atomic_save(obj, path):
    tmp = path + ".tmp"; torch.save(obj, tmp); os.replace(tmp, path)


def train_model(name):
    ckpt = os.path.join(DIRS["checkpoints"], slug(name) + ".pt")
    last = os.path.join(DIRS["checkpoints"], slug(name) + "_last.pt")
    hist_path = os.path.join(DIRS["logs"], slug(name) + "_history.csv")
    done_flag = os.path.join(DIRS["checkpoints"], slug(name) + ".done")
    if CFG["RESUME"] and os.path.exists(done_flag) and os.path.exists(ckpt):
        print(f"[{name}] training already finished")
        return pd.read_csv(hist_path)
    seed_everything(CFG["SEED"])
    model = build_model(name).to(DEVICE, memory_format=torch.channels_last)
    criterion = DiceCELoss(NC, CE_W, CFG["W_CE"], CFG["W_DICE"]).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=CFG["LR"], weight_decay=CFG["WEIGHT_DECAY"])
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    scaler = torch.amp.GradScaler(DEVICE.type, enabled=USE_AMP)
    start_ep, best, wait, hist, elapsed = 0, -1.0, 0, [], 0.0
    if CFG["RESUME"] and os.path.exists(last):
        st = torch.load(last, map_location=DEVICE, weights_only=False)
        model.load_state_dict(st["model"]); opt.load_state_dict(st["opt"])
        sched.load_state_dict(st["sched"]); scaler.load_state_dict(st["scaler"])
        start_ep, best, wait, hist, elapsed = st["epoch"], st["best"], st["wait"], st["hist"], st["elapsed"]
        torch.set_rng_state(st["rng_torch"]); np.random.set_state(st["rng_np"]); random.setstate(st["rng_py"])
        if DEVICE.type == "cuda" and st.get("rng_cuda") is not None:
            torch.cuda.set_rng_state_all(st["rng_cuda"])
        print(f"[{name}] resumed after epoch {start_ep} (best val Dice {best:.4f})")
    t0 = time.time()
    for ep in range(start_ep, CFG["EPOCHS"]):
        if wait >= CFG["PATIENCE"]:
            break
        model.train(); run = 0.0; nseen = 0
        for x, y, _ in loaders["train"]:
            x = x.to(DEVICE, non_blocking=True, memory_format=torch.channels_last); y = y.to(DEVICE, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(DEVICE.type, dtype=torch.float16, enabled=USE_AMP):
                loss = total_loss(model(x), y, criterion)
            scaler.scale(loss).backward()
            scaler.unscale_(opt); nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update()
            run += loss.item() * x.size(0); nseen += x.size(0)
        sched.step()
        va_loss, va_dice, va_cls = validate(model, loaders["valid"], criterion)
        hist.append(dict(epoch=ep + 1, train_loss=run / max(nseen, 1), val_loss=va_loss, val_dice=va_dice,
                         lr=opt.param_groups[0]["lr"], **{f"val_dice_{c}": d for c, d in zip(FG_CLASSES, va_cls)}))
        improved = va_dice > best + 1e-4
        if improved:
            best, wait = va_dice, 0
            _atomic_save({"model": model.state_dict(), "epoch": ep + 1, "val_dice": va_dice}, ckpt)
        else:
            wait += 1
        _atomic_save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                      "scaler": scaler.state_dict(), "epoch": ep + 1, "best": best, "wait": wait, "hist": hist,
                      "elapsed": elapsed + time.time() - t0, "rng_torch": torch.get_rng_state(),
                      "rng_np": np.random.get_state(), "rng_py": random.getstate(),
                      "rng_cuda": torch.cuda.get_rng_state_all() if DEVICE.type == "cuda" else None}, last)
        pd.DataFrame(hist).to_csv(hist_path, index=False)
        print(f"[{name}] ep {ep + 1:3d} | train {hist[-1]['train_loss']:.4f} | val {va_loss:.4f} | val Dice {va_dice:.4f}"
              f"{' *' if improved else ''}")
    h = pd.DataFrame(hist); h["train_time_min"] = (elapsed + time.time() - t0) / 60
    h.to_csv(hist_path, index=False)
    open(done_flag, "w").write(str(len(h)))
    del model; torch.cuda.empty_cache()
    return h


# %% [cell 11]
HISTORY = {}
for name in CFG["MODELS"]:
    HISTORY[name] = train_model(name)
    gc_ = __import__("gc"); gc_.collect()


# %% [cell 12]
# ---------------------------------------------------------------------------
# 9. Test evaluation: segmentation, cell-level and image-level classification
# ---------------------------------------------------------------------------
import gc
from sklearn.manifold import TSNE


def load_best(name):
    m = build_model(name).to(DEVICE)
    ck = torch.load(os.path.join(DIRS["checkpoints"], slug(name) + ".pt"), map_location=DEVICE, weights_only=False)
    m.load_state_dict(ck["model"]); m.eval()
    return m, ck["epoch"]


@torch.no_grad()
def predict_probs(model, x, out_hw):
    with torch.autocast(DEVICE.type, dtype=torch.float16, enabled=USE_AMP):
        p = model(x).float().softmax(1)
        if CFG["TTA"]:
            p = p + model(x.flip(-1)).float().softmax(1).flip(-1)
            p = p + model(x.flip(-2)).float().softmax(1).flip(-2)
            p = p / 3
    return F.interpolate(p, size=out_hw, mode="bilinear", align_corners=False)[0].cpu().numpy()


test_imgs, test_anns = coco["test"]
CELLS_BY_UID = {u: g[g.category.isin(FG_CLASSES)][["bbox", "category"]].values.tolist()
                for u, g in test_anns.groupby("uid")}
test_ds = PapSegDataset(RECORDS["test"], eval_tf)
rng = np.random.default_rng(CFG["SEED"])
DISP_LONG_Q = 640

ROWS, CMS, PIX, PRED_SMALL, BEST_EPOCH, CELL_ROWS, IMG_ROWS = [], {}, {}, {}, {}, [], []
PROB_ABN_SMALL, CELL_FEATS, CELL_CROPS = {}, [], []
for name in CFG["MODELS"]:
    model, BEST_EPOCH[name] = load_best(name)
    keep_small = name in MAIN_MODELS
    feat_store = {}
    hook = model.segmentation_head.register_forward_hook(lambda m_, i_, o_: feat_store.__setitem__("f", i_[0].detach())) \
        if name == PROPOSED else None
    cm_tot = np.zeros((NC, NC), np.int64); pix_p, pix_y = [], []
    if keep_small:
        PRED_SMALL[name] = {}
    for i in tqdm(range(len(test_ds)), desc=SHORT[name], leave=False):
        x, _, _ = test_ds[i]; r = RECORDS["test"][i]
        gt = cv2.imread(r["mask_path"], cv2.IMREAD_UNCHANGED); H, W = gt.shape
        prob = predict_probs(model, x[None].to(DEVICE), gt.shape)
        pred = prob.argmax(0).astype(np.uint8)
        cm = confusion(pred, gt); cm_tot += cm
        row = dict(model=name, image=r["stem"])
        for c in range(1, NC):
            for k, v in metrics_from_cm(cm, c).items():
                row[f"{k}_{CLASS_NAMES[c]}"] = v
        row["mDice"] = np.nanmean([row[f"Dice_{c}"] for c in FG_CLASSES])
        for k, v in binary_metrics(pred > 0, gt > 0).items():
            row[f"{k}_FG"] = v
        row["HD95_FG"] = hd95(pred > 0, gt > 0)
        ROWS.append(row)
        idx = rng.choice(gt.size, size=min(CFG["PIXELS_PER_IMAGE"], gt.size), replace=False)
        pix_p.append(prob.reshape(NC, -1)[:, idx].T.astype(np.float16)); pix_y.append(gt.ravel()[idx])

        # image-level: does the slide contain abnormal cells?
        IMG_ROWS.append(dict(model=name, image=r["stem"], label=int((gt == ABN_IDX).any()),
                             score=float(np.percentile(prob[ABN_IDX], 99.5))))
        # cell-level: classify every annotated cell from the mean class probability inside its box
        cells = CELLS_BY_UID.get(r["image_id"], [])
        if name == PROPOSED and cells:
            model(x[None].to(DEVICE))                          # one plain pass to fill the feature hook
            fmap = feat_store["f"][0].float().cpu().numpy(); fh, fw = fmap.shape[1:]
            img_full = read_rgb(r["img_path"])
        for (bx, by, bw, bh), cat in cells:
            x0, y0 = int(max(0, np.floor(bx))), int(max(0, np.floor(by)))
            x1, y1 = int(min(W, np.ceil(bx + bw))), int(min(H, np.ceil(by + bh)))
            if x1 <= x0 or y1 <= y0:
                continue
            mp = prob[:, y0:y1, x0:x1].reshape(NC, -1).mean(1)
            fgp = mp[1:]
            pred_c = FG_CLASSES[int(np.argmax(fgp))]
            CELL_ROWS.append(dict(model=name, image=r["stem"], true=cat, pred=pred_c,
                                  p_abn=float(mp[ABN_IDX] / (fgp.sum() + 1e-8)), p_fg=float(1 - mp[0]),
                                  detected=bool(1 - mp[0] >= 0.5)))
            if name == PROPOSED:
                fx0, fy0 = int(x0 * fw / W), int(y0 * fh / H)
                fx1, fy1 = max(fx0 + 1, int(np.ceil(x1 * fw / W))), max(fy0 + 1, int(np.ceil(y1 * fh / H)))
                CELL_FEATS.append(fmap[:, fy0:fy1, fx0:fx1].reshape(fmap.shape[0], -1).mean(1))
                pad = int(0.25 * max(x1 - x0, y1 - y0))
                cr = img_full[max(0, y0 - pad):min(H, y1 + pad), max(0, x0 - pad):min(W, x1 + pad)]
                CELL_CROPS.append(cv2.resize(cr, (96, 96), interpolation=cv2.INTER_AREA))
        if keep_small:
            dsz = display_size(H, W, long=DISP_LONG_Q)
            PRED_SMALL[name][r["stem"]] = cv2.resize(pred, dsz, interpolation=cv2.INTER_NEAREST)
            if name == PROPOSED:
                PROB_ABN_SMALL[r["stem"]] = cv2.resize(prob[ABN_IDX], dsz, interpolation=cv2.INTER_AREA).astype(np.float16)
        del prob, pred
    if hook is not None:
        hook.remove()
    CMS[name] = cm_tot
    PIX[name] = (np.concatenate(pix_p), np.concatenate(pix_y))
    del model; gc.collect(); torch.cuda.empty_cache()

PER_IMAGE = pd.DataFrame(ROWS); CELLS = pd.DataFrame(CELL_ROWS); IMGCLS = pd.DataFrame(IMG_ROWS)
PER_IMAGE.to_csv(os.path.join(DIRS["tables"], "per_image_test_metrics.csv"), index=False)
CELLS.to_csv(os.path.join(DIRS["tables"], "per_cell_predictions.csv"), index=False)
print(PER_IMAGE.groupby("model")[["mDice", "Dice_FG", "IoU_FG", "HD95_FG"]].mean().round(4).loc[CFG["MODELS"]])


# %% [cell 13]
# Computational cost
from torch.utils.flop_counter import FlopCounterMode
COST = []
for name in CFG["MODELS"]:
    m = build_model(name).to(DEVICE).eval(); x = torch.randn(1, 3, IMG_H, IMG_W, device=DEVICE)
    with torch.no_grad():
        fc = FlopCounterMode(display=False)
        with fc:
            m(x)
        for _ in range(3):
            m(x)
        if DEVICE.type == "cuda":
            torch.cuda.synchronize()
        reps = 3 if QUICK else 20; t = time.perf_counter()
        for _ in range(reps):
            m(x)
        if DEVICE.type == "cuda":
            torch.cuda.synchronize()
    COST.append(dict(model=name, params_M=count_params(m), GFLOPs=fc.get_total_flops() / 1e9,
                     latency_ms=(time.perf_counter() - t) / reps * 1000, best_epoch=BEST_EPOCH[name],
                     train_time_min=HISTORY[name]["train_time_min"].iloc[0]))
    del m
COST = pd.DataFrame(COST)


# %% [cell 14]
# ---------------------------------------------------------------------------
# 10. Statistics
# ---------------------------------------------------------------------------


def boot_ci(v, n=CFG["N_BOOT"], seed=CFG["SEED"], stat=np.mean):
    v = np.asarray(v, float); v = v[~np.isnan(v)]
    if len(v) == 0:
        return (np.nan, np.nan)
    r = np.random.default_rng(seed)
    return tuple(np.percentile(stat(v[r.integers(0, len(v), (n, len(v)))], axis=1), [2.5, 97.5]))


def mean_sd(s, pct=True, dec=2):
    s = pd.Series(s).dropna(); f = 100 if pct else 1
    return f"{f * s.mean():.{dec}f} ± {f * s.std():.{dec}f}" if len(s) else "n/a"


def to_latex_booktabs(df, caption, label, path, bold_max=(), bold_min=()):
    cols = list(df.columns)
    esc = lambda s: str(s).replace("%", r"\%").replace("_", r"\_").replace("±", r"$\pm$").replace("&", r"\&").replace("Δ", r"$\Delta$").replace("−", "$-$")
    lead = lambda s: float(str(s).split()[0].strip("[,")) if str(s).split() and str(s).split()[0].strip("[,").replace(".", "", 1).lstrip("-").isdigit() else np.nan
    best = {}
    for c in cols:
        if c in bold_max or c in bold_min:
            vals = df[c].map(lead)
            if vals.notna().any():
                best[c] = vals.idxmax() if c in bold_max else vals.idxmin()
    lines = [r"\begin{table}[t]", r"\centering", rf"\caption{{{caption}}}", rf"\label{{{label}}}", r"\small",
             r"\resizebox{\linewidth}{!}{", r"\begin{tabular}{l" + "c" * (len(cols) - 1) + "}", r"\toprule",
             " & ".join(esc(c) for c in cols) + r" \\", r"\midrule"]
    for i, r_ in df.astype(str).iterrows():
        cells = [(r"\textbf{" + esc(v) + "}") if best.get(c) == i else esc(v) for c, v in zip(cols, r_.values)]
        lines.append(" & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}}", r"\end{table}"]
    open(path, "w").write("\n".join(lines))


def save_table(df, name, caption, label, **kw):
    df.to_csv(os.path.join(DIRS["tables"], name + ".csv"), index=False)
    to_latex_booktabs(df.reset_index(drop=True), caption, label, os.path.join(DIRS["tables"], name + ".tex"), **kw)
    display(df) if "display" in globals() else print(df.to_string(index=False))


# Table 2: segmentation, main comparison
t2 = []
for name in MAIN_MODELS:
    d = PER_IMAGE[PER_IMAGE.model == name]; lo, hi = boot_ci(d.mDice)
    row = {"Model": SHORT[name]}
    for c in FG_CLASSES:
        row[f"Dice {c.lower()} (%)"] = mean_sd(d[f"Dice_{c}"])
    row.update({"mDice (%)": mean_sd(d.mDice), "mDice 95% CI": f"[{100 * lo:.2f}, {100 * hi:.2f}]",
                "IoU (%)": mean_sd(d.IoU_FG), "Precision (%)": mean_sd(d.Precision_FG),
                "Recall (%)": mean_sd(d.Recall_FG), "HD95 (px)": mean_sd(d.HD95_FG, pct=False, dec=1)})
    t2.append(row)
TABLE2 = pd.DataFrame(t2)
save_table(TABLE2, "table2_segmentation", f"Test-set segmentation (n = {len(RECORDS['test'])} images), mean $\\pm$ SD over images.",
           "tab:seg", bold_max=[c for c in TABLE2.columns if "%" in c], bold_min=["HD95 (px)"])

# Table 3: paired tests, proposed vs each baseline (per-image mDice)
def rank_biserial(a, b):
    d = np.asarray(a) - np.asarray(b); d = d[d != 0]
    if len(d) == 0:
        return 0.0
    rk = stats.rankdata(np.abs(d)); return float((rk[d > 0].sum() - rk[d < 0].sum()) / rk.sum())


P = PER_IMAGE.pivot(index="image", columns="model", values="mDice")
t3 = []
for b in BASELINES:
    j = P[[PROPOSED, b]].dropna(); diff = j[PROPOSED] - j[b]
    try:
        p = stats.wilcoxon(j[PROPOSED], j[b]).pvalue
    except ValueError:
        p = 1.0
    lo, hi = boot_ci(diff.values)
    t3.append({"Comparison": f"Proposed vs {SHORT[b]}", "n": len(j), "Mean Δ mDice (pp)": f"{100 * diff.mean():+.2f}",
               "95% CI (pp)": f"[{100 * lo:+.2f}, {100 * hi:+.2f}]", "Wins / ties / losses": f"{(diff > 0).sum()}/{(diff == 0).sum()}/{(diff < 0).sum()}",
               "Effect size r": f"{rank_biserial(j[PROPOSED], j[b]):+.2f}", "p_raw": p})
TABLE3 = pd.DataFrame(t3)
order = np.argsort(TABLE3.p_raw.values); m_ = len(order); adj = np.empty(m_); run = 0
for rank_, k in enumerate(order):
    run = max(run, min(1.0, (m_ - rank_) * TABLE3.p_raw.values[k])); adj[k] = run
TABLE3["p (Holm)"] = [f"{v:.1e}" for v in adj]
TABLE3["Sig."] = ["***" if v < 1e-3 else "**" if v < 1e-2 else "*" if v < 0.05 else "n.s." for v in adj]
TABLE3["p_holm"] = adj
save_table(TABLE3.drop(columns=["p_raw", "p_holm"]), "table3_wilcoxon",
           "Paired two-sided Wilcoxon signed-rank tests on per-image mDice with Holm correction; r is the matched-pairs rank-biserial correlation.",
           "tab:wilcoxon")

# Friedman test and average ranks (for the critical-difference diagram)
Pm = P[MAIN_MODELS].dropna()
FRIEDMAN = stats.friedmanchisquare(*[Pm[c].values for c in MAIN_MODELS])
RANKS = Pm.rank(axis=1, ascending=False).mean().sort_values()
Q05 = {2: 1.960, 3: 2.344, 4: 2.569, 5: 2.728, 6: 2.850, 7: 2.949, 8: 3.031, 9: 3.102, 10: 3.164}
CD = Q05[len(MAIN_MODELS)] * math.sqrt(len(MAIN_MODELS) * (len(MAIN_MODELS) + 1) / (6 * len(Pm)))
print(f"Friedman chi2 = {FRIEDMAN.statistic:.2f}, p = {FRIEDMAN.pvalue:.2e}; Nemenyi CD = {CD:.3f}")

# Table 4: cell-level classification (Normal vs Abnormal), with bootstrap CI over images for AUC
from sklearn.metrics import roc_auc_score
ABN_NAME = CLASS_NAMES[ABN_IDX]


def cell_metrics(d):
    y = (d.true == ABN_NAME).values; yp = (d.pred == ABN_NAME).values
    tp, tn = (y & yp).sum(), (~y & ~yp).sum(); fp, fn = (~y & yp).sum(), (y & ~yp).sum()
    sens, spec = _safe(tp, tp + fn), _safe(tn, tn + fp); prec = _safe(tp, tp + fp)
    f1 = _safe(2 * prec * sens, prec + sens) if not np.isnan(prec) and not np.isnan(sens) else np.nan
    auc_ = roc_auc_score(y, d.p_abn) if 0 < y.sum() < len(y) else np.nan
    return dict(acc=(tp + tn) / len(d), bacc=np.nanmean([sens, spec]), sens=sens, spec=spec, prec=prec, f1=f1, auc=auc_,
                det=d.detected.mean())


def boot_cells(d, fn_, n=300 if not QUICK else 30):
    ims = d.image.unique(); g = {k: v for k, v in d.groupby("image")}; r = np.random.default_rng(CFG["SEED"]); vals = []
    for _ in range(n):
        s = pd.concat([g[k] for k in r.choice(ims, len(ims))])
        try:
            vals.append(fn_(s))
        except ValueError:
            pass
    return np.nanpercentile(vals, [2.5, 97.5]) if vals else (np.nan, np.nan)


t4 = []
for name in MAIN_MODELS:
    d = CELLS[CELLS.model == name]; mt = cell_metrics(d)
    lo, hi = boot_cells(d, lambda s: cell_metrics(s)["auc"])
    t4.append({"Model": SHORT[name], "Accuracy (%)": f"{100 * mt['acc']:.2f}", "Balanced acc. (%)": f"{100 * mt['bacc']:.2f}",
               "Sensitivity (%)": f"{100 * mt['sens']:.2f}", "Specificity (%)": f"{100 * mt['spec']:.2f}",
               "Precision (%)": f"{100 * mt['prec']:.2f}", "F1 (%)": f"{100 * mt['f1']:.2f}",
               "AUC": f"{mt['auc']:.3f}", "AUC 95% CI": f"[{lo:.3f}, {hi:.3f}]", "Detection rate (%)": f"{100 * mt['det']:.2f}"})
TABLE4 = pd.DataFrame(t4)
save_table(TABLE4, "table4_cell_classification",
           f"Cell-level classification of {int((CELLS.model == PROPOSED).sum())} annotated test cells (abnormal = positive).",
           "tab:cellcls", bold_max=[c for c in TABLE4.columns if "%" in c or c == "AUC"])

# Table 5: image-level screening (does the slide contain abnormal cells?)
t5 = []
for name in MAIN_MODELS:
    d = IMGCLS[IMGCLS.model == name]; y, s = d.label.values, d.score.values
    if 0 < y.sum() < len(y):
        a = roc_auc_score(y, s); pr = s >= 0.5
        sens, spec = _safe((pr & (y == 1)).sum(), (y == 1).sum()), _safe((~pr & (y == 0)).sum(), (y == 0).sum())
    else:
        a = sens = spec = np.nan
    t5.append({"Model": SHORT[name], "AUC": f"{a:.3f}", "Sensitivity (%)": f"{100 * sens:.2f}", "Specificity (%)": f"{100 * spec:.2f}"})
TABLE5 = pd.DataFrame(t5)
save_table(TABLE5, "table5_image_screening", "Image-level screening: presence of any abnormal cell (score = 99.5th percentile of P(abnormal)).",
           "tab:imgcls", bold_max=["AUC", "Sensitivity (%)", "Specificity (%)"])

# Table 6: cost
TABLE6 = COST[COST.model.isin(MAIN_MODELS)].assign(model=lambda d: d.model.map(SHORT)).rename(columns={
    "model": "Model", "params_M": "Params (M)", "GFLOPs": "GFLOPs", "latency_ms": "Latency (ms)",
    "best_epoch": "Best epoch", "train_time_min": "Train time (min)"}).round(2)
save_table(TABLE6, "table6_cost", f"Computational cost at {IMG_H} $\\times$ {IMG_W} input (batch 1).", "tab:cost")

# Table 7: ablation
if CFG["RUN_ABLATION"]:
    base = PER_IMAGE[PER_IMAGE.model == PROPOSED].set_index("image").mDice
    t7 = []
    for name in [PROPOSED] + ABLATIONS:
        d = PER_IMAGE[PER_IMAGE.model == name].set_index("image"); lo, hi = boot_ci(d.mDice)
        cm_ = CELLS[CELLS.model == name]
        t7.append({"Variant": "Full model" if name == PROPOSED else name, "mDice (%)": mean_sd(d.mDice),
                   "95% CI": f"[{100 * lo:.2f}, {100 * hi:.2f}]",
                   "Δ vs full (pp)": "—" if name == PROPOSED else f"{100 * (d.mDice - base).mean():+.2f}",
                   "Cell AUC": f"{cell_metrics(cm_)['auc']:.3f}",
                   "Params (M)": f"{COST.set_index('model').loc[name, 'params_M']:.1f}"})
    TABLE7 = pd.DataFrame(t7)
    save_table(TABLE7, "table7_ablation", "Ablation of the proposed components (test set).", "tab:ablation")


# %% [cell 15]
# ---------------------------------------------------------------------------
# Verify absent-class handling and reproduce the mDice results in Table 6(a)
# ---------------------------------------------------------------------------

# 1. Test the Dice routine using small masks with known class presence.
cases = []

for class_idx in range(1, NC):
    class_name = CLASS_NAMES[class_idx]
    background = np.zeros((2, 2), dtype=np.uint8)
    foreground = background.copy()
    foreground[0, 0] = class_idx

    for case, pred, gt in [
        ("Absent from reference and prediction", background, background),
        ("Absent from reference only", foreground, background),
        ("Absent from prediction only", background, foreground),
        ("Present in both, perfect overlap", foreground, foreground),
    ]:
        dice = metrics_from_cm(confusion(pred, gt), class_idx)["Dice"]
        cases.append({"Class": class_name, "Case": case, "Dice": dice})

print("Absent-class handling")
print(pd.DataFrame(cases).to_string(index=False, na_rep="NaN"))

# 2. Recalculate mDice from the original per-image class scores.
dice_cols = [f"Dice_{class_name}" for class_name in FG_CLASSES]
audit = []

for model in MAIN_MODELS:
    d = PER_IMAGE.loc[PER_IMAGE["model"] == model]
    class_dice = d[dice_cols]
    recalculated = class_dice.mean(axis=1, skipna=True)

    # Verify every per-image score against the stored mDice.
    np.testing.assert_allclose(
        recalculated.to_numpy(),
        d["mDice"].to_numpy(),
        rtol=0,
        atol=1e-7,
        equal_nan=True,
        err_msg=f"Per-image mDice mismatch for {SHORT[model]}",
    )

    lo, hi = boot_ci(recalculated)
    audit.append({
        "Model": SHORT[model],
        **{f"{c} valid images": class_dice[c].count() for c in dice_cols},
        "Valid mDice images": recalculated.count(),
        "Mean of class means (%)": f"{100 * class_dice.mean().mean():.3f}",
        "Recalculated mDice (%)": mean_sd(recalculated),
        "mDice 95% CI (%)": f"[{100 * lo:.2f}, {100 * hi:.2f}]",
    })

print("\nRecalculated Table 6(a) metrics")
print(pd.DataFrame(audit).to_string(index=False))


# %% [cell 16]
# ---------------------------------------------------------------------------
# 11. Robustness to acquisition artefacts (test split, network resolution, no TTA)
# ---------------------------------------------------------------------------
from skimage.color import rgb2hed, hed2rgb


def hed_shift(im, s, r):
    hed = rgb2hed(im); hed = hed * (1 + r.uniform(-s, s, 3)) + r.uniform(-s, s, 3) * 0.05
    return (np.clip(hed2rgb(hed), 0, 1) * 255).astype(np.uint8)


def _jpeg(im, q):
    ok, b = cv2.imencode(".jpg", cv2.cvtColor(im, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, q])
    return cv2.cvtColor(cv2.imdecode(b, 1), cv2.COLOR_BGR2RGB)


def _lowres(im, f):
    h, w = im.shape[:2]
    return cv2.resize(cv2.resize(im, (max(8, int(w * f)), max(8, int(h * f))), interpolation=cv2.INTER_AREA), (w, h), interpolation=cv2.INTER_LINEAR)


CORRUPTIONS = {
    "Gaussian noise":   lambda im, s, r: np.clip(im + r.normal(0, [6, 12, 20][s - 1], im.shape), 0, 255).astype(np.uint8),
    "Defocus blur":     lambda im, s, r: cv2.GaussianBlur(im, (0, 0), [1.0, 2.0, 3.0][s - 1]),
    "JPEG compression": lambda im, s, r: _jpeg(im, [40, 20, 10][s - 1]),
    "Contrast loss":    lambda im, s, r: np.clip((im.astype(np.float32) - im.mean()) * [0.75, 0.5, 0.3][s - 1] + im.mean(), 0, 255).astype(np.uint8),
    "Stain variation":  lambda im, s, r: hed_shift(im, [0.2, 0.4, 0.6][s - 1], r),
    "Resolution loss":  lambda im, s, r: _lowres(im, [0.6, 0.4, 0.25][s - 1]),
}
norm = A.Compose([A.Normalize(), ToTensorV2()])


@torch.no_grad()
def robust_eval(model, corr, sev):
    dices = []
    batch_x, batch_y = [], []
    for i, (img, msk) in enumerate(test_ds.items):
        r = np.random.default_rng(1000 + i)
        im = img if sev == 0 else CORRUPTIONS[corr](img, sev, r)
        batch_x.append(norm(image=im)["image"]); batch_y.append(msk)
        if len(batch_x) == 8 or i == len(test_ds.items) - 1:
            xb = torch.stack(batch_x).to(DEVICE)
            with torch.autocast(DEVICE.type, dtype=torch.float16, enabled=USE_AMP):
                pr = model(xb).argmax(1).cpu().numpy()
            for p_, y_ in zip(pr, batch_y):
                cm = confusion(p_, y_)
                dices.append(np.nanmean([metrics_from_cm(cm, c)["Dice"] for c in range(1, NC)]))
            batch_x, batch_y = [], []
    return float(np.nanmean(dices))


ROB = []
if CFG["RUN_ROBUSTNESS"]:
    for name in MAIN_MODELS:
        model, _ = load_best(name)
        clean = robust_eval(model, None, 0)
        for corr in CORRUPTIONS:
            ROB.append(dict(model=name, corruption=corr, severity=0, mDice=clean))
            for sev in (1, 2, 3):
                ROB.append(dict(model=name, corruption=corr, severity=sev, mDice=robust_eval(model, corr, sev)))
        del model; gc.collect(); torch.cuda.empty_cache()
    ROB = pd.DataFrame(ROB)
    ROB.to_csv(os.path.join(DIRS["tables"], "robustness_raw.csv"), index=False)
    clean = ROB[ROB.severity == 0].groupby("model").mDice.first()
    corr_mean = ROB[ROB.severity > 0].groupby(["model", "corruption"]).mDice.mean().unstack()
    t8 = pd.DataFrame({"Model": [SHORT[m] for m in MAIN_MODELS],
                       "Clean mDice (%)": [f"{100 * clean[m]:.2f}" for m in MAIN_MODELS]})
    for c in CORRUPTIONS:
        t8[c + " (%)"] = [f"{100 * corr_mean.loc[m, c]:.2f}" for m in MAIN_MODELS]
    t8["Mean corrupted (%)"] = [f"{100 * corr_mean.loc[m].mean():.2f}" for m in MAIN_MODELS]
    t8["Relative robustness (%)"] = [f"{100 * corr_mean.loc[m].mean() / clean[m]:.1f}" for m in MAIN_MODELS]
    TABLE8 = t8
    save_table(TABLE8, "table8_robustness", "Robustness: mDice under synthetic acquisition artefacts (mean over three severities).",
               "tab:robust", bold_max=[c for c in t8.columns if c != "Model"])


# %% [cell 17]
# ---------------------------------------------------------------------------
# 12. Figures
# ---------------------------------------------------------------------------
import matplotlib.patheffects as pe
VIS_CLASS = {c: ("#E6007E" if "abnormal" in c.lower() else "#00B0FF" if "normal" in c.lower() else "#FFD600") for c in FG_CLASSES}
ERR_COLORS = {"Correct": "#1B7837", "Wrong class": "#FFD000", "False positive": "#FF6D00", "False negative": "#AA00FF"}
CLS_RGB = {CLS_IDX[c]: np.array(mpl.colors.to_rgb(v), np.float32) for c, v in VIS_CLASS.items()}
ERR_RGB = {k: np.array(mpl.colors.to_rgb(v), np.float32) for k, v in ERR_COLORS.items()}


def grey_bg(img, lo=0.35):
    g = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255
    return np.repeat((lo + (1 - lo) * g)[..., None], 3, 2)


def paint(base01, label_map, colors, alpha):
    out = base01.copy()
    for k, m in label_map.items():
        out[m] = (1 - alpha) * out[m] + alpha * colors[k]
    return np.clip(out, 0, 1)


def err_cases(pred, gt):
    return {"Correct": (pred == gt) & (gt > 0), "Wrong class": (pred > 0) & (gt > 0) & (pred != gt),
            "False positive": (pred > 0) & (gt == 0), "False negative": (pred == 0) & (gt > 0)}


def label_ax(ax, img, mask, alpha=0.55, lw=0.8):
    ax.imshow(paint(grey_bg(img), {c: mask == c for c in CLS_RGB}, CLS_RGB, alpha))
    for c, rgb in CLS_RGB.items():
        m = mask == c
        if m.any():
            cs = ax.contour(m, levels=[0.5], colors=[tuple(rgb)], linewidths=lw)
            cs.set(path_effects=[pe.Stroke(linewidth=lw + 0.9, foreground="black"), pe.Normal()])
    image_axis(ax)


def error_ax(ax, img, pred, gt):
    ax.imshow(paint(grey_bg(img, 0.5), err_cases(pred, gt), ERR_RGB, 0.9)); image_axis(ax)


def prob_ax(ax, img, p, gt_m=None):
    a_ = np.clip(p * 1.5, 0, 0.9)[..., None].astype(np.float32)
    ax.imshow(np.clip((1 - a_) * grey_bg(img) + a_ * mpl.colormaps["inferno"](p.astype(np.float32))[..., :3], 0, 1))
    if gt_m is not None and gt_m.any():
        cs = ax.contour(gt_m, levels=[0.5], colors="white", linewidths=0.6, linestyles="--")
        cs.set(path_effects=[pe.Stroke(linewidth=1.3, foreground="black"), pe.Normal()])
    image_axis(ax)


def zoom_window(gt, frac=0.36):
    H, W = gt.shape; h, w = int(H * frac), int(W * frac)
    fg = (gt == ABN_IDX) if (gt == ABN_IDX).any() else (gt > 0)
    if not fg.any():
        return (H - h) // 2, (W - w) // 2, h, w
    dens = cv2.boxFilter(fg.astype(np.float32), -1, (w, h), normalize=False, borderType=cv2.BORDER_CONSTANT)
    ys, xs = np.nonzero(dens >= dens.max() * 0.98); cy, cx = int(np.median(ys)), int(np.median(xs))
    return int(np.clip(cy - h // 2, 0, H - h)), int(np.clip(cx - w // 2, 0, W - w)), h, w


def grid_figure(n_r, n_c, aspect, width=FIG_W["double"], left=0.03, right=0.995, top_in=0.2, bot_in=0.45, wsp=0.02, hsp=0.03):
    axw = (right - left) * width / (n_c + (n_c - 1) * wsp); axh = axw * aspect
    Hf = top_in + bot_in + n_r * axh + (n_r - 1) * hsp * axh
    fig, axs = plt.subplots(n_r, n_c, figsize=(width, Hf), squeeze=False)
    fig.subplots_adjust(left=left, right=right, top=1 - top_in / Hf, bottom=bot_in / Hf, wspace=wsp, hspace=hsp)
    return fig, axs


def style_axes(ax, grid_axis="y"):
    ax.grid(axis=grid_axis, color="0.92", lw=0.5); ax.set_axisbelow(True)


def mcol(n):
    return MODEL_COLOR.get(n, "0.5")


def mlw(n):
    return 1.8 if n == PROPOSED else 1.0


# %% [cell 18]
# Fig. 4  Learning curves: validation mDice of all models, loss of the proposed model
fig, axs = plt.subplots(1, 2, figsize=(FIG_W["double"], 64 * MM), constrained_layout=True, gridspec_kw=dict(width_ratios=[1.5, 1]))
ax = axs[0]
for n in MAIN_MODELS:
    h = HISTORY[n]; y_ = 100 * pd.Series(h.val_dice).ewm(alpha=0.35, adjust=False).mean()
    ax.plot(h.epoch, y_, color=mcol(n), lw=mlw(n), label=SHORT[n], zorder=3 if n == PROPOSED else 2)
    b = h.val_dice.idxmax(); ax.plot(h.epoch[b], 100 * h.val_dice[b], "o", ms=3.5 if n != PROPOSED else 5, mfc=mcol(n), mec="white", mew=0.5, zorder=4)
ax.set(xlabel="Epoch", ylabel="Validation mDice (%)"); style_axes(ax); panel_label(ax, "a")
ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.2), fontsize=6.5, columnspacing=0.8)
ax = axs[1]; h = HISTORY[PROPOSED]
ax.plot(h.epoch, h.train_loss, color="#0072B2", lw=1.3, label="Training")
ax.plot(h.epoch, h.val_loss, color="#E69F00", lw=1.3, label="Validation")
ax.axvline(h.epoch[h.val_dice.idxmax()], color="0.5", ls="--", lw=0.6)
ax.set(xlabel="Epoch", ylabel="Loss", title=SHORT[PROPOSED]); style_axes(ax); ax.legend(); panel_label(ax, "b")
for a_ in axs:
    a_.xaxis.set_major_locator(mpl.ticker.MaxNLocator(integer=True, nbins=6))
save_figure(fig, "Fig4_learning_curves"); plt.show()


# %% [cell 19]
# Fig. 5  Baseline comparison: (a) mean and 95% CI per metric, (b) paired per-image gain of the proposed model
metrics5 = [(f"Dice_{c}", f"Dice\n{c.lower()}") for c in FG_CLASSES] + [("mDice", "mDice"), ("IoU_FG", "IoU"), ("Recall_FG", "Recall"), ("Precision_FG", "Precision")]
fig = plt.figure(figsize=(FIG_W["double"], 92 * MM), constrained_layout=True)
gs = fig.add_gridspec(1, 2, width_ratios=[1.35, 1])
ax = fig.add_subplot(gs[0])
nm = len(MAIN_MODELS); off = np.linspace(-0.32, 0.32, nm)
for k, n in enumerate(MAIN_MODELS):
    d = PER_IMAGE[PER_IMAGE.model == n]
    for j, (col, _) in enumerate(metrics5):
        v = d[col].dropna().values; m = 100 * v.mean(); lo, hi = boot_ci(v)
        ax.errorbar(j + off[k], m, yerr=[[m - 100 * lo], [100 * hi - m]], fmt="o", ms=3.8 if n != PROPOSED else 5.5,
                    color=mcol(n), mec="black" if n == PROPOSED else mcol(n), mew=0.6, elinewidth=0.9, capsize=1.5,
                    label=SHORT[n] if j == 0 else None, zorder=4 if n == PROPOSED else 3)
ax.set_xticks(range(len(metrics5)), [m[1] for m in metrics5]); ax.set_ylabel("Test score (%), mean and 95% CI")
style_axes(ax); ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.1), fontsize=6.5, columnspacing=0.8)
panel_label(ax, "a")

ax = fig.add_subplot(gs[1]); rng5 = np.random.default_rng(3)
for i, b in enumerate(BASELINES[::-1]):
    j = P[[PROPOSED, b]].dropna(); diff = 100 * (j[PROPOSED] - j[b]).values
    ax.scatter(diff, i + rng5.uniform(-0.18, 0.18, len(diff)), s=4, color=mcol(b), alpha=0.5, lw=0)
    lo, hi = boot_ci(diff / 100); ax.plot([100 * lo, 100 * hi], [i, i], color="black", lw=1.6, zorder=4)
    ax.plot(diff.mean(), i, "D", ms=4.5, mfc=mcol(b), mec="black", mew=0.7, zorder=5)
    sig = TABLE3.set_index("Comparison").loc[f"Proposed vs {SHORT[b]}", "Sig."]
    ax.text(1.02, i, sig, transform=mpl.transforms.blended_transform_factory(ax.transAxes, ax.transData),
            va="center", fontsize=7, fontweight="bold" if sig != "n.s." else "normal")
ax.axvline(0, color="0.3", lw=0.8, ls="--")
ax.set_yticks(range(len(BASELINES)), [f"vs {SHORT[b]}" for b in BASELINES[::-1]])
ax.set_xlabel("Per-image mDice gain of the proposed model (pp)"); style_axes(ax, "x"); panel_label(ax, "b")
ax.text(0.98, 1.01, "Holm-adjusted\nWilcoxon", transform=ax.transAxes, ha="right", va="bottom", fontsize=6, color="0.3")
save_figure(fig, "Fig5_baseline_comparison"); plt.show()


# %% [cell 20]
# Fig. 6  Critical-difference diagram (Friedman test with Nemenyi post hoc)
k_ = len(MAIN_MODELS); ranks = RANKS.sort_values(); names = list(ranks.index); half = (k_ + 1) // 2
n_side = max(half, k_ - half)
fig, ax = plt.subplots(figsize=(FIG_W["onehalf"], (22 + 5 * n_side) * MM))
ax.set_xlim(k_ + 2.6, 1 - 2.6); ax.set_ylim(0.62 - 0.12 * n_side, 1.1); ax.axis("off")
ax.hlines(1.0, 1, k_, color="black", lw=0.8)
for t in range(1, k_ + 1):
    ax.vlines(t, 1.0, 1.04, color="black", lw=0.8); ax.text(t, 1.06, str(t), ha="center", va="bottom", fontsize=7)
ax.plot([1, 1 + CD], [1.2, 1.2], color="black", lw=1.2, clip_on=False)
ax.text(1 + CD / 2, 1.22, f"CD = {CD:.2f}", ha="center", va="bottom", fontsize=6.5, clip_on=False)
for i, n in enumerate(names):
    r_ = ranks[n]
    if i < half:                                   # best ranks: labels on the right
        y_ = 0.62 - i * 0.12; x_end = 1 - 0.3; ha = "left"
    else:
        y_ = 0.62 - (k_ - 1 - i) * 0.12; x_end = k_ + 0.3; ha = "right"
    ax.plot([r_, r_, x_end], [1.0, y_, y_], color=mcol(n), lw=1.4 if n == PROPOSED else 0.8)
    ax.text(x_end + (-0.08 if ha == "left" else 0.08), y_, f"{SHORT[n]} ({r_:.2f})", ha=ha, va="center", fontsize=7,
            fontweight="bold" if n == PROPOSED else "normal", color=mcol(n) if n == PROPOSED else "black")
cliques = []
for i in range(k_):
    j = max(jj for jj in range(i, k_) if ranks[names[jj]] - ranks[names[i]] <= CD)
    if j > i and not any(i >= a and j <= b for a, b in cliques):
        cliques.append((i, j))
for q, (a, b) in enumerate(cliques):
    yb = 0.92 - q * 0.06
    ax.plot([ranks[names[a]] - 0.04, ranks[names[b]] + 0.04], [yb, yb], color="black", lw=2.4, solid_capstyle="round")
ax.set_title("Average rank on per-image mDice (lower is better)", fontsize=7, pad=30)
fig.text(0.5, 0.01, f"Friedman $\\chi^2$ = {FRIEDMAN.statistic:.1f}, p = {FRIEDMAN.pvalue:.1e}. Bars join models that do not differ (Nemenyi, $\\alpha$ = 0.05).",
         ha="center", va="bottom", fontsize=6.3, color="0.3")
save_figure(fig, "Fig6_critical_difference"); plt.show()


# %% [cell 21]
# Fig. 7  Qualitative comparison of all models on zoomed regions with abnormal cells
rec_test = {r["stem"]: r for r in RECORDS["test"]}
pd_ = PER_IMAGE[PER_IMAGE.model == PROPOSED].set_index("image").mDice.dropna()
has_abn = IMGCLS[(IMGCLS.model == PROPOSED)].set_index("image").label
cand = pd_[pd_.index.isin(has_abn[has_abn == 1].index)] if (has_abn == 1).sum() >= 4 else pd_
cand = cand.sort_values()
cases7 = list(dict.fromkeys(cand.index[np.round(np.array([0.9, 0.65, 0.4, 0.15]) * (len(cand) - 1)).astype(int)]))
cols7 = ["Image", "Ground truth"] + [SHORT[n] for n in MAIN_MODELS]
fig, axs = grid_figure(len(cases7), len(cols7), 0.75, top_in=0.3, bot_in=0.42)
for i, stem in enumerate(cases7):
    r = rec_test[stem]; dsz = display_size(r["h"], r["w"], long=DISP_LONG_Q)
    img = cv2.resize(read_rgb(r["img_path"]), dsz, interpolation=cv2.INTER_AREA)
    gt = cv2.resize(cv2.imread(r["mask_path"], cv2.IMREAD_UNCHANGED), dsz, interpolation=cv2.INTER_NEAREST)
    y0, x0, h, w = zoom_window(gt); crop = lambda a: a[y0:y0 + h, x0:x0 + w]
    ax = axs[i, 0]; ax.imshow(img); image_axis(ax); ax.set_aspect("auto")
    ax.add_patch(mpl.patches.Rectangle((x0, y0), w, h, fill=False, ec="white", lw=1.0))
    ax.add_patch(mpl.patches.Rectangle((x0, y0), w, h, fill=False, ec="black", lw=0.4, ls="--"))
    ax.set_ylabel(f"Case {i + 1}", fontsize=7, labelpad=2)
    label_ax(axs[i, 1], crop(img), crop(gt)); axs[i, 1].set_aspect("auto")
    for j, n in enumerate(MAIN_MODELS):
        ax = axs[i, 2 + j]; label_ax(ax, crop(img), crop(PRED_SMALL[n][stem])); ax.set_aspect("auto")
        v = PER_IMAGE[(PER_IMAGE.model == n) & (PER_IMAGE.image == stem)].mDice.iloc[0]
        ax.text(0.03, 0.04, f"{100 * v:.1f}", transform=ax.transAxes, fontsize=5.5, color="white",
                bbox=dict(boxstyle="round,pad=0.15", fc="black" if n != PROPOSED else mcol(PROPOSED), ec="none", alpha=0.75))
        if n == PROPOSED:
            for s_ in ax.spines.values():
                s_.set_visible(True); s_.set_edgecolor(mcol(PROPOSED)); s_.set_linewidth(1.2)
for j, c in enumerate(cols7):
    axs[0, j].set_title(c, fontsize=6, pad=3, fontweight="bold" if c == SHORT[PROPOSED] else "normal",
                        color=mcol(PROPOSED) if c == SHORT[PROPOSED] else "black")
h7 = [Patch(facecolor=VIS_CLASS[c], edgecolor="black", lw=0.5, label=c.capitalize()) for c in FG_CLASSES]
fig.legend(handles=h7, loc="lower center", ncol=len(h7), bbox_to_anchor=(0.5, 0.0), title="Per-image mDice (%) shown in each panel", title_fontsize=6.5)
save_figure(fig, "Fig7_qualitative_comparison"); plt.show()


# %% [cell 22]
# Fig. 8  Proposed model in detail: image, ground truth, prediction, error map and P(abnormal) on zoomed regions
cases8 = list(dict.fromkeys(cand.index[np.round(np.linspace(0.95, 0.05, 5) * (len(cand) - 1)).astype(int)]))
cols8 = ["Image", "Ground truth", "Prediction", "Error map", f"P({ABN_NAME.capitalize()})"]
fig, axs = grid_figure(len(cases8), len(cols8), 0.75, right=0.925, bot_in=0.5)
for i, stem in enumerate(cases8):
    r = rec_test[stem]; dsz = display_size(r["h"], r["w"], long=DISP_LONG_Q)
    img = cv2.resize(read_rgb(r["img_path"]), dsz, interpolation=cv2.INTER_AREA)
    gt = cv2.resize(cv2.imread(r["mask_path"], cv2.IMREAD_UNCHANGED), dsz, interpolation=cv2.INTER_NEAREST)
    pr = PRED_SMALL[PROPOSED][stem]; pa = PROB_ABN_SMALL[stem].astype(np.float32)
    y0, x0, h, w = zoom_window(gt); crop = lambda a: a[y0:y0 + h, x0:x0 + w]
    ax = axs[i, 0]; ax.imshow(img); image_axis(ax); ax.set_aspect("auto")
    ax.add_patch(mpl.patches.Rectangle((x0, y0), w, h, fill=False, ec="white", lw=1.0))
    ax.add_patch(mpl.patches.Rectangle((x0, y0), w, h, fill=False, ec="black", lw=0.4, ls="--"))
    ax.set_ylabel(f"Case {i + 1}", fontsize=7, labelpad=2)
    label_ax(axs[i, 1], crop(img), crop(gt)); label_ax(axs[i, 2], crop(img), crop(pr))
    v = PER_IMAGE[(PER_IMAGE.model == PROPOSED) & (PER_IMAGE.image == stem)].mDice.iloc[0]
    axs[i, 2].text(0.03, 0.04, f"mDice {100 * v:.1f}%", transform=axs[i, 2].transAxes, fontsize=6, color="white",
                   bbox=dict(boxstyle="round,pad=0.2", fc="black", ec="none", alpha=0.7))
    error_ax(axs[i, 3], crop(img), crop(pr), crop(gt)); prob_ax(axs[i, 4], crop(img), crop(pa), crop(gt) == ABN_IDX)
    for a_ in axs[i, 1:]:
        a_.set_aspect("auto")
for j, c in enumerate(cols8):
    axs[0, j].set_title(c, fontsize=7, pad=3)
cax = axs[-1, -1].inset_axes([1.05, 0, 0.07, len(cases8) + (len(cases8) - 1) * 0.03])
cb = fig.colorbar(mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(0, 1), cmap="inferno"), cax=cax)
cb.set_label("Predicted probability"); cb.outline.set_linewidth(0.5)
hc = [Patch(facecolor=VIS_CLASS[c], edgecolor="black", lw=0.5, label=c.capitalize()) for c in FG_CLASSES]
he = [Patch(facecolor=v, edgecolor="black", lw=0.5, label=k) for k, v in ERR_COLORS.items()]
fig.legend(handles=hc, title="Class", loc="lower center", bbox_to_anchor=(0.25, 0.0), ncol=len(hc), title_fontsize=7)
fig.legend(handles=he, title="Error map", loc="lower center", bbox_to_anchor=(0.63, 0.0), ncol=len(he), title_fontsize=7)
save_figure(fig, "Fig8_proposed_segmentation"); plt.show()


# %% [cell 23]
# Fig. 9  Robustness: mDice versus artefact severity
if CFG["RUN_ROBUSTNESS"]:
    fig, axs = plt.subplots(2, 3, figsize=(FIG_W["double"], 100 * MM), constrained_layout=True, sharex=True, sharey=True)
    for ax, corr, L in zip(axs.ravel(), CORRUPTIONS, "abcdef"):
        for n in MAIN_MODELS:
            d = ROB[(ROB.model == n) & (ROB.corruption == corr)].sort_values("severity")
            ax.plot(d.severity, 100 * d.mDice, "-o", color=mcol(n), lw=mlw(n), ms=3 if n != PROPOSED else 4,
                    mec="white", mew=0.4, label=SHORT[n], zorder=3 if n == PROPOSED else 2)
        ax.set_title(corr); ax.set_xticks([0, 1, 2, 3], ["Clean", "1", "2", "3"]); style_axes(ax); panel_label(ax, L)
    for ax in axs[1]:
        ax.set_xlabel("Severity")
    for ax in axs[:, 0]:
        ax.set_ylabel("mDice (%)")
    h_, l_ = axs[0, 0].get_legend_handles_labels()
    fig.legend(h_, l_, loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=len(MAIN_MODELS), fontsize=6.5)
    save_figure(fig, "Fig9_robustness"); plt.show()


# %% [cell 24]
# Fig. 10  Cell-level classification: confusion matrices and ROC curves
best_base = max(BASELINES, key=lambda n: cell_metrics(CELLS[CELLS.model == n])["auc"])
fig = plt.figure(figsize=(FIG_W["double"], 68 * MM), constrained_layout=True)
gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1.35])
for k, (n, L) in enumerate([(PROPOSED, "a"), (best_base, "b")]):
    ax = fig.add_subplot(gs[k]); d = CELLS[CELLS.model == n]
    cmx = pd.crosstab(d.true, d.pred).reindex(index=FG_CLASSES, columns=FG_CLASSES, fill_value=0).values
    cmn = 100 * cmx / cmx.sum(1, keepdims=True).clip(min=1)
    ax.imshow(cmn, cmap="Blues", vmin=0, vmax=100)
    for a_ in range(len(FG_CLASSES)):
        for b_ in range(len(FG_CLASSES)):
            ax.text(b_, a_, f"{cmn[a_, b_]:.1f}%\n(n = {cmx[a_, b_]})", ha="center", va="center", fontsize=6.5,
                    color="white" if cmn[a_, b_] > 55 else "black")
    ax.set_xticks(range(len(FG_CLASSES)), [c.capitalize() for c in FG_CLASSES])
    ax.set_yticks(range(len(FG_CLASSES)), [c.capitalize() for c in FG_CLASSES])
    ax.set_xlabel("Predicted"); ax.set_ylabel("True" if k == 0 else "")
    ax.set_title(SHORT[n], color=mcol(n) if n == PROPOSED else "black", fontweight="bold" if n == PROPOSED else "normal")
    for s_ in ax.spines.values():
        s_.set_visible(False)
    ax.tick_params(length=0); panel_label(ax, L)
ax = fig.add_subplot(gs[2])
for n in MAIN_MODELS:
    d = CELLS[CELLS.model == n]; y = (d.true == ABN_NAME).values
    if 0 < y.sum() < len(y):
        fpr, tpr, _ = roc_curve(y, d.p_abn); ax.plot(fpr, tpr, color=mcol(n), lw=mlw(n), label=f"{SHORT[n]} ({auc(fpr, tpr):.3f})",
                                                    zorder=3 if n == PROPOSED else 2)
ax.plot([0, 1], [0, 1], ls="--", color="0.6", lw=0.6)
ax.set(xlabel="False positive rate", ylabel="True positive rate", xlim=(0, 1), ylim=(0, 1.01), title="Cell-level ROC (abnormal)")
ax.set_aspect("equal"); style_axes(ax, "both"); panel_label(ax, "c")
ax.legend(title="AUC", loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=6, title_fontsize=6.5)
save_figure(fig, "Fig10_cell_classification"); plt.show()


# %% [cell 25]
# Fig. 11  Feature space of the proposed model (t-SNE of per-cell decoder features) and example cells
if len(CELL_FEATS) > 10:
    Fm = np.array(CELL_FEATS); dP = CELLS[CELLS.model == PROPOSED].reset_index(drop=True)
    Fm = (Fm - Fm.mean(0)) / (Fm.std(0) + 1e-6)
    Z = TSNE(n_components=2, perplexity=min(30, max(5, len(Fm) // 5)), init="pca", random_state=CFG["SEED"]).fit_transform(Fm)
    fig = plt.figure(figsize=(FIG_W["double"], 92 * MM), constrained_layout=True)
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 1.15])
    ax = fig.add_subplot(gs[0])
    for c in FG_CLASSES:
        ok = (dP.true == c) & (dP.pred == c); bad = (dP.true == c) & (dP.pred != c)
        ax.scatter(Z[ok, 0], Z[ok, 1], s=6, color=VIS_CLASS[c], alpha=0.7, lw=0, label=f"{c.capitalize()}, correct")
        ax.scatter(Z[bad, 0], Z[bad, 1], s=14, facecolor="none", edgecolor=VIS_CLASS[c], lw=0.8, marker="X" if False else "o",
                   label=f"{c.capitalize()}, misclassified")
    ax.set(xticks=[], yticks=[], xlabel="t-SNE 1", ylabel="t-SNE 2"); ax.legend(fontsize=6.3, loc="best", markerscale=1.2)
    panel_label(ax, "a")
    # example cells: correct and wrong for each class, ranked by confidence
    groups = []
    for c in FG_CLASSES[::-1]:
        is_abn = c == ABN_NAME
        tp_ = dP[(dP.true == c) & (dP.pred == c)].sort_values("p_abn", ascending=not is_abn).index[:6]
        fn_ = dP[(dP.true == c) & (dP.pred != c)].sort_values("p_abn", ascending=is_abn).index[:6]
        groups += [(f"{c.capitalize()}: correct", tp_, VIS_CLASS[c]), (f"{c.capitalize()}: misclassified", fn_, "#FF6D00")]
    groups = [g_ for g_ in groups if len(g_[1]) > 0]
    sub = gs[1].subgridspec(max(1, len(groups)), 6, wspace=0.05, hspace=0.15)
    for gi, (title, idxs, col) in enumerate(groups):
        for k in range(6):
            ax = fig.add_subplot(sub[gi, k]); image_axis(ax)
            if k < len(idxs):
                ax.imshow(CELL_CROPS[idxs[k]])
                for s_ in ax.spines.values():
                    s_.set_visible(True); s_.set_edgecolor(col); s_.set_linewidth(1.4)
                ax.text(0.5, 0.02, f"{dP.p_abn[idxs[k]]:.2f}", transform=ax.transAxes, ha="center", va="bottom", fontsize=5.5,
                        color="white", bbox=dict(boxstyle="round,pad=0.1", fc="black", ec="none", alpha=0.6))
            if k == 0:
                ax.set_ylabel(title, fontsize=6, rotation=0, ha="right", va="center", labelpad=4)
    fig.text(0.78, 1.0, f"(b)  Example cells; value = P({ABN_NAME.lower()})", fontsize=7, ha="center", va="bottom", fontweight="bold")
    save_figure(fig, "Fig11_feature_space_and_cells"); plt.show()


# %% [cell 26]
# Fig. 12  Ablation (dot plot with 95% CI) and accuracy-cost trade-off
fig, axs = plt.subplots(1, 2, figsize=(FIG_W["double"], 66 * MM), constrained_layout=True, gridspec_kw=dict(width_ratios=[1, 1.1]))
ax = axs[0]
if CFG["RUN_ABLATION"]:
    vs = [PROPOSED] + ABLATIONS; base_m = 100 * PER_IMAGE[PER_IMAGE.model == PROPOSED].mDice.mean()
    ax.axvline(base_m, color=mcol(PROPOSED), ls="--", lw=0.8, zorder=1)
    for i, n in enumerate(vs):
        v = PER_IMAGE[PER_IMAGE.model == n].mDice.dropna().values; m = 100 * v.mean(); lo, hi = boot_ci(v)
        ax.plot([100 * lo, 100 * hi], [i, i], color=mcol(n), lw=2.2, solid_capstyle="round", zorder=2)
        ax.plot(m, i, "o", ms=7 if n == PROPOSED else 6, mfc=mcol(n), mec="black", mew=0.7, zorder=3)
        ax.text(100 * hi + 0.4, i, f"{m:.1f}" + ("" if n == PROPOSED else f"  ({m - base_m:+.1f})"), va="center", fontsize=6.5,
                fontweight="bold" if n == PROPOSED else "normal")
    ax.set_yticks(range(len(vs)), ["Full model"] + ABLATIONS); ax.invert_yaxis(); ax.set_ylim(len(vs) - 0.5, -0.5)
    ax.set_xlabel("Test mDice (%), mean and 95% CI"); style_axes(ax, "x"); ax.margins(x=0.12)
panel_label(ax, "a")

ax = axs[1]
pts = []
for n in MAIN_MODELS:
    c_ = COST.set_index("model").loc[n]; v = PER_IMAGE[PER_IMAGE.model == n].mDice.dropna(); lo, hi = boot_ci(v)
    ax.errorbar(c_.GFLOPs, 100 * v.mean(), yerr=[[100 * (v.mean() - lo)], [100 * (hi - v.mean())]], fmt="none", ecolor=mcol(n), lw=0.8, capsize=2)
    ax.scatter(c_.GFLOPs, 100 * v.mean(), s=10 + 3 * c_.params_M, color=mcol(n), ec="black", lw=0.6 if n == PROPOSED else 0.3, zorder=3)
    pts.append((n, c_.GFLOPs, 100 * v.mean()))
# simple label de-overlap: stack labels of nearby points vertically
xr = max(p_[1] for p_ in pts) - min(p_[1] for p_ in pts) + 1e-6; yr = max(p_[2] for p_ in pts) - min(p_[2] for p_ in pts) + 1e-6
placed = []
for n, x_, y_ in sorted(pts, key=lambda t: -t[2]):
    lx, dy = x_ + 0.07 * xr, 0.0
    while any(abs(lx - px) / xr < 0.30 and abs(y_ + dy - py) / yr < 0.08 for px, py in placed):
        dy -= 0.08 * yr
    placed.append((lx, y_ + dy))
    ax.annotate(SHORT[n], (x_, y_), xytext=(lx, y_ + dy), textcoords="data", fontsize=6, va="center",
                fontweight="bold" if n == PROPOSED else "normal",
                arrowprops=dict(arrowstyle="-", lw=0.4, color="0.5", shrinkA=0, shrinkB=4))
ax.set(xlabel=f"GFLOPs ({IMG_H}×{IMG_W})", ylabel="Test mDice (%)"); style_axes(ax, "both"); ax.margins(x=0.3, y=0.3)
ax.set_ylim(max(0, ax.get_ylim()[0]), min(100, ax.get_ylim()[1]))
ax.text(0.98, 0.03, "Marker area scales with parameters", transform=ax.transAxes, ha="right", fontsize=6, color="0.4")
panel_label(ax, "b")
save_figure(fig, "Fig12_ablation_and_cost"); plt.show()


# %% [cell 27]
# Fig. 12  Ablation and accuracy-cost trade-off
fig, axs = plt.subplots(1, 2, figsize=(FIG_W["double"], 66 * MM), constrained_layout=True, gridspec_kw=dict(width_ratios=[1, 1.1]))
ax = axs[0]
if CFG["RUN_ABLATION"]:
    vs = [PROPOSED] + ABLATIONS; base_m = PER_IMAGE[PER_IMAGE.model == PROPOSED].mDice.mean()
    for i, n in enumerate(vs):
        v = PER_IMAGE[PER_IMAGE.model == n].mDice.dropna().values; m = 100 * v.mean(); lo, hi = boot_ci(v)
        ax.barh(i, m, color=mcol(n), height=0.62, edgecolor="black" if n == PROPOSED else "none", lw=0.6)
        ax.errorbar(m, i, xerr=[[m - 100 * lo], [100 * hi - m]], color="black", lw=0.8, capsize=2)
        ax.text(100 * hi + 0.6, i, f"{m:.1f}" + ("" if n == PROPOSED else f"  ({m - 100 * base_m:+.1f})"), va="center", fontsize=6.5)
    ax.set_yticks(range(len(vs)), ["Full model"] + ABLATIONS); ax.invert_yaxis()
    ms_ = 100 * PER_IMAGE[PER_IMAGE.model.isin(vs)].groupby("model").mDice.mean()
    ax.set_xlim(max(0, ms_.min() - 15), min(100, ms_.max() + 18)); ax.set_xlabel("Test mDice (%)"); style_axes(ax, "x")
panel_label(ax, "a")
ax = axs[1]
for n in MAIN_MODELS:
    c_ = COST.set_index("model").loc[n]; v = PER_IMAGE[PER_IMAGE.model == n].mDice.dropna(); lo, hi = boot_ci(v)
    ax.errorbar(c_.GFLOPs, 100 * v.mean(), yerr=[[100 * (v.mean() - lo)], [100 * (hi - v.mean())]], fmt="none", ecolor=mcol(n), lw=0.8, capsize=2)
    ax.scatter(c_.GFLOPs, 100 * v.mean(), s=10 + 3 * c_.params_M, color=mcol(n), ec="black", lw=0.6 if n == PROPOSED else 0.3, zorder=3)
    ax.annotate(SHORT[n], (c_.GFLOPs, 100 * v.mean()), xytext=(5, 3), textcoords="offset points", fontsize=6,
                fontweight="bold" if n == PROPOSED else "normal")
ax.set(xlabel=f"GFLOPs ({IMG_H}×{IMG_W})", ylabel="Test mDice (%)"); style_axes(ax, "both"); ax.margins(x=0.2, y=0.25)
ax.set_ylim(max(0, ax.get_ylim()[0]), min(100, ax.get_ylim()[1]))
ax.text(0.98, 0.03, "Marker area scales with parameters", transform=ax.transAxes, ha="right", fontsize=6, color="0.4")
panel_label(ax, "b")
save_figure(fig, "Fig12_ablation_and_cost"); plt.show()


# %% [cell 28]
# Cell A: seed-aware training and loading (replaces train_model / load_best; main-run file names unchanged)
def run_tag(name, seed=None):
    """Checkpoint stem. seed=None is the main run (keeps the original file names)."""
    return slug(name) if seed is None else f"{slug(name)}_seed{seed}"


def train_model(name, seed=None):
    tag = run_tag(name, seed)
    ckpt = os.path.join(DIRS["checkpoints"], tag + ".pt")
    last = os.path.join(DIRS["checkpoints"], tag + "_last.pt")
    hist_path = os.path.join(DIRS["logs"], tag + "_history.csv")
    done_flag = os.path.join(DIRS["checkpoints"], tag + ".done")
    if CFG["RESUME"] and os.path.exists(done_flag) and os.path.exists(ckpt):
        print(f"[{tag}] training already finished")
        return pd.read_csv(hist_path)
    seed_everything(CFG["SEED"] if seed is None else seed)
    if seed is not None:
        loaders["train"].generator.manual_seed(seed)
    model = build_model(name).to(DEVICE, memory_format=torch.channels_last)
    criterion = DiceCELoss(NC, CE_W, CFG["W_CE"], CFG["W_DICE"]).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=CFG["LR"], weight_decay=CFG["WEIGHT_DECAY"])
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    scaler = torch.amp.GradScaler(DEVICE.type, enabled=USE_AMP)
    start_ep, best, wait, hist, elapsed = 0, -1.0, 0, [], 0.0
    if CFG["RESUME"] and os.path.exists(last):
        st = torch.load(last, map_location=DEVICE, weights_only=False)
        model.load_state_dict(st["model"]); opt.load_state_dict(st["opt"])
        sched.load_state_dict(st["sched"]); scaler.load_state_dict(st["scaler"])
        start_ep, best, wait, hist, elapsed = st["epoch"], st["best"], st["wait"], st["hist"], st["elapsed"]
        torch.set_rng_state(st["rng_torch"]); np.random.set_state(st["rng_np"]); random.setstate(st["rng_py"])
        if DEVICE.type == "cuda" and st.get("rng_cuda") is not None:
            torch.cuda.set_rng_state_all(st["rng_cuda"])
        print(f"[{tag}] resumed after epoch {start_ep} (best val Dice {best:.4f})")
    t0 = time.time()
    for ep in range(start_ep, CFG["EPOCHS"]):
        if wait >= CFG["PATIENCE"]:
            break
        model.train(); run = 0.0; nseen = 0
        for x, y, _ in loaders["train"]:
            x = x.to(DEVICE, non_blocking=True, memory_format=torch.channels_last); y = y.to(DEVICE, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(DEVICE.type, dtype=torch.float16, enabled=USE_AMP):
                loss = total_loss(model(x), y, criterion)
            scaler.scale(loss).backward()
            scaler.unscale_(opt); nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update()
            run += loss.item() * x.size(0); nseen += x.size(0)
        sched.step()
        va_loss, va_dice, va_cls = validate(model, loaders["valid"], criterion)
        hist.append(dict(epoch=ep + 1, train_loss=run / max(nseen, 1), val_loss=va_loss, val_dice=va_dice,
                         lr=opt.param_groups[0]["lr"], **{f"val_dice_{c}": d for c, d in zip(FG_CLASSES, va_cls)}))
        improved = va_dice > best + 1e-4
        if improved:
            best, wait = va_dice, 0
            _atomic_save({"model": model.state_dict(), "epoch": ep + 1, "val_dice": va_dice}, ckpt)
        else:
            wait += 1
        _atomic_save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                      "scaler": scaler.state_dict(), "epoch": ep + 1, "best": best, "wait": wait, "hist": hist,
                      "elapsed": elapsed + time.time() - t0, "rng_torch": torch.get_rng_state(),
                      "rng_np": np.random.get_state(), "rng_py": random.getstate(),
                      "rng_cuda": torch.cuda.get_rng_state_all() if DEVICE.type == "cuda" else None}, last)
        pd.DataFrame(hist).to_csv(hist_path, index=False)
        print(f"[{tag}] ep {ep + 1:3d} | train {hist[-1]['train_loss']:.4f} | val {va_loss:.4f} | val Dice {va_dice:.4f}"
              f"{' *' if improved else ''}")
    h = pd.DataFrame(hist); h["train_time_min"] = (elapsed + time.time() - t0) / 60
    h.to_csv(hist_path, index=False)
    open(done_flag, "w").write(str(len(h)))
    del model; torch.cuda.empty_cache()
    return h


def load_best(name, seed=None):
    m = build_model(name).to(DEVICE)
    ck = torch.load(os.path.join(DIRS["checkpoints"], run_tag(name, seed) + ".pt"), map_location=DEVICE, weights_only=False)
    m.load_state_dict(ck["model"]); m.eval()
    return m, ck["epoch"]

# %%
# ---------------------------------------------------------------------------
# 13. Multi-seed repetition (seed 42 = main run, reused)
# ---------------------------------------------------------------------------
MS_SEEDS = [42, 7, 2024] if not QUICK else [42, 7]
MS_MODELS = [PROPOSED, "SegFormer (MiT-B2)", "U-Net++ (R34)"]      # add more names to widen the study


def seed_arg(sd):
    return None if sd == CFG["SEED"] else sd


@torch.no_grad()
def eval_light(model):
    """Per-image mDice / foreground Dice / IoU and per-cell P(abnormal), same protocol as Section 9."""
    rows, cells = [], []
    for i in range(len(test_ds)):
        x, _, _ = test_ds[i]; r = RECORDS["test"][i]
        gt = cv2.imread(r["mask_path"], cv2.IMREAD_UNCHANGED); H, W = gt.shape
        prob = predict_probs(model, x[None].to(DEVICE), gt.shape); pred = prob.argmax(0).astype(np.uint8)
        cm = confusion(pred, gt)
        d = {CLASS_NAMES[c]: metrics_from_cm(cm, c)["Dice"] for c in range(1, NC)}
        bm = binary_metrics(pred > 0, gt > 0)
        rows.append(dict(image=r["stem"], mDice=np.nanmean(list(d.values())), Dice_FG=bm["Dice"], IoU_FG=bm["IoU"]))
        for (bx, by, bw, bh), cat in CELLS_BY_UID.get(r["image_id"], []):
            x0, y0 = int(max(0, np.floor(bx))), int(max(0, np.floor(by)))
            x1, y1 = int(min(W, np.ceil(bx + bw))), int(min(H, np.ceil(by + bh)))
            if x1 > x0 and y1 > y0:
                mp = prob[:, y0:y1, x0:x1].reshape(NC, -1).mean(1); fgp = mp[1:]
                cells.append(dict(image=r["stem"], true=cat, pred=FG_CLASSES[int(np.argmax(fgp))],
                                  p_abn=float(mp[ABN_IDX] / (fgp.sum() + 1e-8)), detected=bool(1 - mp[0] >= 0.5)))
        del prob, pred
    return pd.DataFrame(rows), pd.DataFrame(cells)


MS_IMG, MS_CELL = [], []
for sd in MS_SEEDS:
    for name in MS_MODELS:
        train_model(name, seed_arg(sd))
        model, _ = load_best(name, seed_arg(sd))
        di, dc = eval_light(model)
        MS_IMG.append(di.assign(model=name, seed=sd)); MS_CELL.append(dc.assign(model=name, seed=sd))
        del model; gc.collect(); torch.cuda.empty_cache()
MS_IMG = pd.concat(MS_IMG, ignore_index=True); MS_CELL = pd.concat(MS_CELL, ignore_index=True)
MS_IMG.to_csv(os.path.join(DIRS["tables"], "multiseed_per_image.csv"), index=False)

per_seed = []
for (n, sd), d in MS_IMG.groupby(["model", "seed"]):
    cmx = cell_metrics(MS_CELL[(MS_CELL.model == n) & (MS_CELL.seed == sd)])
    per_seed.append(dict(model=n, seed=sd, mDice=100 * d.mDice.mean(), Dice_FG=100 * d.Dice_FG.mean(),
                         IoU_FG=100 * d.IoU_FG.mean(), cell_AUC=cmx["auc"], cell_BAcc=100 * cmx["bacc"]))
PER_SEED = pd.DataFrame(per_seed)
PER_SEED.round(3).to_csv(os.path.join(DIRS["tables"], "multiseed_per_seed.csv"), index=False)

t10 = []
for n in MS_MODELS:
    d = PER_SEED[PER_SEED.model == n]
    row = {"Model": SHORT[n], "Seeds": len(d)}
    for col, lab, dec in [("mDice", "mDice (%)", 2), ("Dice_FG", "Dice FG (%)", 2), ("IoU_FG", "IoU (%)", 2),
                          ("cell_AUC", "Cell AUC", 3), ("cell_BAcc", "Cell bal. acc. (%)", 2)]:
        row[lab] = f"{d[col].mean():.{dec}f} ± {d[col].std(ddof=1):.{dec}f}" if len(d) > 1 else f"{d[col].mean():.{dec}f}"
    row["mDice per seed"] = ", ".join(f"{v:.2f}" for v in d.sort_values("seed").mDice)
    t10.append(row)
TABLE10 = pd.DataFrame(t10)
save_table(TABLE10, "table10_multiseed", f"Repeatability over {len(MS_SEEDS)} training seeds (mean $\\pm$ SD across seeds).",
           "tab:multiseed", bold_max=["mDice (%)", "Dice FG (%)", "IoU (%)", "Cell AUC", "Cell bal. acc. (%)"])

# paired tests on per-image mDice averaged over seeds
avg = MS_IMG.groupby(["model", "image"]).mDice.mean().unstack(0)
t11 = []
for b in [m for m in MS_MODELS if m != PROPOSED]:
    j = avg[[PROPOSED, b]].dropna(); diff = j[PROPOSED] - j[b]
    try:
        p = stats.wilcoxon(j[PROPOSED], j[b]).pvalue
    except ValueError:
        p = 1.0
    lo, hi = boot_ci(diff.values)
    seed_wins = sum(PER_SEED[(PER_SEED.model == PROPOSED) & (PER_SEED.seed == sd)].mDice.iloc[0] >
                    PER_SEED[(PER_SEED.model == b) & (PER_SEED.seed == sd)].mDice.iloc[0] for sd in MS_SEEDS)
    t11.append({"Comparison": f"Proposed vs {SHORT[b]}", "Mean Δ mDice (pp)": f"{100 * diff.mean():+.2f}",
                "95% CI (pp)": f"[{100 * lo:+.2f}, {100 * hi:+.2f}]", "Image wins": f"{(diff > 0).sum()}/{len(diff)}",
                "Seed wins": f"{seed_wins}/{len(MS_SEEDS)}", "Effect size r": f"{rank_biserial(j[PROPOSED], j[b]):+.2f}", "p_raw": p})
TABLE11 = pd.DataFrame(t11)
if len(TABLE11):
    order = np.argsort(TABLE11.p_raw.values); m_ = len(order); adj = np.empty(m_); run = 0
    for rank_, k in enumerate(order):
        run = max(run, min(1.0, (m_ - rank_) * TABLE11.p_raw.values[k])); adj[k] = run
    TABLE11["p (Holm)"] = [f"{v:.1e}" for v in adj]
    TABLE11["Sig."] = ["***" if v < 1e-3 else "**" if v < 1e-2 else "*" if v < 0.05 else "n.s." for v in adj]
    save_table(TABLE11.drop(columns="p_raw"), "table11_multiseed_tests",
               "Paired Wilcoxon tests on per-image mDice averaged over seeds (Holm-corrected).", "tab:multiseed_tests")

# %%
# Fig. 13  Repeatability across seeds
fig, axs = plt.subplots(1, 2, figsize=(FIG_W["double"], 62 * MM), constrained_layout=True, gridspec_kw=dict(width_ratios=[1, 1.2]))
for ax, col, lab, L in [(axs[0], "mDice", "Test mDice (%)", "a"), (axs[1], "cell_AUC", "Cell-level AUC", "b")]:
    for i, n in enumerate(MS_MODELS):
        d = PER_SEED[PER_SEED.model == n].sort_values("seed"); v = d[col].values
        ax.scatter(np.full(len(v), i) + np.linspace(-0.12, 0.12, len(v)), v, s=22, color=mcol(n), ec="black", lw=0.5, zorder=3)
        ax.hlines(v.mean(), i - 0.28, i + 0.28, color=mcol(n), lw=2.2 if n == PROPOSED else 1.4, zorder=2)
        if len(v) > 1:
            ax.vlines(i, v.mean() - v.std(ddof=1), v.mean() + v.std(ddof=1), color="black", lw=0.8, zorder=1)
        ax.text(i + 0.32, v.mean(), f"{v.mean():.{1 if col == 'mDice' else 3}f}", va="center", fontsize=6.5,
                fontweight="bold" if n == PROPOSED else "normal")
    ax.set_xticks(range(len(MS_MODELS)), [SHORT[n] for n in MS_MODELS], fontsize=6.5)
    ax.set_ylabel(lab); ax.set_xlim(-0.6, len(MS_MODELS) - 0.2); style_axes(ax); panel_label(ax, L)
axs[0].set_title(f"Seeds: {', '.join(map(str, MS_SEEDS))}  (line = mean, bar = ±1 SD)", fontsize=6.5, color="0.35")
save_figure(fig, "Fig13_multiseed"); plt.show()

# %%
# ---------------------------------------------------------------------------
# 14. Cross-dataset (external) testing, zero-shot: no retraining, no fine-tuning
#   FORMAT "coco": folder with images + one COCO json (segmentation, cell and image metrics)
#   FORMAT "cric": CRIC Cervix style CSV with nucleus x/y and Bethesda class (cell and image metrics)
# ---------------------------------------------------------------------------
EXT = dict(
    NAME="CRIC Cervix",
    FORMAT="cric",                                              # "cric" or "coco"
    IMG_DIR=os.environ.get("CERVIX_EXT_IMG", "/content/gdrive/MyDrive/MTA-TransUNet/external/CRIC/images"),
    ANN=os.environ.get("CERVIX_EXT_ANN", "/content/gdrive/MyDrive/MTA-TransUNet/external/CRIC/classifications.csv"),
    # map external class names (lower case) to "normal" / "abnormal"; anything not listed is skipped
    CLASS_MAP={"nilm": "normal", "negative for intraepithelial lesion": "normal", "normal": "normal",
               "asc-us": "abnormal", "asc-h": "abnormal", "lsil": "abnormal", "hsil": "abnormal", "scc": "abnormal",
               "abnormal": "abnormal"},
    CELL_BOX=None,           # box side (px) around a nucleus point; None = median box side of our training cells
    RESCALE=1.0,             # resize external images by this factor before inference (match magnification)
    MAX_IMAGES=None,         # limit for a quick check
)
if os.environ.get("CERVIX_EXT_FORMAT"):
    EXT["FORMAT"] = os.environ["CERVIX_EXT_FORMAT"]
RUN_EXTERNAL = os.path.exists(EXT["ANN"]) and os.path.isdir(EXT["IMG_DIR"])
print("External dataset found:", RUN_EXTERNAL, "|", EXT["ANN"])

_fg_lower = {c.lower(): c for c in FG_CLASSES}


def _canon(name):
    t = EXT["CLASS_MAP"].get(str(name).strip().lower())
    return _fg_lower.get(t) if t else None


def load_external():
    """Return list of dict(path, cells=[(x0, y0, x1, y1, cls)], mask or None)."""
    items = []
    if EXT["FORMAT"] == "coco":
        js = json.load(open(EXT["ANN"]))
        cats = {c["id"]: c["name"] for c in js["categories"]}
        anns = pd.DataFrame(js["annotations"])
        for im in js["images"]:
            p_ = os.path.join(EXT["IMG_DIR"], os.path.basename(im["file_name"]))
            if not os.path.exists(p_):
                continue
            a = anns[anns.image_id == im["id"]] if len(anns) else anns
            h, w = im.get("height"), im.get("width")
            if not h:
                h, w = cv2.imread(p_).shape[:2]
            mask = np.zeros((h, w), np.uint8); cells = []
            for _, rr in a.iterrows():
                cls = _canon(cats.get(rr.category_id))
                if cls is None:
                    continue
                cv2.fillPoly(mask, _polygons(rr.to_dict()), int(CLS_IDX[cls]))
                x, y, bw, bh = rr.bbox; cells.append((x, y, x + bw, y + bh, cls))
            items.append(dict(path=p_, cells=cells, mask=mask))
    else:
        df = pd.read_csv(EXT["ANN"])
        low = {c.lower(): c for c in df.columns}
        fcol = next(low[c] for c in low if "image" in c and ("name" in c or "file" in c))
        ccol = next(low[c] for c in low if "bethesda" in c or c in ("class", "label", "category"))
        xcol = next(low[c] for c in low if c.endswith("_x") or c == "x")
        ycol = next(low[c] for c in low if c.endswith("_y") or c == "y")
        side = EXT["CELL_BOX"] or float(INST[INST.split == "train"].side_px.median())
        for fname, g in df.groupby(fcol):
            p_ = os.path.join(EXT["IMG_DIR"], os.path.basename(str(fname)))
            if not os.path.exists(p_):
                continue
            cells = []
            for _, rr in g.iterrows():
                cls = _canon(rr[ccol])
                if cls is None:
                    continue
                cx, cy = float(rr[xcol]), float(rr[ycol])
                cells.append((cx - side / 2, cy - side / 2, cx + side / 2, cy + side / 2, cls))
            items.append(dict(path=p_, cells=cells, mask=None))
    if EXT["MAX_IMAGES"]:
        items = items[:EXT["MAX_IMAGES"]]
    print(f"{EXT['NAME']}: {len(items)} images, {sum(len(i['cells']) for i in items)} labelled cells, "
          f"classes {pd.Series([c[4] for i in items for c in i['cells']]).value_counts().to_dict()}")
    return items


@torch.no_grad()
def eval_external(model, items):
    rows, cells, imgs = [], [], []
    for it in items:
        img = read_rgb(it["path"])
        if EXT["RESCALE"] != 1.0:
            img = cv2.resize(img, None, fx=EXT["RESCALE"], fy=EXT["RESCALE"], interpolation=cv2.INTER_AREA)
        H, W = img.shape[:2]; s_ = EXT["RESCALE"]
        im_in = cv2.resize(apply_clahe(img) if CFG["USE_CLAHE"] else img, (IMG_W, IMG_H), interpolation=cv2.INTER_AREA)
        x = eval_tf(image=im_in)["image"][None].to(DEVICE)
        prob = predict_probs(model, x, (H, W)); pred = prob.argmax(0).astype(np.uint8)
        stem = os.path.splitext(os.path.basename(it["path"]))[0]
        if it["mask"] is not None:
            gt = it["mask"] if s_ == 1.0 else cv2.resize(it["mask"], (W, H), interpolation=cv2.INTER_NEAREST)
            cm = confusion(pred, gt)
            d = [metrics_from_cm(cm, c)["Dice"] for c in range(1, NC)]
            bm = binary_metrics(pred > 0, gt > 0)
            rows.append(dict(image=stem, mDice=np.nanmean(d), Dice_FG=bm["Dice"], IoU_FG=bm["IoU"]))
        for (x0, y0, x1, y1, cls) in it["cells"]:
            x0, y0 = int(max(0, x0 * s_)), int(max(0, y0 * s_)); x1, y1 = int(min(W, x1 * s_)), int(min(H, y1 * s_))
            if x1 <= x0 or y1 <= y0:
                continue
            mp = prob[:, y0:y1, x0:x1].reshape(NC, -1).mean(1); fgp = mp[1:]
            cells.append(dict(image=stem, true=cls, pred=FG_CLASSES[int(np.argmax(fgp))],
                              p_abn=float(mp[ABN_IDX] / (fgp.sum() + 1e-8)), detected=bool(1 - mp[0] >= 0.5)))
        labs = [c[4] for c in it["cells"]]
        if labs:
            imgs.append(dict(image=stem, label=int(ABN_NAME in labs), score=float(np.percentile(prob[ABN_IDX], 99.5))))
        del prob, pred
    return pd.DataFrame(rows), pd.DataFrame(cells), pd.DataFrame(imgs)


EXT_SEG, EXT_CELL, EXT_IMG = [], [], []
if RUN_EXTERNAL:
    ext_items = load_external()
    for name in MAIN_MODELS:
        model, _ = load_best(name)
        a_, b_, c_ = eval_external(model, ext_items)
        EXT_SEG.append(a_.assign(model=name)); EXT_CELL.append(b_.assign(model=name)); EXT_IMG.append(c_.assign(model=name))
        del model; gc.collect(); torch.cuda.empty_cache()
    EXT_SEG = pd.concat(EXT_SEG, ignore_index=True) if any(len(x) for x in EXT_SEG) else pd.DataFrame()
    EXT_CELL = pd.concat(EXT_CELL, ignore_index=True); EXT_IMG = pd.concat(EXT_IMG, ignore_index=True)
    EXT_CELL.to_csv(os.path.join(DIRS["tables"], "external_per_cell.csv"), index=False)

    t12 = []
    for n in MAIN_MODELS:
        dc = EXT_CELL[EXT_CELL.model == n]; mt = cell_metrics(dc); lo, hi = boot_cells(dc, lambda s: cell_metrics(s)["auc"])
        internal = cell_metrics(CELLS[CELLS.model == n])["auc"]
        row = {"Model": SHORT[n], "Cell AUC": f"{mt['auc']:.3f}", "AUC 95% CI": f"[{lo:.3f}, {hi:.3f}]",
               "Δ AUC vs internal": f"{mt['auc'] - internal:+.3f}", "Balanced acc. (%)": f"{100 * mt['bacc']:.2f}",
               "Sensitivity (%)": f"{100 * mt['sens']:.2f}", "Specificity (%)": f"{100 * mt['spec']:.2f}",
               "F1 (%)": f"{100 * mt['f1']:.2f}"}
        di = EXT_IMG[EXT_IMG.model == n]
        row["Image AUC"] = f"{roc_auc_score(di.label, di.score):.3f}" if 0 < di.label.sum() < len(di) else "n/a"
        if len(EXT_SEG):
            ds = EXT_SEG[EXT_SEG.model == n]
            row["mDice (%)"] = mean_sd(ds.mDice); row["IoU (%)"] = mean_sd(ds.IoU_FG)
        t12.append(row)
    TABLE12 = pd.DataFrame(t12)
    save_table(TABLE12, "table12_external",
               f"Zero-shot external validation on {EXT['NAME']} ({EXT_CELL[EXT_CELL.model == PROPOSED].shape[0]} cells; no retraining).",
               "tab:external", bold_max=[c for c in TABLE12.columns if ("%" in c or "AUC" in c) and "Δ" not in c and "CI" not in c])

# %%
# Fig. 14  External validation: ROC, internal-to-external AUC shift, example predictions
if RUN_EXTERNAL:
    fig = plt.figure(figsize=(FIG_W["double"], 112 * MM), constrained_layout=True)
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 0.5])
    ax = fig.add_subplot(gs[0, 0])
    for n in MAIN_MODELS:
        d = EXT_CELL[EXT_CELL.model == n]; y = (d.true == ABN_NAME).values
        if 0 < y.sum() < len(y):
            fpr, tpr, _ = roc_curve(y, d.p_abn)
            ax.plot(fpr, tpr, color=mcol(n), lw=mlw(n), label=f"{SHORT[n]} ({auc(fpr, tpr):.3f})", zorder=3 if n == PROPOSED else 2)
    ax.plot([0, 1], [0, 1], ls="--", color="0.6", lw=0.6)
    ax.set(xlabel="False positive rate", ylabel="True positive rate", xlim=(0, 1), ylim=(0, 1.01), title=f"{EXT['NAME']}: cell ROC")
    ax.set_aspect("equal"); style_axes(ax, "both"); ax.legend(title="AUC", fontsize=5.5, title_fontsize=6, loc="lower right"); panel_label(ax, "a")

    ax = fig.add_subplot(gs[0, 1])
    gap = {n: (cell_metrics(CELLS[CELLS.model == n])["auc"], cell_metrics(EXT_CELL[EXT_CELL.model == n])["auc"]) for n in MAIN_MODELS}
    for n, (a_in, a_ex) in gap.items():
        ax.plot([0, 1], [a_in, a_ex], "-o", color=mcol(n), lw=mlw(n), ms=4 if n != PROPOSED else 5.5, mec="white", mew=0.5,
                zorder=3 if n == PROPOSED else 2)
    ys = sorted(((v[1], n) for n, v in gap.items()), reverse=True); rng_y = max(1e-3, max(v for v, _ in ys) - min(v for v, _ in ys))
    lab_y, prev = {}, None
    for v, n in ys:                                          # keep labels at least 7% of the range apart
        y_ = v if prev is None else min(v, prev - 0.07 * rng_y); lab_y[n] = prev = y_
    for n, (a_in, a_ex) in gap.items():
        ax.annotate(SHORT[n], (1, a_ex), xytext=(1.08, lab_y[n]), textcoords="data", va="center", fontsize=6,
                    fontweight="bold" if n == PROPOSED else "normal", arrowprops=dict(arrowstyle="-", lw=0.4, color="0.6"))
    ax.set_xticks([0, 1], ["Internal test", EXT["NAME"]]); ax.set_xlim(-0.2, 1.8); ax.set_ylabel("Cell-level AUC")
    ax.set_title("Generalisation gap"); style_axes(ax); panel_label(ax, "b")

    ax = fig.add_subplot(gs[0, 2])
    dP = EXT_CELL[EXT_CELL.model == PROPOSED]
    cmx = pd.crosstab(dP.true, dP.pred).reindex(index=FG_CLASSES, columns=FG_CLASSES, fill_value=0).values
    cmn = 100 * cmx / cmx.sum(1, keepdims=True).clip(min=1)
    ax.imshow(cmn, cmap="Blues", vmin=0, vmax=100)
    for a_ in range(len(FG_CLASSES)):
        for b_ in range(len(FG_CLASSES)):
            ax.text(b_, a_, f"{cmn[a_, b_]:.1f}%\n(n = {cmx[a_, b_]})", ha="center", va="center", fontsize=6.5,
                    color="white" if cmn[a_, b_] > 55 else "black")
    ax.set_xticks(range(len(FG_CLASSES)), [c.capitalize() for c in FG_CLASSES]); ax.set_yticks(range(len(FG_CLASSES)), [c.capitalize() for c in FG_CLASSES])
    ax.set(xlabel="Predicted", ylabel="True"); ax.set_title(f"Proposed on {EXT['NAME']}", color=mcol(PROPOSED))
    for s_ in ax.spines.values():
        s_.set_visible(False)
    ax.tick_params(length=0); panel_label(ax, "c")

    # example external images: image with labelled cells, and P(abnormal) of the proposed model
    ex_imgs = EXT_IMG[(EXT_IMG.model == PROPOSED)].sort_values("label", ascending=False).image.tolist()[:3]
    model, _ = load_best(PROPOSED)
    sub = gs[1, :].subgridspec(1, 2 * len(ex_imgs), wspace=0.03)
    for k, stem in enumerate(ex_imgs):
        it = next(i for i in ext_items if os.path.splitext(os.path.basename(i["path"]))[0] == stem)
        img = read_rgb(it["path"]); H, W = img.shape[:2]
        im_in = cv2.resize(apply_clahe(img) if CFG["USE_CLAHE"] else img, (IMG_W, IMG_H), interpolation=cv2.INTER_AREA)
        prob = predict_probs(model, eval_tf(image=im_in)["image"][None].to(DEVICE), (H, W))
        dsz = display_size(H, W, long=640); sc = dsz[0] / W
        img_s = cv2.resize(img, dsz, interpolation=cv2.INTER_AREA); pa = cv2.resize(prob[ABN_IDX], dsz, interpolation=cv2.INTER_AREA)
        a1 = fig.add_subplot(sub[0, 2 * k]); a1.imshow(img_s); image_axis(a1)
        for (x0, y0, x1, y1, cls) in it["cells"]:
            a1.add_patch(mpl.patches.Rectangle((x0 * sc, y0 * sc), (x1 - x0) * sc, (y1 - y0) * sc, fill=False, ec=VIS_CLASS[cls], lw=0.9,
                                               path_effects=[pe.Stroke(linewidth=1.8, foreground="black"), pe.Normal()]))
        a2 = fig.add_subplot(sub[0, 2 * k + 1]); prob_ax(a2, img_s, pa)
        if k == 0:
            a1.set_title("(d)  External image, reference cells", fontsize=6.5, loc="left")
            a2.set_title(f"P({ABN_NAME.lower()}), proposed", fontsize=6.5)
        del prob
    del model; torch.cuda.empty_cache()
    save_figure(fig, "Fig14_external_validation"); plt.show()


# %% [cell 29]
# ---------------------------------------------------------------------------
# 14a. External dataset: fetch CRIC Cervix from its official figshare record once, keep it on Drive
#   Source: CRIC Cervix Cell Classification, figshare collection 4960286 (Rezende et al., Sci Data 2021)
#   Drive keeps the original downloads; each session unzips them to fast local disk.
# ---------------------------------------------------------------------------
import requests, zipfile, hashlib, shutil

CRIC = dict(
    COLLECTION=4960286,
    DRIVE_DIR=os.path.join(os.path.dirname(CFG["DATA_ROOT"].rstrip("/")), "external", "CRIC"),   # .../MTA-TransUNet/external/CRIC
    LOCAL_DIR="/content/cric_cache" if IN_COLAB else os.path.join(CFG["OUT_DIR"], "cric_cache"),
)


def _md5(path, chunk=1 << 20):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def figshare_fetch(collection, raw_dir):
    """Download every file of a figshare collection once (resumable, checksum-verified)."""
    os.makedirs(raw_dir, exist_ok=True)
    flag = os.path.join(raw_dir, ".download_complete")
    if os.path.exists(flag):
        print(f"CRIC already on Drive: {raw_dir}")
        return
    api = "https://api.figshare.com/v2"
    arts = requests.get(f"{api}/collections/{collection}/articles", params={"page_size": 100}, timeout=60)
    arts.raise_for_status()
    files = []
    for a in arts.json():
        meta = requests.get(f"{api}/articles/{a['id']}", timeout=60); meta.raise_for_status()
        files += meta.json().get("files", [])
    print(f"figshare collection {collection}: {len(files)} files")
    for f in files:
        out = os.path.join(raw_dir, f["name"])
        if os.path.exists(out) and os.path.getsize(out) == f.get("size", -1):
            continue
        tmp = "/content/_dl.part" if IN_COLAB else out + ".part"       # stream to local disk, then move to Drive
        with requests.get(f["download_url"], stream=True, timeout=120) as r:
            r.raise_for_status()
            with open(tmp, "wb") as fh, tqdm(total=f.get("size"), unit="B", unit_scale=True, desc=f["name"], leave=False) as bar:
                for chunk in r.iter_content(1 << 20):
                    fh.write(chunk); bar.update(len(chunk))
        if f.get("computed_md5") and _md5(tmp) != f["computed_md5"]:
            raise IOError(f"checksum mismatch for {f['name']}; rerun the cell to retry")
        shutil.move(tmp, out)
    open(flag, "w").write(json.dumps([{k: f.get(k) for k in ("name", "size", "computed_md5")} for f in files], indent=1))
    print("download complete")


def _walk_cells(obj, ctx=None):
    """Yield flat records for every JSON object that carries nucleus coordinates."""
    ctx = dict(ctx or {})
    if isinstance(obj, dict):
        ctx.update({k: v for k, v in obj.items() if not isinstance(v, (dict, list))})
        if any("nucleus_x" in k.lower() for k in obj):
            yield ctx
        for v in obj.values():
            if isinstance(v, (dict, list)):
                yield from _walk_cells(v, ctx)
    elif isinstance(obj, list):
        for v in obj:
            yield from _walk_cells(v, ctx)


def _normalise_table(df):
    low = {c.lower(): c for c in df.columns}
    pick = lambda cond: next((low[c] for c in low if cond(c)), None)
    f = pick(lambda c: "image" in c and ("name" in c or "file" in c))
    c = pick(lambda c: "bethesda" in c) or pick(lambda c: c in ("class", "label", "category"))
    x = pick(lambda c: "nucleus_x" in c) or pick(lambda c: c.endswith("_x") or c == "x")
    y = pick(lambda c: "nucleus_y" in c) or pick(lambda c: c.endswith("_y") or c == "y")
    if None in (f, c, x, y):
        raise ValueError(f"CRIC annotation columns not recognised: {list(df.columns)}")
    return df[[f, c, x, y]].set_axis(["image_filename", "bethesda_system", "nucleus_x", "nucleus_y"], axis=1)


def prepare_cric(raw_dir, local_dir):
    """Unzip to local disk, flatten images into one folder, write one normalised annotation CSV."""
    os.makedirs(local_dir, exist_ok=True)
    ann_out = os.path.join(local_dir, "cric_cells.csv"); img_out = os.path.join(local_dir, "images")
    if os.path.exists(ann_out) and os.path.isdir(img_out) and len(os.listdir(img_out)):
        return img_out, ann_out
    ext_dir = os.path.join(local_dir, "extracted"); os.makedirs(ext_dir, exist_ok=True)
    for fn in os.listdir(raw_dir):
        p_ = os.path.join(raw_dir, fn)
        if zipfile.is_zipfile(p_):
            with zipfile.ZipFile(p_) as z:
                z.extractall(ext_dir)
        elif not fn.startswith("."):
            shutil.copy(p_, ext_dir)
    img_paths, tables = {}, []
    for root, _, fs in os.walk(ext_dir):
        for fn in fs:
            p_ = os.path.join(root, fn); lo = fn.lower()
            if lo.endswith(IMG_EXT):
                img_paths.setdefault(fn, p_)
            elif lo.endswith(".csv"):
                try:
                    tables.append(_normalise_table(pd.read_csv(p_)))
                except (ValueError, pd.errors.ParserError):
                    pass
            elif lo.endswith(".json"):
                try:
                    recs = list(_walk_cells(json.load(open(p_))))
                    if recs:
                        tables.append(_normalise_table(pd.DataFrame(recs)))
                except (ValueError, json.JSONDecodeError):
                    pass
    if not tables:
        raise FileNotFoundError(f"No CRIC classification table (CSV/JSON with nucleus_x) found under {ext_dir}")
    ann = pd.concat(tables).drop_duplicates().reset_index(drop=True)
    ann["image_filename"] = ann.image_filename.map(lambda s: os.path.basename(str(s)))
    os.makedirs(img_out, exist_ok=True)
    for fn, p_ in img_paths.items():
        dst = os.path.join(img_out, fn)
        if not os.path.exists(dst):
            os.symlink(p_, dst)
    ann = ann[ann.image_filename.isin(img_paths)]
    ann.to_csv(ann_out, index=False)
    print(f"CRIC prepared: {ann.image_filename.nunique()} images, {len(ann)} cells | "
          f"{ann.bethesda_system.value_counts().to_dict()}")
    return img_out, ann_out


if not os.environ.get("CERVIX_EXT_ANN"):          # an explicit path (e.g. a COCO set) overrides the CRIC fetch
    try:
        raw = os.path.join(CRIC["DRIVE_DIR"], "raw")
        figshare_fetch(CRIC["COLLECTION"], raw)
        img_dir, ann_csv = prepare_cric(raw, CRIC["LOCAL_DIR"])
        os.environ["CERVIX_EXT_IMG"], os.environ["CERVIX_EXT_ANN"] = img_dir, ann_csv
        os.environ["CERVIX_EXT_FORMAT"] = "cric"
    except Exception as e:
        print("CRIC fetch/prepare skipped:", type(e).__name__, e)

# %%
# ---------------------------------------------------------------------------
# 14. Cross-dataset (external) testing, zero-shot: no retraining, no fine-tuning
#   FORMAT "coco": folder with images + one COCO json (segmentation, cell and image metrics)
#   FORMAT "cric": CRIC Cervix style CSV with nucleus x/y and Bethesda class (cell and image metrics)
# ---------------------------------------------------------------------------
EXT = dict(
    NAME="CRIC Cervix",
    FORMAT="cric",                                              # "cric" or "coco"
    IMG_DIR=os.environ.get("CERVIX_EXT_IMG", "/content/gdrive/MyDrive/MTA-TransUNet/external/CRIC/images"),
    ANN=os.environ.get("CERVIX_EXT_ANN", "/content/gdrive/MyDrive/MTA-TransUNet/external/CRIC/classifications.csv"),
    # map external class names (lower case) to "normal" / "abnormal"; anything not listed is skipped
    CLASS_MAP={"nilm": "normal", "negative for intraepithelial lesion": "normal", "negative for intraepithelial lesion or malignancy": "normal", "normal": "normal",
               "asc-us": "abnormal", "asc-h": "abnormal", "lsil": "abnormal", "hsil": "abnormal", "scc": "abnormal",
               "abnormal": "abnormal"},
    CELL_BOX=None,           # box side (px) around a nucleus point; None = median box side of our training cells
    RESCALE=1.0,             # resize external images by this factor before inference (match magnification)
    MAX_IMAGES=None,         # limit for a quick check
)
if os.environ.get("CERVIX_EXT_FORMAT"):
    EXT["FORMAT"] = os.environ["CERVIX_EXT_FORMAT"]
RUN_EXTERNAL = os.path.exists(EXT["ANN"]) and os.path.isdir(EXT["IMG_DIR"])
print("External dataset found:", RUN_EXTERNAL, "|", EXT["ANN"])

_fg_lower = {c.lower(): c for c in FG_CLASSES}


def _canon(name):
    t = EXT["CLASS_MAP"].get(str(name).strip().lower())
    return _fg_lower.get(t) if t else None


def load_external():
    """Return list of dict(path, cells=[(x0, y0, x1, y1, cls)], mask or None)."""
    items = []
    if EXT["FORMAT"] == "coco":
        js = json.load(open(EXT["ANN"]))
        cats = {c["id"]: c["name"] for c in js["categories"]}
        anns = pd.DataFrame(js["annotations"])
        for im in js["images"]:
            p_ = os.path.join(EXT["IMG_DIR"], os.path.basename(im["file_name"]))
            if not os.path.exists(p_):
                continue
            a = anns[anns.image_id == im["id"]] if len(anns) else anns
            h, w = im.get("height"), im.get("width")
            if not h:
                h, w = cv2.imread(p_).shape[:2]
            mask = np.zeros((h, w), np.uint8); cells = []
            for _, rr in a.iterrows():
                cls = _canon(cats.get(rr.category_id))
                if cls is None:
                    continue
                cv2.fillPoly(mask, _polygons(rr.to_dict()), int(CLS_IDX[cls]))
                x, y, bw, bh = rr.bbox; cells.append((x, y, x + bw, y + bh, cls))
            items.append(dict(path=p_, cells=cells, mask=mask))
    else:
        df = pd.read_csv(EXT["ANN"])
        low = {c.lower(): c for c in df.columns}
        fcol = next(low[c] for c in low if "image" in c and ("name" in c or "file" in c))
        ccol = next(low[c] for c in low if "bethesda" in c or c in ("class", "label", "category"))
        xcol = next(low[c] for c in low if c.endswith("_x") or c == "x")
        ycol = next(low[c] for c in low if c.endswith("_y") or c == "y")
        side = EXT["CELL_BOX"] or float(INST[INST.split == "train"].side_px.median())
        for fname, g in df.groupby(fcol):
            p_ = os.path.join(EXT["IMG_DIR"], os.path.basename(str(fname)))
            if not os.path.exists(p_):
                continue
            cells = []
            for _, rr in g.iterrows():
                cls = _canon(rr[ccol])
                if cls is None:
                    continue
                cx, cy = float(rr[xcol]), float(rr[ycol])
                cells.append((cx - side / 2, cy - side / 2, cx + side / 2, cy + side / 2, cls))
            items.append(dict(path=p_, cells=cells, mask=None))
    if EXT["MAX_IMAGES"]:
        items = items[:EXT["MAX_IMAGES"]]
    print(f"{EXT['NAME']}: {len(items)} images, {sum(len(i['cells']) for i in items)} labelled cells, "
          f"classes {pd.Series([c[4] for i in items for c in i['cells']]).value_counts().to_dict()}")
    return items


@torch.no_grad()
def eval_external(model, items):
    rows, cells, imgs = [], [], []
    for it in items:
        img = read_rgb(it["path"])
        if EXT["RESCALE"] != 1.0:
            img = cv2.resize(img, None, fx=EXT["RESCALE"], fy=EXT["RESCALE"], interpolation=cv2.INTER_AREA)
        H, W = img.shape[:2]; s_ = EXT["RESCALE"]
        im_in = cv2.resize(apply_clahe(img) if CFG["USE_CLAHE"] else img, (IMG_W, IMG_H), interpolation=cv2.INTER_AREA)
        x = eval_tf(image=im_in)["image"][None].to(DEVICE)
        prob = predict_probs(model, x, (H, W)); pred = prob.argmax(0).astype(np.uint8)
        stem = os.path.splitext(os.path.basename(it["path"]))[0]
        if it["mask"] is not None:
            gt = it["mask"] if s_ == 1.0 else cv2.resize(it["mask"], (W, H), interpolation=cv2.INTER_NEAREST)
            cm = confusion(pred, gt)
            d = [metrics_from_cm(cm, c)["Dice"] for c in range(1, NC)]
            bm = binary_metrics(pred > 0, gt > 0)
            rows.append(dict(image=stem, mDice=np.nanmean(d), Dice_FG=bm["Dice"], IoU_FG=bm["IoU"]))
        for (x0, y0, x1, y1, cls) in it["cells"]:
            x0, y0 = int(max(0, x0 * s_)), int(max(0, y0 * s_)); x1, y1 = int(min(W, x1 * s_)), int(min(H, y1 * s_))
            if x1 <= x0 or y1 <= y0:
                continue
            mp = prob[:, y0:y1, x0:x1].reshape(NC, -1).mean(1); fgp = mp[1:]
            cells.append(dict(image=stem, true=cls, pred=FG_CLASSES[int(np.argmax(fgp))],
                              p_abn=float(mp[ABN_IDX] / (fgp.sum() + 1e-8)), detected=bool(1 - mp[0] >= 0.5)))
        labs = [c[4] for c in it["cells"]]
        if labs:
            imgs.append(dict(image=stem, label=int(ABN_NAME in labs), score=float(np.percentile(prob[ABN_IDX], 99.5))))
        del prob, pred
    return pd.DataFrame(rows), pd.DataFrame(cells), pd.DataFrame(imgs)


EXT_SEG, EXT_CELL, EXT_IMG = [], [], []
if RUN_EXTERNAL:
    ext_items = load_external()
    for name in MAIN_MODELS:
        model, _ = load_best(name)
        a_, b_, c_ = eval_external(model, ext_items)
        EXT_SEG.append(a_.assign(model=name)); EXT_CELL.append(b_.assign(model=name)); EXT_IMG.append(c_.assign(model=name))
        del model; gc.collect(); torch.cuda.empty_cache()
    EXT_SEG = pd.concat(EXT_SEG, ignore_index=True) if any(len(x) for x in EXT_SEG) else pd.DataFrame()
    EXT_CELL = pd.concat(EXT_CELL, ignore_index=True); EXT_IMG = pd.concat(EXT_IMG, ignore_index=True)
    EXT_CELL.to_csv(os.path.join(DIRS["tables"], "external_per_cell.csv"), index=False)

    t12 = []
    for n in MAIN_MODELS:
        dc = EXT_CELL[EXT_CELL.model == n]; mt = cell_metrics(dc); lo, hi = boot_cells(dc, lambda s: cell_metrics(s)["auc"])
        internal = cell_metrics(CELLS[CELLS.model == n])["auc"]
        row = {"Model": SHORT[n], "Cell AUC": f"{mt['auc']:.3f}", "AUC 95% CI": f"[{lo:.3f}, {hi:.3f}]",
               "Δ AUC vs internal": f"{mt['auc'] - internal:+.3f}", "Balanced acc. (%)": f"{100 * mt['bacc']:.2f}",
               "Sensitivity (%)": f"{100 * mt['sens']:.2f}", "Specificity (%)": f"{100 * mt['spec']:.2f}",
               "F1 (%)": f"{100 * mt['f1']:.2f}"}
        di = EXT_IMG[EXT_IMG.model == n]
        row["Image AUC"] = f"{roc_auc_score(di.label, di.score):.3f}" if 0 < di.label.sum() < len(di) else "n/a"
        if len(EXT_SEG):
            ds = EXT_SEG[EXT_SEG.model == n]
            row["mDice (%)"] = mean_sd(ds.mDice); row["IoU (%)"] = mean_sd(ds.IoU_FG)
        t12.append(row)
    TABLE12 = pd.DataFrame(t12)
    save_table(TABLE12, "table12_external",
               f"Zero-shot external validation on {EXT['NAME']} ({EXT_CELL[EXT_CELL.model == PROPOSED].shape[0]} cells; no retraining).",
               "tab:external", bold_max=[c for c in TABLE12.columns if ("%" in c or "AUC" in c) and "Δ" not in c and "CI" not in c])

# %%
# Fig. 14  External validation: ROC, internal-to-external AUC shift, example predictions
if RUN_EXTERNAL:
    fig = plt.figure(figsize=(FIG_W["double"], 112 * MM), constrained_layout=True)
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 0.5])
    ax = fig.add_subplot(gs[0, 0])
    for n in MAIN_MODELS:
        d = EXT_CELL[EXT_CELL.model == n]; y = (d.true == ABN_NAME).values
        if 0 < y.sum() < len(y):
            fpr, tpr, _ = roc_curve(y, d.p_abn)
            ax.plot(fpr, tpr, color=mcol(n), lw=mlw(n), label=f"{SHORT[n]} ({auc(fpr, tpr):.3f})", zorder=3 if n == PROPOSED else 2)
    ax.plot([0, 1], [0, 1], ls="--", color="0.6", lw=0.6)
    ax.set(xlabel="False positive rate", ylabel="True positive rate", xlim=(0, 1), ylim=(0, 1.01), title=f"{EXT['NAME']}: cell ROC")
    ax.set_aspect("equal"); style_axes(ax, "both"); ax.legend(title="AUC", fontsize=5.5, title_fontsize=6, loc="lower right"); panel_label(ax, "a")

    ax = fig.add_subplot(gs[0, 1])
    gap = {n: (cell_metrics(CELLS[CELLS.model == n])["auc"], cell_metrics(EXT_CELL[EXT_CELL.model == n])["auc"]) for n in MAIN_MODELS}
    for n, (a_in, a_ex) in gap.items():
        ax.plot([0, 1], [a_in, a_ex], "-o", color=mcol(n), lw=mlw(n), ms=4 if n != PROPOSED else 5.5, mec="white", mew=0.5,
                zorder=3 if n == PROPOSED else 2)
    ys = sorted(((v[1], n) for n, v in gap.items()), reverse=True); rng_y = max(1e-3, max(v for v, _ in ys) - min(v for v, _ in ys))
    lab_y, prev = {}, None
    for v, n in ys:                                          # keep labels at least 7% of the range apart
        y_ = v if prev is None else min(v, prev - 0.07 * rng_y); lab_y[n] = prev = y_
    for n, (a_in, a_ex) in gap.items():
        ax.annotate(SHORT[n], (1, a_ex), xytext=(1.08, lab_y[n]), textcoords="data", va="center", fontsize=6,
                    fontweight="bold" if n == PROPOSED else "normal", arrowprops=dict(arrowstyle="-", lw=0.4, color="0.6"))
    ax.set_xticks([0, 1], ["Internal test", EXT["NAME"]]); ax.set_xlim(-0.2, 1.8); ax.set_ylabel("Cell-level AUC")
    ax.set_title("Generalisation gap"); style_axes(ax); panel_label(ax, "b")

    ax = fig.add_subplot(gs[0, 2])
    dP = EXT_CELL[EXT_CELL.model == PROPOSED]
    cmx = pd.crosstab(dP.true, dP.pred).reindex(index=FG_CLASSES, columns=FG_CLASSES, fill_value=0).values
    cmn = 100 * cmx / cmx.sum(1, keepdims=True).clip(min=1)
    ax.imshow(cmn, cmap="Blues", vmin=0, vmax=100)
    for a_ in range(len(FG_CLASSES)):
        for b_ in range(len(FG_CLASSES)):
            ax.text(b_, a_, f"{cmn[a_, b_]:.1f}%\n(n = {cmx[a_, b_]})", ha="center", va="center", fontsize=6.5,
                    color="white" if cmn[a_, b_] > 55 else "black")
    ax.set_xticks(range(len(FG_CLASSES)), [c.capitalize() for c in FG_CLASSES]); ax.set_yticks(range(len(FG_CLASSES)), [c.capitalize() for c in FG_CLASSES])
    ax.set(xlabel="Predicted", ylabel="True"); ax.set_title(f"Proposed on {EXT['NAME']}", color=mcol(PROPOSED))
    for s_ in ax.spines.values():
        s_.set_visible(False)
    ax.tick_params(length=0); panel_label(ax, "c")

    # example external images: image with labelled cells, and P(abnormal) of the proposed model
    ex_imgs = EXT_IMG[(EXT_IMG.model == PROPOSED)].sort_values("label", ascending=False).image.tolist()[:3]
    model, _ = load_best(PROPOSED)
    sub = gs[1, :].subgridspec(1, 2 * len(ex_imgs), wspace=0.03)
    for k, stem in enumerate(ex_imgs):
        it = next(i for i in ext_items if os.path.splitext(os.path.basename(i["path"]))[0] == stem)
        img = read_rgb(it["path"]); H, W = img.shape[:2]
        im_in = cv2.resize(apply_clahe(img) if CFG["USE_CLAHE"] else img, (IMG_W, IMG_H), interpolation=cv2.INTER_AREA)
        prob = predict_probs(model, eval_tf(image=im_in)["image"][None].to(DEVICE), (H, W))
        dsz = display_size(H, W, long=640); sc = dsz[0] / W
        img_s = cv2.resize(img, dsz, interpolation=cv2.INTER_AREA); pa = cv2.resize(prob[ABN_IDX], dsz, interpolation=cv2.INTER_AREA)
        a1 = fig.add_subplot(sub[0, 2 * k]); a1.imshow(img_s); image_axis(a1)
        for (x0, y0, x1, y1, cls) in it["cells"]:
            a1.add_patch(mpl.patches.Rectangle((x0 * sc, y0 * sc), (x1 - x0) * sc, (y1 - y0) * sc, fill=False, ec=VIS_CLASS[cls], lw=0.9,
                                               path_effects=[pe.Stroke(linewidth=1.8, foreground="black"), pe.Normal()]))
        a2 = fig.add_subplot(sub[0, 2 * k + 1]); prob_ax(a2, img_s, pa)
        if k == 0:
            a1.set_title("(d)  External image, reference cells", fontsize=6.5, loc="left")
            a2.set_title(f"P({ABN_NAME.lower()}), proposed", fontsize=6.5)
        del prob
    del model; torch.cuda.empty_cache()
    save_figure(fig, "Fig14_external_validation"); plt.show()


# %% [cell 30]
# ---------------------------------------------------------------------------
# 15. External validation on Mendeley LBC (liquid-based cytology, different hospital, scanner and preparation)
#   Hussain et al., Data in Brief 2020; Mendeley Data doi:10.17632/zddtpgzv63 (CC BY-NC-ND 4.0)
#   963 images, 2048 x 1536, classes NILM / LSIL / HSIL / SCC (image-level labels only)
#   Zero-shot: no retraining, no fine-tuning, decision threshold fixed on OUR validation split.
# ---------------------------------------------------------------------------
import re, zipfile, hashlib, shutil, requests
from sklearn.metrics import roc_auc_score

LBC = dict(
    DATASET="zddtpgzv63", VERSION=4,
    DRIVE_DIR=os.path.join(os.path.dirname(CFG["DATA_ROOT"].rstrip("/")), "external", "LBC"),   # .../MTA-TransUNet/external/LBC
    LOCAL_DIR="/content/lbc_cache" if IN_COLAB else os.path.join(CFG["OUT_DIR"], "lbc_cache"),
    N_BOOT=2000 if not QUICK else 100,
)
LBC_RAW = os.path.join(LBC["DRIVE_DIR"], "raw"); os.makedirs(LBC_RAW, exist_ok=True)
GRADES = ["NILM", "LSIL", "HSIL", "SCC"]


def _stream(url, out, desc):
    tmp = ("/content/" if IN_COLAB else os.path.dirname(out) + "/") + "_dl.part"
    with requests.get(url, stream=True, timeout=120, allow_redirects=True) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0)) or None
        with open(tmp, "wb") as fh, tqdm(total=total, unit="B", unit_scale=True, desc=desc, leave=False) as bar:
            for chunk in r.iter_content(1 << 20):
                fh.write(chunk); bar.update(len(chunk))
    shutil.move(tmp, out)


def fetch_lbc():
    """Download once to Drive. Tries the Mendeley full-dataset zip, then the public file API.
    If both fail, put the zip from the dataset page into LBC_RAW by hand; it is then used as is."""
    flag = os.path.join(LBC_RAW, ".download_complete")
    have = [f for f in os.listdir(LBC_RAW) if not f.startswith(".")]
    if os.path.exists(flag) or any(zipfile.is_zipfile(os.path.join(LBC_RAW, f)) for f in have if f.lower().endswith(".zip")):
        print("LBC already on Drive:", LBC_RAW); return True
    ds, v = LBC["DATASET"], LBC["VERSION"]
    # route 1: full-dataset zip served by Mendeley's download cache
    try:
        out = os.path.join(LBC_RAW, f"{ds}-{v}.zip")
        _stream(f"https://prod-dcd-datasets-cache-zipfiles.s3.eu-west-1.amazonaws.com/{ds}-{v}.zip", out, "LBC zip")
        if zipfile.is_zipfile(out):
            open(flag, "w").write("zip"); print("LBC downloaded (dataset zip)"); return True
        os.remove(out)
    except Exception as e:
        print("dataset zip route failed:", type(e).__name__)
    # route 2: public API, file by file (folders walked recursively)
    try:
        api = "https://data.mendeley.com/public-api/datasets"
        folders = requests.get(f"{api}/{ds}/folders/{v}", timeout=60).json()
        fl = [{"id": "root", "path": ""}]
        byid = {f["id"]: f for f in folders}
        for f in folders:
            parts, cur = [], f
            while cur:
                parts.append(cur["name"]); cur = byid.get(cur.get("parent_id"))
            fl.append({"id": f["id"], "path": "/".join(reversed(parts))})
        n = 0
        for fo in fl:
            files = requests.get(f"{api}/{ds}/files", params={"folder_id": fo["id"], "version": v}, timeout=60).json()
            for f in files:
                url = (f.get("content_details") or {}).get("download_url")
                if not url:
                    continue
                out = os.path.join(LBC_RAW, fo["path"], f["filename"]); os.makedirs(os.path.dirname(out), exist_ok=True)
                if not (os.path.exists(out) and os.path.getsize(out) == f.get("size", -1)):
                    _stream(url, out, f["filename"])
                n += 1
        if n:
            open(flag, "w").write("api"); print(f"LBC downloaded ({n} files via API)"); return True
    except Exception as e:
        print("API route failed:", type(e).__name__)
    print(f"Automatic download failed. Download the zip from https://data.mendeley.com/datasets/{ds}/{v} "
          f"and place it in {LBC_RAW} (no need to unzip). Then rerun this cell.")
    return False


def grade_of(path):
    """Class from folder names (robust to naming like 'High squamous intra-epithelial lesion')."""
    for part in reversed(os.path.dirname(path).lower().replace("_", " ").replace("-", " ").split(os.sep)):
        if re.search(r"\bnilm\b|negative|normal", part):
            return "NILM"
        if re.search(r"\bhsil\b|\bhigh\b", part):
            return "HSIL"
        if re.search(r"\blsil\b|\blow\b", part):
            return "LSIL"
        if re.search(r"\bscc\b|carcinoma", part):
            return "SCC"
    return None


def prepare_lbc():
    local = LBC["LOCAL_DIR"]; lst = os.path.join(local, "lbc_images.csv")
    if os.path.exists(lst):
        return pd.read_csv(lst)
    ext = os.path.join(local, "extracted"); os.makedirs(ext, exist_ok=True)
    for root, _, fs in os.walk(LBC_RAW):
        for fn in fs:
            p_ = os.path.join(root, fn)
            if zipfile.is_zipfile(p_):
                with zipfile.ZipFile(p_) as z:
                    z.extractall(ext)
            elif fn.lower().endswith(IMG_EXT):
                dst = os.path.join(ext, os.path.relpath(p_, LBC_RAW)); os.makedirs(os.path.dirname(dst), exist_ok=True)
                shutil.copy(p_, dst)
    changed = True
    while changed:                                                  # zips inside zips
        changed = False
        for root, _, fs in os.walk(ext):
            for fn in fs:
                p_ = os.path.join(root, fn)
                if fn.lower().endswith(".zip") and zipfile.is_zipfile(p_):
                    with zipfile.ZipFile(p_) as z:
                        z.extractall(os.path.splitext(p_)[0])
                    os.remove(p_); changed = True
    rows = [dict(path=os.path.join(r, f), grade=grade_of(os.path.join(r, f)))
            for r, _, fs in os.walk(ext) for f in fs if f.lower().endswith(IMG_EXT) and not f.startswith("._")]
    df = pd.DataFrame(rows).dropna().reset_index(drop=True)
    df["label"] = (df.grade != "NILM").astype(int)
    df.to_csv(lst, index=False)
    return df


RUN_LBC = bool(CFG.get("RUN_LBC_ORIGINAL", False)) and fetch_lbc()
if RUN_LBC:
    LBC_DF = prepare_lbc()
    print(f"LBC: {len(LBC_DF)} images | {LBC_DF.grade.value_counts().reindex(GRADES).to_dict()}")
    RUN_LBC = LBC_DF.label.nunique() == 2
    if not RUN_LBC:
        print("Could not find both normal and abnormal images; check folder names with LBC_DF.path.head()")

# %%
# 15b. Independence check (perceptual hash) between LBC and all our images
def dhash(path, size=8):
    g = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if g is None:
        return None
    g = cv2.resize(g, (size + 1, size), interpolation=cv2.INTER_AREA)
    return np.packbits((g[:, 1:] > g[:, :-1]).ravel())


def _hash_all(paths):
    with ThreadPoolExecutor(8) as ex:
        hs = list(ex.map(dhash, paths))
    keep = [(p_, h) for p_, h in zip(paths, hs) if h is not None]
    return [k for k, _ in keep], np.unpackbits(np.stack([h for _, h in keep]), axis=1).astype(np.uint8)


if RUN_LBC:
    k1, A_ = _hash_all([r["img_path"] for s_ in CFG["SPLITS"] for r in RECORDS[s_]])
    k2, B_ = _hash_all(LBC_DF.path.tolist())
    D = A_.shape[1] - (A_ @ B_.T + (1 - A_) @ (1 - B_).T)
    n_dup = int((D <= 6).any(0).sum())
    print(f"LBC images that duplicate an internal image (Hamming <= 6): {n_dup} of {len(k2)}")
    if n_dup:
        bad = {k2[j] for j in np.nonzero((D <= 6).any(0))[0]}
        LBC_DF = LBC_DF[~LBC_DF.path.isin(bad)].reset_index(drop=True)
        print(f"excluded; {len(LBC_DF)} independent LBC images remain")

# %%
# 15c. Zero-shot evaluation. Image score = 99.5th percentile of P(abnormal), as in Table 5.
#      Threshold per model = Youden-optimal on OUR validation split (fixed before seeing LBC).
@torch.no_grad()
def image_score(model, img_rgb):
    im = cv2.resize(apply_clahe(img_rgb) if CFG["USE_CLAHE"] else img_rgb, (IMG_W, IMG_H), interpolation=cv2.INTER_AREA)
    p = predict_probs(model, eval_tf(image=im)["image"][None].to(DEVICE), (IMG_H, IMG_W))
    return float(np.percentile(p[ABN_IDX], 99.5)), p[ABN_IDX]


if RUN_LBC:
    models = {n: load_best(n)[0] for n in MAIN_MODELS}
    # thresholds from our validation split
    THR = {}
    val_items = PapSegDataset(RECORDS["valid"], eval_tf).items
    for n, m in models.items():
        s_, y_ = [], []
        for img, msk in val_items:
            x = eval_tf(image=img)["image"][None].to(DEVICE)
            s_.append(float(np.percentile(predict_probs(m, x, (IMG_H, IMG_W))[ABN_IDX], 99.5))); y_.append(int((msk == ABN_IDX).any()))
        fpr, tpr, th = roc_curve(y_, s_)
        THR[n] = float(th[np.argmax(tpr - fpr)]) if len(set(y_)) == 2 else 0.5
    print("thresholds (from our validation split):", {SHORT[k]: round(v, 3) for k, v in THR.items()})
    # LBC scores: image outer loop, models inner loop (each image read once)
    SC = {n: np.zeros(len(LBC_DF)) for n in MAIN_MODELS}
    EX_MAPS = {}
    ex_idx = {g: LBC_DF.index[LBC_DF.grade == g][0] for g in GRADES if (LBC_DF.grade == g).any()}
    for i, r in tqdm(LBC_DF.iterrows(), total=len(LBC_DF), desc="LBC"):
        img = read_rgb(r.path)
        for n, m in models.items():
            SC[n][i], pmap = image_score(m, img)
            if n == PROPOSED and i in ex_idx.values():
                EX_MAPS[r.grade] = (cv2.resize(img, (IMG_W, IMG_H), interpolation=cv2.INTER_AREA), pmap.astype(np.float32))
    del models; gc.collect(); torch.cuda.empty_cache()
    y = LBC_DF.label.values
    pd.DataFrame({"path": LBC_DF.path, "grade": LBC_DF.grade, **{SHORT[n]: SC[n] for n in MAIN_MODELS}}).to_csv(
        os.path.join(DIRS["tables"], "lbc_image_scores.csv"), index=False)

    rngb = np.random.default_rng(CFG["SEED"]); BOOT = rngb.integers(0, len(y), (LBC["N_BOOT"], len(y)))
    ok_b = [b for b in BOOT if 0 < y[b].sum() < len(b)]

    def boot_auc(s):
        return np.array([roc_auc_score(y[b], s[b]) for b in ok_b])

    AUCB = {n: boot_auc(SC[n]) for n in MAIN_MODELS}
    t13 = []
    for n in MAIN_MODELS:
        s = SC[n]; pr = s >= THR[n]; a = roc_auc_score(y, s)
        sens, spec = pr[y == 1].mean(), (~pr[y == 0]).mean()
        row = {"Model": SHORT[n], "AUC": f"{a:.3f}", "AUC 95% CI": f"[{np.percentile(AUCB[n], 2.5):.3f}, {np.percentile(AUCB[n], 97.5):.3f}]",
               "Balanced acc. (%)": f"{100 * (sens + spec) / 2:.2f}", "Sensitivity (%)": f"{100 * sens:.2f}", "Specificity (%)": f"{100 * spec:.2f}"}
        for g in GRADES[1:]:
            mk = (LBC_DF.grade == g).values
            row[f"Detected {g} (%)"] = f"{100 * pr[mk].mean():.1f}" if mk.any() else "n/a"
        t13.append(row)
    TABLE13 = pd.DataFrame(t13)
    save_table(TABLE13, "table13_lbc_external",
               f"Zero-shot external validation on Mendeley LBC ({len(y)} images; {int(y.sum())} abnormal). Thresholds fixed on our validation split.",
               "tab:lbc", bold_max=[c for c in TABLE13.columns if c not in ("Model", "AUC 95% CI")])

    # paired bootstrap: AUC difference, proposed vs each baseline
    t14 = []
    for b in BASELINES:
        d = AUCB[PROPOSED] - AUCB[b]; obs = roc_auc_score(y, SC[PROPOSED]) - roc_auc_score(y, SC[b])
        p = min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean()))
        t14.append({"Comparison": f"Proposed vs {SHORT[b]}", "Δ AUC": f"{obs:+.3f}",
                    "95% CI": f"[{np.percentile(d, 2.5):+.3f}, {np.percentile(d, 97.5):+.3f}]", "p_raw": max(p, 1 / len(d))})
    TABLE14 = pd.DataFrame(t14)
    order = np.argsort(TABLE14.p_raw.values); m_ = len(order); adj = np.empty(m_); run = 0
    for rk, k in enumerate(order):
        run = max(run, min(1.0, (m_ - rk) * TABLE14.p_raw.values[k])); adj[k] = run
    TABLE14["p (Holm)"] = [f"{v:.1e}" for v in adj]
    TABLE14["Sig."] = ["***" if v < 1e-3 else "**" if v < 1e-2 else "*" if v < 0.05 else "n.s." for v in adj]
    save_table(TABLE14.drop(columns="p_raw"), "table14_lbc_auc_tests",
               "Paired bootstrap comparison of external AUC on Mendeley LBC (Holm-corrected).", "tab:lbc_tests")

# %%
# Fig. 15  External validation on Mendeley LBC
if RUN_LBC:
    fig = plt.figure(figsize=(FIG_W["double"], 112 * MM), constrained_layout=True)
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 0.5])
    ax = fig.add_subplot(gs[0, 0])
    for n in MAIN_MODELS:
        fpr, tpr, _ = roc_curve(y, SC[n])
        ax.plot(fpr, tpr, color=mcol(n), lw=mlw(n), label=f"{SHORT[n]} ({auc(fpr, tpr):.3f})", zorder=3 if n == PROPOSED else 2)
        s_ = SC[n] >= THR[n]
        ax.plot((s_[y == 0]).mean(), s_[y == 1].mean(), "o", ms=3.5 if n != PROPOSED else 5, mfc=mcol(n), mec="black", mew=0.5, zorder=4)
    ax.plot([0, 1], [0, 1], ls="--", color="0.6", lw=0.6)
    ax.set(xlabel="False positive rate", ylabel="True positive rate", xlim=(0, 1), ylim=(0, 1.01), title="Image-level ROC (abnormal vs NILM)")
    ax.set_aspect("equal"); style_axes(ax, "both"); ax.legend(title="AUC", fontsize=5.5, title_fontsize=6, loc="lower right"); panel_label(ax, "a")

    ax = fig.add_subplot(gs[0, 1])
    gx = np.arange(len(GRADES)); off = np.linspace(-0.33, 0.33, len(MAIN_MODELS))
    for k, n in enumerate(MAIN_MODELS):
        pr = SC[n] >= THR[n]
        vals = [100 * (~pr[(LBC_DF.grade == "NILM").values]).mean()] + [100 * pr[(LBC_DF.grade == g).values].mean() for g in GRADES[1:]]
        ax.bar(gx + off[k], vals, width=0.8 / len(MAIN_MODELS), color=mcol(n), edgecolor="black" if n == PROPOSED else "none", lw=0.5, label=SHORT[n])
    ax.set_xticks(gx, ["NILM\n(specificity)"] + [f"{g}\n(sensitivity)" for g in GRADES[1:]], fontsize=6.5)
    ax.set(ylabel="Correct at fixed threshold (%)", ylim=(0, 105), title="Per-grade performance"); style_axes(ax); panel_label(ax, "b")

    ax = fig.add_subplot(gs[0, 2])
    rng_ = np.random.default_rng(0)
    for k, g in enumerate(GRADES):
        v = SC[PROPOSED][(LBC_DF.grade == g).values]
        if not len(v):
            continue
        ax.boxplot(v, positions=[k], widths=0.5, showfliers=False, medianprops=dict(color="black", lw=1),
                   boxprops=dict(lw=0.6), whiskerprops=dict(lw=0.6), capprops=dict(lw=0.6))
        ax.scatter(k + rng_.uniform(-0.18, 0.18, len(v)), v, s=3, alpha=0.5, lw=0,
                   color=VIS_CLASS[FG_CLASSES[0]] if g == "NILM" else mcol(PROPOSED))
    ax.axhline(THR[PROPOSED], color="0.3", ls="--", lw=0.7); ax.text(-0.45, THR[PROPOSED] + 0.01, "threshold", fontsize=5.5, va="bottom", ha="left", color="0.3")
    ax.set_xticks(range(len(GRADES)), GRADES); ax.set(ylabel="Image score, proposed", ylim=(-0.02, 1.02), title="Score by Bethesda grade")
    style_axes(ax); panel_label(ax, "c")

    sub = gs[1, :].subgridspec(1, 2 * len(EX_MAPS), wspace=0.03)
    for k, g in enumerate([g for g in GRADES if g in EX_MAPS]):
        img_s, pm = EX_MAPS[g]
        a1 = fig.add_subplot(sub[0, 2 * k]); a1.imshow(img_s); image_axis(a1); a1.set_title(g, fontsize=7)
        a2 = fig.add_subplot(sub[0, 2 * k + 1]); prob_ax(a2, img_s, pm); a2.set_title(f"P({ABN_NAME.lower()})", fontsize=6.5)
        if k == 0:
            a1.text(-0.04, 1.08, "(d)", transform=a1.transAxes, fontsize=8, fontweight="bold", ha="right")
    h_, l_ = fig.axes[1].get_legend_handles_labels()
    fig.legend(h_, l_, loc="lower center", bbox_to_anchor=(0.5, -0.03), ncol=len(MAIN_MODELS), fontsize=6)
    save_figure(fig, "Fig15_external_LBC"); plt.show()


# %% [cell 31]
if RUN_LBC:
    ext = os.path.join(LBC["LOCAL_DIR"], "extracted")
    listed = set(pd.read_csv(os.path.join(LBC["LOCAL_DIR"], "lbc_images.csv")).path)
    for r, _, fs in os.walk(ext):
        n = sum(f.lower().endswith(IMG_EXT) for f in fs)
        if n: print(n, r)
        for f in fs:
            if os.path.join(r, f) not in listed: print("  not used:", f)


# %% [cell 34]
print("Outputs in", CFG["OUT_DIR"])
for sub in ["figures", "tables"]:
    print(sub, sorted(os.listdir(DIRS[sub])))
