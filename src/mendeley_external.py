"""Zero-shot external validation on the Mendeley LBC Roboflow export (MTA-TransUNet, SegFormer, U-Net++).

Self-contained: needs only the checkpoints produced by src/run_pipeline.py and the Roboflow export.
    CERVIX_ROOT=data/cric python src/mendeley_external.py
The few-shot adaptation experiment at the end is not part of the paper; set MEN_ADAPT=1 to run it.
"""
# %%
# Mendeley LBC (Roboflow export) external test: data check, overlap check with our dataset, zero-shot on 3 models
import os, re, glob, json, shutil, zipfile, subprocess, gc, numpy as np, pandas as pd, cv2, torch, torch.nn.functional as F
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
from PIL import Image
from concurrent.futures import ThreadPoolExecutor
from sklearn.metrics import roc_auc_score, roc_curve
try:
    import segmentation_models_pytorch as smp
except ImportError:
    subprocess.run(["pip", "install", "-q", "segmentation-models-pytorch"]); import segmentation_models_pytorch as smp

ROOT = os.environ.get("CERVIX_ROOT", "/content/gdrive/MyDrive/MTA-TransUNet/cervical" if os.path.exists("/content") else "data/cric")
if ROOT.startswith("/content/gdrive") and not os.path.exists("/content/gdrive/MyDrive"):
    from google.colab import drive; drive.mount("/content/gdrive")
RES = f"{ROOT}/results_q1_proposed"; CK, OUT, FIG = f"{RES}/checkpoints", f"{RES}/tables", f"{RES}/figures"
PARENT = os.path.dirname(ROOT)
MEN = os.environ.get("CERVIX_MEN_DRIVE") or next((p for p in glob.glob(f"{PARENT}/*") if os.path.basename(p).lower() == "mendeley-dataset"),
                                                   f"{PARENT}/Mendeley-dataset")
LOCAL = "/content/men" if os.path.exists("/content") else "/tmp/men"
IMG_H, IMG_W, SEED, N_BOOT, HAM = 384, 512, 42, 1000, 6
MODELS = {"MTA-TransUNet (proposed)": "w_o_scSE_attention", "SegFormer": "SegFormer__MiT_B2", "U-Net++": "U_Net____R34"}
COL = {"MTA-TransUNet (proposed)": "#C0392B", "SegFormer": "#E69F00", "U-Net++": "#56B4E9"}
proposed = "MTA-TransUNet (proposed)"
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu"); AMP = DEV.type == "cuda"
IMX = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
for d in (OUT, FIG):
    os.makedirs(d, exist_ok=True)

if not os.path.isdir(LOCAL):                                             # copy once per session (zip or folder)
    zips = glob.glob(f"{MEN}/*.zip")
    if zips:
        for z in zips:
            zipfile.ZipFile(z).extractall(LOCAL)
    else:
        shutil.copytree(MEN, LOCAL)


def cls_of(name):
    n = name.lower()
    return 1 if "abn" in n else 0 if "norm" in n else None


def stem(p):
    b = os.path.basename(p)
    return re.split(r"\.rf\.", b)[0] if ".rf." in b else os.path.splitext(b)[0]


BOX = {}                                                                 # image path -> list of (x1, y1, x2, y2, label) in pixels
for js in glob.glob(f"{LOCAL}/**/*.json", recursive=True):               # COCO
    try:
        c = json.load(open(js))
    except Exception:
        continue
    if not isinstance(c, dict) or "annotations" not in c:
        continue
    cm = {k["id"]: cls_of(k["name"]) for k in c["categories"]}; im = {i["id"]: os.path.join(os.path.dirname(js), i["file_name"]) for i in c["images"]}
    for p in im.values():
        BOX.setdefault(p, [])
    for a in c["annotations"]:
        if cm.get(a["category_id"]) is not None and a.get("bbox"):
            x, y, w, h = a["bbox"]; BOX[im[a["image_id"]]].append((x, y, x + w, y + h, cm[a["category_id"]]))
if not BOX:
    for yml in glob.glob(f"{LOCAL}/**/data.yaml", recursive=True):       # YOLO (boxes or polygons)
        import yaml
        nm = yaml.safe_load(open(yml))["names"]; nm = list(nm.values()) if isinstance(nm, dict) else nm
        for t in glob.glob(f"{os.path.dirname(yml)}/**/labels/*.txt", recursive=True):
            ip = next((q for e in IMX for q in [t.replace("/labels/", "/images/")[:-4] + e] if os.path.exists(q)), None)
            if ip is None:
                continue
            W, H = Image.open(ip).size; BOX[ip] = []
            for ln in open(t).read().split("\n"):
                v = ln.split()
                if len(v) < 5 or cls_of(nm[int(v[0])]) is None:
                    continue
                q = np.array(v[1:], float)
                if len(q) == 4:
                    cx, cy, w, h = q; x1, y1, x2, y2 = cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2
                else:
                    x1, y1, x2, y2 = q[0::2].min(), q[1::2].min(), q[0::2].max(), q[1::2].max()
                BOX[ip].append((x1 * W, y1 * H, x2 * W, y2 * H, cls_of(nm[int(v[0])])))
if not BOX:
    for x in glob.glob(f"{LOCAL}/**/*.xml", recursive=True):             # Pascal VOC
        r = ET.parse(x).getroot()
        ip = next((q for e in IMX for q in [x[:-4] + e] if os.path.exists(q)), None)
        if ip is None:
            continue
        BOX[ip] = [(*[float(o.find("bndbox").find(k).text) for k in ("xmin", "ymin", "xmax", "ymax")], cls_of(o.find("name").text))
                   for o in r.iter("object") if cls_of(o.find("name").text) is not None]
assert BOX, f"no COCO / YOLO / VOC annotations found under {MEN}"

seen, ITEMS = set(), []                                                  # one copy per source image (drops Roboflow augmented copies)
for p in sorted(BOX):
    s = stem(p)
    if s in seen or not os.path.exists(p) or not BOX[p]:
        continue
    seen.add(s); b = BOX[p]
    ITEMS.append(dict(path=p, id=s, boxes=b, label=int(any(k[4] == 1 for k in b)), size=Image.open(p).size))
SZ = pd.Series([f"{w}x{h}" for w, h in (i["size"] for i in ITEMS)]).value_counts()
print(f"annotated files: {len(BOX)} | unique source images with boxes: {len(ITEMS)} | image sizes: {SZ.head(3).to_dict()}")
if len(SZ) == 1 and SZ.index[0] in ("640x640", "416x416", "512x512"):
    print("WARNING: images were resized to a square by Roboflow; re-export at original size if possible")


def dhash(path, size=8):
    g = cv2.imread(path, cv2.IMREAD_REDUCED_GRAYSCALE_4)
    return None if g is None else np.packbits((cv2.resize(g, (size + 1, size), interpolation=cv2.INTER_AREA)[:, 1:] >
                                                cv2.resize(g, (size + 1, size), interpolation=cv2.INTER_AREA)[:, :-1]).ravel())


def hash_all(paths):
    with ThreadPoolExecutor(8) as ex:
        hs = list(ex.map(dhash, paths))
    k = [(p, h) for p, h in zip(paths, hs) if h is not None]
    return [p for p, _ in k], np.unpackbits(np.stack([h for _, h in k]), axis=1).astype(np.int32)


OWN = [p for s in ("train", "valid", "test") for p in glob.glob(f"{ROOT}/{s}/*") if p.lower().endswith(IMX)]
N0 = len(ITEMS); n_dup = 0
if OWN:
    _, A = hash_all(OWN); kb, B = hash_all([i["path"] for i in ITEMS])
    D = A.shape[1] - (A @ B.T + (1 - A) @ (1 - B).T); bad = {kb[j] for j in np.nonzero((D <= HAM).any(0))[0]}
    n_dup = len(bad); ITEMS = [i for i in ITEMS if i["path"] not in bad]
    print(f"overlap with our dataset ({len(OWN)} images): {n_dup} Mendeley images within Hamming <= {HAM}; excluded")
else:
    print(f"WARNING: our images not found under {ROOT}/train|valid|test; overlap check skipped")
NC = sum(len(i["boxes"]) for i in ITEMS); NA = sum(k[4] for i in ITEMS for k in i["boxes"])
T20a = pd.DataFrame([{"Source images with boxes": N0, "Duplicates of our data (excluded)": n_dup if OWN else "not checked",
                      "Test images": len(ITEMS), "Abnormal images": sum(i["label"] for i in ITEMS),
                      "Cells (boxes)": NC, "Abnormal cells": NA, "Normal cells": NC - NA, "Image size": SZ.index[0]}])
T20a.to_csv(f"{OUT}/table20a_mendeley_data.csv", index=False); print(T20a.to_string(index=False), "\n")

# Zero-shot: no retraining, no threshold tuning. Cell score = mean class probability inside each box (as SIPaKMeD).


def build(name):
    if name == proposed:
        class Net(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.net = smp.Unet("mit_b2", encoder_weights=None, classes=3, aux_params=dict(classes=1, dropout=0.2))
                self.segmentation_head = self.net.segmentation_head

            def forward(self, x):
                out = self.net(x)
                return out if self.training else out[0]
        return Net()
    if name == "SegFormer":
        return smp.Segformer("mit_b2", encoder_weights=None, classes=3)
    return smp.UnetPlusPlus("resnet34", encoder_weights=None, classes=3)


def load(name, path=None):
    m = build(name); m.load_state_dict(torch.load(path or f"{CK}/{MODELS[name]}.pt", map_location="cpu", weights_only=False)["model"])
    return m.to(DEV).eval()


MEAN, STD = np.array([0.485, 0.456, 0.406]) * 255, np.array([0.229, 0.224, 0.225]) * 255


def prep(it):
    W, H = it["size"]
    im = cv2.imread(it["path"], cv2.IMREAD_REDUCED_COLOR_2 if max(W, H) > 1600 else cv2.IMREAD_COLOR)
    lab = cv2.cvtColor(cv2.cvtColor(im, cv2.COLOR_BGR2RGB), cv2.COLOR_RGB2LAB)
    lab[..., 0] = cv2.createCLAHE(2.0, (8, 8)).apply(lab[..., 0])
    im = cv2.resize(cv2.cvtColor(lab, cv2.COLOR_LAB2RGB), (IMG_W, IMG_H), interpolation=cv2.INTER_AREA)
    sx, sy = IMG_W / W, IMG_H / H
    bx = [(int(x1 * sx), int(y1 * sy), max(int(np.ceil(x2 * sx)), int(x1 * sx) + 1), max(int(np.ceil(y2 * sy)), int(y1 * sy) + 1), l)
          for x1, y1, x2, y2, l in it["boxes"]]
    return im, bx


@torch.no_grad()
def evaluate(model, items, name, keep=()):
    cells, imgs, ex = [], [], {}
    for it in items:
        im, bx = prep(it)
        x = torch.from_numpy(((im - MEAN) / STD).transpose(2, 0, 1)).float()[None].to(DEV)
        with torch.autocast(DEV.type, dtype=torch.float16, enabled=AMP):
            p = (model(x).float().softmax(1) + model(x.flip(-1)).float().softmax(1).flip(-1)
                 + model(x.flip(-2)).float().softmax(1).flip(-2))[0].cpu().numpy() / 3
        imgs.append(dict(model=name, image=it["id"], label=it["label"], score=float(np.percentile(p[2], 99.5))))
        fg = p.argmax(0) > 0
        for x1, y1, x2, y2, l in bx:
            mp = p[:, y1:y2, x1:x2].reshape(3, -1).mean(1)
            cells.append(dict(model=name, image=it["id"], y=l, p_abn=float(mp[2] / (mp[1] + mp[2] + 1e-8)), pred=int(mp[2] > mp[1]),
                              detected=bool(fg[y1:y2, x1:x2].mean() >= 0.25)))
        if it["id"] in keep:
            ex[it["id"]] = (im, p[2].astype(np.float16), bx)
        del x, p
    return pd.DataFrame(cells), pd.DataFrame(imgs), ex


def boot_auc(C, ims, draws):
    grp = C.groupby("image").indices; y, s = C.y.values, C.p_abn.values; out = []
    for b in draws:
        ix = np.concatenate([grp[ims[j]] for j in b if ims[j] in grp])
        out.append(roc_auc_score(y[ix], s[ix]) if 0 < y[ix].sum() < len(ix) else np.nan)
    return np.array(out)


def holm(p):
    p = np.asarray(p); o = np.argsort(p); adj = np.empty(len(p)); run = 0
    for k, i in enumerate(o):
        run = max(run, min(1.0, (len(p) - k) * p[i])); adj[i] = run
    return adj


def summary(C, I, ab):
    y, s, pr = C.y.values, C.p_abn.values, C.pred.values
    tp, tn = ((pr == 1) & (y == 1)).sum(), ((pr == 0) & (y == 0)).sum()
    fp, fn = ((pr == 1) & (y == 0)).sum(), ((pr == 0) & (y == 1)).sum()
    se, sp, pc = tp / max(tp + fn, 1), tn / max(tn + fp, 1), tp / max(tp + fp, 1)
    return {"Images": I.image.nunique(), "Cells": len(C), "Cell AUC": f"{roc_auc_score(y, s):.3f}",
            "AUC 95% CI": f"[{np.nanpercentile(ab, 2.5):.3f}, {np.nanpercentile(ab, 97.5):.3f}]",
            "Accuracy (%)": f"{100 * (tp + tn) / len(y):.2f}", "Balanced acc. (%)": f"{50 * (se + sp):.2f}",
            "Sensitivity (%)": f"{100 * se:.2f}", "Specificity (%)": f"{100 * sp:.2f}", "Precision (%)": f"{100 * pc:.2f}",
            "F1 (%)": f"{200 * pc * se / max(pc + se, 1e-8):.2f}", "Detection rate (%)": f"{100 * C.detected.mean():.2f}",
            "Image AUC": f"{roc_auc_score(I.label, I.score):.3f}" if I.label.nunique() == 2 else "n/a"}



# %%
rk = np.random.default_rng(SEED)
KEEP = [i["id"] for i in rk.permutation([i for i in ITEMS if i["label"] == 1])[:2]] + \
       [i["id"] for i in rk.permutation([i for i in ITEMS if i["label"] == 0])[:1]]
CC, II, EX = [], [], {}
for n in MODELS:                                                        # one model in memory at a time
    m = load(n); c_, i_, e_ = evaluate(m, ITEMS, n, keep=KEEP if n == proposed else ())
    CC.append(c_); II.append(i_); EX.update(e_); del m; gc.collect(); torch.cuda.empty_cache()
    print(f"{n}: {len(c_)} cells done")
CC, II = pd.concat(CC, ignore_index=True), pd.concat(II, ignore_index=True)
CC.to_csv(f"{OUT}/mendeley_zeroshot_cells.csv", index=False)
ims = sorted(CC.image.unique()); r = np.random.default_rng(SEED); draws = [r.integers(0, len(ims), len(ims)) for _ in range(N_BOOT)]
AB = {n: boot_auc(CC[CC.model == n], ims, draws) for n in MODELS}
T20 = pd.DataFrame([{"Model": n, **summary(CC[CC.model == n], II[II.model == n], AB[n])} for n in MODELS])
rows = []
for b in [n for n in MODELS if n != proposed]:
    d = AB[proposed] - AB[b]; d = d[~np.isnan(d)]
    obs = roc_auc_score(CC[CC.model == proposed].y, CC[CC.model == proposed].p_abn) - roc_auc_score(CC[CC.model == b].y, CC[CC.model == b].p_abn)
    rows.append({"Comparison": f"Proposed vs {b}", "Δ cell AUC": f"{obs:+.3f}", "95% CI": f"[{np.percentile(d, 2.5):+.3f}, {np.percentile(d, 97.5):+.3f}]",
                 "p": max(min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean())), 1 / len(d))})
T21 = pd.DataFrame(rows); ph = holm(T21.p); T21["p (Holm)"] = [f"{v:.1e}" for v in ph]
T21["Sig."] = ["***" if v < 1e-3 else "**" if v < 1e-2 else "*" if v < 0.05 else "n.s." for v in ph]; T21 = T21.drop(columns="p")
T20.to_csv(f"{OUT}/table20_mendeley_zeroshot.csv", index=False); T21.to_csv(f"{OUT}/table21_mendeley_zeroshot_tests.csv", index=False)
for t in (T20, T21):
    print(t.to_string(index=False), "\n")

MM = 1 / 25.4
plt.rcParams.update({"font.size": 7, "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42, "legend.frameon": False})
fig = plt.figure(figsize=(190 * MM, 62 * MM), constrained_layout=True); gs = fig.add_gridspec(1, 4, width_ratios=[1.15, 1, 1, 1])
ax = fig.add_subplot(gs[0, 0])
for n in MODELS:
    d = CC[CC.model == n]; f_, t_, _ = roc_curve(d.y, d.p_abn)
    ax.plot(f_, t_, color=COL[n], lw=1.8 if n == proposed else 1.1, label=f"{n} ({roc_auc_score(d.y, d.p_abn):.3f})")
ax.plot([0, 1], [0, 1], "--", color="0.6", lw=0.6); ax.set_aspect("equal")
ax.set(xlabel="False positive rate", ylabel="True positive rate", title="(a) Cell-level ROC, zero-shot"); ax.legend(title="AUC", loc="lower right", fontsize=6)
for k, key in enumerate([i for i in KEEP if i in EX]):
    im, pm, bx = EX[key]; pm = pm.astype(np.float32); a = np.clip(pm * 1.5, 0, 0.9)[..., None]
    g = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)[..., None] / 255 * 0.65 + 0.35
    ov = (np.clip((1 - a) * g + a * plt.cm.inferno(pm)[..., :3], 0, 1) * 255).astype(np.uint8).copy()
    for x1, y1, x2, y2, l in bx:
        cv2.rectangle(ov, (x1, y1), (x2 - 1, y2 - 1), (230, 0, 126) if l else (0, 200, 120), 2)
    ax = fig.add_subplot(gs[0, k + 1]); ax.imshow(ov); ax.axis("off")
    ax.set_title(f"({'bcd'[k]}) {'Abnormal' if any(b[4] for b in bx) else 'Normal'} image, proposed", fontsize=6.5)
for e, dpi in (("pdf", None), ("png", 300), ("tiff", 600)):
    fig.savefig(f"{FIG}/Fig18_external_Mendeley_zeroshot.{e}", dpi=dpi, bbox_inches="tight",
                **({"pil_kwargs": {"compression": "tiff_lzw"}} if e == "tiff" else {}))
plt.show()

# %%
if os.environ.get("MEN_ADAPT", "0") != "1":
    print("Few-shot adaptation skipped (set MEN_ADAPT=1 to run it).")
    raise SystemExit(0)

# Few-shot adaptation on Mendeley: fine-tune each model on 20% of images (stratified by image label), test on the other 80%.
# Box-supervised targets: inscribed ellipse per box (normal / abnormal), bright slide background = background, rest ignored.
# Same recipe for all models; checkpoints saved every epoch (resumable). Needs Cell 1 only; zero-shot scores read from Table 20 run.
ADAPT_FRAC, EPOCHS, BATCH, LR, IGN = 0.20, int(os.environ.get("MEN_EPOCHS", 15)), 4, 1e-4, 255
ZS = pd.read_csv(f"{OUT}/mendeley_zeroshot_cells.csv")
ra = np.random.default_rng(SEED); TR, TE = [], []
for l in (0, 1):
    g = [i for i in ITEMS if i["label"] == l]; o = ra.permutation(len(g)); k = max(1, int(round(ADAPT_FRAC * len(g))))
    TR += [g[j] for j in o[:k]]; TE += [g[j] for j in o[k:]]
print(f"adaptation split: {len(TR)} training images, {len(TE)} test images")


def target(it):
    im, bx = prep(it); g = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
    lb = np.full(g.shape, IGN, np.uint8); t, _ = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU); lb[g > t] = 0
    for x1, y1, x2, y2, l in sorted(bx, key=lambda b: b[4]):          # abnormal drawn last (wins overlaps)
        cv2.ellipse(lb, ((x1 + x2) // 2, (y1 + y2) // 2), (max(1, (x2 - x1) // 2), max(1, (y2 - y1) // 2)), 0, 0, 360, 2 if l else 1, -1)
    return im, lb


TRAIN = [target(it) for it in TR]; YIMG = [it["label"] for it in TR]
freq = np.bincount(np.concatenate([l[l != IGN] for _, l in TRAIN]), minlength=3).astype(float); freq /= freq.sum()
CEW = torch.tensor(np.minimum((w := 1 / np.sqrt(freq + 1e-8)) / w.mean(), 10.0), dtype=torch.float32, device=DEV)


def loss_fn(out, y, yi):
    aux = None
    if isinstance(out, (tuple, list)):
        out, aux = out
    out = out.float(); ce = F.cross_entropy(out, y, weight=CEW, ignore_index=IGN)
    v = (y != IGN).unsqueeze(1).float(); pr = out.softmax(1) * v; oh = F.one_hot(y.clamp(max=2), 3).permute(0, 3, 1, 2).float() * v
    dice = (2 * (pr * oh).sum((0, 2, 3)) + 1) / (pr.sum((0, 2, 3)) + oh.sum((0, 2, 3)) + 1)
    l = ce + (1 - dice[1:].mean())
    return l + 0.2 * F.binary_cross_entropy_with_logits(aux.float(), yi[:, None]) if aux is not None else l


def to_t(im):
    return torch.from_numpy(((im - MEAN) / STD).transpose(2, 0, 1)).float()


def finetune(name):
    ck = f"{CK}/{MODELS[name]}_mendeley_adapt.pt"
    m = load(name); opt = torch.optim.AdamW(m.parameters(), lr=LR, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EPOCHS); scaler = torch.amp.GradScaler(DEV.type, enabled=AMP); ep0 = 0
    if os.path.exists(ck):
        st = torch.load(ck, map_location=DEV, weights_only=False)
        m.load_state_dict(st["model"]); opt.load_state_dict(st["opt"]); sch.load_state_dict(st["sch"]); ep0 = st["epoch"]
        print(f"[{name}] resumed at epoch {ep0}")
    g = torch.Generator().manual_seed(SEED); rng = np.random.default_rng(SEED + ep0)
    for ep in range(ep0, EPOCHS):
        m.train(); tot = 0.0
        for b in torch.randperm(len(TRAIN), generator=g).split(BATCH):
            xs, ys = [], []
            for i in b.tolist():
                im, lb = TRAIN[i]
                if rng.random() < 0.5:
                    im, lb = im[:, ::-1], lb[:, ::-1]
                if rng.random() < 0.5:
                    im, lb = im[::-1], lb[::-1]
                xs.append(to_t(np.ascontiguousarray(im))); ys.append(torch.from_numpy(np.ascontiguousarray(lb)).long())
            x, y = torch.stack(xs).to(DEV), torch.stack(ys).to(DEV)
            yi = torch.tensor([YIMG[i] for i in b.tolist()], dtype=torch.float32, device=DEV)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(DEV.type, dtype=torch.float16, enabled=AMP):
                loss = loss_fn(m(x), y, yi)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); tot += loss.item() * len(b)
        sch.step()
        torch.save({"model": m.state_dict(), "opt": opt.state_dict(), "sch": sch.state_dict(), "epoch": ep + 1}, ck + ".tmp"); os.replace(ck + ".tmp", ck)
        print(f"[{name}] epoch {ep + 1}/{EPOCHS} loss {tot / len(TRAIN):.4f}")
    del m, opt; gc.collect(); torch.cuda.empty_cache()
    return ck


AD_C, AD_I = [], []
for n in MODELS:
    ck = finetune(n); m = load(n, ck); c_, i_, _ = evaluate(m, TE, n); AD_C.append(c_); AD_I.append(i_)
    del m; gc.collect(); torch.cuda.empty_cache()
AD_C, AD_I = pd.concat(AD_C, ignore_index=True), pd.concat(AD_I, ignore_index=True)
AD_C.to_csv(f"{OUT}/mendeley_adapted_cells.csv", index=False)
te_ids = {it["id"] for it in TE}; ZT = ZS[ZS.image.astype(str).isin(te_ids)]
ims2 = sorted(AD_C.image.unique()); r2 = np.random.default_rng(SEED); draws2 = [r2.integers(0, len(ims2), len(ims2)) for _ in range(N_BOOT)]
ABA = {n: boot_auc(AD_C[AD_C.model == n], ims2, draws2) for n in MODELS}
rows = []
for n in MODELS:
    s = summary(AD_C[AD_C.model == n], AD_I[AD_I.model == n], ABA[n]); z = roc_auc_score(ZT[ZT.model == n].y, ZT[ZT.model == n].p_abn)
    rows.append({"Model": n, "Zero-shot cell AUC": f"{z:.3f}", "Adapted cell AUC": s["Cell AUC"], "AUC 95% CI": s["AUC 95% CI"],
                 "Δ AUC": f"{float(s['Cell AUC']) - z:+.3f}", **{k: s[k] for k in ("Balanced acc. (%)", "Sensitivity (%)", "Specificity (%)",
                                                                                   "F1 (%)", "Detection rate (%)", "Image AUC")}})
T22 = pd.DataFrame(rows); T22.to_csv(f"{OUT}/table22_mendeley_adapted.csv", index=False)
rows = []
for b in [n for n in MODELS if n != proposed]:
    d = ABA[proposed] - ABA[b]; d = d[~np.isnan(d)]
    obs = roc_auc_score(AD_C[AD_C.model == proposed].y, AD_C[AD_C.model == proposed].p_abn) - roc_auc_score(AD_C[AD_C.model == b].y, AD_C[AD_C.model == b].p_abn)
    rows.append({"Comparison": f"Proposed vs {b}", "Δ cell AUC": f"{obs:+.3f}", "95% CI": f"[{np.percentile(d, 2.5):+.3f}, {np.percentile(d, 97.5):+.3f}]",
                 "p": max(min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean())), 1 / len(d))})
T23 = pd.DataFrame(rows); ph = holm(T23.p); T23["p (Holm)"] = [f"{v:.1e}" for v in ph]
T23["Sig."] = ["***" if v < 1e-3 else "**" if v < 1e-2 else "*" if v < 0.05 else "n.s." for v in ph]; T23 = T23.drop(columns="p")
T23.to_csv(f"{OUT}/table23_mendeley_adapted_tests.csv", index=False)
for t in (T22, T23):
    print(t.to_string(index=False), "\n")

MM = 1 / 25.4
plt.rcParams.update({"font.size": 7, "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42, "legend.frameon": False})
fig, axs = plt.subplots(1, 2, figsize=(190 * MM, 72 * MM), constrained_layout=True)
ax = axs[0]
for n in MODELS:
    for C_, ls, tag in [(ZT, ":", "zero-shot"), (AD_C, "-", "adapted")]:
        d = C_[C_.model == n]; f_, t_, _ = roc_curve(d.y, d.p_abn)
        ax.plot(f_, t_, ls=ls, color=COL[n], lw=1.8 if n == proposed else 1.0, label=f"{n}, {tag} ({roc_auc_score(d.y, d.p_abn):.3f})")
ax.plot([0, 1], [0, 1], "--", color="0.7", lw=0.5); ax.set_aspect("equal")
ax.set(xlabel="False positive rate", ylabel="True positive rate", title=f"(a) Held-out {int(100 * (1 - ADAPT_FRAC))}%: before and after adaptation")
ax.legend(fontsize=5.5, loc="lower right")
ax = axs[1]; mets = ["Balanced acc. (%)", "Sensitivity (%)", "Specificity (%)", "F1 (%)"]; off = np.linspace(-0.27, 0.27, len(MODELS))
for k, n in enumerate(MODELS):
    v = [float(T22.loc[T22.Model == n, c].iloc[0]) for c in mets]
    bars = ax.bar(np.arange(len(mets)) + off[k], v, width=0.26, color=COL[n], edgecolor="black" if n == proposed else "none", lw=0.5, label=n)
    ax.bar_label(bars, fmt="%.0f", fontsize=5, padding=1)
ax.set_xticks(range(len(mets)), [c.replace(" (%)", "") for c in mets])
ax.set(ylabel="%", ylim=(0, 108), title="(b) After adaptation, held-out cells"); ax.legend(fontsize=6, loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=3)
for e, dpi in (("pdf", None), ("png", 300), ("tiff", 600)):
    fig.savefig(f"{FIG}/Fig19_Mendeley_adaptation.{e}", dpi=dpi, bbox_inches="tight", **({"pil_kwargs": {"compression": "tiff_lzw"}} if e == "tiff" else {}))
plt.show()
