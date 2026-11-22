"""Match the Roboflow export (959 images) to the original Mendeley release (962 images).

Lists the released images missing from the export, checks for duplicates, and cross-tabulates
Bethesda labels against Roboflow abnormal boxes. Self-contained.
    LBC_DRIVE=data python src/reconcile_mendeley_roboflow.py
"""
# %%
import os, re, glob, json, zipfile, hashlib
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import numpy as np, pandas as pd, cv2

if os.path.exists("/content") and not os.path.exists("/content/gdrive/MyDrive"):
    from google.colab import drive; drive.mount("/content/gdrive")

DRIVE = os.environ.get("LBC_DRIVE", "/content/gdrive/MyDrive/MTA-TransUNet" if os.path.exists("/content") else "data")
ORIG_RAW = os.environ.get("LBC_RAW_DIR", f"{DRIVE}/external/LBC/raw")
ORIG_DIR = os.environ.get("LBC_ORIG_DIR", "/content/lbc_cache/extracted")
RF_SEARCH = [os.environ.get("LBC_RF_DIR", ""), "/content/men", f"{DRIVE}/Mendeley-dataset",
             os.path.join(os.path.dirname(DRIVE), "Mendeley-dataset")]
TAB_DIR = os.environ.get("LBC_TAB_DIR", f"{DRIVE}/cervical/results_q1_proposed/tables")
HAM = 6
IMX = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
GRADES = ["NILM", "LSIL", "HSIL", "SCC"]
RF_NAME = re.compile(r"^(.+?)_(?:jpe?g|png|bmp|tiff?)\.rf\.[0-9a-f]+\.\w+$", re.I)


def unpack(src, dst):
    os.makedirs(dst, exist_ok=True)
    for z in glob.glob(f"{src}/**/*.zip", recursive=True):
        zipfile.ZipFile(z).extractall(dst)
    changed = True
    while changed:
        changed = False
        for z in glob.glob(f"{dst}/**/*.zip", recursive=True):
            zipfile.ZipFile(z).extractall(os.path.splitext(z)[0]); os.remove(z); changed = True


def images_under(root):
    return sorted(p for p in glob.glob(f"{root}/**/*", recursive=True)
                  if p.lower().endswith(IMX) and not os.path.basename(p).startswith("._"))


def grade_of(path):
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


def crop_bars(g, tol=3):
    def flat(v_mean, v_std):
        return (v_std < tol) & ((v_mean < 10) | (v_mean > 245))
    rows = ~flat(g.mean(1), g.std(1)); cols = ~flat(g.mean(0), g.std(0))
    if rows.sum() < 8 or cols.sum() < 8:
        return g, False
    r0, r1 = np.argmax(rows), len(rows) - np.argmax(rows[::-1])
    c0, c1 = np.argmax(cols), len(cols) - np.argmax(cols[::-1])
    padded = (r0 + len(rows) - r1 + c0 + len(cols) - c1) > 4
    return g[r0:r1, c0:c1], padded


def image_info(path, crop=False):
    im = cv2.imread(path, cv2.IMREAD_COLOR)
    if im is None:
        return None, None, False
    g = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY); padded = False
    if crop:
        g, padded = crop_bars(g)
    g = cv2.resize(g, (9, 8), interpolation=cv2.INTER_AREA)
    return hashlib.md5(im.tobytes()).hexdigest(), (g[:, 1:] > g[:, :-1]).ravel().astype(np.uint8), padded


def info_all(paths, crop=False):
    with ThreadPoolExecutor(8) as ex:
        return list(ex.map(lambda p: image_info(p, crop), paths))


def hamming(A, B):
    return (A[:, None, :] != B[None, :, :]).sum(-1).astype(np.int16)


def norm_stem(s):
    return re.sub(r"[^0-9a-z]", "", s.lower())


if not images_under(ORIG_DIR):
    if glob.glob(f"{ORIG_RAW}/**/*.zip", recursive=True):
        print("unpacking the original release ..."); unpack(ORIG_RAW, ORIG_DIR)
    else:
        ORIG_DIR = ORIG_RAW
ORIG = pd.DataFrame({"path": images_under(ORIG_DIR)})
ORIG["grade"] = ORIG.path.map(grade_of)
ORIG = ORIG.dropna(subset=["grade"]).reset_index(drop=True)
assert len(ORIG), f"no original images found under {ORIG_DIR} or {ORIG_RAW}"
inf = info_all(ORIG.path.tolist())
ORIG["pix_md5"] = [i[0] for i in inf]
ok_o = np.array([i[1] is not None for i in inf])
Bo = np.stack([i[1] if i[1] is not None else np.zeros(64, np.uint8) for i in inf])
print(f"Original release: {len(ORIG)} images | {ORIG.grade.value_counts().reindex(GRADES).to_dict()} | "
      f"unreadable: {(~ok_o).sum()}")


RF_DIR, jsons = None, []
for cand in [c for c in RF_SEARCH if c and os.path.isdir(c)]:
    jsons = glob.glob(f"{cand}/**/_annotations.coco.json", recursive=True)
    if not jsons and glob.glob(f"{cand}/*.zip"):
        unpack(cand, "/content/men_rf" if os.path.exists("/content") else "/tmp/men_rf")
        jsons = glob.glob(f"{'/content/men_rf' if os.path.exists('/content') else '/tmp/men_rf'}/**/_annotations.coco.json", recursive=True)
    if jsons:
        RF_DIR = cand; break
assert jsons, f"Roboflow COCO export not found in {RF_SEARCH}; set LBC_RF_DIR"
rows = []
for js in sorted(jsons):
    cj = json.load(open(js)); d = os.path.dirname(js)
    cat = {c["id"]: c["name"].lower() for c in cj["categories"]}
    n_abn, n_nor = Counter(), Counter()
    for a in cj["annotations"]:
        nm = cat.get(a["category_id"], "")
        if "abn" in nm:
            n_abn[a["image_id"]] += 1
        elif "norm" in nm:
            n_nor[a["image_id"]] += 1
    for im in cj["images"]:
        rows.append(dict(split=os.path.basename(d), rf_file=im["file_name"], rf_path=os.path.join(d, im["file_name"]),
                         n_abnormal=n_abn[im["id"]], n_normal=n_nor[im["id"]]))
RF = pd.DataFrame(rows)
inf = info_all(RF.rf_path.tolist(), crop=True)
ok_r = np.array([i[1] is not None for i in inf])
Br = np.stack([i[1] if i[1] is not None else np.zeros(64, np.uint8) for i in inf])
n_pad = sum(i[2] for i in inf)
print(f"Roboflow export ({RF_DIR}): {len(RF)} images | {RF.split.value_counts().to_dict()} | "
      f"letterboxed: {n_pad} ({'stretched to square' if n_pad < len(RF) / 2 else 'letterboxed'})")


D = hamming(Bo, Br); D[~ok_o, :] = 99; D[:, ~ok_r] = 99
or_stem = [norm_stem(os.path.splitext(os.path.basename(p))[0]) for p in ORIG.path]
rf_stem = [norm_stem(m.group(1)) if (m := RF_NAME.match(f)) else norm_stem(os.path.splitext(f)[0]) for f in RF.rf_file]
NAME_HAM = 16
by_stem = {}
for i, s in enumerate(or_stem):
    by_stem.setdefault(s, []).append(i)
pairs = {(int(i), int(j)) for i, j in zip(*np.nonzero(D <= HAM))}
pairs |= {(i, j) for j, s in enumerate(rf_stem) for i in by_stem.get(s, []) if D[i, j] <= NAME_HAM}
I, J = (np.array(v, int) for v in zip(*sorted(pairs))) if pairs else (np.array([], int), np.array([], int))
same = np.array([or_stem[i] == rf_stem[j] for i, j in zip(I, J)], bool)
o2r, r2o = {}, {}
for k in np.lexsort((D[I, J], ~same)):
    i, j = int(I[k]), int(J[k])
    if i not in o2r and j not in r2o:
        o2r[i] = j; r2o[j] = i
ORIG["rf_file"] = [RF.rf_file.iat[o2r[i]] if i in o2r else None for i in range(len(ORIG))]
ORIG["rf_split"] = [RF.split.iat[o2r[i]] if i in o2r else None for i in range(len(ORIG))]
ORIG["hamming"] = [int(D[i, o2r[i]]) if i in o2r else np.nan for i in range(len(ORIG))]
ORIG["rf_n_abnormal"] = [int(RF.n_abnormal.iat[o2r[i]]) if i in o2r else np.nan for i in range(len(ORIG))]
n_name = sum(or_stem[i] == rf_stem[j] for i, j in o2r.items())
print(f"Matched {len(o2r)} of {len(RF)} exported images to an original ({n_name} also by file name)")


Doo = hamming(Bo, Bo); np.fill_diagonal(Doo, 99); Doo[~ok_o, :] = 99; Doo[:, ~ok_o] = 99
miss = []
for i in np.nonzero(ORIG.rf_file.isna().values)[0]:
    px = [j for j in range(len(ORIG)) if j != i and ORIG.pix_md5[i] and ORIG.pix_md5[j] == ORIG.pix_md5[i]]
    near = [int(j) for j in np.nonzero(Doo[i] <= HAM)[0] if j not in px]
    name = lambda j: f"{os.path.basename(ORIG.path[j])} ({ORIG.grade[j]}{', exported' if j in o2r else ''})"
    if not ok_o[i]:
        reason = "unreadable image file"
    elif px:
        reason = "pixel-identical to " + "; ".join(name(j) for j in px)
    elif near:
        reason = "near-duplicate of " + "; ".join(f"{name(j)}, d={Doo[i, j]}" for j in near)
    else:
        reason = "no duplicate in the release"
    miss.append(dict(image=os.path.relpath(ORIG.path[i], ORIG_DIR), grade=ORIG.grade[i], reason=reason,
                     duplicate_exported=any(j in o2r for j in px + near)))
MISSING = pd.DataFrame(miss, columns=["image", "grade", "reason", "duplicate_exported"])
print(f"\nReleased images absent from the export: {len(MISSING)}")
print(MISSING.to_string(index=False) if len(MISSING) else "  none")
unmatched_rf = RF.loc[[j for j in range(len(RF)) if j not in r2o], ["split", "rf_file"]]
print(f"Exported images with no original: {len(unmatched_rf)}")
if len(unmatched_rf):
    print(unmatched_rf.to_string(index=False))
grp = ORIG.dropna(subset=["pix_md5"]).groupby("pix_md5").size()
print(f"Pixel-identical groups in the release: {(grp > 1).sum()} ({int(grp[grp > 1].sum())} images)")


mt = ORIG.dropna(subset=["rf_file"]).copy()
mt["export"] = np.where(mt.rf_n_abnormal > 0, ">=1 abnormal box", "no abnormal box")
print("\nBethesda label (release) x abnormal boxes (export):")
print(pd.crosstab(mt.grade, mt.export, margins=True).reindex(GRADES + ["All"]))


os.makedirs(TAB_DIR, exist_ok=True)
ORIG.drop(columns="pix_md5").to_csv(f"{TAB_DIR}/lbc_roboflow_reconciliation.csv", index=False)
MISSING.to_csv(f"{TAB_DIR}/lbc_missing_from_roboflow.csv", index=False)
print(f"\nSaved lbc_roboflow_reconciliation.csv and lbc_missing_from_roboflow.csv to {TAB_DIR}")

words = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five"}
w = lambda n: words.get(n, str(n))
img = lambda n: "image" if n == 1 else "images"
grades_of = lambda df: " and ".join(f"{w(n)} {g}" for g, n in Counter(df.grade).items())
is_dup = MISSING.duplicate_exported & MISSING.reason.str.startswith("pixel-identical") if len(MISSING) else pd.Series(dtype=bool)
dup, other = MISSING[is_dup], MISSING[~is_dup]
tail = (f" Because the primary analysis used all {len(ORIG)} released images, the omission affects only the "
        f"secondary cell-level analysis ({100 * len(other) / len(ORIG):.1f}% of images).")
if len(unmatched_rf):
    print("\nSome exported images matched no original; inspect them before writing the paper sentence.")
elif not len(MISSING):
    print("\nPaper: All released images were present in the export.")
elif not len(other):
    print(f"\nPaper: Matching by file name and dHash showed that the {w(len(dup))} missing {img(len(dup))} "
          f"({grades_of(dup)}) {'was' if len(dup) == 1 else 'were each'} pixel-identical to another released image "
          "present in the export, consistent with Roboflow's automatic removal of duplicate uploads; no distinct "
          "image was lost.")
elif not len(dup):
    print(f"\nPaper: Matching by file name and dHash identified the {w(len(other))} missing {img(len(other))} "
          f"({grades_of(other)}). {'It did' if len(other) == 1 else 'None'} not duplicate another released image"
          ", and the export records do not state why "
          f"{'it was' if len(other) == 1 else 'they were'} omitted." + tail)
else:
    print(f"\nPaper: Matching by file name and dHash identified the {w(len(MISSING))} missing images. "
          f"{w(len(dup)).capitalize()} of them ({grades_of(dup)}) "
          f"{'was a pixel-identical duplicate of an exported image' if len(dup) == 1 else 'were pixel-identical duplicates of exported images'}"
          ", consistent with Roboflow's automatic removal of duplicate uploads. The export records do "
          f"not state why the other {w(len(other))} ({grades_of(other)}) {'was' if len(other) == 1 else 'were'} omitted." + tail)
