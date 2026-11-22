"""Write JSON manifests of the CRIC and Mendeley Roboflow exports (file lists, sizes, box counts, SHA-256).

    python scripts/make_split_manifest.py --cric data/cric --mendeley data/Mendeley-dataset --out data/splits

Each export folder must contain train/, valid/ and test/ subfolders with a COCO annotation file.
The annotation file is chosen exactly as in the pipeline: _combined_annotations.json first, then
_annotations.coco.json, then the first *.json. Exact duplicate annotations are counted once.
"""
import argparse
import glob
import hashlib
import json
import os
import re
from collections import Counter

SPLITS = ("train", "valid", "test")


def annotation_file(split_dir):
    for name in ("_combined_annotations.json", "_annotations.coco.json"):
        p = os.path.join(split_dir, name)
        if os.path.exists(p):
            return p
    js = sorted(glob.glob(os.path.join(split_dir, "*.json")))
    return js[0] if js else None


def cell_class(name):
    n = str(name).lower()
    return "abnormal" if "abn" in n else "normal" if "norm" in n else None


def source_stem(file_name):
    b = os.path.basename(file_name)
    return re.split(r"\.rf\.", b)[0] if ".rf." in b else os.path.splitext(b)[0]


def sha256(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(chunk), b""):
            h.update(blk)
    return h.hexdigest()


def manifest(export_dir, hashes=True):
    out = {"export_dir": os.path.abspath(export_dir), "splits": {}}
    for sp in SPLITS:
        d = os.path.join(export_dir, sp)
        js = annotation_file(d) if os.path.isdir(d) else None
        if js is None:
            continue
        coco = json.load(open(js))
        cats = {c["id"]: cell_class(c["name"]) for c in coco.get("categories", [])}
        images = {}
        for im in coco["images"]:
            images.setdefault(im["file_name"], im)          # one record per file
        id2file = {im["id"]: im["file_name"] for im in coco["images"]}
        seen, counts = set(), {f: Counter() for f in images}
        for a in coco["annotations"]:
            f = id2file.get(a["image_id"]); c = cats.get(a["category_id"])
            if f is None or c is None:
                continue
            key = (f, c, tuple(round(v, 2) for v in a.get("bbox", [])))
            if key in seen:
                continue
            seen.add(key); counts[f][c] += 1
        recs = []
        for f, im in sorted(images.items()):
            p = os.path.join(d, f)
            recs.append({"file_name": f, "source_stem": source_stem(f), "width": im.get("width"),
                         "height": im.get("height"), "normal": counts[f]["normal"],
                         "abnormal": counts[f]["abnormal"], "exists": os.path.exists(p),
                         **({"sha256": sha256(p)} if hashes and os.path.exists(p) else {})})
        out["splits"][sp] = {
            "annotation_file": os.path.basename(js), "images": len(recs),
            "normal": sum(r["normal"] for r in recs), "abnormal": sum(r["abnormal"] for r in recs),
            "images_with_abnormal": sum(r["abnormal"] > 0 for r in recs),
            "missing_on_disk": sum(not r["exists"] for r in recs), "records": recs}
    allr = [r for s in out["splits"].values() for r in s["records"]]
    out["total"] = {"images": len(allr), "distinct_source_stems": len({r["source_stem"] for r in allr}),
                    "normal": sum(r["normal"] for r in allr), "abnormal": sum(r["abnormal"] for r in allr),
                    "images_with_abnormal": sum(r["abnormal"] > 0 for r in allr),
                    "image_sizes": sorted({f"{r['width']}x{r['height']}" for r in allr})}
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cric", default="data/cric")
    ap.add_argument("--mendeley", default="data/Mendeley-dataset")
    ap.add_argument("--out", default="data/splits")
    ap.add_argument("--no-hash", action="store_true", help="skip SHA-256 (faster)")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    for tag, d in (("cric_roboflow_v1", a.cric), ("mendeley_roboflow_v1", a.mendeley)):
        if not os.path.isdir(d):
            print(f"skip {tag}: {d} not found"); continue
        m = manifest(d, hashes=not a.no_hash)
        path = os.path.join(a.out, tag + ".json")
        json.dump(m, open(path, "w"), indent=1)
        print(f"{tag}: {m['total']['images']} images, {m['total']['normal']} normal / "
              f"{m['total']['abnormal']} abnormal boxes -> {path}")


if __name__ == "__main__":
    main()
