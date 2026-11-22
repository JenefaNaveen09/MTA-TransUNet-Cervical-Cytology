"""Check local copies of the two Roboflow exports against the counts reported in the paper.

    python scripts/verify_data.py --cric data/cric --mendeley data/Mendeley-dataset

Expected values come from configs/datasets.json. Exit code 0 if every check passes.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from make_split_manifest import manifest  # noqa: E402


def check(rows, label, got, want):
    ok = got == want
    rows.append((label, got, want, "PASS" if ok else "FAIL"))
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cric", default="data/cric")
    ap.add_argument("--mendeley", default="data/Mendeley-dataset")
    ap.add_argument("--config", default="configs/datasets.json")
    a = ap.parse_args()
    exp = json.load(open(a.config))
    rows, ok = [], True

    if os.path.isdir(a.cric):
        m, e = manifest(a.cric, hashes=False), exp["cric"]
        for sp, want in e["splits"].items():
            s = m["splits"].get(sp, {})
            for k in ("images", "normal", "abnormal"):
                ok &= check(rows, f"CRIC {sp} {k}", s.get(k), want[k])
            ok &= check(rows, f"CRIC {sp} files missing on disk", s.get("missing_on_disk"), 0)
        ok &= check(rows, "CRIC distinct filename stems", m["total"]["distinct_source_stems"], e["distinct_filename_stems"])
        ok &= check(rows, "CRIC image size", m["total"]["image_sizes"], ["x".join(map(str, e["image_size_wh"]))])
    else:
        rows.append(("CRIC export", a.cric, "folder", "SKIP"))

    if os.path.isdir(a.mendeley):
        m, e = manifest(a.mendeley, hashes=False), exp["mendeley_lbc"]
        for sp, want in e["export_partitions"].items():
            ok &= check(rows, f"Mendeley {sp} images", m["splits"].get(sp, {}).get("images"), want)
        ok &= check(rows, "Mendeley images", m["total"]["images"], e["export_images"])
        ok &= check(rows, "Mendeley normal boxes", m["total"]["normal"], e["boxes"]["normal"])
        ok &= check(rows, "Mendeley abnormal boxes", m["total"]["abnormal"], e["boxes"]["abnormal"])
        ok &= check(rows, "Mendeley images with abnormal box", m["total"]["images_with_abnormal"], e["images_with_abnormal_box"])
    else:
        rows.append(("Mendeley export", a.mendeley, "folder", "SKIP"))

    w = max(len(r[0]) for r in rows)
    for label, got, want, st in rows:
        print(f"{st:4s}  {label:<{w}}  got {got}  expected {want}")
    print("ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
