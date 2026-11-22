# Data

The data are not included in this repository. Both datasets are public; download them as described below.

## 1. CRIC Cervix, Roboflow export (training, validation, internal test)

- Source: **Cervical Cancer 400 Image Cric**, version 1, by Lab 03 AI Center on Roboflow Universe, CC BY 4.0
  https://universe.roboflow.com/lab-03-ai-center-ev3hu/cervical-cancer-400-image-cric/dataset/1
- Download format: **COCO JSON**. The export was produced with auto-orient and a stretch to 1376 x 1020, without augmentation.
- Original data: CRIC Searchable Image Database (https://database.cric.com.br); Rezende et al., *Scientific Data* 8, 151 (2021), doi:10.1038/s41597-021-00933-8.

Unzip so that each split folder holds its images and its COCO file:

```
data/cric/
├── train/   147 images + _annotations.coco.json
├── valid/   148 images + _annotations.coco.json
└── test/    104 images + _annotations.coco.json
```

Expected content: 399 image files from 397 distinct source images (one source image is repeated within train and one within valid); normal/abnormal boxes of 2,725/1,508 (train), 2,384/2,198 (valid), and 1,739/1,020 (test). Classes: normal = NILM; abnormal = ASC-US, LSIL, ASC-H, HSIL, SCC.

## 2. Mendeley LBC, Roboflow export (external validation)

- Source: **Mendeley-dataset**, version 1, by Projet2 on Roboflow Universe, CC BY 4.0
  https://universe.roboflow.com/projet2/mendeley-dataset-0cl0l
- Download format: **COCO JSON** (640 x 640 images).
- Original data: Hussain et al., *Data in Brief* 30, 105589 (2020), doi:10.1016/j.dib.2020.105589; Mendeley Data, version 4, doi:10.17632/zddtpgzv63.4.

Place the downloaded zip, or the extracted folders, in:

```
data/Mendeley-dataset/        Roboflow zip, or train/ valid/ test/ with _annotations.coco.json
```

Expected content: 959 images (671/191/97 in the Roboflow partitions, pooled for evaluation); 4,745 boxes (2,425 normal, 2,320 abnormal); 351 images with at least one abnormal box.

## 3. Optional: original releases

These are used only for the overlap audit and the reconciliation check.

```
data/external/CRIC/           CRIC figshare collection 4960286; downloaded automatically by src/run_pipeline.py
data/external/LBC/raw/        original Mendeley release zip (962 images), for src/reconcile_mendeley_roboflow.py
```

The Mendeley release is described as 963 images, but the public version 4 contains 962 (NILM 612, LSIL 113, HSIL 163, SCC 74).

## Google Colab layout

On Colab, use the same structure under `MyDrive/MTA-TransUNet/`:

```
MyDrive/MTA-TransUNet/
├── cervical/                 = data/cric (train/ valid/ test/); results are written to cervical/results_q1_proposed/
├── Mendeley-dataset/         = data/Mendeley-dataset
└── external/{CRIC,LBC}/      = data/external/
```

## Verify and record the splits

```bash
python scripts/verify_data.py --cric data/cric --mendeley data/Mendeley-dataset
python scripts/make_split_manifest.py --cric data/cric --mendeley data/Mendeley-dataset --out data/splits
```

`make_split_manifest.py` writes `data/splits/cric_roboflow_v1.json` and `data/splits/mendeley_roboflow_v1.json`. Each lists every image file per split with its source stem, size, normal and abnormal box counts, and SHA-256 checksum.
