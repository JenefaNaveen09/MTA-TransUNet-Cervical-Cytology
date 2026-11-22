# MTA-TransUNet: Joint Cell-Region Localisation and Abnormality Classification for Cervical Cytology

Code for the paper *"MTA-TransUNet: Joint Cell-Region Localisation and Abnormality Classification with Auxiliary Supervision for Cervical Cytology"* (J. Archpaul, V. Kandasamy, T. M. Thiyagu, T. Stephan).

MTA-TransUNet combines a MiT-B2 encoder, a five-block U-Net decoder, and an auxiliary image-level abnormality head. It predicts Background, Normal, and Abnormal regions in Pap smear images. The repository reproduces every experiment in the paper:

- training of the proposed model and six baselines under one protocol
- internal evaluation on CRIC Cervix: segmentation, cell-level and image-level classification, statistics, ablations, three-seed repeatability, robustness to six corruptions, and computational cost
- the CRIC overlap audit
- zero-shot external validation on Mendeley liquid-based cytology (LBC)

## Repository layout

```
MTA-TransUNet/
├── README.md
├── LICENSE
├── CITATION.cff
├── requirements.txt
├── configs/
│   ├── default.json              settings of the reported runs
│   ├── quick.json                smoke-test settings (not for results)
│   ├── models.json               model registry: encoders, decoders, parameters, GFLOPs, checkpoint names
│   ├── external_mendeley.json    external-validation protocol
│   └── datasets.json             dataset sources, licences, and expected counts
├── data/
│   ├── README.md                 how to download and arrange the data
│   └── splits/                   JSON manifests written by scripts/make_split_manifest.py
├── notebooks/
│   └── Cervical_MTA_TransUNet.ipynb   full notebook (outputs cleared)
├── src/
│   ├── run_pipeline.py           all internal experiments and the CRIC overlap audit
│   ├── mendeley_external.py      zero-shot Mendeley LBC external validation
│   └── reconcile_mendeley_roboflow.py   match the Roboflow export to the original Mendeley release
├── scripts/
│   ├── verify_data.py            check local data against the counts in the paper
│   ├── make_split_manifest.py    write file lists, box counts and SHA-256 per split
│   └── run_all.sh                run everything in order
└── results/
    └── README.md                 output files and what they contain
```

`src/run_pipeline.py` is generated from the notebook, and each `# %% [cell n]` block is one notebook cell. The script and the notebook run the same code.

## Installation

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

**Tested environment:** Google Colab with an NVIDIA L4 GPU, PyTorch 2.14.1+cu130, segmentation-models-pytorch 0.6.0, albumentations 2.0.8, and mixed-precision (FP16) training. Training all ten models (7 main + 3 ablations) and the three-seed runs takes several GPU-hours. A CPU is enough for `scripts/` and for checking the code with `configs/quick.json`.

ImageNet encoder weights are downloaded automatically by segmentation-models-pytorch from the Hugging Face Hub on first use.

## Data

Both datasets are public. Download instructions and the expected folder layout are in [`data/README.md`](data/README.md).

| Dataset | Role | Source |
|---|---|---|
| CRIC Cervix, Roboflow export v1 (399 images) | training, validation, internal test | [Lab 03 AI Center, Roboflow Universe](https://universe.roboflow.com/lab-03-ai-center-ev3hu/cervical-cancer-400-image-cric/dataset/1), CC BY 4.0; original data: Rezende et al., *Sci. Data* 2021, doi:10.1038/s41597-021-00933-8 |
| Mendeley LBC, Roboflow export v1 (959 images) | zero-shot external validation | [Projet2, Roboflow Universe](https://universe.roboflow.com/projet2/mendeley-dataset-0cl0l), CC BY 4.0; original data: Hussain et al., *Data in Brief* 2020, doi:10.1016/j.dib.2020.105589; Mendeley Data doi:10.17632/zddtpgzv63.4 |
| CRIC Cervix original release (figshare collection 4960286) | overlap audit only | downloaded automatically by `run_pipeline.py` |

After downloading, check your copy:

```bash
python scripts/verify_data.py --cric data/cric --mendeley data/Mendeley-dataset
python scripts/make_split_manifest.py --cric data/cric --mendeley data/Mendeley-dataset --out data/splits
```

The partitions of both exports were used as provided, so the splits are fully defined by the public exports.

## Running

### Locally

```bash
export CERVIX_ROOT=data/cric CERVIX_CONFIG=configs/default.json
python src/run_pipeline.py                     # train and evaluate all models, statistics, figures, overlap audit
python src/mendeley_external.py                # zero-shot external validation (needs the trained checkpoints)
LBC_DRIVE=data python src/reconcile_mendeley_roboflow.py   # optional: needs the original Mendeley release
```

Or run `bash scripts/run_all.sh`. For a quick check that the code runs end to end, use `CERVIX_CONFIG=configs/quick.json` (results are meaningless).

### Google Colab

Put the data on Google Drive under `MyDrive/MTA-TransUNet/` (layout in `data/README.md`), open `notebooks/Cervical_MTA_TransUNet.ipynb`, and run all cells. The notebook mounts Drive automatically. Training resumes from the last saved epoch if the session stops.

### Environment variables

| Variable | Default | Purpose |
|---|---|---|
| `CERVIX_ROOT` | `data/cric` (Colab: `/content/gdrive/MyDrive/MTA-TransUNet/cervical`) | CRIC export folder; results go to `$CERVIX_ROOT/results_q1_proposed` |
| `CERVIX_CONFIG` | none | JSON file that overrides the settings in notebook cell 1 |
| `CERVIX_QUICK` | `0` | `1` = built-in smoke-test settings |
| `CERVIX_MEN_DRIVE` | `<parent of CERVIX_ROOT>/Mendeley-dataset` | Mendeley Roboflow export (folder or zip) |
| `MEN_ADAPT` | `0` | `1` = also run the exploratory few-shot adaptation (not in the paper) |
| `MEN_EPOCHS` | `15` | epochs for the few-shot adaptation |
| `CERVIX_EXT_IMG`, `CERVIX_EXT_ANN`, `CERVIX_EXT_FORMAT` | figshare download | CRIC overlap-audit images, annotations, and format |
| `LBC_DRIVE`, `LBC_RAW_DIR`, `LBC_ORIG_DIR`, `LBC_RF_DIR`, `LBC_TAB_DIR` | see script | paths for `reconcile_mendeley_roboflow.py` |

## Method summary

| Item | Setting |
|---|---|
| Input | CLAHE on the L channel (clip 2.0, 8 × 8 tiles), area resize to 384 × 512, ImageNet normalisation |
| Proposed model | MiT-B2 encoder (ImageNet), U-Net decoder (256/128/64/32/16), auxiliary image head (dropout 0.2) |
| Loss | weighted cross-entropy (weights ∝ 1/√pixel frequency, normalised to mean 1, capped at 10) + soft Dice, plus 0.2 × binary cross-entropy for the auxiliary head |
| Optimisation | AdamW, lr 3 × 10⁻⁴, weight decay 10⁻⁴, batch 8, FP16, gradient clipping at norm 1.0, 3-epoch warm-up then cosine decay, 40 epochs, early stopping after 10 epochs without validation mDice improvement |
| Augmentation | horizontal and vertical flips (p 0.5 each); affine with scale 0.9–1.1, translation ±5 %, rotation ±20° (p 0.5); hue/saturation/value (p 0.5); brightness/contrast ±15 % (p 0.5); Gaussian blur (p 0.1); HED stain jitter (p 0.5); applied to all models |
| Inference | mean of softmax maps over the original, horizontal-flip, and vertical-flip views at network resolution, then one bilinear resize to the original size (internal) |
| Cell score | mean class probability inside each reference box; abnormal if mean P(abnormal) > mean P(normal) |
| Image score | 99.5th percentile of the abnormal-probability map |
| Statistics | bootstrap 95 % CIs (2,000 internal, 1,000 external), Wilcoxon signed-rank with Holm correction and rank-biserial effect size, Friedman and Nemenyi tests, paired bootstrap of AUC differences |
| Seeds | 42 (main run); 42, 7, 2024 (repeatability) |

All baselines use the same data, preprocessing, augmentation, loss, schedule, and checkpoint selection. Encoders, decoders, parameter counts, and GFLOPs are listed in `configs/models.json`.

## Reproducibility notes

- **Checkpoint name of the proposed model.** It was trained under an earlier name, so its checkpoint file is `w_o_scSE_attention.pt`, and the `+ scSE attention` ablation is stored as `MTA_TransUNet.pt`. The mapping is in `CKPT_ALIAS` (notebook cell 9) and `configs/models.json`.
- **Determinism.** Seeds are fixed, but `cudnn.benchmark=True` and mixed precision make GPU runs not bit-exact. Expect small differences in the last decimal places. For bit-exact reruns, set `torch.backends.cudnn.deterministic = True` in cell 1, which makes training slower.
- **Training annotation file.** The loader uses `_combined_annotations.json` if a split folder contains one, otherwise `_annotations.coco.json`. Exact duplicate annotations are removed, so both give the same 4,233 training boxes. `scripts/verify_data.py` checks this.
- **Mendeley external labels.** In the Roboflow export, the box labels follow the image diagnosis: abnormal boxes appear in all 350 LSIL, HSIL, and SCC images and in 1 of 609 NILM images. Cell-level external results therefore largely reflect image-level discrimination. Three released NILM images (NL\_5\_ (15), NL\_6\_ (7), NL\_9\_ (15)) were not annotated in the export.
- **Optional analyses not in the paper.** These are switched off by default: image-level evaluation on the original Mendeley release (`RUN_LBC_ORIGINAL`) and few-shot adaptation (`MEN_ADAPT=1`).
- **Figure 12.** It is produced twice, in cells 26 and 27, as in the original notebook. The second version (bar chart) overwrites the first.

## Citation

```bibtex
@article{archpaul_mtatransunet,
  title   = {MTA-TransUNet: Joint Cell-Region Localisation and Abnormality Classification with Auxiliary Supervision for Cervical Cytology},
  author  = {Archpaul, Jenefa and Kandasamy, Vidhya and Thiyagu, T. M. and Stephan, Thompson},
  journal = {[Journal]},
  year    = {[Year]},
  doi     = {[DOI]}
}
```

Please also cite the datasets (CRIC: Rezende et al., 2021; Mendeley LBC: Hussain et al., 2020) and the two Roboflow exports listed above.

## Licence

The code is released under the MIT License (see `LICENSE`). The datasets keep their own licences: the Roboflow exports are CC BY 4.0, and the original CRIC and Mendeley releases are under the terms given in their repositories. The data are not redistributed here.

## Contact

Thompson Stephan (corresponding author): thompsoncse@gmail.com
