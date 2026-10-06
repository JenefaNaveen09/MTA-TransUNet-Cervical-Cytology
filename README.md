# Cervical Cytology Image Segmentation

**PyTorch research code for multiclass segmentation, model comparison, and external evaluation of cervical cytology images.**

This repository contains a Jupyter notebook for preparing COCO-style annotations, training segmentation networks, evaluating predictions, and producing figures and tables. The workflow assigns background and annotation-derived foreground classes to image pixels. Separate notebook sections evaluate models labelled MTA-TransUNet, SegFormer, and U-Net++ on external SIPaKMeD and Mendeley data.

## Contents

| Path | Description |
| --- | --- |
| `Code/Cervical_new.ipynb` | Main notebook, including training, evaluation, visualization, and external studies |
| `config_all_models.json` | Settings for the five segmentation baselines |
| `config_unet.json`, `config_attention_unet.json` | Settings for the custom U-Net and Attention U-Net |
| `config_unet_resnet34.json`, `config_unetpp_resnet34.json`, `config_deeplabv3plus_resnet34.json` | Settings for ResNet-34 encoder models |
| `config_quick_smoke.json` | Reduced settings for checking the main workflow |
| `config_sipakmed_external.json`, `config_mendeley_external.json` | External evaluation settings and checkpoint identifiers |
| `figures/` | Supplied dataset, preprocessing, learning-curve, and evaluation artwork |
| `logs/` | Saved configuration and two training-history CSV files |

Paths above are relative to the outer `Code/` directory extracted from the archive. Place this README in that directory.

## Implemented workflow

- Read training, validation, and test annotations from COCO-style JSON files.
- Construct semantic masks from polygon annotations and derive class names from the annotated categories.
- Apply optional contrast-limited adaptive histogram equalization (CLAHE), resizing, normalization, and training augmentation.
- Train segmentation models with class-weighted cross-entropy and foreground Dice loss.
- Use AdamW, learning-rate warmup followed by cosine scheduling, and early stopping based on validation Dice.
- Evaluate class-specific and combined foreground segmentation, with optional horizontal and vertical flip test-time augmentation.
- Generate quantitative summaries, paired comparisons, probability maps, qualitative overlays, and publication figures.

### Main segmentation models

| Model | Implementation |
| --- | --- |
| U-Net | Custom convolutional encoder and decoder |
| Attention U-Net | Custom U-Net with additive attention gates |
| U-Net (ResNet-34) | `segmentation_models_pytorch.Unet` |
| U-Net++ (ResNet-34) | `segmentation_models_pytorch.UnetPlusPlus` |
| DeepLabV3+ (ResNet-34) | `segmentation_models_pytorch.DeepLabV3Plus` |

The default notebook configuration selects **U-Net (ResNet-34)**. Change `CFG["MODELS"]` to run other models or the complete baseline comparison.

## Installation

Use a Python environment with Jupyter and PyTorch. A CUDA-capable GPU is preferable for training; the notebook selects CPU when CUDA is unavailable.

```bash
python -m venv .venv
```

Activate the environment:

```bash
# Linux or macOS
source .venv/bin/activate

# Windows PowerShell
.venv\Scripts\Activate.ps1
```

Install the packages imported by the notebook:

```bash
python -m pip install jupyter torch torchvision numpy pandas opencv-python matplotlib scipy scikit-learn scikit-image tqdm Pillow PyWavelets albumentations segmentation-models-pytorch
```

For external SIPaKMeD archives, install `py7zr` if a `7z` executable is unavailable:

```bash
python -m pip install py7zr
```

These are dependency names inferred from the code, rather than a version-pinned environment. Package versions are not supplied in the archive. The notebook uses recent PyTorch APIs, including `torch.utils.flop_counter.FlopCounterMode`, and external evaluation uses `smp.Segformer`; confirm that the installed versions provide these APIs. ImageNet encoder initialization may require downloading pretrained weights.

## Dataset preparation

The primary images and annotation files are not included in the archive. Arrange the dataset as follows:

| Directory | Required contents |
| --- | --- |
| `<dataset-root>/train/` | Training images and COCO annotation JSON |
| `<dataset-root>/valid/` | Validation images and COCO annotation JSON |
| `<dataset-root>/test/` | Test images and COCO annotation JSON |

Place image files directly inside their split directory. The loader resolves each annotation filename by its basename.

For each split, the loader prioritizes `_combined_annotations.json`, then `_annotations.coco.json`, and otherwise selects the first JSON file in sorted order. Annotation JSON must contain `images`, `annotations`, and `categories`. Image records should include `id`, `file_name`, `width`, and `height`; annotation records need `image_id`, `category_id`, and `bbox`, with polygon `segmentation` where available.

**Mask interpretation:** polygon annotations are rasterized as class masks. Empty segmentation fields and RLE annotations fall back to filled bounding boxes; RLE masks are not decoded. Metrics from these fallback masks measure agreement with rectangular labels and should be interpreted accordingly. Later foreground classes overwrite earlier classes where annotations overlap.

## Run the main experiment

1. Open `Code/Cervical_new.ipynb` in Jupyter or upload it to Google Colab.
2. Before the first notebook cell, set the dataset location:

   ```python
   import os
   os.environ["CERVIX_ROOT"] = "/absolute/path/to/cervical"
   ```

   In Colab, use the appropriate mounted Google Drive path. The default path is `/content/gdrive/MyDrive/cervical`.

3. Review the `CFG` dictionary, especially `DATA_ROOT`, `MODELS`, `ENCODER_WEIGHTS`, and `RESUME`.
4. Execute the main workflow cells in order. Run external-study cells separately after providing their datasets and checkpoints.
5. Inspect the saved configuration, training history, generated masks, and evaluation tables before comparing results.

### Configuration files

The supplied JSON files record experiment settings. **The notebook does not automatically load these files.** To use one, add a configuration-loading block after the initial `CFG` definition and quick-mode update, but before `CFG["OUT_DIR"]` and `DIRS` are created:

```python
with open("config_all_models.json", "r", encoding="utf-8") as f:
    CFG.update(json.load(f))

# Override machine-specific paths after loading the supplied configuration.
CFG["DATA_ROOT"] = os.environ["CERVIX_ROOT"]
```

Use a path relative to the notebook kernel's working directory. The JSON files contain author-specific Google Drive paths that must be changed for a new environment. External JSON files document settings for their separate sections; those sections also require editing their own variables.

### Default training settings

| Setting | Default |
| --- | --- |
| Input long side | 512 pixels |
| Batch size | 8 |
| Maximum epochs | 60 |
| Initial learning rate | 0.0003 |
| Weight decay | 0.0001 |
| Warmup | 3 epochs |
| Early-stopping patience | 15 epochs |
| Random seed | 42 |
| Cross-entropy / Dice weights | 1.0 / 1.0 |
| CLAHE | Enabled |
| Flip test-time augmentation | Enabled |
| Encoder initialization | ImageNet for the supported encoder models |
| Bootstrap draws | 2,000 |

The short input dimension is calculated from the median dataset aspect ratio and rounded to a multiple of 32.

### Quick workflow check

Before running the first notebook cell:

```python
import os
os.environ["CERVIX_ROOT"] = "/absolute/path/to/cervical"
os.environ["CERVIX_QUICK"] = "1"
```

Quick mode uses a 128-pixel input long side, batch size 4, three training epochs, no pretrained encoder weights, and reduced sampling. It still requires valid dataset splits. Use it to check data loading and execution, rather than to reproduce full experiment results. Remove this setting or set it to `"0"` for full training.

## Outputs and evaluation

Main-workflow outputs are written to `<dataset-root>/results_q1/`:

| Subdirectory | Generated artifacts |
| --- | --- |
| `checkpoints/` | Saved model checkpoints |
| `logs/` | Configuration and training histories |
| `masks/` | Rasterized annotation masks for each split |
| `tables/` | Dataset summaries, evaluation results, statistical comparisons, and computational-cost summaries |
| `figures/` | Dataset plots, learning curves, Dice distributions, prediction comparisons, probability maps, ROC/PR curves, and confusion matrices |

The main evaluation includes Dice, IoU, precision, recall, specificity, foreground accuracy, and foreground HD95. Pixel-level ROC and precision-recall curves use sampled pixels. The notebook includes bootstrap summaries and paired Wilcoxon signed-rank comparisons with Holm adjustment when multiple models are evaluated.

The figure export function writes PDF, 300 dpi PNG, and 600 dpi LZW-compressed TIFF. Existing figures and histories are supplied artifacts; reproducing them requires the corresponding data, settings, and model runs. No performance values are asserted in this README because the archive does not include a complete numerical results table for all studies.

## External evaluation

Separate notebook sections contain SIPaKMeD full-field evaluation and Mendeley evaluation, including zero-shot and fine-tuning routines.

| Requirement | SIPaKMeD | Mendeley |
| --- | --- | --- |
| Dataset environment variable | `CERVIX_SPK_DRIVE` | `CERVIX_MEN_DRIVE` |
| Checkpoint directory | `<dataset-root>/results_q1_proposed/checkpoints/` | `<dataset-root>/results_q1_proposed/checkpoints/` |
| Model identifiers | `w_o_scSE_attention`, `SegFormer__MiT_B2`, `U_Net____R34` | Same identifiers |

The external loader expects `.pt` files containing a `model` state dictionary. These checkpoints are **not included** in the supplied archive. Main baseline training writes to `results_q1/` and does not produce all checkpoints expected by the external sections.

In the external code, the label **MTA-TransUNet (ours)** refers to a wrapper around `smp.Unet("mit_b2", ...)` with a three-class segmentation output and an auxiliary classification head. SegFormer uses a MiT-B2 encoder, and U-Net++ uses ResNet-34. The supplied main training workflow does not train the model labelled MTA-TransUNet. Match the external constructor and checkpoint architecture before running these sections.

SIPaKMeD processing uses full-field images with contour `.dat` files. The code maps Koilocytotic and Dyskeratotic classes to abnormal labels. Mendeley processing has separate annotation-parsing routines. Inspect the relevant external cell for its expected dataset layout and label mapping.

## Reproducibility notes

- Record the Python, PyTorch, segmentation-models-pytorch, and Albumentations versions, hardware, and complete configuration for each run.
- Set `RESUME=False` and use a separate output directory when starting an independent experiment. With `RESUME=True`, existing checkpoints can cause training to be skipped.
- The default seed is 42, but cuDNN deterministic mode is disabled and benchmarking is enabled. Identical numerical results across environments are not guaranteed.
- The notebook caches resized images and masks in RAM. Reduce input size or use a smaller dataset for initial checks if memory is limited.
- External sections require additional data and compatible trained weights. They should not be treated as an unconditional continuation of the main baseline run.

## Troubleshooting

| Issue | Check |
| --- | --- |
| Missing COCO JSON | Confirm the split directory and annotation filename |
| Missing image files | Confirm image basenames and their placement inside each split |
| Repeated image-ID error | Use the original per-split COCO file when merged annotation blocks cannot be linked safely |
| CUDA memory error | Reduce `BATCH_SIZE` or `LONG_SIDE` |
| Existing training unexpectedly skipped | Review `RESUME` and the checkpoint directory |
| Missing external checkpoint | Supply the required `.pt` file in `results_q1_proposed/checkpoints/` |
| State-dictionary mismatch | Match the checkpoint to its exact model constructor and package version |
| External archive extraction error | Provide `7z` or install `py7zr` |

