# Outputs

All outputs are written to `$CERVIX_ROOT/results_q1_proposed/`:

```
results_q1_proposed/
├── checkpoints/   <model>.pt (best), <model>_last.pt (resume), <model>.done
├── logs/          config.json, <model>_history.csv
├── masks/         rasterised reference masks
├── tables/        .csv and .tex tables
└── figures/       .pdf, .png (300 dpi), and .tiff (600 dpi) figures
```

## Tables

| File | Content | Produced by |
|---|---|---|
| table1_dataset | dataset and split statistics | run_pipeline.py, cell 4 |
| table2_segmentation | Dice, IoU, recall, precision, HD95 of all models | cell 14 |
| table3_wilcoxon | paired Wilcoxon tests, proposed vs each baseline (Holm) | cell 14 |
| table4_cell_classification | cell-level AUC, accuracy, sensitivity, specificity, F1, detection rate | cell 14 |
| table5_image_screening | image-level screening | cell 14 |
| table6_cost | parameters, GFLOPs, latency | cells 13–14 |
| table7_ablation | component ablations | cell 14 |
| table8_robustness | mDice under six corruptions at three severities | cell 16 |
| table10_multiseed, table11_multiseed_tests | three-seed repeatability | cell 28 |
| table12_external | CRIC overlap audit (not independent) | cells 28–29 |
| table13_lbc_external, table14_lbc_auc_tests | original Mendeley release, image level (optional, `RUN_LBC_ORIGINAL`) | cell 30 |
| table20a_mendeley_data | Mendeley export: data and overlap check | mendeley_external.py |
| table20_mendeley_zeroshot | zero-shot cell- and image-level results | mendeley_external.py |
| table21_mendeley_zeroshot_tests | paired bootstrap AUC tests (Holm) | mendeley_external.py |
| table22/23_mendeley_adapted* | few-shot adaptation (optional, `MEN_ADAPT=1`) | mendeley_external.py |
| lbc_roboflow_reconciliation, lbc_missing_from_roboflow | export vs original release | reconcile_mendeley_roboflow.py |

Per-item data: `per_image_test_metrics.csv`, `per_cell_predictions.csv`, `robustness_raw.csv`, `multiseed_per_image.csv`, `multiseed_per_seed.csv`, `external_per_cell.csv`, `mendeley_zeroshot_cells.csv`.

## Figures

| File | Content |
|---|---|
| Fig1_dataset_overview | annotation distributions |
| Fig2_annotated_samples | annotated training images |
| Fig3_preprocessing | CLAHE and resizing |
| Fig4_learning_curves | validation mDice and training loss |
| Fig5_baseline_comparison | metrics with 95 % CI and paired per-image gains |
| Fig6_critical_difference | Friedman–Nemenyi critical-difference diagram |
| Fig7_qualitative_comparison | all models on regions with abnormal cells |
| Fig8_proposed_segmentation | proposed model: prediction, error map, P(abnormal) |
| Fig9_robustness | mDice vs corruption severity |
| Fig10_cell_classification | confusion matrices and ROC curves |
| Fig11_feature_space_and_cells | t-SNE of cell features and example cells |
| Fig12_ablation_and_cost | ablation and accuracy–cost trade-off |
| Fig13_multiseed | three-seed repeatability |
| Fig14_external_validation | CRIC overlap audit |
| Fig15_external_LBC | original Mendeley release (optional) |
| Fig18_external_Mendeley_zeroshot | zero-shot Mendeley evaluation |
| Fig19_Mendeley_adaptation | few-shot adaptation (optional) |

Figure and table numbers here are the file names used by the code and may differ from the numbering in the published paper.
