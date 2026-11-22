#!/usr/bin/env bash
# Reproduce all experiments. Run from the repository root.
set -euo pipefail
export CERVIX_ROOT="${CERVIX_ROOT:-data/cric}"
export CERVIX_CONFIG="${CERVIX_CONFIG:-configs/default.json}"
export MPLBACKEND=Agg

python scripts/verify_data.py --cric "$CERVIX_ROOT" --mendeley "$(dirname "$CERVIX_ROOT")/Mendeley-dataset"
python scripts/make_split_manifest.py --cric "$CERVIX_ROOT" --mendeley "$(dirname "$CERVIX_ROOT")/Mendeley-dataset" --out data/splits
python src/run_pipeline.py          # internal experiments, statistics, figures, CRIC overlap audit
python src/mendeley_external.py     # zero-shot Mendeley LBC external validation
if [ -d "$(dirname "$CERVIX_ROOT")/external/LBC/raw" ]; then
  LBC_DRIVE="$(dirname "$CERVIX_ROOT")" python src/reconcile_mendeley_roboflow.py
fi
echo "Outputs in $CERVIX_ROOT/results_q1_proposed"
