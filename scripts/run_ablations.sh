#!/usr/bin/env bash
# Automated Static Ablation Runner for Variants S0, S1, S2, S3 and Diagnostic S3-ZeroWiener
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "${SCRIPT_DIR}")"
cd "${PROJECT_ROOT}"

ITERS="${1:-80000}"
SEED="${2:-42}"

CONFIGS=(
  "configs/ablations/S0_flow_only.yaml"
  "configs/ablations/S1_daab_mtcsa_no_wiener.yaml"
  "configs/ablations/S2_twin_decoder_tilt.yaml"
  "configs/ablations/S3_lucid_static_full.yaml"
)

echo "=================================================================="
echo " Running LucidNet-Static Ablation Suite (S0 -> S3) | iters=${ITERS} | seed=${SEED}"
echo "=================================================================="

for cfg in "${CONFIGS[@]}"; do
  exp_name="$(basename "${cfg}" .yaml)"
  exp_dir="experiments/ablation_${exp_name}_seed${SEED}"
  echo ""
  echo ">>> [Train] Variant: ${exp_name} (${cfg})"
  python3 scripts/train.py \
    --config "${cfg}" \
    --exp-dir "${exp_dir}" \
    --opts "train.total_iters=${ITERS}" "experiment.seed=${SEED}"

  echo ">>> [Evaluate] Variant: ${exp_name}"
  python3 scripts/evaluate.py \
    --config "${cfg}" \
    --checkpoint "${exp_dir}/checkpoints/best_psnr.pt" \
    --tiled \
    --output-json "${exp_dir}/eval_metrics.json"
done

S3_DIR="experiments/ablation_S3_lucid_static_full_seed${SEED}"
if [[ -f "${S3_DIR}/checkpoints/best_psnr.pt" ]]; then
  echo ""
  echo ">>> [Evaluate] Diagnostic Variant: S3-ZeroWiener (beta_deconv = 0)"
  python3 scripts/evaluate.py \
    --config "configs/ablations/S3_lucid_static_full.yaml" \
    --checkpoint "${S3_DIR}/checkpoints/best_psnr.pt" \
    --zero-wiener \
    --tiled \
    --output-json "${S3_DIR}/eval_metrics_zero_wiener.json"
fi

echo ""
echo "All static ablation experiments completed successfully."
