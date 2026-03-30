#!/bin/bash
# Run MTP benchmarks as weights become available.
# Usage: ./run_benchmarks.sh [--batch-verify]

set -e
cd "$(dirname "$0")"
source .venv/bin/activate

BATCH_FLAG=""
if [[ "$1" == "--batch-verify" ]]; then
    BATCH_FLAG="--batch-verify"
fi

MAX_TOKENS=128

echo "MTP Benchmark Runner"
echo "===================="
echo "Checking for available MTP weights..."
echo

# Model configs: (mlx_model, bf16_source, weight_file)
declare -a MODELS=(
    "mlx-community/Qwen3.5-4B-4bit|Qwen/Qwen3.5-4B|mtp_weights/Qwen_Qwen3.5-4B.safetensors"
    "mlx-community/Qwen3.5-9B-MLX-4bit|Qwen/Qwen3.5-9B|mtp_weights/Qwen_Qwen3.5-9B.safetensors"
    "mlx-community/Qwen3.5-27B-4bit|Qwen/Qwen3.5-27B|mtp_weights/Qwen_Qwen3.5-27B.safetensors"
    "mlx-community/Qwen3.5-35B-A3B-4bit|Qwen/Qwen3.5-35B-A3B|mtp_weights/Qwen_Qwen3.5-35B-A3B.safetensors"
)

READY_MODELS=()
for entry in "${MODELS[@]}"; do
    IFS='|' read -r mlx_model bf16_source weight_file <<< "$entry"
    short=$(echo "$mlx_model" | sed 's|.*/||')
    if [[ -f "$weight_file" ]]; then
        echo "  [READY] $short ($weight_file)"
        READY_MODELS+=("$mlx_model")
    else
        echo "  [WAIT]  $short ($weight_file not found)"
    fi
done

echo
if [[ ${#READY_MODELS[@]} -eq 0 ]]; then
    echo "No models ready. Run extract_mtp_weights.py first."
    exit 1
fi

echo "Running benchmarks for ${#READY_MODELS[@]} model(s)..."
echo

python benchmark.py \
    --model "${READY_MODELS[@]}" \
    --max-tokens $MAX_TOKENS \
    $BATCH_FLAG \
    --output benchmark_results_all.json
