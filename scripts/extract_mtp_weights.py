#!/usr/bin/env python3
"""
Phase 0b: Extract MTP Weights from Original Qwen Models

Downloads MTP weight tensors from the original BF16 Qwen3.5 model
and saves them as a separate safetensors file that can be loaded
alongside the quantized MLX model.

Usage:
    python extract_mtp_weights.py --source Qwen/Qwen3.5-4B --target mlx-community/Qwen3.5-4B-4bit
    python extract_mtp_weights.py --source Qwen/Qwen3.5-35B-A3B --target mlx-community/Qwen3.5-35B-A3B-4bit

This will create a `mtp_weights.safetensors` file in the target model's cache directory.
"""

import argparse
import json
import os
from pathlib import Path

import mlx.core as mx
from huggingface_hub import hf_hub_download
from safetensors.numpy import save_file as np_save_file


def extract_mtp_weights(
    source_repo: str,
    output_path: str = None,
) -> Path:
    """
    Extract MTP weights from a source Qwen3.5 model.

    Args:
        source_repo: HuggingFace repo with BF16 weights (e.g., Qwen/Qwen3.5-4B)
        output_path: Where to save the extracted weights. If None, saves next to
                     the source model cache.

    Returns:
        Path to the saved mtp_weights.safetensors file
    """
    print(f"Extracting MTP weights from {source_repo}")

    # 1. Download and parse the safetensors index
    print("  Downloading safetensors index...")
    try:
        idx_path = hf_hub_download(source_repo, "model.safetensors.index.json")
    except Exception:
        # Single file model (no index)
        print("  No index file - trying single safetensors...")
        sf_path = hf_hub_download(source_repo, "model.safetensors")
        weights = mx.load(sf_path)
        mtp_weights = {k: v for k, v in weights.items() if "mtp" in k.lower()}
        if not mtp_weights:
            raise ValueError(f"No MTP weights found in {source_repo}")
        return _save_mtp_weights(mtp_weights, output_path or "mtp_weights.safetensors")

    with open(idx_path) as f:
        index = json.load(f)

    # 2. Find which shard files contain MTP weights
    mtp_keys = {}
    mtp_files = set()
    for key, filename in index["weight_map"].items():
        if "mtp" in key.lower():
            mtp_keys[key] = filename
            mtp_files.add(filename)

    if not mtp_keys:
        raise ValueError(f"No MTP weights found in {source_repo}")

    print(f"  Found {len(mtp_keys)} MTP tensors across {len(mtp_files)} shard(s)")
    for k in sorted(mtp_keys.keys()):
        print(f"    {k}")

    # 3. Download only the needed shards and extract MTP tensors
    mtp_weights = {}
    for shard_file in sorted(mtp_files):
        print(f"  Downloading {shard_file}...")
        shard_path = hf_hub_download(source_repo, shard_file)
        shard_weights = mx.load(shard_path)
        for key in mtp_keys:
            if mtp_keys[key] == shard_file and key in shard_weights:
                mtp_weights[key] = shard_weights[key]

    print(f"  Extracted {len(mtp_weights)} MTP tensors")

    # 4. Save
    out_path = output_path or "mtp_weights.safetensors"
    return _save_mtp_weights(mtp_weights, out_path)


def _save_mtp_weights(weights: dict, output_path: str) -> Path:
    """Save MTP weights as a safetensors file using MLX."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    # MLX can save safetensors directly
    mx.save_safetensors(str(out), weights)

    total_params = sum(w.size for w in weights.values())
    size_mb = sum(w.nbytes for w in weights.values()) / (1024 * 1024)
    print(f"  Saved to {out} ({total_params:,} params, {size_mb:.1f} MB)")

    # Also print weight info for debugging
    print(f"\n  Weight details:")
    for k in sorted(weights.keys()):
        w = weights[k]
        print(f"    {k}: shape={list(w.shape)} dtype={w.dtype}")

    return out


def get_mtp_save_path(target_repo: str) -> Path:
    """Get the path where MTP weights should be saved for a target MLX model."""
    from huggingface_hub import snapshot_download

    model_path = Path(snapshot_download(
        target_repo,
        allow_patterns=["config.json"],
    ))
    return model_path / "mtp_weights.safetensors"


def main():
    parser = argparse.ArgumentParser(description="Extract MTP weights from Qwen3.5 models")
    parser.add_argument(
        "--source",
        required=True,
        help="Source HF repo with BF16 weights (e.g., Qwen/Qwen3.5-4B)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output path for mtp_weights.safetensors (default: ./mtp_weights/<source_name>.safetensors)",
    )
    args = parser.parse_args()

    if args.output is None:
        safe_name = args.source.replace("/", "_")
        args.output = f"mtp_weights/{safe_name}.safetensors"

    extract_mtp_weights(args.source, args.output)
    print("\nDone! Use this file with vllm_mlx_mtp to enable MTP speculative decoding.")


if __name__ == "__main__":
    main()
