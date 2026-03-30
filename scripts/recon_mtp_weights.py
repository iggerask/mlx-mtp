#!/usr/bin/env python3
"""
Phase 0: MTP Weight Recon Script

Inspects Qwen3.5 MLX model checkpoints to determine:
- Whether MTP weights are present in the safetensors files
- The exact parameter names and shapes
- The config fields related to MTP
- Architecture dimensions for building the MTP head

Usage:
    python recon_mtp_weights.py [model_name]

Default model: mlx-community/Qwen3.5-4B-4bit (smallest for quick recon)
"""

import json
import sys
from pathlib import Path

from huggingface_hub import hf_hub_download, snapshot_download

# Models to check (ordered smallest to largest)
DEFAULT_MODELS = [
    "mlx-community/Qwen3.5-4B-4bit",
    "mlx-community/Qwen3.5-9B-4bit",
    "mlx-community/Qwen3.5-27B-4bit",
    "mlx-community/Qwen3.5-35B-A3B-4bit",
]


def inspect_config(model_name: str) -> dict:
    """Download and inspect config.json for MTP-related fields."""
    print(f"\n{'='*60}")
    print(f"Inspecting: {model_name}")
    print(f"{'='*60}")

    try:
        config_path = hf_hub_download(model_name, "config.json")
    except Exception as e:
        print(f"  ERROR downloading config: {e}")
        return {}

    with open(config_path) as f:
        config = json.load(f)

    # Check both top-level and text_config
    text_config = config.get("text_config", config)

    mtp_fields = {}
    for key in [
        "num_nextn_predict_layers",
        "num_mtp_layers",
        "mtp_num_layers",
        "model_type",
        "num_hidden_layers",
        "hidden_size",
        "vocab_size",
        "intermediate_size",
        "num_attention_heads",
        "num_key_value_heads",
        "full_attention_interval",
        "tie_word_embeddings",
    ]:
        val = text_config.get(key, config.get(key, "NOT FOUND"))
        mtp_fields[key] = val

    print("\n  Config fields:")
    for k, v in mtp_fields.items():
        marker = " <-- MTP" if "nextn" in k or "mtp" in k else ""
        print(f"    {k}: {v}{marker}")

    return {"config": mtp_fields, "full_config": config, "text_config": text_config}


def inspect_weights(model_name: str) -> dict:
    """Download safetensors index and check for MTP weight keys."""
    import mlx.core as mx

    # Download the model files
    try:
        model_path = Path(
            snapshot_download(
                model_name,
                allow_patterns=["*.json", "model*.safetensors"],
            )
        )
    except Exception as e:
        print(f"  ERROR downloading model: {e}")
        return {}

    # Load all weight keys and find MTP ones
    import glob

    weight_files = sorted(glob.glob(str(model_path / "model*.safetensors")))
    print(f"\n  Weight files: {len(weight_files)}")

    all_keys = {}
    mtp_keys = {}

    for wf in weight_files:
        weights = mx.load(wf)
        fname = Path(wf).name
        for k, v in weights.items():
            all_keys[k] = {"shape": list(v.shape), "dtype": str(v.dtype), "file": fname}
            if "mtp" in k.lower() or "nextn" in k.lower():
                mtp_keys[k] = {
                    "shape": list(v.shape),
                    "dtype": str(v.dtype),
                    "file": fname,
                }

    print(f"  Total parameters: {len(all_keys)}")
    print(f"  MTP parameters: {len(mtp_keys)}")

    if mtp_keys:
        print("\n  MTP weight details:")
        # Group by prefix for readability
        prefixes = set()
        for k in sorted(mtp_keys.keys()):
            info = mtp_keys[k]
            print(f"    {k}")
            print(f"      shape={info['shape']}  dtype={info['dtype']}  file={info['file']}")
            # Extract prefix (e.g., "model.mtp_block.0")
            parts = k.split(".")
            for i, p in enumerate(parts):
                if "mtp" in p.lower():
                    prefixes.add(".".join(parts[: i + 2]))
                    break

        print(f"\n  MTP block prefixes: {sorted(prefixes)}")

        # Compute total MTP parameter count
        total_mtp_params = sum(
            1
            for shape in (info["shape"] for info in mtp_keys.values())
            for _ in range(max(1, len(shape)))
        )
        # Better: compute actual element count
        import functools
        import operator

        total_mtp_elements = sum(
            functools.reduce(operator.mul, info["shape"], 1)
            for info in mtp_keys.values()
        )
        total_all_elements = sum(
            functools.reduce(operator.mul, info["shape"], 1)
            for info in all_keys.values()
        )
        pct = 100.0 * total_mtp_elements / total_all_elements if total_all_elements else 0
        print(f"\n  MTP elements: {total_mtp_elements:,} ({pct:.2f}% of total {total_all_elements:,})")
    else:
        print("\n  ⚠ NO MTP WEIGHTS FOUND - this model lacks MTP parameters")
        print("  This is a BLOCKER for MTP speculative decoding.")
        print("  Options:")
        print("    1. Find a different quantization that preserves MTP weights")
        print("    2. Extract MTP weights from BF16 model and merge")
        print("    3. Quantize the model yourself with MTP weights preserved")

    return {"all_keys": all_keys, "mtp_keys": mtp_keys}


def main():
    if len(sys.argv) > 1:
        models = [sys.argv[1]]
    else:
        models = DEFAULT_MODELS

    results = {}
    for model_name in models:
        config_info = inspect_config(model_name)
        weight_info = inspect_weights(model_name)
        results[model_name] = {**config_info, **weight_info}

        has_mtp = bool(weight_info.get("mtp_keys"))
        has_config = config_info.get("config", {}).get(
            "num_nextn_predict_layers", "NOT FOUND"
        ) != "NOT FOUND"

        print(f"\n  Summary for {model_name}:")
        print(f"    MTP config field present: {has_config}")
        print(f"    MTP weights present: {has_mtp}")
        if has_mtp:
            print(f"    MTP weight count: {len(weight_info['mtp_keys'])}")
        print()

    # Final summary
    print("\n" + "=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)
    for model_name, info in results.items():
        has_mtp = bool(info.get("mtp_keys"))
        status = "✓ MTP WEIGHTS PRESENT" if has_mtp else "✗ NO MTP WEIGHTS"
        print(f"  {model_name}: {status}")


if __name__ == "__main__":
    main()
