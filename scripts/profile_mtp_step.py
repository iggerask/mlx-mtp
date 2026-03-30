#!/usr/bin/env python3
"""
Profile MTP decode step to find overhead.
"""

import time
import json
from pathlib import Path
import mlx.core as mx
from huggingface_hub import snapshot_download
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache
from vllm_mlx_mtp.hidden_capture import HiddenStateCapture
from vllm_mlx_mtp.mtp_decoder import MTPConfig, MTPDecoder
from vllm_mlx_mtp.mtp_head import build_mtp_head, load_mtp_weights_from_file
from vllm_mlx_mtp.optimizations import quantize_mtp_head
from mlx_fused_moe.patch_moe_full import patch_moe_full

MODEL_NAME = "mlx-community/Qwen3.5-35B-A3B-4bit"
BF16_SOURCE = "Qwen/Qwen3.5-35B-A3B"
MTP_WEIGHTS = Path("mtp_weights/Qwen_Qwen3.5-35B-A3B.safetensors")


def main():
    print("Loading model...")
    model, tokenizer = load(MODEL_NAME)
    patch_moe_full(model, verbose=False)

    print("Loading MTP head...")
    bf16_dir = Path(snapshot_download(BF16_SOURCE, allow_patterns=["config.json"]))
    cfg = json.loads((bf16_dir / "config.json").read_text())
    weights = load_mtp_weights_from_file(MTP_WEIGHTS)
    mtp_head = build_mtp_head(weights, cfg, norm_shift=True)
    quantize_mtp_head(mtp_head, bits=4, group_size=64)
    mx.eval(mtp_head.parameters())

    # Run generation with timing
    prompt = "Write a Python function that implements merge sort:"
    prompt_arr = mx.array(tokenizer.encode(prompt))

    mtp_cfg = MTPConfig(
        num_speculative_tokens=1,
        batch_verify=True,
        lazy_draft=True,
        zero_replay=True,
    )
    dec = MTPDecoder(model, mtp_head, mtp_cfg)
    cache = make_prompt_cache(model)

    # Warmup
    for tok in dec.generate(prompt_arr, cache, max_tokens=20, temperature=0.0):
        pass
    dec.cleanup()

    # Timed run
    cache = make_prompt_cache(model)
    dec2 = MTPDecoder(model, mtp_head, mtp_cfg)

    t0 = time.perf_counter()
    tokens = list(dec2.generate(prompt_arr, cache, max_tokens=200, temperature=0.0))
    t_total = time.perf_counter() - t0

    s = dec2.stats
    print(f"\nTokens: {len(tokens)}")
    print(f"Total time: {t_total*1000:.0f}ms")
    print(f"Tok/s: {len(tokens)/t_total:.0f}")
    print(f"Prefill: {s.prefill_time*1000:.0f}ms")
    print(f"Decode: {(t_total - s.prefill_time)*1000:.0f}ms")
    print(f"Steps: {s.total_steps}")
    print(f"Tok/step: {s.tokens_per_step:.2f}")
    print(f"Accept rate: {s.acceptance_rate:.0%}")
    print(f"ms/step: {(t_total - s.prefill_time)/s.total_steps*1000:.2f}ms")
    print(f"ms/token: {(t_total - s.prefill_time)/len(tokens)*1000:.2f}ms")

    # Theoretical best
    model_2tok_ms = 17.4  # measured
    theoretical_ms_per_step = model_2tok_ms  # verify cost
    theoretical_tok_per_step = s.tokens_per_step
    theoretical_tok_s = theoretical_tok_per_step / (theoretical_ms_per_step / 1000)
    actual_tok_s = len(tokens) / (t_total - s.prefill_time)
    overhead_pct = (1 - actual_tok_s / theoretical_tok_s) * 100

    print(f"\nTheoretical (model-only): {theoretical_tok_s:.0f} tok/s")
    print(f"Actual: {actual_tok_s:.0f} tok/s")
    print(f"MTP overhead: {overhead_pct:.0f}%")

    dec2.cleanup()


if __name__ == "__main__":
    main()
