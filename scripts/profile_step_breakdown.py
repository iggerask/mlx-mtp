#!/usr/bin/env python3
"""
Profile where time goes in each MTP decode step.
Instruments the actual hot path to find overhead.
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

    # First, measure raw 2-token forward (no MTP overhead)
    print("\n=== Raw model forward timing ===")
    prompt = "Write a Python function that implements merge sort:"
    prompt_arr = mx.array(tokenizer.encode(prompt))
    cache = make_prompt_cache(model)
    logits = model(prompt_arr[None], cache=cache)
    mx.eval(logits, *[c.state for c in cache if hasattr(c, "state")])
    tok = mx.argmax(logits[:, -1, :], axis=-1)
    mx.eval(tok)

    # Time raw 1-token forward
    input_1 = mx.array([[tok.item()]])
    times_1tok = []
    for _ in range(5):  # warmup
        mx.eval(model(input_1, cache=cache))
    for _ in range(30):
        t0 = time.perf_counter()
        mx.eval(model(input_1, cache=cache))
        times_1tok.append((time.perf_counter() - t0) * 1000)
    print(f"  1-token forward: {min(times_1tok):.2f}ms (min of 30)")

    # Time raw 2-token forward
    input_2 = mx.array([[tok.item(), tok.item()]])
    cache2 = make_prompt_cache(model)
    logits = model(prompt_arr[None], cache=cache2)
    mx.eval(logits, *[c.state for c in cache2 if hasattr(c, "state")])
    times_2tok = []
    for _ in range(5):
        mx.eval(model(input_2, cache=cache2))
    for _ in range(30):
        t0 = time.perf_counter()
        mx.eval(model(input_2, cache=cache2))
        times_2tok.append((time.perf_counter() - t0) * 1000)
    print(f"  2-token forward: {min(times_2tok):.2f}ms (min of 30)")

    # Time MTP head forward
    capture = HiddenStateCapture(model)
    cache3 = make_prompt_cache(model)
    logits = model(prompt_arr[None], cache=cache3)
    hidden = capture.get_hidden_state()
    mx.eval(logits, hidden, *[c.state for c in cache3 if hasattr(c, "state")])
    tok_arr = mx.argmax(logits[:, -1, :], axis=-1)
    mx.eval(tok_arr)
    last_h = hidden[:, -1:, :]

    embed_fn = model.language_model.model.embed_tokens
    lm_head = model.language_model.lm_head

    times_mtp = []
    for _ in range(5):
        tok_embed = embed_fn(tok_arr[None])
        mtp_h = mtp_head(last_h, tok_embed)
        draft_logits = lm_head(mtp_h)
        draft_tok = mx.argmax(draft_logits[:, -1, :], axis=-1)
        mx.eval(draft_tok)
    for _ in range(30):
        t0 = time.perf_counter()
        tok_embed = embed_fn(tok_arr[None])
        mtp_h = mtp_head(last_h, tok_embed)
        draft_logits = lm_head(mtp_h)
        draft_tok = mx.argmax(draft_logits[:, -1, :], axis=-1)
        mx.eval(draft_tok)
        times_mtp.append((time.perf_counter() - t0) * 1000)
    print(f"  MTP head draft: {min(times_mtp):.2f}ms (min of 30)")

    # Time the full lazy graph: MTP draft + 2-token verify (no eval boundary)
    times_lazy = []
    for _ in range(5):
        tok_embed = embed_fn(tok_arr[None])
        mtp_h = mtp_head(last_h, tok_embed)
        draft_logits = lm_head(mtp_h)
        draft = mx.argmax(draft_logits[:, -1, :], axis=-1)
        verify_input = mx.concatenate([tok_arr.reshape(1, 1), draft.reshape(1, 1)], axis=1)
        vlogits = model(verify_input, cache=cache3)
        vhidden = capture.get_hidden_state()
        verified = mx.argmax(vlogits[:, 0, :], axis=-1)
        bonus = mx.argmax(vlogits[:, 1, :], axis=-1)
        mx.eval(draft, verified, bonus, vlogits, vhidden)
    for _ in range(30):
        t0 = time.perf_counter()
        tok_embed = embed_fn(tok_arr[None])
        mtp_h = mtp_head(last_h, tok_embed)
        draft_logits = lm_head(mtp_h)
        draft = mx.argmax(draft_logits[:, -1, :], axis=-1)
        verify_input = mx.concatenate([tok_arr.reshape(1, 1), draft.reshape(1, 1)], axis=1)
        vlogits = model(verify_input, cache=cache3)
        vhidden = capture.get_hidden_state()
        verified = mx.argmax(vlogits[:, 0, :], axis=-1)
        bonus = mx.argmax(vlogits[:, 1, :], axis=-1)
        mx.eval(draft, verified, bonus, vlogits, vhidden)
        times_lazy.append((time.perf_counter() - t0) * 1000)
    print(f"  Full lazy (MTP+verify): {min(times_lazy):.2f}ms (min of 30)")

    # Time Python overhead: .item() calls + branching
    times_item = []
    dummy = mx.array([42])
    mx.eval(dummy)
    for _ in range(1000):
        t0 = time.perf_counter()
        _ = dummy.item()
        times_item.append((time.perf_counter() - t0) * 1e6)
    print(f"\n  .item() call: {min(times_item):.1f}μs (min of 1000)")

    # Now time the actual MTP generate loop
    print("\n=== Full MTP generate (K=1 ZR) ===")
    mtp_cfg = MTPConfig(
        num_speculative_tokens=1,
        batch_verify=True,
        lazy_draft=True,
        zero_replay=True,
    )

    # Warmup
    cache_w = make_prompt_cache(model)
    dec_w = MTPDecoder(model, mtp_head, mtp_cfg)
    for tok in dec_w.generate(prompt_arr, cache_w, max_tokens=20, temperature=0.0):
        pass
    dec_w.cleanup()

    # Timed run
    cache4 = make_prompt_cache(model)
    dec = MTPDecoder(model, mtp_head, mtp_cfg)
    t0 = time.perf_counter()
    tokens = list(dec.generate(prompt_arr, cache4, max_tokens=200, temperature=0.0))
    t_total = time.perf_counter() - t0

    s = dec.stats
    decode_time = t_total - s.prefill_time
    ms_per_step = decode_time / s.total_steps * 1000
    ms_per_tok = decode_time / len(tokens) * 1000

    print(f"  Tokens: {len(tokens)}")
    print(f"  Total: {t_total*1000:.0f}ms (prefill {s.prefill_time*1000:.0f}ms)")
    print(f"  Decode: {decode_time*1000:.0f}ms")
    print(f"  Steps: {s.total_steps}")
    print(f"  Tok/step: {s.tokens_per_step:.2f}")
    print(f"  Accept: {s.acceptance_rate:.0%}")
    print(f"  ms/step: {ms_per_step:.2f}ms")
    print(f"  ms/tok: {ms_per_tok:.2f}ms")
    print(f"  tok/s: {len(tokens)/decode_time:.0f}")

    # Compare theoretical
    raw_2tok = min(times_2tok)
    theoretical_ms_per_step = raw_2tok  # 2-token forward is the verify cost
    theoretical_tok_per_s = s.tokens_per_step / (theoretical_ms_per_step / 1000)
    actual_tok_per_s = len(tokens) / decode_time
    overhead_ms = ms_per_step - raw_2tok
    overhead_pct = overhead_ms / ms_per_step * 100

    print(f"\n=== Overhead Analysis ===")
    print(f"  Raw 2-token forward: {raw_2tok:.2f}ms")
    print(f"  Actual ms/step: {ms_per_step:.2f}ms")
    print(f"  Overhead per step: {overhead_ms:.2f}ms ({overhead_pct:.0f}%)")
    print(f"  Theoretical tok/s: {theoretical_tok_per_s:.0f}")
    print(f"  Actual tok/s: {actual_tok_per_s:.0f}")
    print(f"  Gap: {(1 - actual_tok_per_s/theoretical_tok_per_s)*100:.0f}%")

    # Breakdown estimate
    print(f"\n  Estimated breakdown:")
    print(f"    2-tok forward (verify): {raw_2tok:.2f}ms")
    print(f"    MTP head draft (graph): ~{min(times_mtp):.2f}ms (overlapped in lazy graph)")
    print(f"    Lazy graph overhead: {min(times_lazy) - raw_2tok:.2f}ms")
    print(f"    ZR capture (GDN split): included in verify")
    n_items = 3  # draft.item(), verified.item(), bonus.item() or similar
    item_overhead = min(times_item) * n_items / 1000
    print(f"    .item() calls ({n_items}×): ~{item_overhead:.2f}ms")
    print(f"    Python/yield/loop: ~{overhead_ms - (min(times_lazy) - raw_2tok) - item_overhead:.2f}ms")

    dec.cleanup()


if __name__ == "__main__":
    main()
