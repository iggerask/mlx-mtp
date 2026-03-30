# MTP Optimization Benchmark Report — Qwen3.5-35B-A3B-4bit

**Date:** 2026-03-28
**Hardware:** Apple Silicon (Mac)
**Model:** `mlx-community/Qwen3.5-35B-A3B-4bit` (quantized backbone)
**MTP Weights:** BF16, extracted from `Qwen/Qwen3.5-35B-A3B`
**Max Tokens:** 128 per generation
**Runs per prompt:** 2 (best-of-N)

---

## Executive Summary

| Method | Avg tok/s | Speedup | MTP Head Size | Acceptance Rate |
|--------|-----------|---------|---------------|-----------------|
| Baseline (autoregressive) | 64.6 | 1.00x | — | — |
| **MTP BF16** | 69.6 | **1.08x** | 1689 MB | 80% |
| **MTP Q4 (4-bit head)** | 70.4 | **1.09x** | 475 MB (28%) | 76% |
| Prompt Lookup | 66.1 | 1.02x | — | 46% |
| Shared Expert d=1 | 56.8 | 0.88x | — | ~100% |
| Shared Expert d=3 | 51.9 | 0.80x | — | ~100% |

**Winner: MTP with 4-bit quantized head** — matches BF16 throughput while using 72% less memory for the MTP head.

---

## Methods Tested

### 1. MTP Speculative Decoding (BF16 & Q4)

Uses the model's Multi-Token Prediction head to draft 1 token, then batch-verifies with the main model. Two variants:

- **BF16:** Full-precision MTP head (1689 MB)
- **Q4:** 4-bit quantized MTP head via `nn.quantize()` (475 MB, 28% of BF16)

### 2. Prompt Lookup Decoding

Builds an n-gram index over the input prompt tokens. When recent output matches an n-gram in the prompt, predicts the continuation for free (no model call). Most effective when the model is paraphrasing or summarizing its own input.

### 3. Shared-Expert-Only Self-Speculation

For the MoE architecture (256 experts, top-8 routing), patches all MoE layers to only use the shared expert during drafting. This skips the expensive expert routing. All 40 transformer layers still run. Two draft depths tested: d=1 and d=3.

---

## Detailed Results

### Per-Category Speedup vs Baseline

| Method | Code | Prose | Short | Summarization |
|--------|------|-------|-------|---------------|
| MTP BF16 | 1.11x | 0.92x | **1.18x** | 1.09x |
| MTP Q4 | 1.06x | 1.02x | **1.20x** | 1.08x |
| Prompt Lookup | 1.00x | 0.96x | **1.19x** | 0.93x |
| Shared Expert d=1 | 0.86x | 0.86x | 0.90x | 0.90x |
| Shared Expert d=3 | 0.79x | 0.78x | 0.82x | 0.83x |

### Per-Category Acceptance Rates

| Method | Code | Prose | Short | Summarization |
|--------|------|-------|-------|---------------|
| MTP BF16 | 81% | 78% | 87% | 74% |
| MTP Q4 | 76% | 72% | 85% | 71% |
| Prompt Lookup | 39% | 20% | 83% | 40% |

### Peak Results

| Metric | Best Method | Value | Prompt |
|--------|-------------|-------|--------|
| Highest tok/s | Prompt Lookup | **91.6 t/s** (1.43x) | "The capital of France is" |
| Best MTP | MTP Q4 | **81.7 t/s** (1.28x) | "The capital of France is" |
| Best code | MTP BF16 | **76.7 t/s** (1.20x) | deepClone JavaScript |
| Best summarization | MTP BF16 | **62.5 t/s** (1.15x) | Text summarization |

---

## Analysis

### MTP Quantization: BF16 vs Q4

The 4-bit quantized MTP head delivers **nearly identical throughput** to the BF16 head while consuming **72% less memory**. The acceptance rate drops slightly (76% vs 80%) due to quantization noise, but the faster head evaluation compensates.

| | BF16 Head | Q4 Head |
|---|-----------|---------|
| Head memory | 1689 MB | 475 MB |
| Avg tok/s | 69.6 | 70.4 |
| Avg acceptance | 80% | 76% |

**Recommendation:** Always use the Q4 head — it's strictly better on memory with no throughput penalty.

### Prompt Lookup Decoding

Prompt lookup is **highly situational**. It excels when output tokens overlap with the input prompt (1.43x on "The capital of France is" where the model echoes content). On most prompts it provides minimal benefit (0.89x-1.03x) because the model generates novel content that doesn't match prompt n-grams.

Best use cases:
- Summarization of short, repetitive input
- Fill-in-the-blank / completion tasks
- Code rewriting where output largely mirrors input

### Shared-Expert-Only Self-Speculation: Why It Fails

Despite near-perfect draft accuracy (~100% acceptance), shared-expert-only drafting is **slower than baseline** (0.80-0.88x). The reason:

1. **The model is memory-bandwidth bound**, not compute bound. The 35B-A3B backbone reads ~19 GB per token at ~1.3 TB/s bandwidth.
2. Shared-expert-only mode still runs all 40 transformer layers — it only skips the routing computation within MoE blocks.
3. The expert routing in MoE is already efficient (top-8 of 256 uses ~3% of expert parameters), so skipping it saves minimal bandwidth.
4. The overhead of running the model twice (draft + verify) dwarfs the per-token savings.

**Conclusion:** Self-speculative decoding via shared experts is not viable for memory-bandwidth-bound MoE inference on Apple Silicon. The draft model needs to be **much cheaper** (like the MTP head, which is ~1.7 GB vs ~19 GB backbone) to amortize the verification cost.

---

## Recommendations for Qwen3.5-35B-A3B on Apple Silicon

1. **Use MTP with Q4 head** (1.09x average, up to 1.28x) — best general-purpose acceleration
2. **Combine with prompt lookup** for summarization/rewrite tasks — the two are complementary (MTP handles novel content, prompt lookup handles echoed content)
3. **Stick with K=1 draft depth** — multi-token MTP (K>1) degrades accuracy too fast to be worth it
4. **Don't bother with self-speculative decoding** — the model is bandwidth-bound, so running it twice is always slower

### Tokens/s Across All Model Sizes (K=1 MTP, batch verify)

| Model | Baseline t/s | MTP t/s | Speedup | Accept Rate |
|-------|-------------|---------|---------|-------------|
| Qwen3.5-4B | 69.9 | 69.2 | 0.99x | 68% |
| Qwen3.5-9B | 44.4 | 44.8 | 1.01x | 71% |
| Qwen3.5-27B | 14.3 | 16.0 | **1.12x** | 74% |
| Qwen3.5-35B-A3B | 64.6 | 70.4 | **1.09x** | 76% |

MTP speculative decoding provides the largest gains on **larger, more bandwidth-bound models** (27B and 35B-A3B), while smaller models that are already fast see less benefit.
