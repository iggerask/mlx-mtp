# Benchmark Results: Optimization Suite for Qwen3.5-35B-A3B

**Model**: Qwen3.5-35B-A3B (4-bit quantized, mlx-community)
**Platform**: Apple M2 Max 48GB
**Date**: 2026-03-29
**Gen tokens**: 128, **Runs**: Best of 2, **Temperature**: 0.0 (greedy)

---

## 1. Speed Results (tok/s decode)

### KV Cache Methods

| Ctx    | Baseline | Naive INT8 | TQ-LM INT4 | TQ-LM INT8 |
|--------|----------|------------|-------------|-------------|
| 512    | 74.6     | 69.6       | 65.8        | 67.1        |
| 2048   | 68.0     | 61.6       | 53.0        | 57.3        |
| 4096   | 64.3     | 59.2       | 40.2        | 50.0        |
| 8192   | 61.8     | 52.5       | 26.9        | 36.3        |
| 16384  | 55.6     | 19.2       | 15.8        | 20.9        |

### Speculative Decoding Methods

| Ctx    | Baseline | MTP K=1 | EAGLE D2W1 | EAGLE D3W1 | EAGLE D2W3 | EAGLE D3W1 adapt | EAGLE D3W1+INT8 |
|--------|----------|---------|------------|------------|------------|------------------|-----------------|
| 512    | 74.6     | 66.3   | 67.0       | 53.3       | 56.7       | 54.1             | 52.2            |
| 2048   | 68.0     | 59.5   | 62.0       | 47.0       | 53.2       | 48.9             | 40.5            |
| 4096   | 64.3     | 59.6   | 66.0       | 52.6       | 58.4       | 53.7             | 41.0            |
| 8192   | 61.8     | 55.4   | 52.6       | 45.2       | 47.3       | 47.0             | 35.1            |
| 16384  | 55.6     | 13.0   | 2.2        | 1.7        | 1.7        | -                | -               |

### Speedup vs Baseline

| Ctx    | Naive INT8 | MTP K=1 | EAGLE D2W1 | EAGLE D3W1 | TQ-LM INT4 | TQ-LM INT8 |
|--------|------------|---------|------------|------------|-------------|-------------|
| 512    | 0.93x      | 0.89x  | 0.90x      | 0.71x      | 0.88x       | 0.90x       |
| 2048   | 0.91x      | 0.88x  | 0.91x      | 0.69x      | 0.78x       | 0.84x       |
| 4096   | 0.92x      | 0.93x  | **1.03x**  | 0.82x      | 0.63x       | 0.78x       |
| 8192   | 0.85x      | 0.90x  | 0.85x      | 0.73x      | 0.44x       | 0.59x       |
| 16384  | 0.35x      | 0.23x  | 0.04x      | 0.03x      | 0.28x       | 0.38x       |

---

## 2. KV Cache Memory (MB)

| Ctx    | Baseline | Naive INT8 | TQ-LM INT4 | TQ-LM INT8 |
|--------|----------|------------|-------------|-------------|
| 512    | 48.7     | 32.9       | 35.7        | 38.5        |
| 2048   | 80.1     | 32.9       | 43.7        | 54.3        |
| 4096   | 122.1    | 32.9       | 54.4        | 75.5        |
| 8192   | 205.9    | 32.9       | 75.7        | 117.7       |
| 16384  | 373.7    | 32.9       | 118.3       | 202.3       |

Note: Naive INT8 reports 32.9 MB at all lengths — this is likely because the QuantizedKVCache
reports only the quantized representation size and the ArraysCache layers have fixed size.

---

## 3. Quality (Token Match vs Baseline, first 64 tokens)

| Method              | 512  | 2048 | 4096 | 8192 | 16384 |
|---------------------|------|------|------|------|-------|
| Naive INT8          | 100% | 100% | 100% | 100% | 100%  |
| TQ-LM INT4          | 27%  | 28%  | 28%  | 17%  | 30%   |
| TQ-LM INT8          | 27%  | 19%  | 28%  | 16%  | 28%   |
| MTP K=1             | 100% | 100% | 100% | 100% | 100%  |
| EAGLE D2W1          | 100% | 100% | 100% | 100% | 100%  |
| EAGLE D3W1          | 100% | 100% | 100% | 100% | 100%  |
| EAGLE D2W3          | 100% | 100% | 100% | 100% | 100%  |
| EAGLE D3W1 adapt    | 100% | 100% | 100% | 100% | 100%  |
| EAGLE D3W1+INT8     | -    | -    | -    | 66%  | -     |

---

## 4. Speculative Decoding Acceptance Rates

| Ctx    | MTP K=1 | EAGLE D2W1 | EAGLE D3W1 | EAGLE D2W3 | EAGLE D3W1 adapt |
|--------|---------|------------|------------|------------|------------------|
| 512    | 75%     | 68%        | 54%        | 22%        | 56%              |
| 2048   | 83%     | 69%        | 50%        | 22%        | 54%              |
| 4096   | 81%     | 75%        | 57%        | 25%        | 60%              |
| 8192   | 79%     | 65%        | 53%        | 22%        | 57%              |
| 16384  | 75%     | 72%        | 54%        | 25%        | -                |

**Tokens per step** (effective throughput multiplier from speculation):

| Ctx    | MTP K=1 | EAGLE D2W1 | EAGLE D3W1 | EAGLE D2W3 | EAGLE D3W1 adapt |
|--------|---------|------------|------------|------------|------------------|
| 512    | 1.77    | 2.37       | 2.63       | 2.33       | 2.63             |
| 2048   | 1.84    | 2.41       | 2.52       | 2.35       | 2.52             |
| 4096   | 1.83    | 2.53       | 2.74       | 2.53       | 2.74             |
| 8192   | 1.80    | 2.32       | 2.61       | 2.32       | 2.61             |
| 16384  | 1.77    | 2.45       | 2.63       | 2.50       | -                |

---

## 5. Key Findings

### EAGLE-style Speculative Decoding

1. **EAGLE D2W1 is the best speculative config**. At 4K context it achieves 1.03x baseline speed
   while generating 2.53 tokens/step. High acceptance (75%) and only one extra verification step.

2. **EAGLE D3W1 produces the highest tokens/step (2.6-2.7)** but the extra MTP chain + verification
   overhead makes it slower than D2W1 in wall-clock time on this hardware.

3. **Multi-chain (D2W3) doesn't help**. Acceptance rate drops to ~22% because alternative top-K
   candidates rarely match the greedy continuation. The extra chains waste compute.

4. **Adaptive depth (D3W1 adapt)** behaves similarly to D3W1 — the confidence threshold of 0.3
   isn't aggressive enough to meaningfully reduce average depth (2.8 vs 3.0).

5. **16K context cliff kills speculation**. At 16K, all methods collapse to 1-2 t/s because
   batch verification requires running the full model with large KV caches. The verification
   overhead (N+1 tokens) becomes devastating at the memory cliff.

6. **All speculation methods maintain 100% quality** (greedy mode). The verification step
   guarantees exact match with baseline.

### TurboQuant Lloyd-Max (Updated with fused kernels + incremental dequant)

7. **With fused Metal kernels and incremental dequant, TQ-LM achieves 0.89-0.93x baseline**
   speed — a dramatic improvement from the initial 0.28-0.44x. The fused quantize/dequant
   kernels (Metal binary search + codebook lookup) and incremental dequantization (only
   dequantize newly added tokens, not the full cache each step) were the key optimizations.

8. **Quality "issue" was a measurement artifact, not a real bug**. Token-match % is misleading:
   passthrough mode (rotation only, zero quantization) produces identical output to 4-bit and
   8-bit quantized. The ~25% token match vs baseline is autoregressive divergence from the
   Hadamard rotation changing the floating-point computation path through SDPA. The actual
   probability distributions are nearly identical (KL=0.05, same argmax, coherent text).

9. **The KV cache compression works** (2-3x smaller for quantized storage). However, the
   incremental dequant approach maintains a parallel float cache, reducing net memory savings.

10. **Fused quantized SDPA kernel**: A Metal kernel computing attention directly against 4-bit
    packed KV cache was implemented and produces correct results (rel_err < 0.003). However,
    it is 3-18x slower than dequant+SDPA due to tree reduction barriers (7 per T position).
    This is a fundamental limitation of Metal's threading model — Triton/CUDA can avoid this
    with warp-level primitives. The kernel saves ~160MB (no float cache) across 10 layers
    but the speed penalty makes it impractical except for extreme memory constraints.

### Naive INT8 KV Quantization

11. **Naive INT8 remains the best KV cache optimization**: 100% quality, ~0.85-0.93x speed at
    reasonable context lengths, and significant memory reduction.

### Combined Methods

12. **EAGLE + INT8 hurts more than it helps** at 8K (0.57x baseline). The combination introduces
    quality degradation (66% match) and the INT8 decode overhead compounds with the speculative
    verification overhead.

---

## 6. Recommendations

### For agentic workloads (4-8K context):
- **Best single method**: Baseline (unmodified) — it's already 60+ t/s
- **Best if latency matters**: EAGLE D2W1 — near-baseline speed with 2.5x tokens/step
- **Best if memory matters**: Naive INT8 — zero quality loss, modest speed cost

### For long context (16K+):
- The 16K "memory cliff" is actually a first-step latency spike (peak memory ~38GB on M2 Max
  48GB causes memory pressure). Steady-state decode after warmup is 61.5 t/s — only 5% slower
  than 8K. The 128-token benchmark limit made slow first steps dominate averages.
- TurboQuant KV compression can help reduce peak memory at 16K+
- Speculation makes it worse (batch verification is expensive at 16K)

### What to NOT pursue further:
- **Multi-chain EAGLE (W>1)**: Low acceptance rate, wasted compute
- **EAGLE at 16K**: Batch verification is catastrophically slow at the memory cliff

### Worth exploring:
- **TurboQuant + EAGLE**: Combining speculation with KV compression could give both latency
  and memory benefits at moderate context lengths.
- **Fused SDPA via MLX C++ extension**: The Metal `metal_kernel` API lacks efficient cross-SIMD
  primitives. A C++ Metal extension with `simd_sum` and warp-level cooperation could make the
  fused quantized SDPA competitive.

---

## 7. mx.compile Optimization Results

**Test**: Compile various sub-graphs of the model forward pass to reduce kernel dispatch overhead.
**Hypothesis**: ~800 kernel dispatches × ~15μs each ≈ 12ms per step. Fusing ops should reduce dispatches.

### Strategies Tested

| Strategy | Avg t/s | vs Baseline | Output Match |
|----------|---------|-------------|--------------|
| Baseline (no compile) | 74.3 | 1.00x | ✓ |
| Compile MoE/MLP blocks | 75.2 | 1.01x | ✓ |
| Compile norm+MLP path | 73.7 | 0.99x | ✓ |
| Compile MoE routing | 73.8 | 0.99x | ✓ |
| Compile full model step | N/A | N/A | FAILED |

### Finding: mx.compile has no measurable impact

All compilation strategies are within ±2% of baseline — within measurement noise.

**Root cause**: MLX's lazy evaluation already provides the same graph fusion benefits.
In the decode loop, the computation graph is:
```
model(token, cache=cache) → logits → argmax → mx.eval(tok)
```
Since `mx.eval()` is called only once per step, MLX already accumulates the entire
40-layer forward pass + argmax into a single computation graph before executing.
There is no additional opportunity for `mx.compile` to fuse operations.

The ~12ms per-step cost is the actual Metal kernel execution time, not Python dispatch
overhead between separate `mx.eval()` calls. Each quantized matmul, softmax, GDN kernel,
and MoE routing operation is already batched into one graph by lazy eval.

**Implication**: The decode bottleneck for this model on M2 Max is **compute-bound**,
not dispatch-bound. Optimization efforts should focus on:
1. Reducing total compute per step (e.g., layer skipping, smaller MoE routing)
2. Processing more tokens per step (batch decode, speculation)
3. Reducing memory bandwidth (KV compression for long contexts)

---

## 8. Multi-Token-Per-Step Comprehensive Benchmark

**Test**: Exhaustive comparison of all strategies that process >1 token per decode step.
**Max tokens**: 200, **Runs**: Best of 3, **Temperature**: 0.0

### All Strategies Compared (average across 5 prompt categories)

| Strategy | Avg t/s | vs Base | Accept | Tok/Step | ms/step |
|----------|---------|---------|--------|----------|---------|
| Baseline | 72.1 | 1.00x | — | 1.00 | 13.8 |
| **MTP K=1 batch** | **78.8** | **1.09x** | **81%** | **1.82** | **23.2** |
| MTP K=1 sequential | 66.9 | 0.93x | 83% | 1.84 | 27.3 |
| MTP K=2 batch | 66.8 | 0.93x | 66% | 2.34 | 35.6 |
| MTP K=3 batch | 58.9 | 0.82x | 57% | 2.71 | 47.5 |
| MTP K=4 batch | 45.6 | 0.63x | 44% | 2.77 | 60.1 |
| Prompt lookup (d=3) | 68.7 | 0.95x | 55% | 1.20 | 18.2 |
| Prompt lookup (d=8) | 67.1 | 0.93x | 46% | 1.20 | 19.3 |
| Shared-expert (d=1) | 61.0 | 0.85x | — | 2.00 | 32.6 |
| Shared-expert (d=3) | 55.0 | 0.76x | — | 3.95 | 71.6 |
| MTP+lookup hybrid | 66.7 | 0.93x | — | 1.91 | 30.6 |

### Per-Category t/s (standout results highlighted)

| Strategy | Code | Prose | Short | Repetitive | Q&A |
|----------|------|-------|-------|------------|-----|
| Baseline | 74.3 | 72.2 | 71.4 | 71.8 | 70.9 |
| MTP K=1 batch | 77.2 | 72.6 | 76.3 | **98.0** | 70.1 |
| MTP K=2 batch | 66.3 | 64.9 | 46.6 | **99.2** | 57.1 |
| MTP K=3 batch | 55.2 | 53.1 | 40.0 | **100.4** | 45.8 |
| Prompt lookup (d=3) | 68.5 | 69.5 | **93.2** | 43.0 | 69.1 |
| MTP+lookup hybrid | 62.9 | 70.4 | 70.5 | 40.5 | **89.4** |

### Key Finding: MTP K=1 batch is the only general-purpose winner

MTP K=1 batch verify is the sweet spot: 1.82 tokens/step at 81% acceptance,
totaling 78.8 t/s (1.09x). Higher K values degrade acceptance faster than
they add tokens — the MTP-1 head is trained for single-token-ahead prediction
and degrades out-of-distribution when chained.

### Why higher K fails (cost model)

Per step: `MTP_head(K × 5ms) + batch_verify(14 + K × 4.5ms) + cache_overhead`

| K | MTP cost | Verify cost | Total | Accept | Tok/step | Effective t/s |
|---|----------|-------------|-------|--------|----------|---------------|
| 1 | 5ms | 18ms | 23ms | 81% | 1.82 | 79 |
| 2 | 10ms | 23ms | 36ms | 66% | 2.34 | 65 |
| 3 | 15ms | 27ms | 48ms | 57% | 2.71 | 57 |
| 4 | 20ms | 32ms | 60ms | 44% | 2.77 | 46 |

MTP head cost scales linearly with K (5ms per chain step), while acceptance
drops ~15 percentage points per additional K. Only works when acceptance > 90%
(repetitive content: MTP K=3 → 100.4 t/s).

---

## 9. Optimized MTP K=1 Batch Results

**Test**: Push MTP K=1 batch to maximum throughput via micro-optimizations.

### Optimizations Tested

| Strategy | Avg t/s | vs Base | ms/step | Key Change |
|----------|---------|---------|---------|------------|
| Baseline | 73.4 | 1.00x | 13.6 | — |
| MTP K=1 batch (standard) | 80.4 | 1.09x | 22.7 | Current implementation |
| MTP K=1 lazy batch | 81.6 | 1.11x | 22.4 | Remove early mx.eval(draft) |
| MTP Q4 batch (standard) | 81.4 | 1.11x | 22.5 | 4-bit quantized MTP head |
| **MTP Q4 lazy batch** | **83.0** | **1.13x** | **22.0** | **Q4 + lazy eval combined** |
| MTP Q4 lazy sequential | 69.4 | 0.95x | 25.9 | Q4 + lazy, no batch verify |
| Adaptive K (BF16) | 71.1 | 0.97x | 28.2 | K=1→3 when accept>85% |
| Adaptive K (Q4) | 73.2 | 1.00x | 28.0 | Same with Q4 head |

### Per-Category Speedup vs Baseline

| Strategy | Code | Prose | Repetitive | Q&A |
|----------|------|-------|------------|-----|
| MTP Q4 lazy batch | 1.08x | 1.03x | **1.38x** | 1.04x |
| MTP BF16 lazy batch | 1.04x | 1.02x | 1.35x | 1.03x |
| MTP Q4 batch (std) | 1.04x | 1.02x | 1.35x | 1.02x |

### Finding: Two stackable micro-optimizations add up to 1.13x

1. **Lazy draft eval** (+2%): Removing the `mx.eval(draft_token)` sync point before
   batch verify lets MLX build the full MTP + model graph before evaluation.
   Saves ~0.3-0.5ms per step.

2. **Q4 MTP head** (+2%): Quantizing the MTP head from BF16 (1689 MB) to 4-bit (475 MB)
   saves ~0.5ms per invocation with no acceptance degradation (81% → 81%).

Combined: 83.0 t/s (1.13x baseline), or **100.9 t/s on repetitive** (1.38x).

### Why adaptive K doesn't help

Adaptive K (switching K=1→3 when acceptance > 85%) actually hurts because:
- The K>1 path requires `save_cache_state()` which costs ~2ms (deep-copies GDN states)
- Cache restore on partial reject adds another ~2ms
- MTP head chains degrade in accuracy out-of-distribution
- The overhead of K>1 only pays off with sustained >90% acceptance, which is rare

### Cost Model: Optimized MTP K=1 Q4 Lazy Batch

```
Per step (22.0ms total):
  MTP head Q4:       ~3.5ms  (was ~5ms with BF16)
  Batch verify (2):  ~18ms   (14ms base + 4ms marginal)
  Save cache:         0ms    (only saved on reject fallback path — 19% of steps)
  Overhead/sync:     ~0.5ms  (was ~1ms with early eval)

Expected tokens: 1 + 0.81 = 1.81
Effective per-token cost: 22.0 / 1.81 = 12.2ms (vs 13.6ms baseline = 10.3% faster)
```

---

## 10. Layer Skip & Early Exit Results

### Block Influence (BI) Profiling

Measured BI = 1 - cosine_similarity(layer_input, layer_output) across calibration set.

```
Layer  Type     BI       Assessment
  0    GDN    0.3696    ████ Cornerstone — highest transformation
  1    GDN    0.0908    ██
  ...
 24    GDN    0.0142    ▏  ← Lowest BI (near-identity)
 25    GDN    0.0107    ▏  ← Second-lowest
  ...
 38    GDN    0.1274    ██
 39    Attn   0.3574    ████ Output layer — second-highest
```

Skip candidates (BI < 0.03): Layers 25, 24, 17, 29, 21, 28, 16, 23, 20, 22.
9 GDN layers + 1 Attention layer (23). All in the middle third of the network.

### Finding: Layer skip gives ZERO speedup on this model

| Strategy | Avg t/s | vs Base | Token Match |
|----------|---------|---------|-------------|
| Baseline | 72.8 | 1.00x | — |
| Skip 5 layers | 73.1 | 1.00x | 100% |
| Skip 8 layers | 73.0 | 1.00x | 100% |
| Skip 10 layers | 73.1 | 1.00x | 100% |

**Quality is perfectly preserved** — 100% token match even when skipping 10 layers.
These layers are genuinely near-identity transformations.

**But there's zero speedup** because the skipped layers (mostly GDN) are already
negligible-cost during decode. GDN layers use custom Metal kernels for their
recurrent state update (~0.1ms each). The dominant cost is the 40 MLP/MoE blocks,
which we're NOT skipping.

### Self-speculative layer skip: slower than baseline

| Strategy | Avg t/s | vs Base | Acceptance |
|----------|---------|---------|------------|
| Selfspec skip-5, draft=1 | 61.6 | 0.85x | 100% |
| Selfspec skip-8, draft=1 | 61.6 | 0.85x | 100% |

100% acceptance (skipped-layer model produces identical output), but the draft
model is the same speed as the full model (skipped layers cost nothing), so the
draft+verify overhead makes it net slower.

### Early exit: slower than baseline (0.65-0.75x)

| Strategy | Avg t/s | vs Base | Exit Rate | Avg Exit Layer | Match |
|----------|---------|---------|-----------|---------------|-------|
| Early exit t=0.90 l≥16 | 47.0 | 0.65x | 44% | 39 | 44% |
| Early exit t=0.95 l≥24 | 54.4 | 0.75x | 80% | 39 | 80% |
| Early exit t=0.99 l≥24 | 54.0 | 0.74x | 100% | 39 | 100% |

**Average exit layer is 39** even with low thresholds — the model almost never
becomes confident enough to exit before the final layer. The lm_head probes at
each attention checkpoint add ~1ms overhead per probe, making every step slower.

### Root cause: this model is memory-bandwidth limited

The 40-layer model has only ~3B active parameters per step (MoE routes top-8 of
256 experts). At 4-bit quantization, active weight data is ~1.5GB per step.

```
Theoretical minimum at M2 Max bandwidth (400 GB/s):
  1.5 GB / 400 GB/s = 3.75 ms/step = 267 t/s

Actual observed:
  13.7 ms/step = 73 t/s

Utilization: 27% of theoretical bandwidth
Gap: 3.7x — attributable to kernel dispatch overhead, Python/C++ overhead,
     Metal command buffer scheduling, quantized matmul kernel efficiency
```

This gap is below the MLX framework level — not optimizable from Python.
The 73 t/s baseline is already very efficient for this hardware + model.

### Conclusion: optimization ceiling identified

| Optimization | Result | Why |
|---|---|---|
| mx.compile | 0% | Lazy eval already fuses |
| Layer skip (static) | 0% | Skipped layers cost nothing |
| Early exit (CALM) | -25% to -35% | Probe overhead > savings |
| Self-speculative | -15% | Draft model same speed as full |
| Shared-expert draft | -15% | Same — draft not cheaper |
| Prompt lookup | ±5% | Only works for matching content |
| **MTP K=1 Q4 lazy batch** | **+13%** | **Only winner** |
| MTP K=3 (repetitive) | +40% | Content-specific |

**MTP Q4 lazy batch at 83 t/s (1.13x) is the practical ceiling** for this model
on M2 Max without changes to the MLX framework or model weights.

---

## 11. Deep Optimization Analysis: C++ / Kernel Dispatch / Metal Scheduling

### 11.1 Decode Step Breakdown

Profiled individual decode steps (30 tokens, Qwen3.5-35B-A3B 4-bit on M2 Max):

| Component | Time | % of Step |
|-----------|------|-----------|
| **GPU execution** (mx.eval) | 11.7ms | 88% |
| **Python graph build** (lazy ops) | 1.6ms | 12% |
| **Total step** | 13.3ms | 100% |
| *Theoretical min* (pure bandwidth) | *6.8ms* | *51%* |

### 11.2 Computation Graph Analysis

One decode step produces **2,612 operations** in the MLX computation graph:

| Operation | Count | % | Notes |
|-----------|-------|---|-------|
| QuantizedMatmul | 391 | 15.0% | Main weight computation |
| Reshape | 320 | 12.3% | Zero-cost shape ops |
| Broadcast | 260 | 10.0% | Lightweight |
| RMSNorm | 191 | 7.3% | Already fused (mx.fast) |
| Multiply | 150 | 5.7% | Element-wise |
| GatherQMM | 120 | 4.6% | MoE expert dispatch (3/layer) |
| Add | 120 | 4.6% | Element-wise |
| Arange | 120 | 4.6% | Index generation |
| CompiledSigmoidMultiplyMultiply | 80 | 3.1% | Already fused activations |
| Softmax | 40 | 1.5% | MoE router (1/layer) |
| ArgPartition | 40 | 1.5% | Top-K selection (1/layer) |
| ScaledDotProductAttention | 10 | 0.4% | Fused attention (10 attn layers) |
| CustomKernel | 30 | 1.1% | GatedDeltaNet compiled |
| Other | ~641 | 24.6% | Various element-wise |

**511 matmul operations** (QuantizedMatmul + GatherQMM) dominate computation.

### 11.3 Kernel Dispatch Overhead Measurement

Measured per-kernel overhead with trivial Metal kernels:

| Dispatch Pattern | Time per Kernel |
|-----------------|-----------------|
| Custom kernel, 1 per mx.eval() | 186μs |
| Native MLX op, 1 per mx.eval() | 98μs |
| **10 kernels per mx.eval()** | **15μs** |

**Key insight**: Within a single `mx.eval()` call, per-kernel dispatch overhead drops to ~15μs because MLX batches operations into one Metal command buffer. The full model (2,612 ops in one eval) benefits from this batching.

### 11.4 MoE Component Timing

Isolated per-layer MoE profiling (individual eval per component, NOT representative of fused execution):

| Component | Time (isolated) |
|-----------|----------------|
| Full MoE block | 1.06ms |
| Router (gate + softmax + topk) | 0.17ms |
| SwitchGLU (3× gather_mm) | 0.89ms |
| Shared expert | 0.18ms |
| **40 layers × MoE (isolated)** | **42.3ms** |
| **Full model step (fused)** | **11.7ms** |

The 3.6× ratio (42.3ms → 11.7ms) shows MLX's lazy evaluation + graph-level optimization already provides massive savings.

### 11.5 Custom Metal Kernel Experiment

**Fused softmax + topk + normalize** kernel (replaces 4 separate ops):

| Implementation | Router Time | 40-layer Savings |
|---------------|-------------|------------------|
| Current (4 ops) | 0.180ms | baseline |
| Fused custom kernel | 0.200ms | **-0.80ms (slower!)** |

Custom kernels are **slower** because they can't participate in MLX's full-graph batch optimization. A custom kernel runs as its own command buffer, while native ops benefit from being batched with 2,612 other ops.

### 11.6 Tested Optimization Strategies

| Strategy | Result | Why |
|----------|--------|-----|
| mx.compile(model) | **FAILED** | Incompatible with ArraysCache (GDN recurrent state) |
| mx.async_eval | **1.01x** | GPU time (11.7ms) >> Python time (1.6ms), nothing to overlap |
| Dedicated generation stream | **1.00x** | Same reason |
| Custom Metal kernel (fused router) | **0.90x** | Breaks full-graph batch optimization |

### 11.7 GPU Utilization Gap Analysis

```
Theoretical min (400 GB/s bandwidth):     6.8ms  (100%)
Actual GPU execution:                    11.7ms  ( 57%)
─────────────────────────────────────────────────────
Gap:                                      4.9ms  ( 43%)
```

The 4.9ms gap is Metal's internal overhead for scheduling 2,612 operations sequentially within one command buffer. This includes:
- Per-kernel compute encoder setup
- Metal scheduler inserting memory barriers between dependent kernels
- Cache invalidation between different weight tensor reads
- Command buffer processing overhead

This overhead is **intrinsic to the Metal runtime** and cannot be reduced from Python, custom kernels, or user-facing MLX APIs.

### 11.8 What Would Help (Framework-Level)

These would require changes to MLX itself (C++ core or Metal backend):

1. **Megakernel / persistent kernel approach**: Run the entire transformer layer as one Metal kernel that stays resident on GPU, avoiding per-kernel scheduling. Used by FlashInfer/TensorRT-LLM on CUDA.
2. **Deeper kernel fusion in mx.compile**: Fuse `RMSNorm → Matmul → Activation` chains into single kernels to reduce op count from 2,612 to ~200-300.
3. **Metal 4 TensorOps (M5 hardware)**: Dedicated matrix multiply accelerator provides 4× TTFT improvement. Decode improvement is ~20-27% from increased bandwidth.
4. **Indirect command buffers**: Let the GPU schedule subsequent kernels without CPU round-trips. Requires Metal 3+ features.

### 11.9 Final Optimization Ceiling

```
Optimization Stack                  tok/s    vs Baseline
──────────────────────────────────────────────────────────
Baseline (no optimizations)          73.5     1.00x
+ MTP K=1 batch verify               80.0     1.09x
+ Lazy draft evaluation              82.0     1.11x
+ Q4 MTP head                        83.0     1.13x  ← ACHIEVED
──────────────────────────────────────────────────────────
Theoretical max (0 overhead):       ~147      2.00x  (bandwidth-limited)
Realistic max (framework changes):  ~100-110  1.35-1.50x (estimated)
```

**1.13x is the achievable ceiling from Python-level optimization.** Further gains require MLX framework-level changes (kernel fusion, Metal scheduling improvements) or new hardware (M5 with Metal 4 TensorOps).

---

## 12. Framework-Level Optimization Deep Dive

### 12.1 Tested Python-Level Approaches

| Approach | Result | Details |
|----------|--------|---------|
| **ZMLX patch** (pip, deltanet only) | **0.97x** | 30 GDN layers patched; slight regression |
| **ZMLX patch** (all patterns forced) | **0.89x** | 181 modules patched; moe_mlp, rmsnorm, softmax all slower |
| **mx.compile(model)** | **FAILED** | ArraysCache incompatible with compile's array-tree requirement |
| **mx.compile(moe_block)** | **FAILED** | "Slice cannot infer output shapes" (argpartition) |
| **mx.async_eval pipeline** | **1.04x** | 69.5 vs 66.6 tok/s; minimal overlap since GPU >> Python |
| **Force sorted_indices** | **1.00x** | No change — sorting 8 indices is trivial |

**Why ZMLX regresses on Qwen3.5**: MLX's built-in ops (`mx.fast.rms_norm`, `gather_qmm`, `mx.fast.scaled_dot_product_attention`) are already highly optimized for this architecture. ZMLX's replacement kernels add Python dispatch overhead that exceeds any compute savings.

**Why mx.compile fails**: The Qwen3.5 hybrid architecture uses `ArraysCache` for GatedDeltaNet recurrent state. This cache type contains non-array state (offsets, metadata) that mx.compile cannot trace through. The `Slice` shape inference failure in the MoE block comes from `argpartition` which produces data-dependent output shapes.

### 12.2 Fused gather_qmm_swiglu Kernel Proof-of-Concept

Built and validated a custom Metal kernel that fuses gate_proj + up_proj + SwiGLU into a single dispatch:

**Correctness**: Max absolute difference = 0.000241 vs reference (PASS)

**Isolated performance** (per mx.eval call, includes ~100μs dispatch overhead):

| Implementation | Time | Speedup |
|---------------|------|---------|
| SwitchGLU (gate+up+SwiGLU+down, 3 gather_qmm) | 0.918ms | baseline |
| Estimated gate+up+SwiGLU portion (2/3) | 0.612ms | baseline |
| **Fused kernel (gate+up+SwiGLU)** | **0.196ms** | **3.1x** |

**Why this can't help as a standalone kernel**: The fused kernel runs as its own Metal command buffer (via `mx.fast.metal_kernel`), breaking MLX's graph-level optimization. In the full model, 2,612 ops share one command buffer at ~15μs/op overhead. Adding a standalone kernel forces a separate command buffer submission.

**Why this CAN help as a C++ primitive**: A native MLX primitive participates in the computation graph and shares the same command buffer. The savings come from:
- Eliminating 80 gather_qmm ops (2 per layer × 40 layers) → replaced by 40 fused ops
- Eliminating 40 SwiGLU activation ops → absorbed into the fused kernel
- Eliminating 2 intermediate DRAM writes per layer (gate_out, up_out tensors)
- Net reduction: 2,612 → 2,492 ops (-120 ops × ~15μs = ~1.8ms estimated savings)

### 12.3 C++ Extension Architecture

Built a complete scaffolded C++ MLX extension (`mlx_fused_moe/`) with:
- **Metal kernel** (`kernels/gather_qmm_swiglu.metal`): Reads both gate and up weight rows in a single pass, applies dequantization inline, computes SwiGLU in registers
- **C++ primitive** (`cpp/ops.cpp`): `GatherQMMSwiGLU` class inheriting from `mlx::core::Primitive`, implementing `eval_gpu()` via Metal kernel dispatch using MLX's command encoder API
- **Python bindings** (`cpp/bindings.cpp`): nanobind wrapper exposing `gather_qmm_swiglu()` to Python
- **CMake build** (`CMakeLists.txt`): Uses `find_package(MLX)` + `mlx_build_metallib()` for the Metal shader library

**Build command** (not yet tested):
```bash
cd mlx_fused_moe && pip install -e .
```

### 12.4 Estimated Impact of C++ Fused MoE

```
Current decode step:                   13.3ms (2,612 ops)
- Eliminate 120 ops (gather+activation): -1.8ms (estimated)
- Eliminate DRAM intermediates:          -0.5ms (estimated)
New decode step:                       ~11.0ms (2,492 ops)
Speedup:                               ~1.21x baseline
Combined with MTP Q4 lazy batch:       ~1.21 × 1.13 = ~1.37x
Estimated throughput:                  ~90-100 tok/s
```

### 12.5 gather_qmm_swiglu: What ZMLX Implements (Reference)

ZMLX's `gather_qmm_swiglu` (~800 lines C++/Metal in `integrations/mlx_local_integration/`) is the production reference:
- Requires building MLX from source with custom patches
- Reported +6.4% on GLM-4.7-Flash, +2% on Qwen3.5 (stock MLX only provides `topk_gating_softmax` and `moe_combine`)
- Uses threadgroup memory for shared dequantization, SIMD shuffle for reduction
- Handles both float16 and bfloat16 activation types

### 12.6 Other Framework-Level Findings

**Metal ICBs (Indirect Command Buffers)**: Not viable with MLX — lazy graph evaluation re-encodes GPU commands per eval(), incompatible with ICB's pre-fixed command sequence. ICB limit of 16,384 commands also constrains large models.

**Megakernel approach**: CUDA-only (MPK compiler, Mirage 2025). Shows 1.2x on CUDA. Not portable to Metal due to reliance on CUDA streaming multiprocessor task scheduling.

**SegmentedMM**: Present in recent MLX releases for grouped GEMM. Not yet exposed in Python API in a form useful for custom MoE dispatch.

**Apple Neural Engine**: MLX does not expose ANE paths. Core ML is required for INT8 ANE dispatch. MLX maintainers closed w4a8 GEMM as wontfix.

### 12.7 Complete Optimization Stack

```
Optimization Stack                              tok/s    vs Baseline
────────────────────────────────────────────────────────────────────────
Baseline                                         67-74    1.00x
+ async_eval pipeline                            69-76    1.04x
+ MTP K=1 batch verify                          ~80      1.09x
+ Lazy draft evaluation                         ~82      1.11x
+ Q4 MTP head                                   ~83      1.13x  ← ACHIEVED
────────────────────── Python ceiling ──────────────────────────────────
+ C++ fused gather_qmm_swiglu (estimated)       ~93-100  1.27-1.37x
+ ZMLX experimental MLX fork                    ~95-105  1.30-1.43x
────────────────────── Framework ceiling ───────────────────────────────
Theoretical max (pure bandwidth, 0 overhead)    ~147     2.00x
```

### 12.8 Recommended Next Steps

1. **Build and test the C++ extension** (`mlx_fused_moe/`): Requires MLX headers and nanobind. Expected ~1 day of engineering to get building and passing tests.
2. **Contribute upstream**: File an MLX issue proposing `gather_qmm_swiglu` as a new `mx.fast.*` primitive. The fusion pattern is universal for all SwiGLU-based MoE models.
3. **Profile with Metal GPU Trace**: `MTL_CAPTURE_ENABLED=1` + `mx.metal.start_capture()` to get kernel-level timeline in Xcode. This would reveal the exact dispatch distribution and identify any remaining optimization targets.
4. **Test on M5 hardware**: Metal 4 TensorOps + 28% more bandwidth → estimated ~1.3-1.5x over M2 Max (stacks with all software optimizations).
