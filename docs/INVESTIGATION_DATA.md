# Speculative Decoding on Apple Silicon: Investigation Data

**Model**: Qwen3.5-35B-A3B (MoE, 256 experts, top-8 routing, 40 layers)
**Quantization**: 4-bit (mlx-community)
**Hardware**: Apple M2 Max 48GB (400 GB/s memory bandwidth)
**Framework**: MLX 0.31.1
**Baseline**: 70-74 t/s decode (greedy, temperature=0)

---

## Phase 1: MTP Head Implementation

### What is MTP?

Multi-Token Prediction: the model ships with a small auxiliary head trained to predict the next token from the backbone's hidden states. We use it as a speculative drafter -- draft one token cheaply, verify it with the full model, accept or reject.

### The Critical Bug: Concatenation Order

The MTP head takes `(hidden_state, token_embedding)` as input. Initial implementation used `[hidden, embed]` concat order. Acceptance was 42.9%.

Reading vLLM's reference implementation revealed the correct order is `[embed, hidden]`. After fixing:

| State | Acceptance | tok/step | tok/s |
|---|---|---|---|
| Before fix | 42.9% | 1.48 | 50.4 |
| After fix | 90.4% | 1.92 | 97.2 |

### The Norm Shift (+1)

Qwen3.5 raw checkpoints store RMSNorm weights as `(weight - 1)`. The mlx-lm sanitize function adds +1 for the main model, but MTP weights are loaded separately and bypass sanitization. Missing this caused silent accuracy degradation.

### MoE Architecture in the MTP Head

The 35B-A3B MTP head itself contains a full MoE layer (256 experts + router + shared expert gate). Required implementing `MTPMoEMLP` class and converting 785 individual expert weight tensors into stacked `SwitchLinear` format (20 tensors).

### Sequential vs Batch Verification

Two verification strategies:

- **Sequential**: Process token_0, then check draft. Bit-exact with baseline. 0.93x (slower due to MTP overhead).
- **Batch**: Process `[token_0, draft]` together. Faster (1.05x) but GatedDeltaNet recurrent layers produce slightly different numerical results when processing 2 tokens at once vs one at a time. The difference comes from the Metal kernel processing the recurrence as a batch vs the Python loop processing token-by-token.

Decision: use batch for throughput, sequential only for correctness validation.

---

## Phase 2: Cross-Model Scaling Analysis

### MTP Scales with Model Size

| Model | Parameters | Base t/s | MTP Batch Ratio | Acceptance |
|---|---|---|---|---|
| Qwen3.5-0.8B | 0.8B | 199.5 | 0.86x | 34% |
| Qwen3.5-2B | 2B | 131.4 | 0.93x | 54% |
| Qwen3.5-4B | 4B | 69.9 | 0.99x | 68% |
| Qwen3.5-9B | 9B | 44.4 | 1.01x | 71% |
| Qwen3.5-27B | 27B | 14.3 | **1.12x** | 74% |
| Qwen3.5-35B-A3B | 3B active (MoE) | 70.2 | **1.05x** | 76% |

**Key insight**: The MTP head is a fixed ~5ms cost per step. As models get larger (more ms/step for the main forward), the MTP overhead becomes proportionally smaller. Below ~4B, the head costs more than it saves.

### Multi-Token Speculation (K > 1)

Chaining the MTP head K times for deeper speculation:

| K | MTP Cost | Verify Cost | Total | Acceptance | Tok/Step | Effective t/s |
|---|---|---|---|---|---|---|
| 1 | 5ms | 18ms | 23ms | 81% | 1.82 | 79 |
| 2 | 10ms | 23ms | 36ms | 66% | 2.34 | 65 |
| 3 | 15ms | 27ms | 48ms | 57% | 2.71 | 57 |
| 4 | 20ms | 32ms | 60ms | 44% | 2.77 | 46 |

Acceptance drops ~15 percentage points per additional K because the MTP head was trained on backbone hidden states, not its own outputs. Chaining it makes predictions increasingly out-of-distribution.

Exception: repetitive content (counting sequences, code patterns) maintains >90% acceptance at K=2-3, giving up to 1.52x speedup.

---

## Phase 3: Failed Optimization Attempts

### Shared-Expert Drafting (0.85x -- FAILED)

Idea: patch MoE layers to skip expert routing and use only the shared expert for cheap drafting.

Result: 0.85x despite ~100% acceptance. Running the 40-layer model twice (even cheaply for draft) is always slower than running it once. The model is memory-bandwidth bound -- each forward pass loads ~1.5GB of weights regardless of compute complexity.

### Small-Model Draft (FAILED)

Using Qwen3.5-4B as draft model for 35B: only 2-5% token agreement. The models are too architecturally different despite sharing a tokenizer.

### Layer Skip / Early Exit (0% to -35% -- FAILED)

Block Influence profiling identified 10 near-identity layers (BI < 0.03). Skipping them maintained 100% token match but gave 0% speedup -- the skipped layers (mostly GDN recurrent) cost ~0.1ms each, negligible vs MoE layers.

Early exit (CALM-style): average exit layer was 39/40 even with aggressive thresholds. The lm_head probes added ~1ms overhead per probe, making every step slower (0.65-0.75x).

### mx.compile (FAILED)

ArraysCache (GDN recurrent state) is incompatible with mx.compile's array-tree tracing requirement. Compiling individual blocks showed 0% gain because MLX's lazy evaluation already batches all 2,612 ops into a single Metal command buffer.

### Custom Metal Kernels via mx.fast.metal_kernel (SLOWER)

A fused softmax+topk+normalize kernel was slower than the 4 separate ops it replaced (0.90x). Custom kernels run as separate command buffers, breaking MLX's graph-level batch optimization.

**This was the key learning: any custom kernel must participate in the native MLX computation graph via a C++ primitive, not via `mx.fast.metal_kernel`.**

---

## Phase 4: Profiling the Decode Step

### Cost Breakdown (13.3ms per step)

| Component | Time | % |
|---|---|---|
| GPU execution (mx.eval) | 11.7ms | 88% |
| Python graph build (lazy ops) | 1.6ms | 12% |
| **Total step** | **13.3ms** | 100% |
| *Theoretical min (pure bandwidth)* | *6.8ms* | *51%* |

### Computation Graph: 2,612 Operations

| Operation | Count | % |
|---|---|---|
| QuantizedMatmul | 391 | 15.0% |
| GatherQMM (MoE expert dispatch) | 120 | 4.6% |
| ScaledDotProductAttention | 10 | 0.4% |
| CustomKernel (GatedDeltaNet) | 30 | 1.1% |
| Everything else | ~2,061 | 78.9% |

511 matmul operations dominate. MoE layers account for ~60-80% of GPU time.

### GPU Utilization Gap

```
Theoretical min (400 GB/s): 6.8ms  (100%)
Actual GPU execution:      11.7ms  ( 57%)
Gap:                        4.9ms  ( 43%)
```

The 4.9ms gap is Metal's internal scheduling overhead for 2,612 sequential operations within one command buffer. This is intrinsic to the Metal runtime.

### Per-Component MoE Timing (Isolated)

| Component | Time (isolated) |
|---|---|
| Full MoE block | 1.06ms |
| Router (gate + softmax + topk) | 0.17ms |
| SwitchGLU (3x gather_qmm) | 0.89ms |
| Shared expert | 0.18ms |

40 layers x 1.06ms = 42.3ms isolated, but only 11.7ms fused in the full graph (3.6x reduction from lazy evaluation).

---

## Phase 5: MTP Q4 + Lazy Batch (Python Ceiling: 1.13x)

### Stackable Micro-Optimizations

| Optimization | Avg t/s | vs Base | Mechanism |
|---|---|---|---|
| Baseline | 73.4 | 1.00x | -- |
| MTP K=1 batch | 80.4 | 1.09x | Batch verify `[token_0, draft]` |
| + Lazy draft | 82.0 | 1.11x | Remove `mx.eval(draft)` sync point |
| + Q4 MTP head | **83.0** | **1.13x** | Quantize MTP head 1689MB -> 475MB |

**Lazy draft**: Instead of evaluating the draft token before building the verify graph, we let MLX include the MTP head computation in the same graph as the model forward. Saves ~0.5ms per step.

**Q4 MTP head**: 4-bit quantization reduces memory bandwidth for the draft step by 72%. Acceptance drops marginally (80% -> 76%) but total throughput improves because the draft is faster.

### Cost Model at 1.13x

```
Per step (22.0ms total):
  MTP head Q4:       ~3.5ms
  Batch verify (2):  ~18ms  (14ms base + 4ms marginal for 2nd token)
  Overhead/sync:     ~0.5ms

Expected tokens: 1 + 0.81 = 1.81
Effective per-token cost: 22.0 / 1.81 = 12.2ms (vs 13.6ms baseline)
```

### 1.13x = Python Ceiling

| Optimization | Result | Why |
|---|---|---|
| mx.compile | 0% | Lazy eval already fuses |
| Layer skip | 0% | Skipped layers cost nothing |
| Early exit | -25% to -35% | Probe overhead > savings |
| Self-speculative | -15% | Draft model same speed |
| Shared-expert draft | -15% | Same reason |
| Custom Metal kernel | -10% | Breaks graph optimization |
| **MTP K=1 Q4 lazy batch** | **+13%** | **Only winner** |

---

## Phase 6: Fused MoE SIMD Metal Kernels (C++ Extension)

### Why C++ Extension?

Custom Metal kernels via `mx.fast.metal_kernel` break graph optimization (-10%). A C++ `UnaryPrimitive` participates natively in the MLX computation graph, sharing the same Metal command buffer.

### Kernel 1: gather_qmm_swiglu (Gate + Up + SwiGLU)

Fuses two `gather_qmm` calls (gate_proj, up_proj) and SwiGLU activation into one Metal dispatch per MoE layer. Eliminates 120 ops from the graph (2 gather_qmm + 1 activation per layer x 40 layers).

**Architecture**:
- Grid: `(n_tokens, ceil(output_dim/8), top_k)` threadgroups of 64
- 2 simdgroups x 32 threads, each simdgroup computes 4 output rows
- Cooperative x-vector loading: each thread loads 16 values, 32 threads cover 512 elements
- Pre-division trick for 4-bit dequant: `x[i+j] = x[i+j] / (16^j)`, then mask packed uint16 weights
- `simd_sum` for cross-thread reduction
- SiLU applied in-register: `silu(gate_val) * up_val`

**Evolution**:

| Version | Isolated Speedup | Full Model Impact |
|---|---|---|
| Naive per-thread | 2.0-2.3x | **0.96x (regression)** |
| SIMD-tiled (matching MLX qmv_fast pattern) | 2.5x | **1.046x** |

The naive kernel was actually slower in the full model because it under-utilized SIMD lanes. The rewrite matched MLX's own `qmv_fast` tiling pattern (cooperative loading, simd_sum reduction, 8 output rows per threadgroup).

### Kernel 2: gather_qmm_down_reduce (Down + Score-Weighted Sum)

Fuses `gather_qmm(down_proj)` + score multiplication + expert sum into one dispatch. Loops over all 8 experts inside one threadgroup, accumulating score-weighted results.

| Config | Avg t/s | vs Baseline |
|---|---|---|
| baseline | 74.0 | 1.00x |
| fused_gate_up only | 76.3 | 1.03x |
| fused_gate_up + down_reduce | 77.1 | **1.04x** |

Down-proj fusion adds ~1% on top of gate+up. The gains are modest because the down_proj is a smaller matrix (512 -> 2048 vs 2048 -> 512 for gate/up).

### Fused GDN Projections (0% -- FAILED)

Attempted fusing the 4 GDN input projections (qkv, z, a, b) that all read the same input vector. No speedup because MLX's built-in `qmv_fast` already handles single-matrix quantized GEMV efficiently. The 4 individual dispatches are not a bottleneck.

### Combined Results (Fused MoE + MTP)

| Config | Avg t/s | vs Base |
|---|---|---|
| baseline | 72.6 | 1.00x |
| fused MoE only | 74.1 | 1.02x |
| MTP K=1 only | 81.6 | 1.12x |
| **fused MoE + MTP K=1** | **84.0** | **1.16x** |

### Build System

C++ extension using cmake + nanobind:

**Key gotchas**:
1. nanobind version must match MLX's (2.10.2 for MLX 0.31.1)
2. `NB_DOMAIN mlx` required for cross-module type sharing
3. `mlx/primitives.h` must be explicitly included
4. `allocator::malloc()` not `allocator::malloc_or_wait()`
5. Metallib auto-discovered from .so location at runtime

---

## Phase 7: K=2 Optimization Attempts

### Why K=2 Underperforms

K=2 batch verify processes 3 tokens, loading 3x8=24 expert weight matrices per MoE layer (~21ms). With 67% acceptance at depth 2, the expected value:

```
K=2 batch:
  All accepted (45%):     21ms -> 3 tokens = 7.0ms/tok
  Partial accept (22%):   21ms + 14ms replay -> 2 tokens = 17.5ms/tok
  Full reject (33%):      21ms + 14ms replay -> 1 token = 35.0ms/tok
  Weighted average:        ~12.6ms/tok

K=1 batch:
  Accept (81%):           18ms -> 2 tokens = 9.0ms/tok
  Reject (19%):           18ms + 14ms replay -> 1 token = 32.0ms/tok
  Weighted average:        ~12.4ms/tok
```

K=2 only wins if acceptance is consistently > 85%.

### Cascade K=2: Avoid the 3-Token Batch

**Idea**: Use K=1's 2-token batch verify `[token_0, draft1]`, then check position 1's logits for draft2 "for free" (already computed). If draft2 matches: one extra 1-token forward for bonus. If not: identical cost to K=1.

| Config | Avg t/s | vs Base | Tok/Step | Accept |
|---|---|---|---|---|
| k2_batch (old) | 69.5 | 0.95x | 2.34 | 67% |
| **k2_cascade** | **73.8** | **1.01x** | 2.32 | 65% |

+6.2% over old K=2. On QA (diverse text): +23% because rejected draft2 costs nothing.

Still doesn't beat K=1 (73.8 vs 82.8) because the extra forward when both accepted costs ~14ms for only +1 token.

### Adaptive K: Dynamic K Selection

Rolling acceptance window (last 20 drafts). When acceptance > threshold, use K=2 (cascade), otherwise K=1.

| Config | Avg t/s | vs Base |
|---|---|---|
| K=1 fixed | 82.8 | 1.13x |
| Adaptive (threshold 90%) | 80.3 | 1.10x |
| Adaptive (threshold 80%) | 77.8 | 1.06x |

The 90% threshold stays in K=1 mode most of the time (correctly). The 80% threshold is too aggressive in upgrading to K=2.

---

## Phase 8: The Replay Forward Problem

### The Cost of Rejection

When a draft is rejected, the cache contains the wrong token's contribution to the GDN recurrent state. We must:

1. Restore the full cache to pre-verify state (`save_cache_state` / `restore_cache_state`)
2. Replay accepted tokens through the model (~14ms for 1 token)

The 14ms replay is the memory-bandwidth floor for loading ~1.5GB of active weights through 40 layers. It equals one full baseline decode step.

### K=1 Cost Structure

```
Accept (81%): 18ms  (2-token batch verify)         -> 2 tokens
Reject (19%): 18ms + 14ms (verify + replay)         -> 1 token
              ^^^^^^^^^^^^
              32ms for 1 token on rejection
```

Average per-step: 18 + 0.19 * 14 = 20.7ms for 1.81 tokens = 11.4ms/tok

If we could eliminate the replay: 18ms for 1.81 tokens = 9.9ms/tok (+15% theoretical)

---

## Phase 9: Lossy Reject Attempts (FAILED)

### Variant A: Keep Wrong Draft in GDN State

On rejection, trim KV cache by 1 (correct) but don't restore GDN recurrent state (leave the wrong draft's contribution).

**Quality test results** (100 tokens, greedy, 4 prompts):

| Prompt | K=1 Diverge At | K=2 Diverge At |
|---|---|---|
| Fibonacci code | token 5 | token 3 |
| Relativity | token 8 | token 5 |
| Counting 1-30 | EXACT MATCH | token 97 |
| TCP vs UDP | token 46 | token 5 |

**Unacceptable**. The wrong draft token actively pushes GDN state in the wrong direction. With 30 GDN layers each amplifying the error, divergence begins within 3-8 tokens on diverse content.

### Variant B: Restore GDN to Pre-Verify (Missing token_0)

On rejection, restore GDN state from the pre-verify snapshot (before token_0 was processed), but keep token_0's KV cache entry.

| Prompt | K=1 Diverge At | K=2 Diverge At |
|---|---|---|
| Fibonacci code | token 11 | token 3 |
| Relativity | token 8 | token 6 |
| Counting 1-30 | EXACT MATCH | token 66 |
| TCP vs UDP | token 46 | token 7 |

**Still unacceptable**. Missing token_0 in the GDN state creates a KV/GDN desynchronization -- the attention layers have token_0's contribution but the GDN layers don't. The delta-rule gating does NOT attenuate the missing-token error fast enough.

### Root Cause Analysis

The GDN recurrent state update is:
```
state_new = g * state_old + beta * (v ⊗ k)
```

Where g (gating) is ~0.9-0.99 per step. A wrong/missing token contribution persists in the state for many steps (exponential decay with rate g). With 30 GDN layers, each propagating the error, even small corruption at one layer compounds through subsequent layers.

**Conclusion**: Lossy rejection is fundamentally incompatible with GDN architectures. The recurrent state requires exact restoration.

---

## Phase 10: Zero-Replay via GDN Intermediate Capture

### The Idea

During the 2-token batch verify `[token_0, draft1]`, the GDN recurrence processes token_0 then draft1 sequentially internally. If we can capture the intermediate state (after token_0, before draft1), we can restore to it on rejection without replaying.

### Attempt 1: Python Ops Path (1.06x -- TOO SLOW)

The Metal kernel processes the entire sequence in one GPU dispatch -- we can't intercept intermediate state. But the Python ops path uses a loop over tokens, where we CAN intercept.

**Approach**: Monkey-patch `gated_delta_update` to force the Python ops path during capture. Save recurrent state before the last token.

**Problem**: The ops path is significantly slower than the Metal kernel. Every verify step (not just rejected ones) pays the penalty.

| Config | Standard | Ops Path Zero-Replay | Delta |
|---|---|---|---|
| K=1 | 1.14x | 1.06x | **-8%** |
| all_k1 | 1.21x | 1.10x | **-11%** |

The ops path adds ~3-5ms to every verify forward. The replay saving (19% x 14ms = 2.7ms average) doesn't compensate.

On repetitive content (98% acceptance, almost no replays): 95.0 vs 79.1 t/s = 17% slower. Pure penalty.

### Attempt 2: Split Metal Kernel (1.27x -- SUCCESS)

**Key insight**: Instead of forcing the ops path for the entire T-token batch, split it into T individual Metal kernel calls. Each call processes one token using the full Metal kernel optimization. We capture the intermediate state between calls.

```python
def _split_gated_delta_update(q, k, v, a, b, A_log, dt_bias, state, mask, use_kernel):
    if capture_enabled and T > 1:
        for t in range(T):
            y_t, state = original_gated_delta_update(
                q[:, t:t+1], k[:, t:t+1], v[:, t:t+1], ...,
                state, ..., use_kernel=True,  # Metal kernel!
            )
            if t < T - 1:
                intermediates.append(mx.array(state))
        return concat(ys), state
    else:
        return original_gated_delta_update(q, k, v, ..., state, mask, use_kernel)
```

The overhead per split: one extra Metal kernel dispatch per GDN layer per verify step. With ~30 GDN layers and ~0.05ms dispatch overhead, that's ~1.5ms additional. But the replay saving is 19% x 14ms = 2.7ms. Net gain: ~1.2ms per step.

### Conv State Restoration

GDN layers also have a conv sliding window state (cache[0], kernel_size=4, stores last 3 rows of QKV projections). On rejection, this must also be restored to the intermediate position.

**Formula** for intermediate conv state at position p after N verify tokens, with kernel_size-1=3:
```
token_start = max(0, 3 - N)
intermediate = concat(
    saved_conv[p+1:],                                    # old context rows
    current_conv[token_start : token_start + p + 1],     # accepted token rows
)
```

Before the ops path fix (conv not restored), quality was significantly worse. After restoring both recurrent AND conv state, the remaining divergence is only from batch-size-dependent floating point precision in quantized matmuls (batch of 2 vs 1 in Metal GPU tiling).

### Self-Consistency Test

Zero-replay produces bit-identical output when run twice with the same input (greedy sampling). The divergence vs standard path is from the computational path difference, not state corruption.

### Final Results: Split Metal Kernel Zero-Replay

| Config | Avg t/s | vs Base | Accept | Tok/Step |
|---|---|---|---|---|
| baseline | 70.9 | 1.00x | -- | 1.00 |
| mtp_k1 (standard) | 80.7 | 1.14x | 81% | 1.82 |
| **zr_k1** | **85.9** | **1.21x** | 80% | 1.81 |
| all_k1 (fused, standard) | 84.4 | 1.19x | 81% | 1.82 |
| **all_zr_k1** | **90.4** | **1.27x** | 79% | 1.80 |

Per-category (all_zr_k1):

| Category | Base t/s | all_zr_k1 t/s | Speedup |
|---|---|---|---|
| code | 72.0 | 90.2 | 1.26x |
| prose | 70.1 | 85.5 | 1.22x |
| repetitive | 70.5 | 100.6 | 1.43x |
| qa | 71.1 | 86.2 | 1.24x |

### Approach Comparison

| Zero-Replay Method | zr_k1 | all_zr_k1 | Why |
|---|---|---|---|
| Ops path (forced Python) | 1.06x | 1.10x | Ops path ~3-5ms slower than Metal per verify |
| **Split Metal kernel** | **1.21x** | **1.27x** | Individual Metal calls have negligible overhead |

### K=2 and K=3 with Zero-Replay

| Config | Avg t/s | vs Base | Accept | Tok/Step |
|---|---|---|---|---|
| **all_zr_k1** | **90.4** | **1.27x** | 79% | 1.80 |
| all_zr_k2 | 83.0 | 1.18x | 64% | 2.30 |
| all_zr_k3 | 60.6 | 0.86x | 37% | 2.11 |

K=1 remains optimal. The MTP head's accuracy degrades too sharply at depth 2+ to offset the verify cost.

---

## Complete Optimization Timeline

```
Optimization Stack                              tok/s    vs Baseline
──────────────────────────────────────────────────────────────────────
Baseline                                         70-74    1.00x
+ MTP K=1 batch verify                          ~80      1.09x
+ Lazy draft evaluation                         ~82      1.11x
+ Q4 MTP head                                   ~83      1.13x
────────────────────── Python ceiling ──────────────────────────────
+ C++ fused MoE SIMD kernels (gate+up+down)     ~84      1.19x
+ Zero-replay (split Metal kernel)               90.4     1.27x
────────────────────── Current best ───────────────────────────────
Theoretical max (pure bandwidth, 0 overhead)    ~147      2.00x
```

### What Didn't Work (Summary)

| Attempt | Result | Root Cause |
|---|---|---|
| Shared-expert drafting | 0.85x | Model is memory-bandwidth bound; 2 forwards always > 1 forward |
| Small-model drafting | FAILED | 2-5% token agreement between 4B and 35B |
| Layer skip | 0% | Skipped layers already cost nothing (~0.1ms each) |
| Early exit (CALM) | -25% to -35% | Model rarely exits early; probe overhead |
| mx.compile | 0% | Lazy eval already fuses; ArraysCache incompatible |
| Custom Metal kernel (metal_kernel API) | -10% | Breaks graph batching (separate command buffer) |
| ZMLX fork | -3% to -11% | MLX built-in ops already optimal for this arch |
| Lossy reject (keep wrong state) | N/A | Severe quality degradation (diverge in 3-8 tokens) |
| Lossy reject (restore pre-verify) | N/A | Missing token causes KV/GDN desync |
| Zero-replay via ops path | 1.06x | Ops path 3-5ms slower than Metal per verify |
| Fused GDN projections | 0% | qmv_fast already optimal for single GEMV |
| K=2 batch verify | 0.95x | 3-token batch too expensive at 64% acceptance |
| K=3 | 0.86x | 37% acceptance, head accuracy too low |
| Adaptive K (80% threshold) | 1.06x | Too aggressive in upgrading to K=2 |

### What Worked

| Optimization | Contribution | Mechanism |
|---|---|---|
| MTP K=1 batch verify | +9% | Speculate 1 token, verify in batch |
| Lazy draft eval | +2% | Remove sync point, single MLX graph |
| Q4 MTP head | +2% | 72% less memory bandwidth for draft |
| C++ fused gate+up+SwiGLU | +3% | Eliminate 120 ops from graph |
| C++ fused down+reduce | +1% | Eliminate expert sum ops |
| Zero-replay (split Metal) | +8% | Eliminate 14ms replay on rejection |
| **Total** | **+27%** | **70.9 -> 90.4 t/s** |

---

## Hardware Context

### M2 Max Memory Bandwidth Analysis

```
Active weights per step: ~1.5GB (4-bit, 3B active params in MoE)
M2 Max bandwidth: 400 GB/s
Theoretical minimum: 1.5GB / 400 GB/s = 3.75ms = 267 t/s

Actual: 13.3ms = 75 t/s (28% utilization)
With all optimizations: 11.1ms = 90 t/s (34% utilization)
```

The 66% gap between theoretical and actual is Metal runtime overhead (command buffer scheduling, memory barriers, kernel dispatch). This is not addressable from user code.

### What Would Help (Framework / Hardware Level)

1. **MLX kernel fusion** (mx.compile for MoE): Fuse `RMSNorm -> Matmul -> Activation` chains into single kernels. Would reduce 2,612 ops to ~200-300.
2. **Metal megakernel**: Run entire transformer layer as one persistent Metal kernel. Used by FlashInfer/TensorRT-LLM on CUDA.
3. **Metal 4 TensorOps (M5)**: Dedicated matrix multiply accelerator. Estimated ~20-27% decode improvement.
4. **Indirect command buffers**: Let GPU schedule subsequent kernels without CPU round-trips.

---

## EAGLE Comparison (from Earlier Phase)

For completeness, EAGLE-style tree speculation was also tested:

| Method | 4K Context t/s | vs Baseline | Accept | Tok/Step |
|---|---|---|---|---|
| Baseline | 64.3 | 1.00x | -- | 1.00 |
| MTP K=1 | 59.6 | 0.93x | 81% | 1.83 |
| EAGLE D2W1 | **66.0** | **1.03x** | 75% | 2.53 |
| EAGLE D3W1 | 52.6 | 0.82x | 57% | 2.74 |

EAGLE with depth-2 width-1 was the only config that beat baseline, and only at 4K context. At 16K context, all speculation methods collapsed due to memory pressure.

Note: the MTP results here are from an earlier implementation phase (before lazy batch, Q4, fused kernels, zero-replay). The final MTP implementation significantly outperforms these early numbers.

---

## KV Cache Experiments (from Earlier Phase)

| Method | 512ctx | 2048ctx | 4096ctx | 8192ctx | Quality |
|---|---|---|---|---|---|
| Baseline | 74.6 | 68.0 | 64.3 | 61.8 | 100% |
| Naive INT8 | 69.6 | 61.6 | 59.2 | 52.5 | 100% |
| TQ-LM INT4 (fused) | 65.8 | 53.0 | 40.2 | 26.9 | ~27% match |
| TQ-LM INT8 (fused) | 67.1 | 57.3 | 50.0 | 36.3 | ~25% match |

TurboQuant-LM quality "issue" was a measurement artifact -- the Hadamard rotation changes the floating-point computation path through SDPA, causing autoregressive divergence. Actual output quality (KL divergence, coherence) was fine.

A fused quantized SDPA Metal kernel was also implemented (attention directly against 4-bit packed KV) -- correct results but 3-18x slower due to Metal's tree reduction barriers. Saves ~160MB but impractical for speed.
