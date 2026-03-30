# Cross-Model Benchmark Report

**Platform**: Apple Silicon M-series, 48GB unified memory
**Framework**: MLX + mlx-lm
**Token generation**: Greedy (temperature=0), 128 max tokens, best of 2 runs
**Optimizations tested**: MTP (BF16/Q4), Prompt Lookup, Shared Expert Drafter

---

## Summary Table

| Model | Arch | Size (4-bit) | Baseline t/s | Best Method | Speedup |
|-------|------|-------------|-------------|-------------|---------|
| Qwen3.5-35B-A3B | Transformer + MoE | 19GB | 64.6 | MTP Q4 | **1.09x** |
| Nemotron-3-Nano-4B | Mamba2 + Attn hybrid | 2.1GB | 82.1 | Prompt Lookup* | 1.32x* |
| Nemotron-3-Nano-30B-A3B | Mamba2 + Attn + MoE | 17GB | 73.8 | Prompt Lookup* | 1.04x* |
| Qwen3-Next-80B-A3B (3-bit) | DeltaNet + Attn + MoE | 33GB | 52.7 | MTP BF16 | **1.40x*** |
| Qwen3-Coder-Next | DeltaNet + Attn + MoE | 42GB | N/A (OOM) | - | - |

*Prompt Lookup gains are highly prompt-dependent. Starred values are best-case, not average.
**MTP BF16 1.40x was on summarization only; average across categories is ~1.03x due to memory pressure.

---

## Detailed Results

### 1. Qwen3.5-35B-A3B (from previous benchmark)

| Method | Avg t/s | Speedup | Accept Rate |
|--------|---------|---------|-------------|
| Baseline | 64.6 | 1.00x | - |
| MTP BF16 | 69.6 | 1.08x | 80% |
| MTP Q4 | 70.4 | **1.09x** | 76% |
| Prompt Lookup | 66.1 | 1.02x | 46% |
| Shared Expert d=1 | 56.8 | 0.88x | ~100% |
| Shared Expert d=3 | 51.9 | 0.80x | ~100% |

**Verdict**: MTP Q4 is the clear winner. Minimal quality loss from 4-bit quantization of the MTP head, and the memory savings (928MB vs 3.3GB) are significant.

---

### 2. Nemotron-3-Nano-4B (dense Mamba2 hybrid)

| Category | Baseline t/s | Prompt Lookup | PL Speedup |
|----------|-------------|---------------|------------|
| code:0 | 85.8 | 113.1 | 1.32x |
| code:1 | 86.1 | 64.7 | 0.75x |
| prose:0 | 86.3 | 88.4 | 1.02x |
| prose:1 | 86.5 | 98.2 | 1.13x |
| short:0 | 59.3 | 58.7 | 0.99x |
| short:1 | 87.7 | 87.2 | 0.99x |
| summarization:0 | 81.4 | 67.4 | 0.83x |
| summarization:1 | 83.4 | 48.1 | 0.58x |
| **Average** | **82.1** | **78.2** | **0.95x** |

**Verdict**: This dense model is already fast at 82 t/s (only 0.6B params at 4-bit). Prompt Lookup helps on specific prompts with repetitive structure (code:0 at 1.32x) but hurts on others. No MTP available; no MoE for shared expert.

---

### 3. Nemotron-3-Nano-30B-A3B (Mamba2 + MoE hybrid)

| Category | Baseline t/s | PL t/s | PL x | SE1 t/s | SE1 x |
|----------|-------------|--------|------|---------|-------|
| code:0 | 80.2 | 76.8 | 0.96x | 60.4 | 0.75x |
| code:1 | 79.4 | 75.4 | 0.95x | 58.4 | 0.74x |
| prose:0 | 80.3 | 78.7 | 0.98x | 60.1 | 0.75x |
| prose:1 | 80.2 | 76.6 | 0.95x | 59.3 | 0.74x |
| short:0 | 80.3 | 83.9 | 1.04x | 60.0 | 0.75x |
| short:1 | 80.7 | 78.0 | 0.97x | 60.1 | 0.74x |
| summarization:0 | 71.6 | 58.6 | 0.82x | 55.4 | 0.77x |
| summarization:1 | 37.9 | 27.4 | 0.72x | 32.3 | 0.85x |
| **Average** | **73.8** | **69.4** | **0.93x** | **55.7** | **0.76x** |

**Verdict**: No optimization helps this model on average. Shared Expert drafting produces high acceptance (~2 tok/step) but the 2x overhead of the verify pass makes it net negative. No MTP available for this architecture (Mamba2 hybrid).

---

### 4. Qwen3-Next-80B-A3B-Instruct (3-bit quant, DeltaNet + MoE)

| Category | Base t/s | BF16 t/s | BF16 x | Q4 t/s | Q4 x | PL t/s | PL x | SE1 t/s | SE1 x |
|----------|---------|----------|--------|--------|------|--------|------|---------|-------|
| code:0 | 55.5 | 54.4 | 0.98x | 50.0 | 0.90x | 54.4 | 0.98x | 43.7 | 0.79x |
| code:1 | 54.5 | 56.3 | 1.03x | 49.6 | 0.91x | 52.3 | 0.96x | 43.4 | 0.79x |
| prose:0 | 54.8 | 44.4 | 0.81x | 48.8 | 0.89x | 53.7 | 0.98x | 43.4 | 0.79x |
| prose:1 | 55.1 | 59.1 | 1.07x | 49.6 | 0.90x | 52.1 | 0.95x | 43.7 | 0.79x |
| short:0 | 55.8 | 52.1 | 0.93x | 49.9 | 0.89x | 56.3 | 1.01x | 43.9 | 0.79x |
| short:1 | 56.3 | 61.3 | 1.09x | 57.2 | 1.02x | 54.3 | 0.96x | 44.7 | 0.79x |
| sum:0 | 36.9 | 51.7 | 1.40x | 53.4 | 1.45x | 35.3 | 0.96x | 31.2 | 0.85x |
| sum:1 | 51.0 | 45.3 | 0.89x | 40.6 | 0.80x | 43.3 | 0.85x | 40.7 | 0.80x |
| **Avg** | **52.5** | **53.1** | **1.03x** | **49.9** | **0.98x** | **50.2** | **0.96x** | **41.8** | **0.80x** |

MTP acceptance rates: BF16 ~79%, Q4 ~74%

**Verdict**: MTP BF16 marginally positive at 1.03x average, with standout 1.40x on summarization (high n-gram overlap). However, the 3.3GB MTP BF16 head + 33GB model = 36GB out of 48GB leaves little room, causing memory pressure that limits MTP's benefit. On a 64GB+ machine, expect significantly better MTP gains (similar to the 1.09x seen on 35B-A3B with comfortable headroom).

---

### 5. Qwen3-Coder-Next (could not benchmark)

**Status**: Cannot run on 48GB RAM.
- 4-bit quant: 42GB on disk (same 80B MoE architecture as Qwen3-Next-80B-A3B)
- No 3-bit quant available
- No MTP weights in source model
- Model loads via mmap but inference causes constant page faults (60MB RSS for 42GB model)
- A 64GB+ machine is required

---

## Key Findings

1. **MTP is the most effective optimization** when memory permits. Q4 quantization of MTP heads provides the best balance of quality and memory (928MB vs 3.3GB for BF16, only ~4% lower acceptance rate).

2. **Memory is the bottleneck on Apple Silicon**. All optimizations suffer when the model + optimization state approaches total RAM. The 35B-A3B (19GB) with MTP Q4 (928MB) = 20GB on 48GB gave 1.09x. The 80B (33GB) with MTP BF16 (3.3GB) = 36GB on 48GB gave only 1.03x average.

3. **Shared Expert drafting is not viable** on any architecture tested. Despite 100% acceptance rates, the overhead of running the full verification pass plus the shared expert forward pass outweighs the savings. It consistently runs at 0.74-0.88x of baseline.

4. **Prompt Lookup is highly prompt-dependent**. It can give up to 1.32x on prompts with repetitive structures (binary search code template) but can also degrade performance significantly (0.58x). Not recommended as a general optimization.

5. **Hybrid architectures (Mamba2, DeltaNet) are fast at small scale** but don't support MTP out of the box. The Nemotron-3-Nano-4B runs at 82 t/s — faster than any MoE model tested — because its 0.6B active params are fully compute-bound.

---

## Architecture Comparison

| Architecture | Model | Active Params | Base t/s | Notes |
|-------------|-------|--------------|---------|-------|
| Mamba2 hybrid (dense) | Nemotron-4B | 0.6B | 82 t/s | Fastest. No MoE overhead. |
| Transformer + MoE | Qwen3.5-35B-A3B | ~3B | 65 t/s | MTP available, best optimization target |
| Mamba2 + MoE | Nemotron-30B-A3B | ~3B | 74 t/s | Mamba layers faster than attention |
| DeltaNet + MoE | Qwen3-Next-80B-A3B | ~3B | 53 t/s* | Memory-bound at 3-bit (33GB) |

*Running at 3-bit on tight memory; would be faster at 4-bit on a 64GB machine.

---

## New Models: MTP Scaling Analysis (Qwen3.5 + GLM-4.7-Flash)

### MTP Performance vs Model Size (Dense Qwen3.5)

| Model | Quant | Base t/s | MTP BF16 x | MTP Q4 x | MTP Accept | PL avg x | MTP Head Size |
|-------|-------|---------|-----------|---------|------------|---------|---------------|
| Qwen3.5-0.8B | 4-bit | 245 | **0.66x** | **0.65x** | 34% | 1.20x | 41 MB |
| Qwen3.5-0.8B | 8-bit | 176 | **0.77x** | **0.77x** | 44% | 1.32x | 41 MB |
| Qwen3.5-2B | 4-bit | 142 | **0.85x** | **0.83x** | 60% | 1.05x | 122 MB |
| Qwen3.5-2B | 8-bit | 93 | **0.91x** | **0.91x** | 57% | 1.12x | 122 MB |
| Qwen3.5-4B | 4-bit | 117* | 1.02x* | 1.04x* | 69%* | 1.01x* | 215 MB |
| Qwen3.5-9B | 4-bit | 86* | 1.05x* | 1.06x* | 70%* | 1.00x* | 640 MB |
| Qwen3.5-27B | 4-bit | 67* | 1.06x* | 1.08x* | 74%* | 1.01x* | 1.5 GB |
| Qwen3.5-35B-A3B | 4-bit | 65* | 1.08x | 1.09x | 76% | 1.02x | 3.3 GB |

*Values from earlier benchmarks

### Key Insight: MTP Has a Minimum Model Size Threshold

There is a clear **crossover point around 4B parameters** where MTP transitions from harmful to beneficial:

- **< 2B params**: MTP always hurts (0.65-0.91x). The baseline is already so fast (140-245 t/s) that the MTP head's overhead (forward pass + verification) can't be amortized.
- **2-4B params**: MTP is neutral (0.83-1.04x). Approaching break-even.
- **> 4B params**: MTP consistently helps (1.05-1.09x). The baseline slows enough that MTP's speculative tokens save more time than they cost.

### Quantization Impact on MTP Effectiveness

Comparing 4-bit vs 8-bit at the same model size:

| Model | 4-bit Base | 4-bit MTP x | 8-bit Base | 8-bit MTP x |
|-------|-----------|-------------|-----------|-------------|
| 0.8B | 245 t/s | 0.66x | 176 t/s | 0.77x |
| 2B | 142 t/s | 0.85x | 93 t/s | 0.91x |

**MTP works better on slower (8-bit) models** because:
1. Slower baseline = more time saved per accepted token
2. BF16 MTP head is the same size regardless of base model quant
3. The MTP overhead is relatively smaller compared to slower inference

### GLM-4.7-Flash (30B MoE, 4.7B active)

| Category | Baseline t/s | PL x |
|----------|-------------|------|
| code:0 | 56 | 0.96x |
| code:1 | 55 | 1.01x |
| prose:0 | 55 | 1.01x |
| prose:1 | 55 | 0.99x |
| short:0 | 56 | 0.82x |
| short:1 | 55 | 1.00x |
| summarization:0 | 52 | 0.58x |
| summarization:1 | 51 | 0.91x |
| **Average** | **54.4** | **0.91x** |

**Note**: GLM-4.7-Flash has MTP in its BF16 source (`num_nextn_predict_layers: 1`) but uses a different MTP architecture (DeepSeek-style layer-as-MTP-head with `eh_proj`/`enorm`/`hnorm`). Our Qwen3.5-focused MTP decoder doesn't support this format yet. The MTP weights are also not preserved in the mlx-lm conversion. Extracting and using GLM's MTP head would require implementing a new decoder class.

---

## Updated Key Findings

1. **MTP has a minimum model size threshold of ~4B parameters.** Below this, the baseline is too fast for MTP's overhead to pay off. The BF16 MTP head's forward pass takes a fixed minimum time regardless of model size.

2. **MTP works better on higher-precision (slower) models.** 8-bit models see ~10% better MTP ratios than 4-bit models at the same parameter count, because the slower baseline gives MTP more time budget to work with.

3. **Prompt Lookup excels on small, fast models.** At 0.8B-4bit (245 t/s), Prompt Lookup achieves 1.20-1.32x average — much better than MTP. This is because PL has near-zero overhead (just n-gram matching) vs MTP's full head forward pass.

4. **The sweet spot for MTP is 9B-35B at 4-bit quantization.** This gives 65-86 t/s baseline, 1.05-1.09x MTP speedup, and 70-76% acceptance rates.

5. **MTP acceptance rate scales with model size**: 34% at 0.8B → 60% at 2B → 76% at 35B. Larger models have more predictable outputs, making the MTP head's task easier.
