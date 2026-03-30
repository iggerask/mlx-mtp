# MoE Serving Optimizations: Research Synthesis (2024-2026)

**Target model**: Qwen3.5-35B-A3B (35B total params, ~3B active, 64 experts, top-4 routing)
**Target platform**: Apple Silicon (MLX), but CUDA results noted where relevant
**Research date**: March 2026
**Confidence codes**: [H] high, [M] medium, [L] low

---

## Executive Summary

MoE inference research has exploded since late 2023, driven by the popularity of DeepSeek, Qwen-MoE, and Mixtral. Six orthogonal optimization axes have emerged: (1) expert offloading with predictive caching, (2) mixed-precision expert quantization, (3) expert pruning/merging, (4) speculative decoding adapted for MoE, (5) KV cache compression, and (6) attention kernel improvements. For a single-user, Apple Silicon deployment of Qwen3.5-35B-A3B:

- **Highest-impact, lowest-risk**: Expert offloading with predictive prefetch (HOBBIT/MoE-Infinity pattern) combined with 4-bit weight quantization — already practical via mlx-lm today.
- **High-impact, medium effort**: MoE-Spec-style expert budgeting during speculative decoding verification; pairs naturally with MTP heads already in scope for this project.
- **High-impact, requires custom work**: MxMoE mixed-precision assignment (hot experts at higher bit, cold at lower) — no MLX implementation exists but the algorithm is not complex.
- **Useful but limited on unified memory**: KV cache quantization (TurboQuant, KIVI) reduces memory pressure but Apple Silicon's unified DRAM means no CPU offload benefit for KV.
- **Near-term MLX ecosystem**: oMLX and vllm-mlx are bringing paged KV cache, prefix sharing, and SSD caching to Apple Silicon inference; these complement rather than replace mlx-lm.

---

## 1. Expert Offloading and Caching

### 1.1 MoE-Infinity
**Paper**: "MoE-Infinity: Activation-Aware Expert Offloading for Efficient MoE Serving" (arXiv:2401.14361, 2024)
**URL**: https://arxiv.org/html/2401.14361v2

**Mechanism**: Identifies the *temporal locality* of expert activations across sequences. Experts activated for one token are very likely to be activated again for subsequent tokens in the same or nearby sequences. MoE-Infinity uses sequence-level expert activation tracing to build a reuse-aware prefetch schedule. Inactive experts are offloaded to host memory; likely-to-be-needed experts are prefetched asynchronously.

| Dimension | Result |
|---|---|
| Speedup | 4–20x latency reduction vs naive offloading |
| Memory | Fits models that would otherwise require 8x the GPU RAM |
| Quality | Zero degradation (no approximation) |
| Implementation | Requires custom runtime; not native to MLX |
| Apple Silicon | Unified memory means CPU/GPU bandwidth is better than PCIe — concept applies but offload to SSD is the relevant variant |

[H] The 4–20x range is real but context-dependent: longer sequences with high temporal locality get the most benefit.

---

### 1.2 ExpertFlow
**Paper**: "ExpertFlow: Optimized Expert Activation and Token Allocation for Efficient Mixture-of-Experts Inference" (arXiv:2410.17954, Oct 2024)
**URL**: https://arxiv.org/abs/2410.17954

**Mechanism**: Uses a lightweight predictor to forecast routing paths *before* computation begins, enabling real-time error correction in the expert cache. The predictor operates at the token level, watching which experts are selected in earlier layers to predict later-layer selections (exploiting cross-layer locality). Also reorders token processing to maximize cache reuse.

| Dimension | Result |
|---|---|
| Speedup | 2–10x inference speed increase |
| Memory | 75.4% average GPU memory savings; peak 93.72% |
| Cache hit ratio | Up to 91.96% |
| Quality | Near-zero degradation (predictor errors are corrected) |
| Implementation | Not natively available in MLX |
| Apple Silicon | Predictor is cheap; concept directly applicable |

[H] The predictor is the key insight — it shifts expert loading from reactive to proactive.

---

### 1.3 Pre-gated MoE (Microsoft Research, ISCA 2024)
**Paper**: "Pre-gated MoE: An Algorithm-System Co-Design for Fast and Scalable Mixture-of-Expert Inference"
**URL**: https://www.microsoft.com/en-us/research/wp-content/uploads/2024/05/isca24_pregated_moe_camera_ready.pdf

**Mechanism**: Modifies the MoE routing structure so that the gate function at layer N selects experts for layer N+1, not layer N. This eliminates the sequential dependency where you must finish layer N before you know which experts to load for layer N+1. The CPU→GPU expert migration latency is fully overlapped with expert execution.

| Dimension | Result |
|---|---|
| Speedup | Overlaps CPU-GPU transfer entirely with compute; latency ~= compute-only case |
| Memory | No reduction |
| Quality | Requires fine-tuning to adopt the pre-gate architecture; slight task-dependent quality change |
| Implementation | Architecture modification, not a drop-in optimization |
| Apple Silicon | The overlap principle applies even on unified memory when offloading to SSD (oMLX pattern) |

[M] Requires model fine-tuning. For Qwen3.5-35B-A3B as-is, not directly applicable without retraining.

---

### 1.4 AdapMoE
**Paper**: "AdapMoE: Adaptive Sensitivity-based Expert Gating and Management for Efficient MoE Inference" (ICCAD 2024)
**URL**: https://arxiv.org/html/2408.10284v1

**Mechanism**: Three components: (i) sensitivity-based adaptive gating that lets tokens use a dynamic number of experts rather than a fixed top-k, (ii) expert prediction and adaptive prefetching using the insight that activations predict expert selections 2–3 layers ahead, (iii) dynamic programming cache allocation. A small auxiliary predictive layer handles the bootstrapping problem for first-layer prediction.

| Dimension | Result |
|---|---|
| Speedup | 1.35x over all prior expert management methods |
| Memory | Reduces effective active params by allowing tokens to use fewer experts when confident |
| Quality | Adaptive gating introduces a small tunable quality-speed tradeoff |
| Implementation | Requires auxiliary predictive layer; not native in MLX |
| Apple Silicon | Multi-layer lookahead is directly applicable; the auxiliary layer is lightweight |

[M] The 1.35x is modest but additive on top of other optimizations.

---

### 1.5 HOBBIT
**Paper**: "HOBBIT: A Mixed Precision Expert Offloading System for Fast MoE Inference" (arXiv:2411.01433, Nov 2024)
**URL**: https://arxiv.org/abs/2411.01433

**Mechanism**: Combines three hierarchical strategies: (1) token-level: on a cache miss, loads a low-precision (int4) version of the expert instead of waiting for the full-precision expert — 4x faster to load; (2) layer-level: prefetches experts for the next layer while the current layer runs; (3) sequence-level: multi-dimensional eviction policy that considers both recency and cross-request frequency. Built on top of llama.cpp; tested on consumer hardware.

| Dimension | Result |
|---|---|
| Speedup | Up to 9.93x over state-of-the-art MoE offloading |
| Memory | Designed for memory-constrained devices (consumer GPUs, edge) |
| Quality | Cache-miss int4 experts introduce marginal quality loss on misses only |
| Implementation | llama.cpp-based; not in MLX but the algorithm is portable |
| Apple Silicon | High relevance — directly targets the edge/consumer case; the int4 cache-miss trick is ideal for the SSD-offload scenario |

[H] The most directly applicable offloading system for Apple Silicon single-device deployment. The int4 fallback on cache miss is a clever latency-hiding trick.

---

### 1.6 FlashMoE (SSD-focused)
**Paper**: "FlashMoE: Reducing SSD I/O Bottlenecks via ML-Based Cache Replacement for Mixture-of-Experts Inference on Edge Devices" (arXiv:2601.17063, Jan 2026)
**URL**: https://arxiv.org/abs/2601.17063

**Mechanism**: Stores inactive experts on NVMe SSD rather than DRAM. Uses a lightweight ML-based cache replacement policy that combines recency and frequency signals — outperforms both LRU and LFU. Targets the case where even DRAM is insufficient to hold all experts.

| Dimension | Result |
|---|---|
| Speedup | 2.6x over existing MoE inference systems; 51% better cache hit rate vs LRU/LFU |
| Memory | Minimal DRAM footprint — essentially streams experts from SSD |
| Quality | No degradation |
| Implementation | Edge-focused; Jan 2026 paper, no production implementation yet |
| Apple Silicon | Directly relevant — M-series Macs have fast NVMe; MLX already touches this via oMLX |

[M] Newest paper in this space. The ML-based cache policy is a modest but meaningful improvement over naive LRU.

---

### 1.7 SP-MoE and MoE-SpeQ (Offloading + Speculative)
**SP-MoE**: "Speculative Decoding and Prefetching for Accelerating MoE-based Model Inference" (arXiv:2510.10302, Oct 2025)
**URL**: https://arxiv.org/abs/2510.10302

**MoE-SpeQ**: "Speculative Quantized Decoding with Proactive Expert Prefetching and Offloading for Mixture-of-Experts" (arXiv:2511.14102, Nov 2025)
**URL**: https://arxiv.org/abs/2511.14102

**Mechanism (SP-MoE)**: Combines speculative decoding with expert prefetching. The key insight: when using speculative decoding, the draft model's expert activation pattern strongly predicts the target model's pattern. SP-MoE uses this correspondence to prefetch target model experts while running the draft model. A cutoff-layer policy prevents over-prefetching; asynchronous I/O threads hide loading latency.

**Mechanism (MoE-SpeQ)**: A 4-bit quantized draft model predicts expert selections for full-precision parent with >90% accuracy. An "Amortization Roofline Model" adaptive governor tunes speculation depth to hardware bandwidth. Combines speculative decoding, expert quantization, and proactive prefetching.

| Dimension | SP-MoE | MoE-SpeQ |
|---|---|---|
| Speedup | 1.07–3.5x TPOT over SOTA | 2.34–4.8x over SOTA offloading |
| Memory | Higher peak during verification | Draft model is small (4-bit) |
| Quality | No degradation | Marginal from int4 draft |
| Implementation | Not in MLX | Not in MLX |
| Apple Silicon | Directly applicable to SSD-offload scenario | Same; draft-model approach maps well |

[H] MoE-SpeQ is particularly relevant for this project: the draft model (could be the MTP head) predicts expert selections, enabling proactive prefetch. This is a natural integration with MTP speculative decoding work.

---

## 2. Expert Quantization

### 2.1 MxMoE (ICML 2025)
**Paper**: "MxMoE: Mixed-precision Quantization for MoE with Accuracy and Performance Co-Design" (arXiv:2505.05799, ICML 2025)
**URL**: https://arxiv.org/html/2505.05799v1
**Code**: https://github.com/cat538/MxMoE

**Mechanism**: Rather than assigning one bit-width per expert (as earlier work did), MxMoE assigns different bit-widths at the *linear block* level within each expert. Two key insights: (1) linear blocks differ in quantization sensitivity within a single expert; (2) high-activation-frequency experts should be at higher precision since errors accumulate more. The bit-width assignment is formulated as an Integer Linear Program (ILP) and solved once per model. Generates mixed-precision GroupGEMM kernels via template-based generation.

| Dimension | Result |
|---|---|
| Speedup | Up to 3.4x over full precision |
| Quality | 2.4 lower Wikitext-2 perplexity than GPTQ at 2.25-bit average |
| Memory | Roughly proportional to average bit-width (e.g., ~50% reduction at 4-bit avg) |
| Implementation | CUDA-focused; no MLX port |
| Apple Silicon | ILP-based assignment could run once and results applied; custom MLX GroupGEMM needed |

[H] The linear-block granularity insight is validated at ICML. For Qwen3.5-35B-A3B with 64 experts, some experts see orders of magnitude more traffic than others — this is highly exploitable.

---

### 2.2 HOBBIT Mixed-Precision (also in section 1.5)

As noted above, HOBBIT's key contribution to quantization is specifically for the *offload path*: cache-miss experts are loaded at int4 rather than fp16, giving 4x faster load times for cold experts while hot (cached) experts stay at full precision. This is a practical approximation of the "hot = high precision, cold = low precision" principle.

---

### 2.3 MoPEQ
**Paper**: "MoPEQ: Mixture of Mixed Precision Quantized Experts" (arXiv:2509.02512, Sep 2025)
**URL**: https://arxiv.org/html/2509.02512v1

**Mechanism**: Post-training quantization that assigns optimal bit-width to each expert by analyzing per-expert sensitivity via Hessian trace approximation. Similar experts are clustered together. Unlike MxMoE, operates at expert granularity rather than linear-block granularity.

| Dimension | Result |
|---|---|
| Speedup | Not primary focus (accuracy-oriented) |
| Quality | Maintains performance while reducing average bit-width |
| Memory | Proportional to average bit-width |
| Implementation | Research code only |
| Apple Silicon | Concept applies; requires custom mixed-bitwidth kernel |

[M] The Hessian sensitivity approach is well-understood; the expert-level granularity is coarser than MxMoE but simpler to implement.

---

### 2.4 MoEQuant
**Paper**: "MoEQuant: Enhancing Quantization for Mixture-of-Experts Large Language Models" (arXiv:2505.03804, 2025)
**URL**: https://arxiv.org/pdf/2505.03804

**Mechanism**: Two techniques: Expert Block-wise Subgroup Search (EBSS) searches for optimal quantization groupings within experts; Activation-Guided Quantization (AGQ) uses activation statistics to guide precision decisions.

| Dimension | Result |
|---|---|
| Speedup | Enables lower bit-width without quality loss |
| Quality | Improved vs standard quantization at same bit-width |
| Memory | Depends on target bit-width |
| Implementation | Research code |
| Apple Silicon | Applicable; uses standard PTQ approach |

[M] Solid incremental improvement on standard quantization; EBSS is the more novel idea.

---

### 2.5 Current Practical State on MLX

The mlx-community already hosts Qwen3.5-35B-A3B at 4-bit, 5.5-bit, and 8-bit. The 5.5-bit model achieves ~96 tok/s at 1000 tokens on a capable M-series machine with ~22.3 GiB memory. **This is already using global uniform quantization** — the research above points toward per-expert or per-block mixed precision as the next step, but requires custom MLX kernel work.

---

## 3. Expert Pruning and Merging

### 3.1 REAP: Router-weighted Expert Activation Pruning
**Paper**: "REAP the Experts: Why Pruning Prevails for One-Shot MoE compression" (arXiv:2510.13999, Oct 2025)
**URL**: https://arxiv.org/abs/2510.13999

**Mechanism**: One-shot (no fine-tuning) expert pruning that considers both router gate-values and expert activation norms to minimize reconstruction error. Key finding: for *generative* tasks, pruning outperforms merging. Merging techniques suffer from an irreducible error due to loss of routing specialization.

| Dimension | Result |
|---|---|
| Speedup | Proportional to reduction ratio (50% pruning → ~2x fewer expert ops) |
| Quality | Best-in-class at 50% compression on generative benchmarks; merging shows >20% MMLU drop at same ratio |
| Memory | Linear reduction with pruning ratio |
| Implementation | One-shot; applies to any MoE |
| Apple Silicon | Directly applicable; no training required |

[H] Important result: merging is competitive on discriminative tasks but pruning wins for generation. For a coding/chat use case, prefer pruning. The one-shot nature makes this practical without retraining infrastructure.

---

### 3.2 PuzzleMoE
**Paper**: "PuzzleMoE: Efficient Compression of Large Mixture-of-Experts Models via Sparse Expert Merging and Bit-packed inference" (arXiv:2511.04805, Nov 2025)
**URL**: https://arxiv.org/abs/2511.04805

**Mechanism**: Training-free compression that identifies element-wise weight redundancy and specialization across experts, then merges only the redundant components while preserving specialized parts (hence "sparse merging"). Combines with bit-packed inference for further speedup. Evaluated on Qwen3-MoE-30B-A3B specifically.

| Dimension | Result |
|---|---|
| Speedup | Up to 1.28x inference speedup |
| Quality | Best accuracy at 50% compression; +16.7% MMLU vs prior methods |
| Memory | 50% compression → ~50% parameter reduction |
| Implementation | Training-free; code likely available |
| Apple Silicon | Directly applicable; no training required; evaluated on Qwen3 architecture |

[H] The Qwen3-MoE-30B-A3B evaluation is directly relevant to Qwen3.5-35B-A3B. The 1.28x speedup is modest but the memory savings at 50% compression are significant for fitting in M-series memory.

---

### 3.3 MoE-I2
**Paper**: "MoE-I2: Compressing Mixture of Experts Models through Inter-Expert Pruning and Intra-Expert Low-Rank Decomposition" (EMNLP 2024 Findings)
**URL**: https://aclanthology.org/2024.findings-emnlp.612/

**Mechanism**: Two-stage: first prune entire experts (inter-expert), then apply low-rank decomposition to the surviving experts (intra-expert). Requires fine-tuning to recover performance after compression.

| Dimension | Result |
|---|---|
| Speedup | Reduces both expert count and per-expert compute |
| Quality | Requires fine-tuning; recovers well with tuning |
| Memory | Double reduction: fewer experts + smaller per-expert weights |
| Implementation | Requires fine-tuning infrastructure |
| Apple Silicon | Algorithm is platform-agnostic; fine-tuning on Apple Silicon is feasible via MLX for smaller models |

[M] Most powerful compression option but the fine-tuning requirement raises the barrier significantly for a personal-scale deployment.

---

### 3.4 DiEP: Differentiable Expert Pruning
**Paper**: "DiEP: Adaptive Mixture-of-Experts Compression through Differentiable Expert Pruning" (arXiv:2509.16105, Sep 2025)
**URL**: https://arxiv.org/html/2509.16105v1

**Mechanism**: Learns per-expert importance scores via differentiable optimization, allowing non-uniform pruning ratios across layers. More nuanced than fixed-ratio methods.

| Dimension | Result |
|---|---|
| Speedup | Layer-adaptive; can be higher than uniform pruning |
| Quality | Better than uniform pruning at same compression ratio |
| Memory | Non-uniform reduction |
| Implementation | Requires training pass |
| Apple Silicon | Feasible but requires training infrastructure |

[M] The non-uniform pruning is theoretically superior but requires more compute to apply.

---

### 3.5 Key Recommendation for Qwen3.5-35B-A3B

At 64 experts with top-4 routing, there is significant redundancy. A practical approach without retraining:

1. Apply REAP or PuzzleMoE at 25-33% compression to remove the least-activated experts
2. Quantize remaining experts to 4-bit (uniform, or MxMoE-style mixed if MLX kernels support it)
3. This could reduce memory from ~22 GiB (5.5-bit) to ~12–15 GiB at similar quality

---

## 4. Speculative Decoding for MoE

### 4.1 EAGLE-3 (NeurIPS 2025)
**Paper**: "EAGLE-3: Scaling up Inference Acceleration of Large Language Models via Training-Time Test" (arXiv:2503.01840, Mar 2025)
**URL**: https://arxiv.org/html/2503.01840v1
**Code**: https://github.com/SafeAILab/EAGLE

**Mechanism**: Unlike EAGLE-1/2 which predict features, EAGLE-3 predicts tokens directly. Uses multi-layer feature fusion via "training-time test" — the draft model is trained to see the same multi-layer context it will see at inference. Abandons feature prediction entirely. Supports Mixtral 8x7B.

| Dimension | Result |
|---|---|
| Speedup | 3–6.5x over vanilla autoregressive; 20–40% improvement over EAGLE-2 |
| Memory | Adds ~0.28B parameter draft model for 8x7B MoE |
| Quality | Lossless (same output distribution as target) |
| Implementation | Not native in MLX; no Apple Silicon port confirmed |
| Apple Silicon | Architecture is platform-agnostic; the draft model would need MLX porting |

[H] EAGLE-3 is the current state-of-the-art for token-level speculative decoding. For MoE specifically, the draft model overhead (0.28B for 8x7B MoE) is modest relative to the gains. The token-prediction approach is also conceptually aligned with MTP heads.

---

### 4.2 MoE-Spec
**Paper**: "MoE-Spec: Expert Budgeting for Efficient Speculative Decoding" (arXiv:2602.16052, Feb 2026)
**URL**: https://arxiv.org/html/2602.16052

**Mechanism**: Addresses a critical problem: speculative decoding with large draft trees activates *many* unique experts during the verification pass, increasing memory pressure and eliminating the speedup gains. MoE-Spec enforces a fixed expert budget during verification — only the top-K most-utility experts are loaded for verification; the long tail is dropped. Training-free; integrates with any speculative decoding pipeline.

Evaluated on Qwen3-30B-A3B, OLMoE-1B-7B, and Mixtral-8x7B across math, code, and summarization.

| Dimension | Result |
|---|---|
| Speedup | 10–30% higher throughput than EAGLE-3 on MoE architectures |
| Memory | Lower peak memory during verification vs naive speculative decoding |
| Quality | Slight quality tradeoff tunable via budget size |
| Implementation | Training-free; integrates with EAGLE or any SD pipeline |
| Apple Silicon | No CUDA-specific operations; directly applicable |

[H] Directly evaluated on Qwen3-30B-A3B, the closest available proxy for Qwen3.5-35B-A3B. The expert budget concept is critical for MoE-specific speculative decoding and resolves the "more speculation = more expert loads = no speedup" problem. **High priority for MLX implementation.**

---

### 4.3 SP-MoE (also in section 1.7)

As described above, SP-MoE prefetches target model experts while running the draft model, exploiting the structural correspondence between draft and target expert activation patterns. The 1.07–3.5x TPOT speedup is on top of speculative decoding gains.

---

### 4.4 MoE-SpeQ (also in section 1.7)

The quantized draft model (4-bit, predicts expert selections at >90% accuracy) is directly relevant to the MTP project: the MTP head already produces next-token predictions; extending it to also predict likely expert activations enables proactive expert prefetching.

---

### 4.5 Medusa
**Paper**: "Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads" (arXiv:2401.10774, 2024)
**URL**: https://arxiv.org/abs/2401.10774

**Mechanism**: Adds multiple LM heads to the target model to predict tokens at positions t+1, t+2, ..., t+k in parallel. Uses tree-based attention for joint verification. No separate draft model needed.

| Dimension | Result |
|---|---|
| Speedup | 2.2–3.6x over vanilla decoding |
| Memory | Adds k small LM heads (negligible) |
| Quality | Slight quality loss from tree sampling |
| Implementation | Requires adding and fine-tuning prediction heads |
| Apple Silicon | Platform-agnostic |

[M] Medusa is simpler than EAGLE but weaker. For Qwen3.5-35B-A3B, the existing MTP heads in the architecture already serve the same function — Medusa heads would be redundant.

---

### 4.6 Lookahead Decoding
**URL**: https://github.com/hao-ai-lab/LookaheadDecoding

**Mechanism**: No separate model. Maintains a 2D sliding window — a "lookahead branch" that generates n-gram candidates in parallel with the main decode, and a "verification branch" that checks them. Requires no training, no auxiliary model.

| Dimension | Result |
|---|---|
| Speedup | 1.5–2.1x (weaker than EAGLE; no training advantage) |
| Memory | Minimal overhead |
| Quality | Lossless |
| Implementation | Lightweight; relatively easy to port |
| Apple Silicon | Directly applicable |

[M] Lookahead is weaker than EAGLE/MTP-based approaches but requires zero training and is easy to implement. Could serve as a baseline or fallback.

---

### 4.7 Cascade Speculative Drafting
**Paper**: "Cascade Speculative Drafting for Even Faster LLM Inference" (arXiv:2312.11462)
**URL**: https://arxiv.org/html/2312.11462v5

**Mechanism**: Hierarchical draft chain: tiny model → small model → medium model → target. Each level filters tokens before the next level verifies, amortizing verification cost across the chain.

| Dimension | Result |
|---|---|
| Speedup | Exceeds single-level speculative decoding when model chain is available |
| Memory | Requires multiple models resident |
| Quality | Lossless |
| Implementation | Requires multiple appropriately-sized models |
| Apple Silicon | Memory constraint makes multi-model approaches challenging on-device |

[L] Less practical for Apple Silicon single-device use case where RAM is the binding constraint.

---

### 4.8 Speculating Experts (Mar 2026)
**Paper**: "Speculating Experts Accelerates Inference for Mixture-of-Experts" (arXiv:2603.19289, Mar 2026)
**URL**: https://arxiv.org/html/2603.19289

A very recent (March 2026) paper specifically on predicting which experts will be needed before the gate decision is made, enabling prefetching to overlap with computation. Specific results not yet widely reported.

[L] Too new to assess confidently; worth monitoring.

---

## 5. KV Cache Optimizations

*Note: Apple Silicon's unified memory means there is no PCIe bandwidth bottleneck for CPU/GPU KV offloading — the main benefit of KV compression on M-series is freeing up DRAM for more context or larger models, not reducing transfer latency.*

### 5.1 TurboQuant (ICLR 2026, Google Research + NYU)
**URL**: https://github.com/0xSero/turboquant
**PyTorch ref impl**: https://github.com/tonbistudio/turboquant-pytorch
**Prior memory entry**: `/Users/ingemarrask/.claude/agent-memory/research-synthesizer/turboquant_mlx.md`

**Mechanism**: Two-stage KV cache quantization. Stage 1: apply a fixed random orthogonal rotation to each K/V vector, then quantize using Lloyd-Max codebook (1D k-means on Beta density — solved once per bit-width). Stage 2: take the residual quantization error, project through a random Gaussian matrix (Johnson-Lindenstrauss), and store only the sign bit. This sign sketch corrects bias in attention score estimates mathematically, making estimates unbiased.

| Dimension | Result |
|---|---|
| Speedup | Up to 8x attention-logit speedup on H100 at 4-bit |
| Memory | 6x+ KV memory reduction at 3-4 bits |
| Quality | <0.1 perplexity degradation (near-lossless) |
| Implementation | PyTorch reference; llama.cpp fork; community MLX evidence |
| Apple Silicon | Reduces DRAM pressure, freeing memory for longer context or larger models |

[H] The rotation trick is the key innovation — it eliminates outlier distributions in K/V vectors without calibration data. The no-calibration property is important for deployment simplicity. MLX community implementation exists (per prior memory).

---

### 5.2 KIVI (ICML 2024)
**Paper**: "KIVI: A Tuning-Free Asymmetric 2-bit Quantization for KV Cache"
**URL**: https://github.com/jy-yuan/KIVI

**Mechanism**: Keys and values have different statistical distributions. KIVI quantizes keys per-channel and values per-token to 2 bits, with a small number of "residual" fp16 entries for the current sliding window. Tuning-free (no calibration).

| Dimension | Result |
|---|---|
| Speedup | Enables ~4x longer context in same memory |
| Memory | ~4x reduction vs fp16 KV cache |
| Quality | <0.3 perplexity on LLaMA-2 at 2-bit |
| Implementation | CUDA-focused; relatively simple algorithm |
| Apple Silicon | Algorithm is directly portable to MLX |

[H] KIVI is mature (ICML 2024), simple, and the asymmetric per-channel/per-token insight is well-validated. Good candidate for MLX implementation.

---

### 5.3 KVQuant (NeurIPS 2024)
**Paper**: "KVQuant: Towards 10 Million Context Length LLM Inference with KV Cache Quantization"
**URL**: https://arxiv.org/abs/2401.18079

**Mechanism**: Combines per-channel key quantization, pre-RoPE key quantization, non-uniform quantization using calibration data, and dense-and-sparse per-vector representation. More complex than KIVI but achieves lower degradation at 3-bit.

| Dimension | Result |
|---|---|
| Speedup | Enables extremely long contexts (tested to 10M tokens) |
| Memory | 3-bit KV: ~5x reduction; <0.1 perplexity degradation |
| Quality | Essentially lossless at 3-bit |
| Implementation | Complex; calibration required |
| Apple Silicon | The pre-RoPE key quantization insight is valuable; full implementation is complex |

[M] KVQuant is more powerful than KIVI but requires calibration and is harder to implement. For practical MLX use, TurboQuant or KIVI are better starting points.

---

### 5.4 ShadowKV
**Paper**: "ShadowKV: KV Cache in Shadows for High-Throughput Long-Context LLM Inference" (arXiv:2410.21465, 2024)
**URL**: https://arxiv.org/abs/2410.21465

**Mechanism**: Stores the *low-rank* key cache on GPU; offloads the full value cache to CPU. During attention, reconstructs approximate keys for coarse selection, then fetches only selected values. Multi-turn capable via low-rank subspace sharing.

| Dimension | Result |
|---|---|
| Speedup | Higher batch throughput at long context |
| Memory | Significant value cache offload; key cache stays compressed on GPU |
| Quality | Recall/selection adds ~73% of total latency per study |
| Implementation | Not in MLX |
| Apple Silicon | CPU/GPU offload on unified memory is fast; but the 73% latency overhead from recall is a concern |

[M] The multi-turn stability (SnapKV fails at turn 2, ShadowKV doesn't) is a strong point for agentic use cases. However, the latency overhead from CPU recall is significant.

---

### 5.5 H2O: Heavy-Hitter Oracle
**Mechanism**: Eviction policy that scores tokens by cumulative attention mass — "heavy hitters" are tokens that have received the most attention across previous queries. These are kept; the rest are evicted. Uses both recency and heavy-hitter signals.

| Dimension | Result |
|---|---|
| Speedup | Enables long context in fixed KV budget |
| Memory | Configurable; typically 20-50% of full KV |
| Quality | Task-dependent; degrades on needle-in-haystack retrieval |
| Implementation | Simple to implement; no calibration |
| Apple Silicon | Directly applicable |

[M] H2O is simple and works well for most generation tasks but fails on retrieval tasks where important tokens are early in context. For agentic coding use cases with long tool outputs, H2O may evict critical earlier context.

---

### 5.6 SnapKV / PyramidKV / AdaKV
**SnapKV**: Prunes tokens using attention statistics gathered during prefill; then uses a fixed compressed cache for decode.
**PyramidKV**: Applies SnapKV-style pruning but with a pyramid strategy — more tokens preserved in lower attention layers, more aggressive compression in higher layers.
**AdaKV**: Per-head adaptive budget allocation rather than uniform budget per layer.

| Method | Key Feature | Quality |
|---|---|---|
| SnapKV | Fast prefill-time pruning | Drops significantly after first turn |
| PyramidKV | Layer-adaptive compression | Better than SnapKV on long docs |
| AdaKV | Head-adaptive budget | Best fine-grained control |

[M] SnapKV's multi-turn failure (per the ShadowKV benchmark comparison) is a serious limitation for agentic use. PyramidKV is the safer choice if these methods are used.

---

### 5.7 MiniKV
**Paper**: "MiniKV: Pushing the Limits of 2-Bit KV Cache via Compression and System Co-Design" (arXiv:2411.18077, 2025)
**URL**: https://arxiv.org/html/2411.18077v3

**Mechanism**: Combines 2-bit KV quantization with token eviction in a unified framework. Achieves 86% KV cache compression while retaining comparable accuracy on LongBench; 48% higher throughput than baselines on A100 with prompts up to 44K tokens.

| Dimension | Result |
|---|---|
| Speedup | 48% higher throughput vs baselines |
| Memory | 86% KV reduction |
| Quality | Comparable to baseline on LongBench |
| Implementation | Research code; CUDA-focused |
| Apple Silicon | Algorithm applicable; no MLX port |

[M] MiniKV's joint quantization+eviction approach is elegant. The 86% compression is aggressive; worth monitoring for quality on reasoning tasks.

---

### 5.8 oMLX SSD KV Caching (Practical Apple Silicon)
**URL**: https://github.com/jundot/omlx

**Mechanism**: Not a compression method — rather, persists KV cache blocks to NVMe SSD across requests. Prefix sharing with copy-on-write blocks. Hot tier in DRAM, cold tier on SSD. For agentic workflows where the same system prompt / tool definitions appear repeatedly, this reduces TTFT dramatically.

| Dimension | Result |
|---|---|
| Speedup | TTFT drops from 30–90s to 1–3s on cached prefixes; 5.8x speedup on text prefix caching |
| Memory | No DRAM reduction; SSD acts as overflow |
| Quality | Lossless (exact KV reuse) |
| Implementation | Available now; built on MLX |
| Apple Silicon | Specifically designed for M-series Macs |

[H] For agentic single-user use cases, SSD KV persistence is probably more impactful than compression in practice. The TTFT improvement from cached system prompts is immediately felt. **Highest practical priority for Apple Silicon.**

---

## 6. Attention Kernel Optimizations

### 6.1 FlashAttention Family (CUDA)
**FlashAttention-3**: Further optimizes for H100 with async scheduling and low-precision.

Apple Silicon does not have an official Dao-AI-Lab FlashAttention port (https://github.com/Dao-AILab/flash-attention/issues/977 — open issue).

---

### 6.2 Metal FlashAttention (Apple Silicon)
**Metal FlashAttention 2.0/2.5**: Community port by Draw Things / Philip Turner.
**URL**: https://engineering.drawthings.ai/p/metal-flashattention-2-0-pushing-forward-on-device-inference-training-on-apple-silicon-fe8aac1ab23c
**PyPI**: https://pypi.org/project/mlx-mfa/

Key facts:
- Version 2.5 with Neural Accelerators delivers up to **4.6x performance improvement on M5 over M4**
- Available as `mlx-mfa` package
- Supports packed/chunked/prefix/speculative inference as explicit runtime capabilities
- Enables 5-second 480p video generation on M5 iPad (16 GiB)

| Dimension | Result |
|---|---|
| Speedup | 4.6x on M5 vs M4; significant vs naive Metal attention |
| Memory | Fused kernel reduces peak memory (no O(n^2) attention materialization) |
| Quality | Exact (numerically equivalent to standard attention) |
| Implementation | Available via mlx-mfa package |
| Apple Silicon | Specifically designed for Metal / Apple Silicon |

[H] Metal FlashAttention is available today and should be the baseline attention implementation for any MLX inference project. The mlx-mfa package provides drop-in integration.

---

### 6.3 Paged Attention / PagedKV in MLX Ecosystem
**Issue in mlx**: https://github.com/ml-explore/mlx/issues/2955
**vllm-mlx**: https://github.com/raullenchai/vllm-mlx — "~77% throughput improvement on Qwen 30B 4-bit" with PagedAttention kernels on Metal

The upstream mlx-lm does not yet have native paged attention as of March 2026 (PR #2955 is open). However:
- **vllm-mlx** has a native MLX backend with paged attention and reports ~400+ tok/s on Apple Silicon
- **oMLX** has a PagedCacheManager with GPU block-based management + SSD overflow

For single-user single-request scenarios (the typical agentic case), paged attention's primary benefit is multi-request batching — less relevant. But for long contexts, paged attention eliminates KV cache fragmentation.

---

### 6.4 Chunked Prefill
**Mechanism**: Rather than prefilling the entire prompt in one pass (which creates a compute-heavy step that blocks decode), split the prefill into fixed-size chunks and interleave with decode steps from ongoing requests. For single-user, this means long prompt prefill doesn't block the first decode step by as much.

| Dimension | Result |
|---|---|
| Speedup | Reduces TTFT for first token on long prompts; marginal for short prompts |
| Memory | No reduction |
| Quality | Lossless |
| Implementation | vllm-mlx and oMLX both support this |
| Apple Silicon | Relevant for agentic workloads with long tool-call outputs being processed |

[M] For single-user agentic use, chunked prefill reduces perceived latency on long contexts. The benefit grows with context length.

---

### 6.5 Ring Attention
**Mechanism**: Distributes long-context attention across multiple devices in a ring topology. Not applicable to single-device Apple Silicon deployments.

[L] Not relevant for single-device use case.

---

## 7. Continuous Batching and Prefill for Single-User Agentic Use

For a single-user agentic deployment on Apple Silicon, the inference profile is:
- Long system prompts + tool definitions (repeated across turns)
- Variable-length tool-call outputs (potentially large JSON)
- Decode-heavy responses (code, explanations)
- Sequential turns with frequent context reuse

**Most impactful optimizations for this profile:**

1. **SSD KV prefix caching** (oMLX) — eliminates repeated prefill of system prompt; 5–30x TTFT reduction on cached turns
2. **Chunked prefill** — prevents long tool outputs from blocking decode start
3. **Metal FlashAttention** (mlx-mfa) — faster attention on every step
4. **Expert prefetching with HOBBIT/MoE-SpeQ approach** — hides expert load latency for MoE models specifically
5. **MoE-Spec expert budgeting** — reduces expert activation overhead during speculative verification

**Less impactful for single-user:**
- Continuous batching (batching across users) — no benefit for single user
- Throughput-oriented optimizations — single-user is latency-bound not throughput-bound

---

## 8. Integration Opportunities for mlx-mtp Project

Given the existing work on MTP speculative decoding for Qwen3.5-35B-A3B:

### Highest priority integrations

| Optimization | Why It Fits | Effort |
|---|---|---|
| MoE-Spec expert budgeting | Fixes the "large draft tree = too many expert loads" problem that otherwise degrades MTP speedup on MoE | Medium |
| MTP heads as expert predictors | The MTP draft prediction can double as expert selection prediction (MoE-SpeQ insight), enabling proactive prefetch | Medium |
| Metal FlashAttention (mlx-mfa) | Drop-in improvement to all attention ops | Low |
| oMLX SSD prefix caching | Complements mlx-lm for agentic workloads | Low (external tool) |

### Medium priority

| Optimization | Why It Fits | Effort |
|---|---|---|
| HOBBIT int4 cache-miss expert trick | For SSD-offload scenarios, loading cold experts at int4 gives 4x faster load | Medium |
| MxMoE mixed precision | Hot experts at 8-bit, cold at 4-bit — requires custom MLX GroupGEMM | High |
| PuzzleMoE 25-33% pruning | Reduces model size without quality loss, leaves more headroom for KV cache | Medium (run once, save pruned model) |

### Lower priority (diminishing returns)

| Optimization | Notes |
|---|---|
| TurboQuant KV compression | Useful but KV cache is smaller than weights for typical context lengths |
| KIVI 2-bit KV | Aggressive; may hurt coding/reasoning quality |
| Expert pruning + fine-tuning | No fine-tuning infrastructure in scope |

---

## References

### Expert Offloading and Caching

- [MoE-Infinity (arXiv:2401.14361)](https://arxiv.org/html/2401.14361v2) — Activation-aware expert offloading, 4–20x latency reduction
- [ExpertFlow (arXiv:2410.17954)](https://arxiv.org/abs/2410.17954) — Predictive routing, 2–10x speedup, 75% GPU memory savings
- [Pre-gated MoE, ISCA 2024](https://www.microsoft.com/en-us/research/wp-content/uploads/2024/05/isca24_pregated_moe_camera_ready.pdf) — Eliminates gate-compute sequential dependency
- [AdapMoE (arXiv:2408.10284)](https://arxiv.org/html/2408.10284v1) — Sensitivity-based gating + multi-layer lookahead prefetch
- [HOBBIT (arXiv:2411.01433)](https://arxiv.org/abs/2411.01433) — Mixed-precision offloading; 9.93x speedup; int4 cache-miss fallback
- [FlashMoE (arXiv:2601.17063)](https://arxiv.org/abs/2601.17063) — ML-based SSD cache replacement; 2.6x speedup; Jan 2026
- [SP-MoE (arXiv:2510.10302)](https://arxiv.org/abs/2510.10302) — Speculative decoding + expert prefetching; 1.07–3.5x TPOT
- [MoE-SpeQ (arXiv:2511.14102)](https://arxiv.org/abs/2511.14102) — Quantized draft for expert prediction; up to 4.8x speedup
- [Speculating Experts (arXiv:2603.19289)](https://arxiv.org/html/2603.19289) — Mar 2026; expert prediction before gate decision
- [Awesome MoE Inference](https://github.com/MoE-Inf/awesome-moe-inference) — Curated paper list

### Expert Quantization

- [MxMoE, ICML 2025 (arXiv:2505.05799)](https://arxiv.org/html/2505.05799v1) — Linear-block-level mixed precision; 3.4x speedup; 2.4 lower PPL than GPTQ
- [MoPEQ (arXiv:2509.02512)](https://arxiv.org/html/2509.02512v1) — Hessian-based per-expert bit-width assignment
- [MoEQuant (arXiv:2505.03804)](https://arxiv.org/pdf/2505.03804) — EBSS + AGQ for MoE quantization
- [HOBBIT (arXiv:2411.01433)](https://arxiv.org/abs/2411.01433) — int4 cache-miss experts in offload scenario

### Expert Pruning and Merging

- [REAP (arXiv:2510.13999)](https://arxiv.org/abs/2510.13999) — Router-weighted pruning; one-shot; pruning beats merging for generation
- [PuzzleMoE (arXiv:2511.04805)](https://arxiv.org/abs/2511.04805) — Sparse merging + bit-packed inference; 1.28x speedup; evaluated on Qwen3-MoE-30B-A3B
- [MoE-I2, EMNLP 2024](https://aclanthology.org/2024.findings-emnlp.612/) — Inter-expert pruning + intra-expert LoRA; requires fine-tuning
- [DiEP (arXiv:2509.16105)](https://arxiv.org/html/2509.16105v1) — Differentiable adaptive pruning

### Speculative Decoding

- [EAGLE-3, NeurIPS 2025 (arXiv:2503.01840)](https://arxiv.org/html/2503.01840v1) — 3–6.5x speedup; MoE support (Mixtral)
- [MoE-Spec (arXiv:2602.16052)](https://arxiv.org/html/2602.16052) — Expert budgeting for speculative verification; 10–30% over EAGLE-3 on MoE; Feb 2026
- [Medusa (arXiv:2401.10774)](https://arxiv.org/abs/2401.10774) — Multiple decoding heads; 2.2–3.6x speedup
- [Cascade Speculative Drafting (arXiv:2312.11462)](https://arxiv.org/html/2312.11462v5) — Hierarchical draft chain

### KV Cache

- [TurboQuant (ICLR 2026)](https://github.com/tonbistudio/turboquant-pytorch) — Rotation + codebook; 6x+ compression; no calibration needed
- [KIVI, ICML 2024](https://github.com/jy-yuan/KIVI) — Asymmetric 2-bit; tuning-free
- [KVQuant, NeurIPS 2024 (arXiv:2401.18079)](https://arxiv.org/abs/2401.18079) — <0.1 PPL at 3-bit; complex
- [ShadowKV (arXiv:2410.21465)](https://arxiv.org/abs/2410.21465) — Low-rank key + offloaded values; multi-turn stable
- [MiniKV (arXiv:2411.18077)](https://arxiv.org/html/2411.18077v3) — 86% KV compression; joint quant+eviction
- [oMLX](https://github.com/jundot/omlx) — SSD KV persistence for Apple Silicon; 5–30x TTFT reduction on cached prefixes

### Attention Kernels / Serving

- [Metal FlashAttention 2.0/2.5](https://engineering.drawthings.ai/p/metal-flashattention-2-0-pushing-forward-on-device-inference-training-on-apple-silicon-fe8aac1ab23c) — Apple Silicon; 4.6x on M5; available as mlx-mfa
- [mlx-mfa on PyPI](https://pypi.org/project/mlx-mfa/) — Installable package
- [mlx paged attention issue #2955](https://github.com/ml-explore/mlx/issues/2955) — Open feature request
- [vllm-mlx](https://github.com/raullenchai/vllm-mlx) — OpenAI-compatible server; paged attention on Metal; ~400+ tok/s
- [KV Cache Optimization Survey (arXiv:2603.20397)](https://arxiv.org/html/2603.20397) — March 2026 survey paper

---

## Gaps and Limitations

1. **No Apple Silicon-specific MoE offloading benchmarks**: All HOBBIT, ExpertFlow, and MoE-Infinity benchmarks are on CUDA hardware. Apple Silicon's unified memory changes the tradeoff (no PCIe bottleneck for CPU/GPU, but SSD is the relevant offload target).

2. **Qwen3.5-35B-A3B architecture specifics**: Most papers benchmark on Mixtral-8x7B, DeepSeek-V2, or Qwen1.5-MoE. PuzzleMoE's Qwen3-MoE-30B-A3B results are the closest available proxy. Architecture differences (64 experts vs 8 or 256) affect offloading behavior.

3. **MoE-Spec not yet available in MLX**: The most critical optimization for MTP+MoE combination (expert budgeting during speculative verification) has no MLX implementation. The algorithm is described in sufficient detail to implement.

4. **Mixed-precision MLX kernels**: Custom GroupGEMM with mixed bitwidth per expert/block does not exist in MLX. This is the main barrier to implementing MxMoE or HOBBIT-style precision switching in the MLX ecosystem.

5. **Quality of reasoning models under compression**: Most compression benchmarks use perplexity or MMLU. Qwen3.5-35B-A3B's strong reasoning capabilities may degrade differently under expert pruning or aggressive KV compression — the "Hold Onto That Thought" 2025 paper flags that KV compression methods that work on standard benchmarks can fail on reasoning chains.
