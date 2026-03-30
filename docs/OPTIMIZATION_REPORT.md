# Optimization Report: Qwen3.5-35B-A3B for Agentic Workflows on Apple Silicon

**Model**: Qwen3.5-35B-A3B (35B total, ~3B active, 64 experts, top-4 routing, hybrid GatedDeltaNet + Attention)
**Platform**: Apple M2 Max 48GB / Apple Silicon (MLX)
**Date**: 2026-03-29

---

## Executive Summary

We researched and evaluated optimizations for serving Qwen3.5-35B-A3B in agentic workloads (tool-calling, long multi-turn sessions, structured JSON output) on Apple Silicon via MLX. The optimizations span six axes: KV cache compression, speculative decoding, MoE expert management, structured output acceleration, context compression, and sparse attention.

### Priority Matrix

| Priority | Optimization | Expected Impact | Implementation Effort | Risk |
|----------|-------------|----------------|----------------------|------|
| 🟢 **P0** | Naive INT8 KV cache (≥8K context) | **2.44x decode speed at 16K** | Zero (built-in mlx-lm) | None |
| 🟢 **P0** | Prefix/KV cache sharing | **2-5x TTFT** for repeated prompts | Medium (serving layer) | None |
| 🟡 **P1** | SuffixDecoding for agentic patterns | **2.5-5.3x decode speed** | Medium (suffix tree + verification) | None |
| 🟡 **P1** | EAGLE-style tree drafting via MTP head | **3-5x decode speed** | Medium (tree attention in MLX) | Low |
| 🟡 **P1** | MoE-Spec expert budgeting | **+16-30% on top of speculation** | Medium (routing hooks) | Low |
| 🟡 **P1** | Parallel tool execution (LLMCompiler) | **3.7x end-to-end latency** | Low (orchestration only) | Low |
| 🟠 **P2** | MxMoE mixed-precision experts | **~2x memory reduction** | High (custom kernels) | Medium |
| 🟠 **P2** | Sparse attention (DUOAttention) | **~50% KV cache at 32K+** | High (head classification + kernels) | Medium |
| 🟠 **P2** | XGrammar structured output | **Near-zero JSON overhead** | High (Metal port) | None |
| 🔴 **P3** | ACON context compression | **26-54% token reduction** | Low (wrapper) | Medium |
| 🔴 **P3** | Expert pruning (REAP/PuzzleMoE) | **25-50% memory reduction** | Medium (one-shot) | Medium |
| 🔴 **P3** | TurboQuant with Lloyd-Max codebooks | **Better than naive quantization** | High (custom codebooks + kernels) | Medium |

---

## 1. KV Cache Quantization (Validated ✅)

### What We Tested

We implemented and benchmarked four KV cache quantization strategies at context lengths 512–16K:

| Strategy | 16K Speed | Quality (token match) | Verdict |
|----------|----------|----------------------|---------|
| Baseline (BF16 KV) | 18.3 t/s | 100% | Collapses at 16K |
| **Naive INT8** | **44.7 t/s (2.44x)** | **100%** | **Winner** |
| Naive INT4 | 36.3 t/s (1.98x) | 66% | Quality loss |
| TurboQuant INT8 | 41.2 t/s (2.25x) | 17% | Bad without codebooks |
| TurboQuant INT4 | 24.7 t/s (1.35x) | 28% | Bad without codebooks |

### Key Findings

1. **Memory cliff at 16K**: Baseline drops from 60 t/s to 18 t/s due to KV cache memory pressure (~654 MB theoretical). INT8 keeps the working set below the thrashing threshold.

2. **TurboQuant needs Lloyd-Max codebooks**: Our Hadamard rotation implementation correctly spreads outlier channels (kurtosis 900→2.9), but mlx's per-group affine quantization can't exploit the resulting Beta distribution. The full TurboQuant algorithm uses pre-computed Lloyd-Max centroids matched to this distribution — without those, the rotation is counterproductive.

3. **INT8 > INT4**: INT8 is faster than INT4 because `mx.quantized_matmul` at 4-bit has more bit-manipulation overhead, and the memory savings from INT4→INT8 don't cross another pressure threshold.

### Recommendation

```python
# Enable immediately for any context ≥ 8K tokens
for token, logprobs in generate_step(
    prompt_tokens, model,
    kv_bits=8, kv_group_size=64, quantized_kv_start=0,
):
    ...
```

### Future: Proper TurboQuant (P3)

A full implementation would require:
- Pre-computing Lloyd-Max centroids for Beta(d/2, (d-1)/2) at head_dim=256
- Custom `turboquant_matmul` Metal kernel for centroid-based dequantization
- Per-vector norm storage instead of per-group scale/zero-point
- Expected result: perplexity-neutral at 3.5 bits (5x compression vs BF16)

---

## 2. Speculative Decoding

### 2.1 EAGLE-Style Tree Drafting via MTP Head (P1) ⭐

**The single highest-leverage optimization not yet implemented.**

Qwen3.5's MTP head is architecturally identical to EAGLE-1's draft model:
```
MTP: pre_fc_norm(h_last) + pre_fc_norm(embed(t)) → cat → fc → transformer_layer → norm → lm_head
EAGLE-1: h_{N-1} + embed(t) → FC → transformer_layer → LM head (shared)
```

Our current MTP implementation does simple sequential speculation (1 draft token → verify). EAGLE's tree attention verifies an entire tree of candidates in one forward pass:

| Method | Speedup | Tokens/cycle | Training Required |
|--------|---------|-------------|-------------------|
| Current MTP (sequential) | 1.09x | 1.76 (76% acceptance) | None (weights exist) |
| EAGLE-1 tree (static) | 3.0-3.5x | ~4 | None (reuse MTP head) |
| EAGLE-2 tree (dynamic) | 3.0-5.0x | ~5 | None beyond EAGLE-1 |
| EAGLE-3 (multi-layer fusion) | 3.0-6.5x | ~6.6 | Moderate (new training) |

**Implementation plan:**
1. Extract hidden states from target model during generation (already done for MTP)
2. Generate tree of draft candidates from MTP head (top-K sampling at each step)
3. Build tree-topology attention mask
4. Verify entire tree in one target model forward pass
5. Accept longest valid prefix

**Complexity**: Medium — the MTP head and weight loading already work. The missing piece is tree attention (custom causal mask) and multi-candidate sampling.

**Reference**: vLLM PR #12755 demonstrates exactly this pattern for DeepSeek-R1's MTP head.

### 2.2 SuffixDecoding for Agentic Patterns (P1)

SuffixDecoding (NeurIPS 2025 Spotlight) is uniquely suited for agentic workloads. It builds a suffix tree from prior prompts/outputs and uses pattern matching to predict next tokens — no draft model needed.

| Workload | Speedup | Why It Works |
|----------|---------|-------------|
| AgenticSQL | **5.3x** | Repeated SQL patterns, JSON wrappers |
| SWE-Bench | **2.5x** | Repeated code patterns, tool schemas |
| vs EAGLE-2/3 | **2.8x faster** | Exploits repetition EAGLE can't see |

**Why it's perfect for agentic workflows:**
- Tool call JSON has identical structure every time (same keys, schema, wrapper)
- Multi-turn sessions repeat system prompt, tool definitions, similar reasoning
- No training required — works with any model immediately
- Quality: zero risk (rejection sampling guarantees identical output)

**Implementation**: Medium — suffix tree is a pure data structure; integration requires modifying the token generation loop for speculative emission and batch verification. Already powers Snowflake ArcticInference in vLLM.

### 2.3 MoE-Spec Expert Budgeting (P1)

Critical for making speculation work with MoE models. Problem: verifying a tree of K draft candidates activates the *union* of all expert subsets — potentially 54 of 64 experts at K=127, negating sparsity benefits.

MoE-Spec caps the expert budget during verification:
- Evaluated on **Qwen3-30B-A3B** (closest proxy to our model)
- +16-30% throughput on top of EAGLE-3 baseline
- Training-free; integrates with any speculative decoding pipeline

### 2.4 Combining Speculation Methods

The methods are composable. Optimal agentic stack:

```
Layer 1: SuffixDecoding (pattern-based, for repetitive JSON/tool structure)
  ↓ fallback when no suffix match
Layer 2: MTP-as-EAGLE tree drafting (model-based, for novel content)
  ↓ during verification
Layer 3: MoE-Spec expert budget (caps verification cost)
```

Expected combined speedup: **4-8x** for typical agentic sessions.

---

## 3. Prefix/KV Cache Sharing (P0)

The highest-ROI optimization for multi-turn agentic workloads, and mathematically lossless.

### The Problem

In agentic sessions, 80-95% of tokens are shared prefix:
- System prompt (~500-2000 tokens)
- Tool definitions (~500-3000 tokens)
- Conversation history (grows per turn)

Without caching, every turn re-processes this entire prefix.

### Solutions

| Technique | Hit Rate (multi-turn) | Implementation |
|-----------|----------------------|---------------|
| SGLang RadixAttention | 75-90% | SGLang only |
| vLLM Automatic Prefix Caching | 40-70% | vLLM, TGI |
| llama.cpp prefix cache | ~50% | Partial |

**For MLX**: No native prefix caching exists. Implementation requires:
1. Content-hashed KV cache blocks
2. LRU eviction policy
3. Radix tree for prefix matching

**Impact**: Up to 5x TTFT improvement for repeated system prompts. For agentic workflows where TTFT matters (every tool call round-trip blocks the user), this is transformative.

---

## 4. MoE Expert Optimization

### 4.1 Mixed-Precision Expert Quantization (P2)

**MxMoE** (ICML 2025): Assigns different bit-widths at the linear-block level within each expert. Hot experts (high activation frequency) get higher precision; cold experts get lower precision.

For Qwen3.5-35B-A3B with 64 experts and top-4 routing:
- Expert activation frequency follows a heavy-tailed distribution
- Top-16 experts may handle 60%+ of tokens → keep at 5-8 bit
- Bottom-16 experts handle <5% of tokens → quantize to 2-3 bit
- Expected: ~50% memory reduction with <0.5 perplexity increase

**Blocker**: Requires custom MLX GroupGEMM kernels for mixed bitwidth.

### 4.2 Expert Pruning (P3)

**REAP** (one-shot, no fine-tuning): Router-weighted expert activation pruning.
- 25-33% expert removal → proportional memory and compute reduction
- Key finding: for generative tasks, pruning > merging

**PuzzleMoE**: Evaluated on Qwen3-MoE-30B-A3B specifically:
- 1.28x inference speedup at 50% compression
- +16.7% MMLU vs prior merging methods
- Training-free

### 4.3 Expert Offloading with Predictive Prefetch (P3)

For models that don't fit entirely in memory:
- **HOBBIT**: Int4 fallback on cache miss (4x faster expert load); up to 9.93x vs naive offloading
- **ExpertFlow**: Cross-layer predictor for expert prefetch; 91.96% cache hit rate
- **FlashMoE**: NVMe SSD-optimized cache replacement (directly relevant for Apple Silicon NVMe)

These are less relevant for our 48GB setup (model fits at 4-bit ~19.5GB), but become critical for 32GB machines or larger models.

---

## 5. Agentic-Specific Optimizations

### 5.1 Parallel Tool Execution (P1)

**LLMCompiler** (ICML 2024): Emits a DAG of tool calls in one generation step. Independent tools execute in parallel.
- 3.7x latency reduction, 6.7x cost savings
- Orthogonal to model-level optimizations — pure orchestration
- No model modifications required

**PASTE** (March 2026): Pattern-aware speculative tool execution. Pre-launches predicted tool calls during LLM generation.
- 48.5% average task completion time reduction
- Sidecar architecture — no changes to LLM needed
- Requires tools to be safely retractable or idempotent

### 5.2 Context Compression (P3)

**ACON** (OpenReview 2026): Gradient-free context compression for long-horizon agents.
- 26-54% token reduction while preserving accuracy
- Specifically designed for tool-calling sessions
- Works as a preprocessing wrapper around any LLM

**LLMLingua-2** (ACL 2024): Token-level importance scoring.
- Up to 20x compression with minimal loss on QA
- Risk: aggressive compression can drop tool schema details

### 5.3 Structured Output / XGrammar (P2)

**XGrammar** (default in vLLM/SGLang): Near-zero overhead JSON constrained decoding.
- Eliminates hallucinated field names in tool calls
- 4,467x faster than previous EBNF-based frameworks
- **Blocker for MLX**: C++/CUDA library, needs Metal port

**SGLang Compressed FSM**: Jump-forward decoding for deterministic JSON tokens.
- Up to 2x latency reduction for structured output
- Treats structural tokens (braces, keys) as cache hits

---

## 6. Long Context (32K+) Attention Optimization

### 6.1 Sparse Attention — DUOAttention (P2)

Classifies attention heads into "retrieval heads" (need full context) and "streaming heads" (work with sliding window):
- ~50% KV cache reduction at 32K with minimal accuracy loss
- Head classification done offline; inference-time routing is straightforward
- **Critical for agents**: retrieval heads must stay active for distant tool results

### 6.2 Hybrid Sparse Strategy (Survey Finding)

Dense attention on W=1024 most recent tokens + sparse attention on remainder:
- ~39% KV cache reduction with near-zero accuracy degradation
- Most practically deployable approach for existing models

### 6.3 For Qwen3.5's Hybrid Architecture

Only 10 of 40 layers use standard attention (the other 30 use GatedDeltaNet linear attention). Sparse attention optimizations would only apply to these 10 layers, limiting the total impact to ~25% of what a pure Transformer would see.

However, the GatedDeltaNet layers' `ArraysCache` already provides O(1) memory per sequence length — the KV cache pressure comes entirely from these 10 attention layers. So the 10 layers are exactly where optimization matters most.

---

## 7. Apple Silicon-Specific Considerations

### Memory Bandwidth is the Bottleneck

Apple Silicon decode is bandwidth-bound, not compute-bound:
- M2 Max: 400 GB/s memory bandwidth
- M4 Max: 546 GB/s
- M2 Ultra: 800 GB/s

At 4-bit quantization, Qwen3.5-35B-A3B requires reading ~19.5 GB of weights per token. On M2 Max at 400 GB/s, theoretical maximum is ~20 tokens/sec for the weight-read alone. Actual throughput is ~75 t/s at short context (due to caching effects and batch prefill).

**Implication**: Any optimization that reduces bytes-read-per-token (quantization, pruning, expert caching) directly translates to speed. Compute-focused optimizations (early exit, layer skip) have less impact.

### Unified Memory Advantage

No CPU↔GPU copy overhead. KV cache, model weights, and orchestration state share the same physical DRAM. This means:
- Expert offloading to "CPU memory" has no benefit (it's the same memory)
- SSD offloading is the relevant variant for Apple Silicon
- Prefix caching is simpler (no device-to-device transfer)

### MLX Performance Ranking (March 2026)

MLX (~230 t/s 7B) > MLC-LLM (~190) > llama.cpp (~150) > Ollama (20-40) > PyTorch MPS (~7-9)

MLX's advantages: `mx.compile` kernel fusion, `mx.fast` hand-tuned Metal kernels (SDPA, RMSNorm, quantized matmul), lazy evaluation for automatic graph optimization.

---

## 8. Implementation Roadmap

### Phase 0: Immediate (Zero Implementation)
- [x] Enable `kv_bits=8` for all contexts ≥ 8K tokens
- [ ] Profile actual memory usage to find the exact threshold where INT8 KV becomes beneficial

### Phase 1: High-Impact, Medium Effort (1-3 weeks each)
- [ ] **EAGLE-style tree attention** for MTP head — extend existing MTP decoder with tree drafting
- [ ] **SuffixDecoding** — build suffix tree from agentic session history
- [ ] **Prefix KV cache sharing** — content-hashed KV blocks with LRU eviction
- [ ] **LLMCompiler-style parallel tools** — orchestration layer, no model changes

### Phase 2: Significant Effort (1-2 months each)
- [ ] **MoE-Spec expert budgeting** — hook into MoE routing for verification-time budget
- [ ] **DUOAttention head classification** — offline profiling + inference-time routing
- [ ] **XGrammar Metal port** — constrained JSON decoding for tool calls

### Phase 3: Research Exploration
- [ ] **Full TurboQuant** with Lloyd-Max codebooks and custom Metal kernels
- [ ] **MxMoE mixed-precision** expert quantization with custom GroupGEMM
- [ ] **REAP/PuzzleMoE** expert pruning — profile expert activation, one-shot prune

---

## 9. Expected Combined Impact

For a typical agentic session (system prompt + tools + 20 turns, reaching 8-16K context):

| Optimization Stack | Estimated Speed | vs Baseline |
|-------------------|----------------|-------------|
| Baseline at 16K | 18.3 t/s | 1.0x |
| + INT8 KV cache | 44.7 t/s | 2.4x |
| + Prefix caching (TTFT only) | 44.7 t/s decode, 5x faster prefill | 2.4x decode |
| + EAGLE tree drafting (MTP head) | ~130-180 t/s effective | ~7-10x |
| + SuffixDecoding (JSON/tool patterns) | ~180-240 t/s effective | ~10-13x |
| + Parallel tool execution | N/A (end-to-end latency) | 3.7x E2E |

*Effective t/s accounts for accepted speculative tokens per verification cycle.*

The speculative decoding numbers are projections based on EAGLE-3 benchmarks on similar-class models. Actual Apple Silicon numbers may be lower due to bandwidth constraints (ReDrafter achieves 2.3x on M2 Ultra vs EAGLE's 3-6.5x on H100).

---

## References

### Papers Cited

1. **TurboQuant**: Zandieh et al., "TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate", ICLR 2026
2. **EAGLE-3**: Li et al., "EAGLE-3: Scaling up Inference Acceleration via Training-Time Test", NeurIPS 2025
3. **MoE-Spec**: "MoE-Spec: Expert Budgeting for Efficient Speculative Decoding", 2026
4. **SuffixDecoding**: "SuffixDecoding: Extreme Speculative Decoding for Emerging AI Applications", NeurIPS 2025
5. **MxMoE**: "MxMoE: Mixed-precision Quantization for MoE", ICML 2025
6. **LLMCompiler**: "An LLM Compiler for Parallel Function Calling", ICML 2024
7. **PASTE**: "Act While Thinking: Accelerating LLM Agents via Pattern-Aware Speculative Tool Execution", 2026
8. **ACON**: "Optimizing Context Compression for Long-horizon LLM Agents", 2026
9. **XGrammar**: "Flexible and Efficient Structured Generation Engine for LLMs", 2024
10. **DUOAttention**: Head-level sparse attention for KV cache reduction, 2025
11. **REAP**: "REAP the Experts: Router-weighted Expert Activation Pruning", 2025
12. **PuzzleMoE**: "Efficient Compression via Sparse Expert Merging", 2025 (evaluated on Qwen3-MoE-30B-A3B)
13. **HOBBIT**: "Mixed Precision Expert Offloading for Fast MoE Inference", 2024
14. **SGLang RadixAttention**: Zheng et al., NeurIPS 2024
15. **LayerSkip**: Elhoushi et al., ACL 2024 (Meta AI)
16. **ReDrafter**: Apple, "Recurrent Drafter for Fast Speculative Decoding", 2024

### Our Benchmark Data

- `BENCHMARK_TURBOQUANT.md` — KV cache quantization results
- `benchmark_turboquant_results.json` — Raw benchmark data
- `vllm_mlx_mtp/turboquant.py` — TurboQuant implementation
- `research/` — Detailed research reports per topic
