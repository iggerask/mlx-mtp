# Agentic LLM Inference Optimizations: Research Survey (2024–2026)

**Compiled:** March 2026
**Scope:** Optimizations targeting agentic LLM inference patterns — repeated tool calls, long context accumulation, structured output, multi-turn growing KV cache, bursty generation.

---

## Executive Summary

Agentic workloads expose bottlenecks that differ fundamentally from batch text generation: they are latency-sensitive (every tool call round-trip blocks the user), prefix-heavy (system prompt + tool definitions + history dominate token count), structured-output-intensive (every tool call and response is JSON), and long-horizon (sessions of 50–200 turns with 32K–128K context). The 2024–2025 research wave has produced production-ready solutions for several of these bottlenecks.

**Highest-ROI interventions for an MLX-based agentic deployment:**

1. **Prefix/KV cache sharing** (RadixAttention / APC) — free speedup if your serving framework supports it; up to 5x throughput gain on repeated system prompts.
2. **Structured output with XGrammar** — near-zero overhead constrained decoding; now the default in vLLM and SGLang; critical for tool-call JSON fidelity.
3. **SuffixDecoding** — model-free speculative decoding designed specifically for agentic repetition patterns; 5.3x speedup, no draft model to train or maintain.
4. **ACON context compression** — gradient-free, works with closed models; 26–54% token reduction on long-horizon agent tasks while preserving accuracy.
5. **Parallel tool execution (LLMCompiler / PASTE)** — orthogonal to model-level optimizations; 3.7x latency reduction by parallelizing independent tool calls.

---

## Research Questions Addressed

1. How do prefix caching techniques work for multi-turn agent loops, and which systems implement them?
2. Which context compression/eviction methods preserve tool-call accuracy across long sessions?
3. How is structured output (JSON/function-call) generation accelerated without quality loss?
4. Can tool execution be pre-fetched or parallelized speculatively?
5. What is the state of the art for draft model distillation, especially for MoE targets?
6. Can adaptive computation (early exit, layer skip) reduce cost on easy agentic tokens?
7. What sparse/sliding-window attention approaches maintain accuracy over 32K+ contexts?
8. What MLX-specific optimizations exist for fast MoE inference on Apple Silicon?

---

## Detailed Findings

### 1. Prompt Caching / Prefix Sharing

#### SGLang RadixAttention

**How it works.** SGLang stores both prompt and generation KV cache in a radix tree — a prefix-indexed data structure supporting efficient insertion, lookup, and LRU eviction. When a new request arrives with a prefix matching a cached subtree, the engine skips recomputation of those tokens entirely. This benefits multi-turn agents directly: the system prompt, tool schema, and prior turn history are typically identical prefixes across requests in the same session.

- **Speedup:** Up to 5x throughput versus systems without prefix reuse; cache hit rates of 75–90% for multi-turn chat vs. 10–20% for vLLM PagedAttention (v1).
- **SGLang v0.4 (Dec 2024)** added a Zero-Overhead Batch Scheduler and Cache-Aware Load Balancer, further increasing prefix cache utilization in multi-session deployments.
- **MLX complexity:** RadixAttention is a serving-layer feature, not a model-layer feature. Porting it to MLX requires building a KV cache management layer with radix tree indexing. No MLX-native implementation exists as of March 2026; llama.cpp has partial prefix caching.
- **Quality risk:** Zero — prefix reuse is mathematically equivalent to full recomputation.
- **Deployment:** Production in SGLang, vLLM v2 (Automatic Prefix Caching).

**Paper:** [SGLang: Efficient Execution of Structured Language Model Programs](https://arxiv.org/abs/2312.07104) (NeurIPS 2024)
**Blog:** [Fast and Expressive LLM Inference with RadixAttention and SGLang](https://lmsys.org/blog/2024-01-17-sglang/)

#### vLLM Automatic Prefix Caching (APC)

vLLM's APC hashes KV cache blocks by their token content. On a cache hit, blocks are reused directly. The vLLM blog describes agentic workloads as the "most extreme case of prefix dominance" — with 10x cost differences between cached and uncached tokens in production. APC is enabled by default in vLLM v0.6+.

- **Speedup:** 2–4x TTFT (time to first token) for long agent prefixes; throughput scales with cache hit rate.
- **MLX complexity:** Medium — requires block-level KV cache management with content hashing.
- **Quality risk:** Zero.
- **Deployment:** Production in vLLM; also in TGI, LMDeploy.

**Docs:** [vLLM Automatic Prefix Caching](https://docs.vllm.ai/en/stable/design/prefix_caching/)

#### Comparison for Multi-Turn Agents

| Technique | Hit Rate (multi-turn) | Eviction Policy | Implementation |
|---|---|---|---|
| SGLang RadixAttention | 75–90% | LRU on radix tree | SGLang only |
| vLLM APC | 40–70% | LRU on KV blocks | vLLM, TGI |
| llama.cpp prefix cache | ~50% (exact match only) | FIFO | llama.cpp, MLX partial |

---

### 2. Context Compression / Eviction

#### LLMLingua Series (Microsoft)

LLMLingua (EMNLP 2023) and its successors compress the input token sequence before sending to the LLM, using a small proxy model to score token importance.

- **LLMLingua-2** (ACL 2024): Trained via data distillation from GPT-4 as a token classification model. 3–6x faster compression than LLMLingua-1; better out-of-domain generalization.
- **LongLLMLingua** (ACL 2024): Targets the "lost in the middle" failure mode; boosts NaturalQuestions performance by 21.4% at ~4x fewer tokens on GPT-3.5-Turbo.
- **Max compression:** Up to 20x with minimal performance loss on standard QA benchmarks.
- **Tool-call risk:** Moderate — aggressive compression can drop tool schema details or prior tool result tokens. LLMLingua-2 is safer than v1 due to classification framing. Rate the risk medium for tool-calling accuracy at >10x compression.
- **MLX complexity:** Low to medium — the compressor runs as a preprocessing step with a small model; output feeds into any inference backend.
- **Deployment:** Available as open-source Python library ([github.com/microsoft/LLMLingua](https://github.com/microsoft/LLMLingua)).

#### ACON: Optimizing Context Compression for Long-Horizon Agents

The most directly relevant paper for agentic use cases (Oct 2025, OpenReview 2026).

ACON frames context compression as a policy optimization problem. Given paired agent trajectories where full context succeeds but compressed context fails, a capable LLM analyzes failure causes and updates compression guidelines in natural language — gradient-free. It compresses both environment observations and interaction histories.

- **Benchmarks:** AppWorld (9-app integration, avg. 42.5 API calls/task), OfficeBench, Multi-objective QA.
- **Results:** 26–54% peak token reduction; AppWorld +32%, OfficeBench +20%, Multi-objective QA +46% over baselines. Smaller compressor distilled from ACON retains >95% accuracy.
- **Quality risk:** Low for well-tuned guidelines; medium if guidelines are not domain-adapted.
- **MLX complexity:** Low — ACON is a wrapper around any LLM; the compressor can run offline.
- **Deployment:** Research code; no production framework integration as of March 2026.

**Paper:** [ACON: Optimizing Context Compression for Long-horizon LLM Agents](https://arxiv.org/abs/2510.00615)

#### InfLLM

InfLLM enables near-infinite context without model retraining by offloading context to external (CPU) memory and using sliding-window attention plus a retrieval mechanism to bring relevant blocks back to GPU at decode time.

- **Quality risk:** Medium — retrieval can miss relevant tool results from many turns ago.
- **Overhead:** Retrieval latency introduced per step; not well-suited for single-machine Apple Silicon deployments where CPU-GPU transfer is within unified memory (but cache locality still matters).
- **MLX complexity:** High — requires custom attention with retrieval hooks.

**Relevant survey:** [A Survey on Large Language Model Acceleration based on KV Cache Management](https://arxiv.org/html/2412.19442v3)

#### AutoCompressor / CEPE / Gist Tokens

These approaches learn to compress context into a fixed set of "summary tokens" or "gist tokens" in embedding space. AutoCompressor and CEPE require fine-tuning the base model to accept compressed representations.

- **Quality risk:** High for out-of-distribution tool schemas — compressed representations may lose exact JSON key names needed for tool call parsing.
- **MLX complexity:** Very high — requires re-training or fine-tuning target model.
- **Recommendation:** Not suitable for production agentic tool-calling without extensive validation. Prefer ACON or LLMLingua-2 for tool-rich agents.

---

### 3. Structured Output Acceleration

#### Outlines + Compressed FSM (SGLang)

SGLang introduced compressed finite state machines (FSM) for constrained decoding in Feb 2024. The key insight: many JSON tokens (keys, structural characters) are deterministic given the schema. Jump-forward decoding skips sampling for these tokens and emits them directly, treating them like a cache hit.

- **Speedup:** Up to 2x latency reduction; 2.5x throughput improvement over guidance+llama.cpp and outlines+vLLM.
- **Implementation:** RadixAttention simplifies jump-forward — the engine terminates the current request and enqueues a new one with the deterministic tokens appended; KV cache is reused automatically.

**Blog:** [Fast JSON Decoding for Local LLMs with Compressed Finite State Machine](https://lmsys.org/blog/2024-02-05-compressed-fsm/)

#### XGrammar (MLC-AI, Nov 2024)

XGrammar is the current state-of-the-art constrained decoding engine. It divides the vocabulary into:
- **Context-independent tokens** — validity can be precomputed at schema compilation time.
- **Context-dependent tokens** — checked at runtime with lightweight operator FSMs (not full PDAs).

This allows grammar compilation to be moved out of the Python hot path (into C with pthread). Achieves near-zero overhead for JSON generation.

- **Speedup:** 4,467x faster first-token time over previous EBNF-based frameworks in extreme cases; 10x faster than other open-source solutions for JSON decoding (per SGLang v0.4 benchmarks).
- **Integration:** Default in vLLM (Dec 2024) and SGLang (v0.4); also in TensorRT-LLM (Jan 2025).
- **MLX complexity:** High — XGrammar is a C++/CUDA library. An MLX port would require reimplementing the token masking layer in Metal. No known MLX integration as of March 2026.
- **Quality risk:** Near-zero — constrained decoding guarantees schema validity by construction.
- **Agentic accuracy note:** Constrained decoding eliminates hallucinated field names in tool calls, which is a significant source of agent errors in free-form generation.

**Paper:** [XGrammar: Flexible and Efficient Structured Generation Engine for Large Language Models](https://arxiv.org/abs/2411.15100)
**Blog:** [Achieving Efficient, Flexible, and Portable Structured Generation with XGrammar](https://blog.mlc.ai/2024/11/22/achieving-efficient-flexible-portable-structured-generation-with-xgrammar)

#### IterGen (ICLR 2025)

IterGen introduces forward/backward generation tied to grammar symbols, with KV cache reuse enabling efficient "undo and redo" when backtracking in constrained generation. Particularly relevant for complex nested JSON schemas.

#### Benchmark

JSONSchemaBench (2025): 10K real-world JSON schemas, 10 datasets of varying complexity. Evaluates Guidance, Outlines, Llamacpp, XGrammar, OpenAI, Gemini. XGrammar leads on overhead and correctness.

**Survey paper:** [Generating Structured Outputs from Language Models: Benchmark and Studies](https://arxiv.org/html/2501.10868v1)

---

### 4. Parallel Tool Execution / Speculative Tool Calls

#### LLMCompiler (ICML 2024, Stanford/Berkeley)

LLMCompiler enables an LLM to emit a DAG of tool calls in one generation step. A Task Fetching Unit dispatches independent tasks in parallel; results are fed back for dependent tasks.

- **Speedup:** Up to 3.7x latency reduction; 6.7x cost savings (fewer LLM calls); ~9% accuracy improvement over ReAct (due to long-horizon planning in the DAG step).
- **Limitation:** Requires the LLM to be capable of DAG-structured output, and downstream tasks must truly be independent.
- **MLX complexity:** Low — orchestration layer; no model modifications required.
- **Quality risk:** Low — correctness is maintained; the risk is LLM errors in DAG planning.

**Paper:** [An LLM Compiler for Parallel Function Calling](https://arxiv.org/abs/2312.04511)
**GitHub:** [SqueezeAILab/LLMCompiler](https://github.com/SqueezeAILab/LLMCompiler)

#### PASTE: Pattern-Aware Speculative Tool Execution (March 2026)

PASTE is the most sophisticated speculative tool execution framework found. It observes that agent applications have stable application-level control flows (recurring tool-call sequences) and predictable parameter dependencies. A Pattern Analyzer abstracts these into invocation sequence templates. An Online Scheduler pre-launches speculative tool calls during LLM generation, validated when the model confirms the call.

- **Speedup:** 48.5% average task completion time reduction; 1.8x tool execution throughput.
- **Overhead:** 1–3 idle CPU cores; ~250 MB additional memory for speculative execution buffers.
- **Deployment:** Sidecar architecture — no changes to LLM or agent framework required.
- **MLX complexity:** Low for the orchestration layer; depends on tool isolation guarantees (speculative calls must be safely retractable or idempotent).
- **Quality risk:** Low if tools are idempotent; medium if tool calls have side effects.

**Paper:** [Act While Thinking: Accelerating LLM Agents via Pattern-Aware Speculative Tool Execution](https://arxiv.org/html/2603.18897)

#### Speculative Actions (Oct 2025)

A lossless framework for faster agentic systems that overlaps tool execution with the next LLM generation step. Validated as lossless (no accuracy degradation by construction).

**Paper:** [Speculative Actions: A Lossless Framework for Faster Agentic Systems](https://arxiv.org/pdf/2510.04371)

#### Sherlock: Reliable and Efficient Agentic Workflow Execution (Nov 2025)

Focuses on reliability and speculative validation of agentic steps. Reduces redundant LLM calls.

**Paper:** [Sherlock: Reliable and Efficient Agentic Workflow Execution](https://arxiv.org/pdf/2511.00330)

---

### 5. Draft Model Distillation for Speculative Decoding

#### EAGLE Series (SafeAILab)

The EAGLE family is the dominant approach for trained draft model speculative decoding.

**EAGLE-1 (ICML 2024):** Extrapolates the second-top-layer feature vectors of the target LLM to predict next tokens. Draft model is small and shares the target model's token vocabulary. Achieved 3–4x speedup.

**EAGLE-2 (EMNLP 2024):** Context-aware dynamic draft trees — the draft model generates a tree of candidates, and acceptance probability is predicted to prune the tree adaptively. 20–40% faster than EAGLE-1; speedup 3.05–4.26x.

**EAGLE-3 (NeurIPS 2025, released March 2025):** Removes the feature prediction constraint. Instead of predicting top-layer features, EAGLE-3 fuses features from multiple intermediate layers (training-time test). This allows the draft model to benefit from more scaling data and produces better draft quality. Achieves 3.0–6.5x speedup vs. vanilla decoding; 20–40% improvement over EAGLE-2.

- **MoE support:** EAGLE has been evaluated on Mixtral 8x7B Instruct. For frontier MoE models (Llama 4, DeepSeek), the SpecForge framework (LMSYS, 2025) enables EAGLE-3 training on complex MoE layers.
- **MLX complexity:** Medium-to-high — requires training a draft model on the target model's hidden states, then running both models in the MLX inference loop with tree verification. No MLX-native EAGLE implementation exists as of March 2026; would require implementing tree attention in MLX.
- **Quality risk:** Very low — provably maintains output distribution under rejection sampling.
- **Integration:** Production in vLLM, TensorRT-LLM, ROCm, NeuronX; AWS SageMaker added EAGLE Nov 2025.

**GitHub:** [SafeAILab/EAGLE](https://github.com/SafeAILab/EAGLE)
**EAGLE-3 paper:** [arxiv.org/abs/2503.01840](https://arxiv.org/html/2503.01840v1)

#### Medusa and Hydra

**Medusa (2024):** Adds multiple decoding heads to the target model's final hidden state, each predicting a token offset ahead. Medusa-1: 2.2x+ speedup (heads only trained). Medusa-2: 2.3–3.6x (full model training with a special recipe).

**Hydra (COLM 2024):** Sequentially-dependent draft heads — each head conditions on the previous head's prediction (via a prefix attention layer), improving draft accuracy. Hydra++ achieves 2.70x over autoregressive decoding.

- **MLX complexity:** Medium — heads attach to the existing model; tree verification requires custom attention logic.
- **MoE note:** Both Medusa and Hydra add heads to the final hidden state, which is the same regardless of MoE routing, so they are architecturally compatible with MoE models.

**Hydra paper:** [arxiv.org/abs/2402.05109](https://arxiv.org/abs/2402.05109)

#### Apple Recurrent Drafter (ReDrafter)

Apple's own speculative decoding approach uses an RNN draft model conditioned on the LLM's hidden states, with dynamic tree attention over beam search candidates to eliminate duplicated prefixes.

- **Speedup:** Up to 2.3x on Apple Silicon (1.37x on M1 Max, higher on M2 Ultra).
- **MLX implementation:** Official MLX reference implementation at [apple/ml-recurrent-drafter](https://github.com/apple/ml-recurrent-drafter).
- **Most MLX-native option** for speculative decoding on Apple Silicon.

**Paper:** [Recurrent Drafter for Fast Speculative Decoding](https://machinelearning.apple.com/research/recurrent-drafter)

#### SuffixDecoding (NeurIPS 2025 Spotlight) — Model-Free, Agentic-Optimized

SuffixDecoding is the most agentic-specific speculative decoding technique. It uses a suffix tree built from prior prompts and outputs to predict the next tokens — no draft model required. Exploits the repetition inherent in agentic patterns (same JSON wrapper, same tool schemas, repeated sub-task patterns in multi-agent pipelines).

- **Speedup:** 5.3x on AgenticSQL, 2.5x on SWE-Bench; 2.8x faster than EAGLE-2/3, 1.9x faster than Token Recycling for these workloads.
- **MLX complexity:** Low-to-medium — suffix tree is a pure data structure; draft tokens come from cache lookup, not model inference. Integration requires modifying the token generation loop to support speculative emission and batch verification.
- **Quality risk:** Zero — rejection sampling guarantees identical output distribution.
- **No training required** — can be applied to any model immediately.
- **Production:** Powers Snowflake ArcticInference in vLLM.

**Paper:** [SuffixDecoding: Extreme Speculative Decoding for Emerging AI Applications](https://arxiv.org/abs/2411.04975)
**Project site:** [suffix-decoding.github.io](https://suffix-decoding.github.io/)

#### Online Speculative Decoding

Online methods update the draft model during inference using the target model's accepted tokens as training signal (self-play). Reduces distribution shift between draft and target.

- **MLX complexity:** High — requires online gradient updates during inference.
- **Recommendation:** Not practical for single-device Apple Silicon inference; skip for MLX deployments.

---

### 6. Adaptive Computation

#### LayerSkip (Meta, ACL 2024)

LayerSkip is an end-to-end solution combining training and inference:

- **Training:** Layer dropout with increasing rates at deeper layers + shared early exit loss across all layers.
- **Inference (early exit):** Stop at an early layer if the exit confidence exceeds a threshold.
- **Self-speculative decoding:** Use early exit as the draft model; verify with remaining layers of the same model. No separate draft model needed; draft and target share all weights and KV cache.

- **Speedup:** 2.16x on summarization, 1.82x on coding, 2.0x on semantic parsing.
- **MLX complexity:** Medium — early exit requires a shared LM head at each layer (or a lightweight exit classifier). The self-speculative mode is attractive for MLX because it avoids loading a second model.
- **Quality risk:** Medium — requires retraining; off-the-shelf models cannot use early exit without the training recipe. For agentic accuracy, the exit threshold must be tuned conservatively for tool-call tokens.
- **Deployment:** Meta-internal; open weights released for some model families.

**Paper:** [LayerSkip: Enabling Early Exit Inference and Self-Speculative Decoding](https://arxiv.org/abs/2404.16710)

#### SWIFT: On-the-Fly Self-Speculative Decoding

SWIFT is a plug-and-play extension to LayerSkip that adaptively selects which intermediate layers to skip at runtime, without requiring the LayerSkip training recipe. It identifies the best "skip set" per input via a lightweight profiling step.

- **MLX complexity:** Medium — layer selection logic; no retraining required on target model.
- **Quality risk:** Medium — dynamic layer selection may occasionally skip layers critical for reasoning.

**OpenReview:** [SWIFT: On-the-Fly Self-Speculative Decoding](https://openreview.net/forum?id=EKJhH5D5wA)

#### CALM (Confident Adaptive Language Modeling)

CALM uses per-token confidence scores at each layer to decide early exit for individual tokens. Focuses on reducing compute for "easy" tokens while using full depth for "hard" tokens.

- **Speedup:** 1.4–2x on standard benchmarks.
- **MLX complexity:** Medium — requires calibration of per-layer confidence classifiers.
- **Quality risk:** Medium for reasoning-heavy tasks; low for repetitive structured tokens.

#### SmartSpec / SkipDecode

SkipDecode implements batch-compatible token-level early exit, addressing the challenge that different tokens in a batch may exit at different layers (ragged batches). SmartSpec extends this to speculative decoding.

- **Agentic relevance:** High — the structured/boilerplate JSON tokens in tool calls are prime candidates for early exit; the reasoning tokens should use full depth.

---

### 7. Memory-Efficient Long Context (32K+)

#### Native Sparse Attention (NSA) — DeepSeek, Feb 2025

NSA is a hardware-aligned, natively trainable sparse attention mechanism from DeepSeek-AI and Peking University, published as an ACL 2025 paper.

**Architecture:** Three parallel attention branches per layer:
1. **Compressed attention** — coarse-grained; attends to compressed representations of distant context blocks.
2. **Selected attention** — fine-grained; attends to top-k selected important token blocks.
3. **Sliding attention** — local context window.

Outputs are combined with learned weights.

- **Speedup:** Substantial speedups over full attention on 64K sequences across decoding, forward pass, and backward pass. Enables economical pretraining with long-context data.
- **MLX complexity:** Very high — requires training with NSA from scratch; cannot be retrofitted to existing dense models. Not suitable for inference-time adaptation.
- **Quality risk:** Low — NSA models match or exceed full attention on general and long-context benchmarks, plus instruction-based reasoning.
- **Agentic note:** The compressed branch maintains global context awareness (tool result history), while the sliding branch handles local formatting. Well-suited for long agent sessions.

**Paper:** [Native Sparse Attention: Hardware-Aligned and Natively Trainable Sparse Attention](https://arxiv.org/abs/2502.11089)

#### SpargeAttn (Feb 2025)

SpargeAttn is an accurate sparse attention method that can be applied at inference time to any existing model — no retraining. Uses offline analysis of attention patterns to identify which tokens can safely be skipped.

- **Speedup:** Significant for long contexts; exact figures depend on sparsity ratio.
- **MLX complexity:** Medium — requires implementing sparse attention kernels in Metal.
- **Quality risk:** Low-to-medium — calibration-dependent.

**Paper:** [SpargeAttn: Accurate Sparse Attention Accelerating Any Model Inference](https://arxiv.org/html/2502.18137v1)

#### The Sparse Frontier (2025 Survey)

A comprehensive 2025 survey evaluated sparse attention tradeoffs (H2O, InfLLM, DUOAttention) across Qwen 2.5, Llama 3.1, Gemma 3 at 16K, 32K, and 64K context lengths.

**Key finding for agents:** Hybrid strategies — dense attention on the W most recent tokens (W = 1,024 typical) plus sparse attention on the remainder — achieve ~39% KV cache reduction with near-zero accuracy degradation. This is the most practically deployable approach for existing models.

**Paper:** [The Sparse Frontier: Sparse Attention Trade-offs in Transformer LLMs](https://arxiv.org/pdf/2504.17768)

#### Long-Context Modeling with Dynamic Hierarchical Sparse Attention (Oct 2025)

Specifically designed for on-device LLMs, targeting the Apple Silicon / mobile deployment scenario. Dynamic hierarchical sparse attention adapts sparsity based on input length.

**Paper:** [arxiv.org/abs/2510.24606](https://arxiv.org/abs/2510.24606)

#### DUOAttention

DUOAttention identifies "retrieval heads" (heads that need full attention for accurate long-range retrieval) and "streaming heads" (heads that work well with sink+sliding-window). Applies full KV cache only to retrieval heads; uses 2-token sink + sliding window for streaming heads.

- **KV cache reduction:** ~50% at 32K context with minimal accuracy loss on retrieval benchmarks.
- **MLX complexity:** Medium — head classification is done offline; inference-time routing is straightforward.
- **Quality risk:** Low for well-calibrated head classification; tool results from distant turns need retrieval heads active.

#### Tool-Call Accuracy Note

Across all sparse attention methods, the critical failure mode for agents is dropping tool result tokens from many turns ago that are referenced in the current turn. Any method with a sliding window must ensure tool output segments are either retained in the global attention sink or protected by a "forced retention" flag. ACON's context compression (Section 2) is preferable to hard eviction for this reason.

---

### 8. Apple Silicon / MLX-Specific Optimizations

#### Unified Memory Architecture — Core Advantage

Apple Silicon's unified memory pool means CPU and GPU share the same physical DRAM. In MLX, tensors require no device copies between CPU and GPU operations. For LLM inference:
- KV cache lives in unified memory; CPU-side orchestration (prefix tree management, tool parsing) shares the same buffer space.
- Memory bandwidth is the primary bottleneck for token generation (decode phase is bandwidth-bound, not compute-bound).
- M4: 120 GB/s; M5: 153 GB/s (28% higher). An M2 Ultra at 192 GB/s can sustain ~190 tokens/sec for a 7B 4-bit model.

**Benchmark ranking (March 2026):** MLX (~230 tok/s) > MLC-LLM (~190 tok/s) > llama.cpp (~150 tok/s) > Ollama (20–40 tok/s) > PyTorch MPS (~7–9 tok/s) for typical 7B-class models.

#### AMX vs GPU in MLX

All Apple M CPUs include at least one AMX (Apple Matrix coprocessor) block for matrix multiplication. However:
- **GPU substantially outperforms AMX** — up to 8,648 words/sec vs. 1,821 words/sec on M1 Max.
- AMX only accelerates matmul; GPU also handles GELU, Softmax, attention, and custom Metal kernels.
- MLX defaults to GPU execution; AMX is used by some CPU-fallback paths.

#### Metal Shader / mx.fast Optimizations

MLX provides `mx.compile` for kernel fusion — multiple GPU kernel launches are fused into a single kernel, reducing dispatch overhead. `mx.fast` sub-package provides hand-tuned Metal implementations of:
- Scaled dot-product attention (flash-attention style)
- Layer normalization, RMS norm
- Quantized matmul (4-bit, 8-bit)

**Custom Metal kernels** can be written for specialized operations (e.g., sparse attention patterns). The GPU supports user-defined compute shaders with full flexibility.

**Paper:** [Benchmarking On-Device Machine Learning on Apple Silicon with MLX](https://arxiv.org/html/2510.18921v1)

#### Neural Accelerator (M5)

The M5's Neural Engine now accelerates matrix-multiplication operations critical for LLM inference. MLX explicitly takes advantage of Neural Accelerators on M5 chips. This is the first time the Neural Engine path is meaningfully integrated with MLX (as of WWDC 2025).

**Apple ML Research:** [Exploring LLMs with MLX and the Neural Accelerators in the M5 GPU](https://machinelearning.apple.com/research/exploring-llms-mlx-m5)

#### MoE on Apple Silicon

For MoE models, the key constraint is expert routing bandwidth. On Apple Silicon:
- Active expert weights must be streamed from unified memory per token; with 2–4 active experts out of many, the effective working set is small.
- MLX handles this well because unified memory means no PCIe transfer for weight loads.
- Benchmark: Qwen 30B-A3B (4-bit quantized) and GPT OSS 20B (MXFP4) run efficiently on M-series chips.
- MLX-LM maintains a 1.43x advantage over llama.cpp for Nemotron-30B-A3B.

**Multi-node MoE:** Research on multi-node expert parallelism on Apple Silicon for larger MoE models (e.g., Llama 4 Scout) is active as of June 2025 ([arxiv.org/abs/2506.23635](https://arxiv.org/html/2506.23635v1)).

#### Speculative Decoding in MLX

- **Apple ReDrafter:** Official MLX implementation; 1.37x on M1 Max, higher on M2 Ultra. Available at [apple/ml-recurrent-drafter](https://github.com/apple/ml-recurrent-drafter).
- **MLX-LM native spec decoding:** Supported via `--draft-model` flag; works with any compatible draft model.
- **LMStudio speculative decoding (beta):** Reports 20–50% speed gains for MLX models; accessible to end users without custom code.
- **SuffixDecoding potential for MLX:** The suffix tree approach requires no draft model training — just a token cache. This is the most practical speculative decoding path for MLX MoE deployments where maintaining a second draft model doubles memory usage.

**Paper:** [Native LLM and MLLM Inference at Scale on Apple Silicon](https://arxiv.org/html/2601.19139v2)

---

## Analysis: Patterns and Implications for MLX MoE Agentic Inference

### Pattern 1: Agentic Workloads Are More Predictable Than General LLM Traffic

SuffixDecoding and PASTE both exploit this: agents repeat patterns (same JSON wrappers, same tool schemas, similar subtask sequences). This predictability is a resource, not a constraint. Model-free speculative decoding (SuffixDecoding) outperforms model-based approaches (EAGLE) precisely because the suffix tree captures agentic repetition better than a neural draft model.

**Implication for MLX MoE:** SuffixDecoding is the highest-priority speculative decoding technique to implement — it requires no second model (critical given MLX memory constraints), exploits agentic patterns directly, and achieves 5.3x speedup on relevant benchmarks.

### Pattern 2: Structured Output and Prefix Caching Are Complementary, Not Competing

XGrammar reduces per-token overhead for structured output; RadixAttention/APC reduces recomputation of shared prefixes. Both are needed. In an agentic loop, the system prompt + tool schemas (prefix caching target) and the tool call JSON (structured output target) are different bottlenecks.

**Implication:** For a production MLX deployment, the nearest equivalent to XGrammar is the `outlines` library (CPU-based token masking). It has overhead but maintains schema validity. The jump-forward optimization (skipping deterministic JSON tokens) is implementable in pure Python for the MLX generation loop.

### Pattern 3: Context Compression Outperforms Hard Eviction for Tool-Calling Agents

Hard KV eviction (H2O, InfLLM) risks dropping tool result tokens needed for future reasoning. ACON's approach — compressing observations and history into informative summaries — is semantically safer. For an agent session of 50+ turns, the recommended stack is: ACON compression (or LLMLingua-2) + sliding-window hybrid attention (dense on last 1K tokens) rather than eviction.

### Pattern 4: Adaptive Computation Suits Agentic Token Profiles

Agentic output is heterogeneous: verbose reasoning tokens (hard, need full depth) interspersed with JSON structural tokens (easy, early exit is safe). LayerSkip's self-speculative mode is particularly well-matched — the early layers draft the JSON boilerplate; full depth verifies and fills reasoning. This can be implemented in MLX without a second model.

### Pattern 5: MoE Models Have Better Theoretical Fit for Apple Silicon Than Dense Models at Same Parameter Count

For a 30B-A3B MoE model, only 3B parameters are active per token. The memory bandwidth requirement for each decode step is proportional to active parameters, not total. This makes bandwidth-limited Apple Silicon more competitive against GPU for MoE models than for equivalent-quality dense models.

---

## Prioritized Implementation Roadmap for MLX MoE Agentic Inference

| Priority | Optimization | Effort | Speedup Potential | Quality Risk |
|---|---|---|---|---|
| 1 | Prefix KV cache (hash-based APC) | Medium | 2–5x TTFT | Zero |
| 2 | Jump-forward decoding for JSON keys | Low-Medium | 1.5–2x for structured output | Zero |
| 3 | SuffixDecoding (suffix tree speculative) | Medium | 3–5x on agentic patterns | Zero |
| 4 | ACON-style context compression | Low | 26–54% token reduction | Low |
| 5 | Hybrid sparse attention (dense last 1K) | High | 1.5–2x at 32K+ context | Low |
| 6 | LLMLingua-2 preprocessing | Low | 3–6x prompt reduction | Medium |
| 7 | LayerSkip self-speculative (if retraining) | Very High | 1.8–2.2x | Medium |
| 8 | Parallel tool calls (LLMCompiler pattern) | Low | 2–4x wall-clock | Low |
| 9 | PASTE speculative tool execution | Medium | 1.5x wall-clock | Low (idempotent tools) |
| 10 | EAGLE-3 draft model | High | 3–6.5x decode | Very Low |

---

## Gaps and Limitations

1. **No MLX-native XGrammar or RadixAttention** — These are the two highest-impact server-side optimizations; neither has an MLX port. Building them would be significant but high-value contributions.
2. **Apple Silicon speculative decoding benchmarks are sparse** — Most published EAGLE/SuffixDecoding numbers are on NVIDIA GPUs. The Apple ReDrafter numbers (1.37x on M1 Max) suggest the unified memory architecture reduces the speculative decoding advantage compared to GPU (because bandwidth savings matter less when memory is already fast and shared).
3. **Tool-call accuracy under compression is understudied** — ACON reports task-level accuracy; no paper reports fine-grained tool parameter error rates under compression. This is a gap for production deployments.
4. **Adaptive computation requires retraining** — LayerSkip, CALM require the training recipe. SWIFT is plug-and-play but less validated. For off-the-shelf MLX models, early exit is not available without fine-tuning.
5. **NSA and sparse attention training from scratch** — The most principled long-context solutions (NSA, sliding window hybrid) require pretraining; retrofitting them to existing models is not straightforward.
6. **PASTE and SuffixDecoding are very recent** — PASTE (March 2026) and SuffixDecoding (NeurIPS 2025) lack long-term production validation. SuffixDecoding's use in Snowflake ArcticInference is a positive signal.

---

## References

### Prefix Caching
- [SGLang: Efficient Execution of Structured Language Model Programs](https://arxiv.org/abs/2312.07104) — Zheng et al., NeurIPS 2024
- [Fast and Expressive LLM Inference with RadixAttention and SGLang](https://lmsys.org/blog/2024-01-17-sglang/) — LMSYS, Jan 2024
- [SGLang v0.4: Zero-Overhead Batch Scheduler](https://lmsys.org/blog/2024-12-04-sglang-v0-4/) — LMSYS, Dec 2024
- [vLLM Automatic Prefix Caching](https://docs.vllm.ai/en/stable/design/prefix_caching/) — vLLM docs
- [KV-Cache Wins You Can See](https://llm-d.ai/blog/kvcache-wins-you-can-see) — llm-d blog

### Context Compression
- [LLMLingua: Compressing Prompts for Accelerated Inference](https://arxiv.org/abs/2310.05736) — Jiang et al., EMNLP 2023
- [LLMLingua-2: Data Distillation for Efficient and Faithful Prompt Compression](https://arxiv.org/html/2403.12968v2) — ACL 2024
- [LongLLMLingua: Accelerating and Enhancing LLMs in Long Context Scenarios](https://aclanthology.org/2024.acl-long.91/) — ACL 2024
- [ACON: Optimizing Context Compression for Long-horizon LLM Agents](https://arxiv.org/abs/2510.00615) — Kang et al., Oct 2025
- [A Survey on LLM Acceleration based on KV Cache Management](https://arxiv.org/html/2412.19442v3) — Dec 2024

### Structured Output
- [Fast JSON Decoding for Local LLMs with Compressed Finite State Machine](https://lmsys.org/blog/2024-02-05-compressed-fsm/) — LMSYS, Feb 2024
- [XGrammar: Flexible and Efficient Structured Generation Engine for LLMs](https://arxiv.org/abs/2411.15100) — MLC-AI, Nov 2024
- [Achieving Efficient, Flexible, and Portable Structured Generation with XGrammar](https://blog.mlc.ai/2024/11/22/achieving-efficient-flexible-portable-structured-generation-with-xgrammar) — MLC blog
- [Generating Structured Outputs from Language Models: Benchmark and Studies](https://arxiv.org/html/2501.10868v1) — Jan 2025
- [Guided Decoding Performance on vLLM and SGLang](https://blog.squeezebits.com/guided-decoding-performance-vllm-sglang) — SqueezeBits

### Parallel Tool Execution
- [An LLM Compiler for Parallel Function Calling](https://arxiv.org/abs/2312.04511) — Kim et al., ICML 2024
- [Act While Thinking: Accelerating LLM Agents via Pattern-Aware Speculative Tool Execution](https://arxiv.org/html/2603.18897) — March 2026
- [Speculative Actions: A Lossless Framework for Faster Agentic Systems](https://arxiv.org/pdf/2510.04371) — Oct 2025
- [Sherlock: Reliable and Efficient Agentic Workflow Execution](https://arxiv.org/pdf/2511.00330) — Nov 2025
- [Optimizing Agentic Language Model Inference via Speculative Tool Calls](https://arxiv.org/pdf/2512.15834) — Dec 2025

### Speculative Decoding / Draft Models
- [EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty](https://arxiv.org/abs/2401.15077) — SafeAILab, ICML 2024
- [EAGLE-2: Faster Inference of Language Models with Dynamic Draft Trees](https://arxiv.org/abs/2406.16858) — SafeAILab, EMNLP 2024
- [EAGLE-3: Scaling up Inference Acceleration via Training-Time Test](https://arxiv.org/html/2503.01840v1) — SafeAILab, NeurIPS 2025
- [SafeAILab/EAGLE GitHub](https://github.com/SafeAILab/EAGLE)
- [SuffixDecoding: Extreme Speculative Decoding for Emerging AI Applications](https://arxiv.org/abs/2411.04975) — NeurIPS 2025 Spotlight
- [Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads](https://arxiv.org/abs/2401.10774) — 2024
- [Hydra: Sequentially-Dependent Draft Heads for Medusa Decoding](https://arxiv.org/abs/2402.05109) — COLM 2024
- [Recurrent Drafter for Fast Speculative Decoding](https://machinelearning.apple.com/research/recurrent-drafter) — Apple, 2024

### Adaptive Computation
- [LayerSkip: Enabling Early Exit Inference and Self-Speculative Decoding](https://arxiv.org/abs/2404.16710) — Meta, ACL 2024
- [SWIFT: On-the-Fly Self-Speculative Decoding for LLM Inference Acceleration](https://openreview.net/forum?id=EKJhH5D5wA) — ICLR 2025

### Long Context / Sparse Attention
- [Native Sparse Attention: Hardware-Aligned and Natively Trainable Sparse Attention](https://arxiv.org/abs/2502.11089) — DeepSeek-AI, Feb 2025; ACL 2025
- [SpargeAttn: Accurate Sparse Attention Accelerating Any Model Inference](https://arxiv.org/html/2502.18137v1) — Feb 2025
- [The Sparse Frontier: Sparse Attention Trade-offs in Transformer LLMs](https://arxiv.org/pdf/2504.17768) — 2025
- [Long-Context Modeling with Dynamic Hierarchical Sparse Attention for On-Device LLMs](https://arxiv.org/abs/2510.24606) — Oct 2025

### Apple Silicon / MLX
- [Exploring LLMs with MLX and the Neural Accelerators in the M5 GPU](https://machinelearning.apple.com/research/exploring-llms-mlx-m5) — Apple, 2025
- [Native LLM and MLLM Inference at Scale on Apple Silicon](https://arxiv.org/html/2601.19139v2) — Jan 2026
- [Benchmarking On-Device Machine Learning on Apple Silicon with MLX](https://arxiv.org/html/2510.18921v1) — Oct 2025
- [When to Choose SGLang Over vLLM: Multi-Turn Conversations and KV Cache Reuse](https://www.runpod.io/blog/sglang-vs-vllm-kv-cache) — Runpod
- [ml-explore/mlx GitHub](https://github.com/ml-explore/mlx)
