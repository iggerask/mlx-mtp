# Squeezing 27% More Speed from an LLM on Apple Silicon

*What worked, what didn't, and why most optimization intuitions are wrong when you're memory-bandwidth bound.*

---

[`Qwen3.5-35B-A3B-4bit`](https://huggingface.co/mlx-community/Qwen3.5-35B-A3B-4bit) decodes at 73 tokens per second on an M4 Pro (48GB). That's already fast — Apple Silicon's unified memory architecture and MLX's lazy evaluation make it competitive with much more expensive setups. But 73 t/s is also only 28% of the chip's theoretical [memory bandwidth](#term-memory-bandwidth). There's a 3.7x gap between what the hardware can do and what the software delivers. Surely some of that is recoverable.

This is the story of trying to close that gap. We got 27% — from 73 to 90 t/s — through a combination of [speculative decoding](#term-speculative-decoding), custom [Metal SIMD kernels](#term-simd), and an architectural trick that exploits the internal structure of the model's hybrid attention-recurrent layers. Along the way, we tried a dozen approaches that failed, and each failure taught us something about why inference optimization on Apple Silicon is fundamentally different from what works on CUDA.

## The Model

We're running [`mlx-community/Qwen3.5-35B-A3B-4bit`](https://huggingface.co/mlx-community/Qwen3.5-35B-A3B-4bit), a 4-bit quantized version of Alibaba's Qwen3.5-35B-A3B. It's a [Mixture-of-Experts](#term-moe) model with 35 billion total parameters but only 3 billion active per token. It routes through 8 of 256 experts per layer, which makes it fast (only 3B weights to load per step) but also means the bottleneck is [memory bandwidth](#term-memory-bandwidth), not compute. At [4-bit quantization](#term-quantization), the model fits in ~18GB and each decode step loads roughly 1.5GB of weights through the memory system.

The architecture is hybrid: 30 of its 40 layers use [GatedDeltaNet](#term-gdn) instead of standard [attention](#term-attention). Only every 4th layer uses full attention with [KV cache](#term-kv-cache). This hybrid design is central to everything that follows — it's both the reason most standard optimizations fail and the key to the one that works.

The original model ships with a [Multi-Token Prediction head](#term-mtp): a small auxiliary network trained alongside the main model to predict the next token from the backbone's hidden states. This is the entry point for speculative decoding. However, the mlx-community quantized version strips the MTP weights to save space — they need to be extracted separately from the original BF16 checkpoint ([`Qwen/Qwen3.5-35B-A3B`](https://huggingface.co/Qwen/Qwen3.5-35B-A3B)) using a weight extraction script. The extracted weights are stored in a single safetensors file (~1.7GB at BF16, or ~475MB after runtime quantization to Q4).

## The First Win

Speculative decoding has a simple loop: draft a token cheaply, verify it with the full model, accept if the draft was right, reject and use the model's answer if it wasn't. The MTP head is the drafter — it takes the model's hidden state and a token embedding, concatenates them, and predicts the next token.

The first step was extracting the MTP head weights from the original BF16 model. The quantized `mlx-community` model strips them, so we download the original [`Qwen/Qwen3.5-35B-A3B`](https://huggingface.co/Qwen/Qwen3.5-35B-A3B) checkpoint and pull out the `mtp.*` tensors — about 785 individual weight tensors covering the MTP head's projections, norms, attention, and a full MoE MLP layer (256 experts + router + shared expert). These get consolidated into a single safetensors file.

Getting the head to produce correct predictions required some care. Qwen3.5 stores [RMSNorm](#term-rmsnorm) weights as `(weight - 1)` in its raw checkpoints — a numerical trick that shifts the distribution closer to zero for better training stability. The mlx-lm model loader has a sanitization step that adds 1 back, but the MTP head weights are loaded separately and bypass this sanitizer. Missing the +1 caused silent accuracy degradation that looked like the head was just "not very good" rather than fundamentally broken — the kind of bug that takes hours to notice because the output is plausible, just worse than it should be.

With the head working correctly: 80% acceptance, 80 t/s, 1.09x baseline. Not bad for a first implementation.

Three micro-optimizations stacked on top of this:

| Optimization | Mechanism | Cumulative |
|---|---|---|
| [Batch verify](#term-batch-verify) | Process `[token_0, draft]` together instead of sequentially | 1.09x |
| Lazy draft eval | Don't sync after drafting — let MLX build one fused [computation graph](#term-computation-graph) for draft+verify | 1.11x |
| [Q4 MTP head](#term-quantization) | Quantize the MTP head from BF16 (1689 MB) to 4-bit (475 MB) | 1.13x |

Why does lazy eval help? MLX uses [lazy evaluation](#term-lazy-eval): operations aren't executed immediately but accumulated into a computation graph. Calling `mx.eval()` sends the entire graph to the GPU at once. If we evaluate the draft token eagerly (to check its value), we force two separate GPU submissions: one for the draft, one for the verify. Keeping both lazy lets them fuse into a single submission, saving ~0.5ms of GPU dispatch overhead per step.

The Q4 head quantizes the BF16 MTP weights to 4-bit at runtime, cutting the head's memory footprint from 1689 MB to 475 MB — 72% less [memory bandwidth](#term-memory-bandwidth) per draft. Acceptance drops marginally (80% to 76%) but the faster draft more than compensates.

At 83 t/s (1.13x), we hit the Python-level optimization ceiling. Every other Python-level approach we tried made things slower. Here's why.

## The Graveyard of Good Ideas

Each failed optimization reveals something fundamental about inference on memory-bandwidth-bound hardware. Understanding why they fail is more valuable than knowing what works.

### "Just skip the unimportant layers"

[Block Influence](#term-block-influence) profiling measures how much each layer transforms its input: `BI = 1 - cosine_similarity(input, output)`. A layer with BI near zero is effectively an identity function — it passes its input through unchanged. We found 10 layers with BI < 0.03, all near-identity. Skipping all 10 preserved 100% token-match with baseline. The model literally produces the same output without them.

Speedup: 0%.

The skipped layers were all [GDN](#term-gdn) (recurrent) layers. During decode, each GDN layer's recurrent update processes one token through a small state matrix — about 0.1ms of work. The [MoE](#term-moe) layers, by contrast, load and multiply through 8 expert weight matrices at ~0.3ms each. The expensive layers are the ones that matter for quality. The cheap ones are the ones we can skip. Removing layers that cost nothing saves nothing.

### "Exit early when the model is confident"

[CALM-style early exit](#term-early-exit): at each layer, run the hidden state through the [LM head](#term-lm-head) (the final projection that converts hidden states to vocabulary probabilities). If the top prediction is confident enough, skip the remaining layers.

Average exit layer: 39 out of 40. The model essentially never becomes confident before the final layer. These models are trained to use their full depth — confidence builds gradually across layers, not in a sudden "aha" moment. Meanwhile, each probe costs ~1ms (the LM head is a large matrix multiply), adding overhead to every step. Result: 0.65x to 0.75x.

### "Use a smaller model as the drafter"

Qwen3.5-4B as a draft model for the 35B: only 2-5% token agreement. Despite sharing a tokenizer, the models have learned completely different probability distributions over next tokens. The 4B model's confident predictions are almost never what the 35B model would have said. This is a fundamental problem with separate-model speculative decoding: the drafter must closely approximate the verifier's distribution, which is hard to guarantee across different model sizes.

### "Draft with the shared expert"

The [MoE](#term-moe) architecture includes a [shared expert](#term-shared-expert) that runs on every token regardless of routing. Monkey-patching the model to use only the shared expert (skipping the router and selected experts) creates a very cheap draft model — same architecture, same weights, just fewer computations.

Acceptance: ~100%. The shared expert alone predicts nearly the same tokens as the full ensemble.

Throughput: 0.85x. Slower than baseline.

This is the critical insight about memory-bandwidth-bound models: **running the model twice is always slower than running it once, regardless of how cheap you make the draft.** Each forward pass loads weight matrices through memory for all 40 layers — projections, norms, routing weights. Whether you use 8 experts or 1, the per-layer overhead is dominated by loading these fixed-cost weights. The shared expert draft is "cheap" in [FLOPs](#term-flops) but still pays nearly the full memory bandwidth cost.

### "Compile the computation graph"

MLX provides `mx.compile()` for [graph-level fusion](#term-graph-fusion), similar to `torch.compile` in PyTorch. The idea is to let the framework analyze the computation graph and fuse multiple operations into more efficient kernels.

Applied to the full model: fails. The GDN layers use `ArraysCache` for their [recurrent state](#term-recurrent-state), which contains non-array metadata that the compiler can't trace through.

Applied to individual MoE blocks: fails. The `argpartition` used for top-K expert routing produces data-dependent output shapes that the shape inference can't resolve.

Applied to smaller sub-graphs (norms, MLPs): 0% to +1%. Within measurement noise.

The reason: MLX's [lazy evaluation](#term-lazy-eval) already provides the same benefit. A full decode step builds a [computation graph](#term-computation-graph) of 2,612 operations, and `mx.eval()` is called exactly once — sending the entire graph to the GPU as a single [Metal command buffer](#term-command-buffer). The graph is already fused at the framework level. There's no additional fusion opportunity for the compiler.

### "Write a custom Metal kernel"

A fused softmax+topk+normalize kernel was implemented using MLX's `mx.fast.metal_kernel` API, replacing 4 separate operations in the MoE router.

Result: 0.90x. 10% slower than the 4 separate operations it replaced.

The `metal_kernel` API executes custom kernels in separate [Metal command buffers](#term-command-buffer), outside of MLX's main computation graph. The 4 native operations share a single command buffer with 2,608 other operations, where per-operation dispatch overhead is ~15 microseconds. A custom kernel in its own buffer pays ~186 microseconds — over 12x more overhead for the same work.

**Any custom kernel must participate in the native MLX graph to be useful. The `metal_kernel` API is not the right path — you need a C++ extension that registers as a native [primitive](#term-primitive).**

### What the graveyard taught us

Every failed approach boils down to the same root cause: this model is [memory-bandwidth bound](#term-memory-bandwidth). Each decode step loads ~1.5GB of weights. The M4 Pro delivers 273 GB/s, giving a theoretical floor of 5.5ms per step (182 t/s). The actual step takes 13.3ms (75 t/s), of which 11.7ms is GPU execution and 1.6ms is Python graph construction.

The gap between theoretical and observed GPU time is Metal's internal overhead: command buffer scheduling, memory barriers between dependent kernels, per-kernel [compute encoder](#term-compute-encoder) setup. This overhead is intrinsic to the Metal runtime and cannot be reduced from user code.

The only path to faster inference is processing more tokens per decode step. [Speculative decoding](#term-speculative-decoding) does this. Everything else either doesn't reduce the dominant cost (layer skip, early exit, graph compilation) or doubles it (two-model drafting, shared-expert drafting).

## Breaking Through the Python Ceiling

The custom Metal kernel failure pointed toward the solution: kernels break graph optimization when they run in separate command buffers, but a C++ [primitive](#term-primitive) that registers with MLX's computation graph shares the buffer. This is the difference between `mx.fast.metal_kernel` (external, separate command buffer) and a `UnaryPrimitive` subclass built as a C++ extension (native, participates in the graph).

We built a fused [MoE](#term-moe) kernel that combines three operations per layer — gate projection, up projection, and [SwiGLU](#term-swiglu) activation — into a single [Metal dispatch](#term-metal-dispatch). Normally these are two separate [gather_qmm](#term-gather-qmm) calls (one for the gate weights, one for the up weights) that each produce an intermediate tensor written to DRAM and read back. The fused kernel reads the input once, computes both projections against 4-bit quantized expert weights, and applies SiLU gating in-register — no intermediate memory round-trip.

### The naive kernel that made things worse

The first version was a straightforward per-thread implementation: each thread computes one output element, independently reading the full 2048-element input vector.

Isolated performance: 2.3x faster than the reference operations.
Full model impact: **0.96x. Four percent slower than doing nothing.**

How can a 2.3x-faster kernel slow down the model? Two reasons. First, isolated benchmarks measure the kernel in its own command buffer — which we just learned adds ~186 microseconds of overhead per dispatch. In the full model, the native operations share a buffer at 15 microseconds each. Second, the naive kernel under-utilized Metal's [SIMD](#term-simd) execution model.

Metal's GPU executes threads in [SIMD groups](#term-simd) of 32 (sometimes called "warps" on NVIDIA hardware). All 32 threads execute the same instruction simultaneously. In the naive kernel, each thread independently loaded the same 2048-element input vector from memory — 32 threads doing 32 redundant reads of the same data. This wastes memory bandwidth, which is exactly the resource we're short on.

### The SIMD-tiled rewrite

The rewrite matched MLX's own `qmv_fast` kernel pattern, which is the gold standard for quantized matrix-vector products on Apple Silicon:

- **[Threadgroup](#term-threadgroup)**: 64 threads = 2 [SIMD groups](#term-simd) of 32
- **Cooperative x-loading**: Each thread loads 16 elements of the input vector. 32 threads in a SIMD group collectively load all 512 elements (the intermediate dimension) in one pass. The input is read from memory exactly once per SIMD group, not 32 times.
- **[SIMD reduction](#term-simd-sum)**: `simd_sum` computes the dot product reduction across all 32 threads in a single hardware instruction, replacing explicit shared-memory accumulation
- **Pre-division trick**: For [4-bit dequantization](#term-quantization), the input values are pre-divided by powers of 16 during the cooperative loading phase. This lets the inner loop mask packed uint16 weight values with simple bitwise operations instead of per-element division — a significant win at the inner-loop level.
- **8 output rows per threadgroup**: Each SIMD group computes 4 output rows, giving good [arithmetic intensity](#term-arithmetic-intensity) (ratio of compute to memory access) per weight load

The rewritten kernel: 2.5x isolated, **1.046x in the full model**. Modest, but real. A second kernel fusing down-projection with score-weighted expert summation added another 1%: 1.04x total from fused kernels.

Combined with MTP speculative decoding: **1.19x** (84.4 t/s). The fused kernels reduce the per-step cost, and MTP amortizes that cost over ~1.82 tokens per step.

But there was still a tax we hadn't addressed.

## The 14ms Replay Tax

When a speculative draft is rejected, the model's cache is in the wrong state. For standard [attention layers](#term-attention), the fix is simple: the [KV cache](#term-kv-cache) stores key-value pairs indexed by position, so you just decrement the position counter to "forget" the rejected token. But for the [GDN recurrent layers](#term-gdn), the state is a compressed summary of all tokens seen so far. The rejected token's contribution has been folded into this summary through the recurrence update. There's no "undo" operation — you can't subtract one token's influence from the accumulated state.

The standard fix: save the entire cache state before verification, and on rejection, restore it and replay the accepted tokens through the model. For K=1 (our best configuration), this means replaying one token — running a full forward pass through all 40 layers. Cost: ~14ms, which is a full baseline decode step.

The cost model makes the penalty clear:

```
Accept (81%):  18ms verify  →  2 tokens  =  9.0 ms/tok
Reject (19%):  18ms verify + 14ms replay  →  1 token  = 32.0 ms/tok
Average:       20.7ms  →  1.81 tokens  =  11.4 ms/tok
```

On 19% of steps, we're paying double. If we could eliminate the replay:

```
Accept (81%):  18ms verify  →  2 tokens  =  9.0 ms/tok
Reject (19%):  18ms verify  →  1 token   = 18.0 ms/tok
Average:       18.0ms  →  1.81 tokens  =  9.9 ms/tok  (+15% theoretical)
```

Before attacking the replay directly, we explored whether speculating deeper (K=2, K=3) could amortize it. K=2 drafts two tokens and verifies three at once. But the 3-token batch loads 3x8=24 expert weight matrices per MoE layer — about 21ms per step. Combined with a drop to 67% acceptance at depth 2 (the [MTP head](#term-mtp) degrades when chained, since it was trained on backbone hidden states, not its own outputs), K=2 was a net loss: 0.95x.

Cascade verification improved this by avoiding the expensive 3-token batch entirely: verify `[token_0, draft1]` as a normal 2-token batch, then check position 1's logits for draft2 for free (those logits are already computed). If draft2 matches: do one extra 1-token forward. If not: identical cost to K=1, no wasted work. This brought K=2 from 0.95x to 1.01x — break-even, but still below K=1.

Adaptive K dynamically switches between K=1 and K=2 based on a rolling window of recent acceptance rates. At a 90% threshold, it correctly stays in K=1 mode for most content and only upgrades for highly repetitive sequences. Result: 1.10x — better, but still below pure K=1's 1.14x.

The path was clear: eliminate the replay forward on K=1 rejection.

## The Failed Shortcuts

Two variants of "lossy rejection" were tested — accepting a slightly corrupted cache state instead of paying for the full replay.

**Variant A: keep the wrong draft in the GDN state, just trim KV.** The [GDN recurrent update](#term-gdn) follows the rule `state = g * state + beta * (v ⊗ k)`, where g is a learned gating factor that controls how much old state is retained vs replaced by new information. When a draft is rejected but its contribution remains in the state, the wrong token's key-value outer product persists, scaled by g ≈ 0.9 to 0.99. With 30 GDN layers each carrying this error forward, quality collapsed within a few tokens:

| Prompt | K=1 First Divergence |
|---|---|
| Fibonacci code | token 5 |
| Theory of relativity | token 8 |
| TCP vs UDP differences | token 46 |
| Counting 1-30 | exact match |

The fibonacci prompt illustrates the corruption vividly — the model produced `if n = if n == 0: return 0 if n == 1: retur`, stuttering as the corrupted GDN state (which "remembers" the wrong draft) fights with the correct KV cache (which only has the accepted tokens).

**Variant B: restore GDN to the pre-verify snapshot (before token_0 was processed), keep token_0's KV entry.** This avoids the wrong draft's contribution but creates a different problem: the attention layers know about token_0 (it's in the KV cache) but the recurrent layers don't (their state was rolled back to before token_0). This KV/GDN desynchronization causes the model to simultaneously "remember" and "forget" the same token through different mechanisms:

| Prompt | K=1 First Divergence |
|---|---|
| Fibonacci code | token 11 |
| Theory of relativity | token 8 |
| TCP vs UDP differences | token 46 |
| Counting 1-30 | exact match |

Slightly better — missing a token is less corrupting than having the wrong one — but still unacceptable for general text. (The counting prompt matches both times because its tokens are so predictable that the model can recover from minor state corruption.)

The root cause is the same for both variants. The GDN gating factor g is close to 1, meaning state perturbations decay slowly — a wrong or missing token's influence persists through dozens of subsequent steps. Each of the 30 GDN layers independently accumulates this error, and the errors compound as they propagate through the full layer stack. Lossy rejection is fundamentally incompatible with recurrent architectures.

## The Solution: Split the Metal Kernel

During a 2-token [batch verify](#term-batch-verify) `[token_0, draft]`, the [GDN](#term-gdn) recurrence processes both tokens sequentially inside a single [Metal kernel dispatch](#term-metal-dispatch). First it updates the state with token_0, then with the draft. The intermediate state — after token_0 but before the draft — exists momentarily inside the GPU's registers but is never written out. Only the final state (after both tokens) gets saved to the cache.

If we could capture that intermediate state, we could restore to it on rejection: exact GDN state after token_0, trim KV by 1, no replay needed.

### Attempt 1: Force the Python path (too slow)

The first attempt monkey-patched `gated_delta_update` to force the Python [ops path](#term-ops-path) during verification. Normally, a multi-token batch goes through the optimized Metal kernel in one GPU dispatch. The ops path instead uses a Python loop that processes tokens one at a time, calling `_gated_delta_step_ops` for each. This lets us capture the state between iterations — exactly the intermediate we need.

It works. The intermediate state is exact. But the ops path is significantly slower than the Metal kernel for multi-token batches. Every verify step (not just rejected ones) pays a 3-5ms penalty:

| Config | Standard | Ops Path Zero-Replay |
|---|---|---|
| K=1 | 1.14x | 1.06x |
| K=1 + fused MoE | 1.19x | 1.10x |

On repetitive content with 98% acceptance (almost no replays to save), the ops path penalty is pure overhead: 95.0 vs 79.1 t/s. We're paying a tax on every step to avoid a cost that only occurs 2% of the time.

### Attempt 2: Split the Metal calls (the breakthrough)

The solution came from a simple observation: we don't need the Python ops path to process tokens individually. We can call the Metal kernel itself with one token at a time. Instead of one call to `gated_delta_update` with a 2-token batch, make two calls with 1-token batches. Each call uses the full Metal kernel optimization — [SIMD lanes](#term-simd), hardware scheduling, everything. We capture the intermediate state between calls.

```python
def _split_gated_delta_update(q, k, v, ..., state, mask, use_kernel):
    B, T, *_ = q.shape
    if capture_enabled and T > 1:
        for t in range(T):
            y_t, state = original_gated_delta_update(
                q[:, t:t+1], ..., state, ...,
                use_kernel=True,   # Full Metal kernel for each token
            )
            if t < T - 1:
                intermediates.append(mx.array(state))  # Capture!
        return concat(ys), state
    else:
        return original_gated_delta_update(q, k, v, ..., state, mask, use_kernel)
```

The overhead is one extra Metal kernel dispatch per GDN layer per verify step. With ~30 GDN layers and ~0.05ms per dispatch, that's ~1.5ms additional cost on every step. The replay saving is 19% of steps times 14ms = 2.7ms average. Net gain: ~1.2ms per step.

Why is this so much better than the ops path? The Python ops path's overhead wasn't from processing tokens individually — it was from using an unoptimized code path for the actual computation. The Metal kernel processes a single token just as efficiently as it processes a batch (for the GDN recurrence, the "batch" is sequential anyway). The only additional cost is the kernel dispatch overhead, which is tiny.

### Restoring the conv state

There's a detail that took a while to get right. GDN layers maintain two pieces of state: the [recurrent state](#term-recurrent-state) (the accumulated delta-rule matrix, captured by the split above) and a [conv sliding window](#term-conv-state) (the last 3 rows of QKV projections, used for a short-range 1D convolution). On rejection, both must be restored to the intermediate position.

The conv state after a 2-token verify contains `[old_row, token_0_qkv, draft_qkv]`. The intermediate (after token_0 only) should contain `[older_row, old_row, token_0_qkv]`. We can reconstruct this from the pre-verify and post-verify states without additional captures:

```
token_start = max(0, kernel_size - 1 - n_verify_tokens)
intermediate_conv = concat(
    saved_conv[position + 1 :],
    current_conv[token_start : token_start + position + 1]
)
```

Before adding conv restoration, quality was significantly degraded — divergence within 16 tokens on some prompts. The conv window is only 4 tokens wide, but even this small corruption was enough to throw off the subsequent QKV projections. After restoring both recurrent and conv state, the remaining numerical difference vs the standard replay path is only from how Metal tiles its quantized matrix multiplies differently for batch-of-2 vs two batch-of-1 calls. This is a [floating-point non-associativity](#term-fp-nonassoc) artifact, not state corruption. Zero-replay produces bit-identical output when run twice with the same input.

## The Final Numbers

The complete optimization stack, each layer building on the previous:

| Optimization | tok/s | vs Baseline | Contribution |
|---|---|---|---|
| Baseline | 70.9 | 1.00x | -- |
| + MTP K=1 batch verify | 80.7 | 1.14x | +14% |
| + Fused MoE SIMD kernels | 84.4 | 1.19x | +5% |
| + Zero-replay (split Metal) | **90.4** | **1.27x** | +8% |

Per content type, the gains are consistent:

| Category | Baseline | Final (all_zr_k1) | Speedup |
|---|---|---|---|
| Code | 72.0 | 90.2 | 1.26x |
| Prose | 70.1 | 85.5 | 1.22x |
| Repetitive | 70.5 | 100.6 | 1.43x |
| Q&A | 71.1 | 86.2 | 1.24x |

K=2 and K=3 with zero-replay were also tested. The replay elimination makes K=2 genuinely viable now (1.18x vs the standard K=2's 0.95x — a dramatic swing), but K=1 still wins because the MTP head's accuracy degrades sharply at depth 2 (64% vs 79% acceptance). At K=3, acceptance drops to 37%. The head was trained to predict one token ahead from backbone hidden states. Chaining it for deeper speculation makes its predictions increasingly [out-of-distribution](#term-ood) — it's seeing its own outputs instead of the backbone's, and it was never trained for that.

| Config | tok/s | vs Base | Acceptance | Tok/Step |
|---|---|---|---|---|
| all_zr_k1 | **90.4** | **1.27x** | 79% | 1.80 |
| all_zr_k2 | 83.0 | 1.18x | 64% | 2.30 |
| all_zr_k3 | 60.6 | 0.86x | 37% | 2.11 |

## What This Means

We moved from 28% to 34% of theoretical memory bandwidth utilization. The remaining 66% gap is Metal's internal scheduling overhead for 2,612 operations per decode step: [command buffer](#term-command-buffer) processing, memory barriers between dependent kernels, per-kernel compute encoder setup. This is intrinsic to the Metal runtime and not addressable from user code.

What would close more of the gap:

- **Deeper kernel fusion in MLX** (fusing RMSNorm, Matmul, and Activation chains): would reduce 2,612 ops to ~200-300, dramatically cutting scheduling overhead. This is an MLX framework-level change.
- **Metal megakernels**: running an entire transformer layer as one persistent kernel, avoiding per-op scheduling. Used by FlashInfer and TensorRT-LLM on CUDA, not yet feasible on Metal.
- **M5 with Metal 4 TensorOps**: Apple's upcoming dedicated matrix multiply accelerator, estimated 20-27% decode improvement from increased effective bandwidth.
- **Better MTP heads**: a 2-layer head trained with curriculum learning on its own outputs could sustain >80% acceptance at K=2, pushing tokens-per-step from 1.8 to ~2.3 and throughput past 100 t/s.

The approach generalizes beyond this specific model. Any hybrid attention-recurrent architecture — Jamba, Zamba, future Mamba-attention hybrids — will face the same replay tax on speculative decoding rejection. The split-kernel intermediate capture technique applies directly to any architecture where the recurrent update can be decomposed into per-token calls, which is true by definition for any sequential recurrence.

The broader lesson is about where optimization leverage exists on memory-bandwidth-bound hardware. Anything that doesn't reduce weight-loading volume or increase tokens-per-load is wasted effort. Layer skipping, early exit, graph compilation, model distillation for drafting — they all fail because they don't touch the dominant cost. Speculative decoding works because it amortizes weight loads across multiple accepted tokens. Custom SIMD kernels work because they reduce intermediate memory round-trips. And zero-replay works because it eliminates a redundant weight load that only existed because the software couldn't capture a value the hardware had already computed.

Sometimes the best optimization is noticing that the answer already exists inside a computation you're about to throw away.

---

## Glossary

<span id="term-memory-bandwidth">**Memory Bandwidth**</span>: The rate at which data can be read from (or written to) memory, measured in GB/s. During LLM decode, the GPU must read the model's weight matrices from memory for every token generated. The M4 Pro has 273 GB/s of bandwidth (48GB unified memory). When the GPU can compute faster than it can read data, the workload is "memory-bandwidth bound" — the bottleneck is data transfer, not arithmetic. This is the case for single-token LLM decode on virtually all current hardware.

<span id="term-speculative-decoding">**Speculative Decoding**</span>: A technique to generate multiple tokens per model forward pass. A cheap "drafter" quickly predicts the next token(s), then the full model verifies those predictions in a single batched forward pass. Correct predictions are accepted for free (the model already computed them during verification); incorrect ones are replaced with the model's own predictions. The net effect is more tokens per unit of compute, at the cost of occasionally wasting work on wrong drafts.

<span id="term-moe">**Mixture-of-Experts (MoE)**</span>: An architecture where each layer contains many "expert" sub-networks (e.g., 256) but only activates a few (e.g., 8) per token. A learned router decides which experts handle each token. This allows models with very large total parameter counts (and thus knowledge capacity) to run at the cost of a much smaller model (only the active experts' weights need to be loaded). The tradeoff: total model size is large (affects download/storage), but per-token compute is small.

<span id="term-gdn">**GatedDeltaNet (GDN)**</span>: A linear recurrent layer that maintains a compressed state matrix, updated with each token via a delta rule: `state = g * state + beta * (v ⊗ k)`. The gating factor `g` controls how quickly old information decays. Unlike attention, which stores explicit key-value pairs for every past token, GDN compresses the entire history into a fixed-size state. This makes it memory-efficient for long sequences but means you can't selectively "undo" individual tokens from the state — it's a lossy compression.

<span id="term-attention">**Attention**</span>: The standard mechanism in transformers where each token attends to all previous tokens via a query-key-value computation. The model stores key and value vectors for every past token in a KV cache. Attention's strength is precise token-level recall; its weakness is that the KV cache grows linearly with sequence length, consuming memory.

<span id="term-kv-cache">**KV Cache**</span>: A buffer storing the key and value vectors computed for each past token in an attention layer. During decode, the model doesn't reprocess the full prompt — it reads keys and values from the cache. The cache is indexed by position, so "undoing" a token is trivial: decrement the position counter. This simplicity is what makes speculative decoding straightforward for pure-attention models, and what makes hybrid models with recurrent state more challenging.

<span id="term-mtp">**Multi-Token Prediction (MTP) Head**</span>: An auxiliary network trained alongside the main model to predict the next token from the backbone's hidden states. Unlike the main model's LM head (which just projects hidden states to vocabulary), the MTP head takes both a hidden state and the current token's embedding as input, giving it richer context. Models that ship with an MTP head (like Qwen3.5, DeepSeek-V3) provide a built-in drafter for speculative decoding — no separate model needed.

<span id="term-quantization">**Quantization (4-bit / Q4)**</span>: Representing model weights with fewer bits than the training precision (typically BF16 = 16 bits). 4-bit quantization stores each weight in 4 bits (16 possible values) plus per-group scale and bias factors for reconstruction. This reduces the model's memory footprint and bandwidth requirements by ~4x, with minimal quality loss for inference. The dequantization happens on-the-fly during matrix multiplication.

<span id="term-simd">**SIMD (Single Instruction, Multiple Data)**</span>: A hardware execution model where 32 threads (on Apple Silicon) execute the same instruction simultaneously on different data. Metal calls these "SIMD groups" (NVIDIA calls them "warps"). Writing efficient Metal kernels means designing for SIMD-width operations: cooperative data loading across 32 threads, `simd_sum` for parallel reductions, and avoiding thread divergence (where different threads in a group need to do different things).

<span id="term-batch-verify">**Batch Verification**</span>: Processing the current token and drafted token(s) together in a single model forward pass, rather than running them sequentially. For a 2-token batch `[token_0, draft]`, the model processes both in one pass and produces logits at both positions. Position 0's logits verify the draft; position 1's logits give a "bonus" next token if the draft is accepted. This amortizes the fixed overhead of a forward pass across more tokens.

<span id="term-lazy-eval">**Lazy Evaluation**</span>: MLX's execution model where operations aren't computed immediately. Instead, they're recorded into a computation graph (a DAG of operations and their dependencies). The graph is only executed when `mx.eval()` is called, at which point the entire graph is sent to the GPU as a single Metal command buffer. This allows automatic operation batching and eliminates Python-level dispatch overhead between operations.

<span id="term-computation-graph">**Computation Graph**</span>: A directed acyclic graph (DAG) representing the sequence of operations needed to compute a result. Each node is an operation (matmul, add, softmax, etc.) and edges represent data dependencies. MLX builds this graph lazily, then executes it all at once. A single decode step for Qwen3.5 produces a graph of 2,612 operations.

<span id="term-command-buffer">**Metal Command Buffer**</span>: The unit of work submitted to Apple's GPU. Contains a sequence of compute commands (kernel dispatches) that the GPU executes in order. When MLX calls `mx.eval()`, it encodes all 2,612 operations from the computation graph into a single command buffer. Operations within one buffer share scheduling resources and can be dispatched with ~15 microseconds of overhead each. Operations in separate buffers pay ~186 microseconds each — the cost of command buffer setup, submission, and synchronization.

<span id="term-rmsnorm">**RMSNorm**</span>: Root Mean Square Layer Normalization. A simplified normalization layer that divides each element by the RMS of the vector, then scales by a learned weight. Qwen3.5 stores these weights shifted by -1 (so a weight of 1.0 is stored as 0.0), which is a training stability trick that requires adjustment during loading.

<span id="term-swiglu">**SwiGLU**</span>: An activation function used in modern transformers: `SwiGLU(x) = SiLU(gate(x)) * up(x)`, where gate and up are separate linear projections of the input. It requires two matrix multiplications (gate and up), making it a natural target for kernel fusion — both projections read the same input, so they can share a single memory load.

<span id="term-gather-qmm">**gather_qmm**</span>: "Gather Quantized Matrix Multiply" — MLX's kernel for MoE inference. Given expert indices (which 8 of 256 experts to use), it gathers the relevant weight rows from the quantized weight tensor and performs the matrix multiply. This avoids materializing the full 256-expert weight matrix in memory.

<span id="term-metal-dispatch">**Metal Kernel Dispatch**</span>: A single GPU compute operation. The CPU encodes a kernel dispatch into a command buffer, specifying the kernel function, threadgroup size, and grid dimensions. The GPU executes the kernel across all threadgroups. Dispatch overhead (~15 microseconds within a shared command buffer) is the cost of the GPU setting up compute encoders, loading the kernel, and scheduling threadgroups.

<span id="term-primitive">**MLX Primitive**</span>: The base class for operations in MLX's computation graph. Native primitives (Add, Matmul, etc.) are implemented in C++ and participate in graph-level optimization. Custom C++ extensions subclass `Primitive` to add new operations that integrate seamlessly — sharing command buffers, participating in lazy evaluation, and being visible to the graph optimizer. This is distinct from `mx.fast.metal_kernel`, which runs outside the graph.

<span id="term-threadgroup">**Threadgroup**</span>: Metal's unit of thread organization. A threadgroup is a block of threads (typically 64 for our kernels) that can share memory and synchronize. Threadgroups are divided into SIMD groups of 32 threads. The kernel's grid dimensions determine how many threadgroups are dispatched.

<span id="term-simd-sum">**simd_sum**</span>: A Metal intrinsic that sums a value across all 32 threads in a SIMD group in a single hardware instruction. Used for dot product reductions: each thread computes a partial sum, then `simd_sum` produces the total. Much faster than explicit shared-memory accumulation with barriers.

<span id="term-block-influence">**Block Influence (BI)**</span>: A metric for measuring how much a transformer layer changes its input: `BI = 1 - cosine_similarity(input, output)`. A BI near 0 means the layer is near-identity (passthrough). Used to identify skip candidates for layer pruning or early exit.

<span id="term-early-exit">**Early Exit (CALM)**</span>: A technique where the model stops processing after an intermediate layer if it's already confident about the next token. Requires running the LM head as a "probe" at intermediate layers to check confidence. Works well on some architectures but fails on models that build confidence gradually across all layers.

<span id="term-lm-head">**LM Head**</span>: The final linear projection layer that converts a hidden state vector (e.g., 2048 dimensions) into vocabulary logits (e.g., 152,064 dimensions — one per possible token). The token with the highest logit is the model's prediction. This is one of the largest single matrix multiplies in the model.

<span id="term-shared-expert">**Shared Expert**</span>: In MoE architectures, an expert that processes every token regardless of routing decisions. It provides a "baseline" computation that the routed experts specialize on top of. Because it runs unconditionally, it can be used as a cheap approximation of the full model — though as we found, "cheap in compute" doesn't mean "cheap in memory bandwidth."

<span id="term-flops">**FLOPs**</span>: Floating-point operations per second — a measure of compute throughput. When a workload is "FLOP-bound," the GPU's arithmetic units are the bottleneck. When it's "memory-bandwidth-bound," the data transfer rate is the bottleneck. Single-token LLM decode is almost always bandwidth-bound because each weight is used exactly once per token (low arithmetic intensity).

<span id="term-graph-fusion">**Graph-Level Fusion**</span>: Combining multiple operations into fewer, more efficient GPU kernel dispatches. For example, fusing `RMSNorm → Linear → SiLU` into one kernel avoids writing intermediate results to memory. MLX's lazy evaluation provides basic fusion by batching all operations into one command buffer. Deeper fusion (combining the operations themselves) requires compiler support or custom kernels.

<span id="term-compute-encoder">**Compute Encoder**</span>: A Metal object that records compute commands (kernel dispatches) into a command buffer. Each kernel dispatch requires the encoder to set the kernel function, bind buffer arguments, and encode the threadgroup/grid dimensions. The encoder setup time contributes to the per-dispatch overhead.

<span id="term-arithmetic-intensity">**Arithmetic Intensity**</span>: The ratio of compute operations (FLOPs) to bytes transferred from memory. High arithmetic intensity means the GPU spends more time computing than waiting for data — good for GPU utilization. Matrix multiplications have intensity proportional to the matrix dimensions. For single-token decode with small batch sizes, intensity is low (each weight is used once), making the workload bandwidth-bound.

<span id="term-recurrent-state">**Recurrent State**</span>: The fixed-size state matrix maintained by GDN layers, representing a compressed summary of all tokens processed so far. Unlike KV cache (which grows with sequence length), recurrent state has constant size regardless of history length. The tradeoff: you can't selectively remove a token's contribution, making speculative decoding rollback more complex.

<span id="term-conv-state">**Conv Sliding Window**</span>: A short buffer (typically 3 rows for kernel_size=4) storing the most recent QKV projections. Used by the 1D convolution in GDN layers for local context modeling before the recurrent update. Must be restored alongside the recurrent state during zero-replay rollback.

<span id="term-ops-path">**Ops Path (Python)**</span>: The fallback implementation of GDN recurrence in Python, using a loop that processes tokens one at a time via `_gated_delta_step_ops`. Slower than the Metal kernel but allows intercepting intermediate states between tokens. The split Metal kernel approach achieves the same interception without the performance penalty.

<span id="term-fp-nonassoc">**Floating-Point Non-Associativity**</span>: The mathematical property that `(a + b) + c` does not always equal `a + (b + c)` in floating-point arithmetic due to rounding. This means GPU operations that accumulate results in different orders (e.g., different GEMM tiling for batch-of-2 vs two batch-of-1) can produce slightly different results. Both are "correct" — they're just different valid roundings of the same mathematical operation.

<span id="term-ood">**Out-of-Distribution (OOD)**</span>: When a model receives inputs that differ from its training data. The MTP head was trained on the backbone model's hidden states but during multi-step speculation (K>1), it receives its own outputs as input — something it never saw during training. This distribution shift causes accuracy to degrade with each additional speculative step.
