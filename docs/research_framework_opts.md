# MLX Framework-Level Optimizations for LLM Inference on Apple Silicon

**Research date**: 2026-03-29
**Scope**: Actionable, implementation-level techniques for decode throughput on MoE models, specifically targeting Qwen3.5-35B-A3B + MTP on M-series hardware.

---

## Executive Summary

Six concrete optimization paths exist, ordered by expected impact and implementation feasibility:

1. **gather_qmm_swiglu C++ primitive** (ZMLX pattern): Fuse gate+up+SwiGLU into one Metal dispatch. Requires custom MLX fork (~800 lines C++). Best proven technique for MoE decode.
2. **mx.compile on the full decode step**: Already available; wrap the entire token-generation graph. GELU fusion alone is 5x. The main blocker for MTP+MoE is dynamic shape — use `shapeless=True` with care.
3. **async_eval pipelining**: Overlap CPU graph construction with GPU execution. Zero new code needed; high payoff in multi-step decode loops.
4. **gather_qmm with sorted_indices**: Stock MLX already has a fused gather+quantized-matmul. Using `sorted_indices=True` enables further kernel optimization. Proven 1.4–3x on MoE prompt phase (PR #2078).
5. **Custom C++ primitive (graph-native)**: Implement a primitive that registers `is_fusable()=true` for element-wise parts; keep matmul parts as opaque. The official extension system (CMake + nanobind + mlx_build_metallib) is complete and documented.
6. **Metal ICBs / persistent kernels**: Theoretically applicable to avoid CPU re-encoding overhead in tight decode loops; no MLX-native implementation exists yet; requires dropping below MLX's abstraction boundary.

---

## 1. MLX C++ Extension Build System

### Architecture

MLX uses a two-layer primitive system:

- **Operations** (`mlx/ops.cpp`): user-facing functions that validate inputs, handle broadcasting, create `array` graph nodes wrapping a Primitive instance.
- **Primitives** (abstract C++ class): define `eval_cpu()`, `eval_gpu()`, `vjp()`, `jvp()`, `vmap()`, and `is_fusable()`.

Every op in the graph is an `array` node whose `.primitive` field points to an instance of some `Primitive` subclass.

### The `Primitive` Contract

Per Awni Hannun (MLX lead), primitives carry a heavy contract: they must support `vmap`, `vjp`, `jvp` transformations. Without these, training workflows break. The recommendation is:

> Use existing ops where possible. Create a new Primitive only when the operation is impossible or extremely inefficient without one.

The `mlx.fast` namespace exists precisely for "fat ops" — fused kernels that wrap existing op compositions but provide custom forward/backward Metal kernels (e.g. `mx.fast.rms_norm`, `mx.fast.rope`, `mx.fast.scaled_dot_product_attention`).

For inference-only use (MTP/decode), you can implement a primitive with `eval_gpu()` only and stub out `vjp()`/`jvp()` with errors — perfectly valid if you never differentiate through it.

### Official Build System

**Directory layout** for a compiled extension:

```
my_mlx_ext/
├── CMakeLists.txt
├── setup.py
├── my_mlx_ext/
│   ├── __init__.py
│   └── ops.py      ← Python bindings shim
├── mlx_ext/
│   ├── ops.h        ← C++ header
│   ├── ops.cpp      ← C++ op + primitive implementation
│   └── kernels/
│       └── ops.metal  ← Metal compute kernels
```

**CMakeLists.txt** (minimal):

```cmake
cmake_minimum_required(VERSION 3.25)
project(mlx_ext LANGUAGES CXX)

find_package(MLX CONFIG REQUIRED)

# Python bindings via nanobind
nanobind_add_module(_ext NB_STABLE_ABI NB_LTO NOMINSIZE
    mlx_ext/ops.cpp
    mlx_ext/bindings.cpp
)
target_link_libraries(_ext PRIVATE mlx)

# Build .metallib
mlx_build_metallib(
    TARGET mlx_ext_metallib
    TITLE "mlx_ext"
    SOURCES mlx_ext/kernels/ops.metal
    DESTINATION ${CMAKE_CURRENT_BINARY_DIR}
)
add_dependencies(_ext mlx_ext_metallib)
```

`mlx_build_metallib()` is provided by MLX's `cmake/extension.cmake`, automatically imported when you call `find_package(MLX)`.

**setup.py**:

```python
from mlx.extension import CMakeExtension, CMakeBuild
from setuptools import setup

setup(
    ext_modules=[CMakeExtension("my_mlx_ext._ext")],
    cmdclass={"build_ext": CMakeBuild},
    package_data={"my_mlx_ext": ["*.metallib", "*.dylib"]},
)
```

### Primitive Implementation Pattern

The `axpby` example (`z = alpha*x + beta*y`) shows the full pattern. GPU eval entry point:

```cpp
void Axpby::eval_gpu(const std::vector<array>& inputs, array& out) {
    auto& s = out.primitive().stream();
    auto& d = metal::device(s.device);

    // Allocate output
    out.set_data(allocator::malloc_or_wait(out.nbytes()));

    // Load kernel from our .metallib
    auto kernel = d.get_kernel("axpby_general_" + get_type_string(out.dtype()),
                               "mlx_ext");
    auto& compute_encoder = d.get_command_encoder(s.index);
    compute_encoder.set_compute_pipeline_state(kernel);
    compute_encoder.set_input_array(inputs[0], 0);  // x
    compute_encoder.set_input_array(inputs[1], 1);  // y
    compute_encoder.set_output_array(out, 2);
    // Set alpha, beta as constants...
    compute_encoder.dispatch_threads(
        MTL::Size(out.size(), 1, 1),
        MTL::Size(std::min(out.size(), (size_t)1024), 1, 1)
    );
}
```

**Benchmark**: Custom `axpby` showed ~2x speedup over `alpha*x + beta*y` composition on 4096×4096 arrays (0.77ms vs 1.56ms).

### Making a Custom Primitive Participate in Fusion

To participate in `mx.compile` fusion:

1. **Override `is_fusable()`** to return `true` — only valid for element-wise ops
2. **Single output**: multi-output primitives cannot be fused
3. **No data-dependent output shapes**: shapes must be deterministic at trace time
4. **Classify**: inherit from `UnaryPrimitive` or `BinaryPrimitive` helpers
5. **Implement `vjp()`/`jvp()`**: even if stubs for inference-only use

**Fusion algorithm internals** (`compile.cpp`):
- Max fusion depth: 11 hops
- Max fusion input arrays: 24
- All ops must be on the same device stream
- Broadcasts are handled via `split_one()` duplication (not a fusion barrier)
- Non-fusible: Matmul, Convolution, Attention, multi-output primitives, data-dependent shapes

For a MoE fused expert kernel: the matmul part (`gather_qmm`) is inherently non-fusible. The SwiGLU activation afterward IS fusible. The optimal pattern is: one large opaque primitive for `gather_qmm_swiglu` (matmul + activation fused in Metal), then let subsequent element-wise ops fuse with `mx.compile`.

**Reference**: [Custom Extensions Docs](https://ml-explore.github.io/mlx/build/html/dev/extensions.html) | [Primitive discussion #1171](https://github.com/ml-explore/mlx/discussions/1171)

---

## 2. ZMLX: What It Fuses and How

**Repo**: https://github.com/Hmbown/ZMLX
**Install**: `pip install zmlx` (no custom MLX build for LFM2 models)

### Operations Fused

| Fusion | Kernel Name | Dispatch Reduction | Notes |
|--------|-------------|-------------------|-------|
| Top-k + Softmax (gating) | `topk_gating_softmax` | 2 → 1 | Decode-only guard |
| Expert weight-and-reduce | `moe_combine` / `moe_combine_exact` | N → 1 | `_exact` uses bfloat16 accumulation to match MLX semantics |
| Gate proj + Up proj + SwiGLU | `gather_qmm_swiglu` | 3 → 1 | **Requires custom MLX fork** (EXPERIMENTAL_MLX.md) |
| Elementwise activation | `elementwise(expr)` helper | N → 1 | Python-string compiled to Metal; e.g. `elementwise("x * sigmoid(x)")` |

### Patching Mechanism

```python
import zmlx
zmlx.patch(model)  # auto-detects architecture, rewires forward() refs
# or:
zmlx.smart_patch(model)  # benchmarks each module, applies only if faster
```

- Architecture detection: checks for LFM2/Qwen/GLM patterns in module class names
- In-place: no weight conversion, just replaces `__call__` references
- Decode guard: fused paths only activate at `seq_len == 1` (decode step), keeping prefill unchanged
- Safe defaults: GLM/Qwen3 variants return `0 modules patched` without the custom MLX build (avoids silent correctness issues)

### Benchmark Results (M4 Max 36GB, greedy decode)

| Model | Stock MLX | ZMLX (pip) | ZMLX (custom MLX) |
|-------|-----------|------------|-------------------|
| LFM2-8B-A1B | baseline | +12.8% | — |
| LFM2-24B-A2B | baseline | +6.0% | — |
| Qwen3.5-35B-A3B | baseline | ~+2% | ~+2% |
| GLM-4.7-Flash | baseline | 0% | +6.4% |

The Qwen3.5 gain is small on stock MLX because the `gather_qmm_swiglu` fusion is gated behind the custom MLX build. The +2% comes from `topk_gating_softmax` and `moe_combine` only.

### The gather_qmm_swiglu Primitive (~800 lines C++/Metal)

Located in `integrations/mlx_local_integration/`. What it does:

```
Inputs: activations (fp16/bf16), gate_weight (4-bit quantized), up_weight (4-bit quantized)
Operation:
  1. Gather rows from gate_weight and up_weight using expert indices
  2. gate_out = dequant(gate_weight) @ activations
  3. up_out   = dequant(up_weight)   @ activations
  4. out = gate_out * sigmoid(gate_out) * up_out   [SiGLU variant]
     or = gate_out * tanh(...)                      [SwiGLU variant]
All in a single Metal dispatch — no intermediate fp16 tensors written to DRAM
```

This is the key insight: the two quantized matmuls share the same dequantized weights in a single kernel, eliminating one full DRAM round-trip per expert per token.

### elementwise() Helper API

```python
from zmlx.api import elementwise

# Compile a string expression into a fused Metal kernel
mish    = elementwise("x * tanh(log(1 + exp(x)))", name="mish")
swiglu  = elementwise("x * sigmoid(x) * y", name="swiglu", arity=2)
geglu   = elementwise("x * gelu(y)", name="geglu", arity=2)
```

This is a Triton-style Python-first kernel authoring API. It compiles string math expressions to Metal, handles caching and autograd. Useful for prototyping fusions without writing Metal code manually.

---

## 3. MLX Kernel Fusion Internals

### What mx.compile Currently Fuses

`@mx.compile` / `mx.compile(fn)` does two things:
1. Simplifies the graph (constant folding, copy elimination, StopGradient removal)
2. Calls `compile_fuse()` to merge element-wise op sequences into `Compiled` primitives

**Fusible** (from `compile.cpp:24–77`):
- All unary element-wise: Sin, Cos, Exp, Log, Relu, Tanh, Sqrt, and all others
- All binary element-wise: Add, Multiply, Divide, Power, etc.
- Shape ops: Reshape, Transpose, Squeeze, ExpandDims
- Indexing: Slice, Take, Gather (note: these are gather-over-index, not gather_mm)

**Non-fusible** (remain as standalone dispatches):
- Matmul, Convolution, Attention (SDPA)
- Any primitive returning `is_fusable() = false`
- Multi-output primitives
- Data-dependent shape operations
- Copy, StopGradient (but these get simplified away first)

### What Breaks Fusion

1. **Any matmul in the middle of an elementwise chain** — creates a fusion boundary. Common in transformer: `linear(x)` breaks the chain, post-linear activation starts a new fused block.
2. **Python control flow** on traced values — graph-dependent branches can't be traced.
3. **Calling `mx.eval()` inside a compiled function** — forces a sync boundary.
4. **Multi-output ops**: if any node in the chain produces multiple outputs, it can't be fused.
5. **Non-matching streams**: ops on CPU and GPU streams can't be fused together.
6. **Shape changes between compilations**: the compiled kernel is keyed by shape. With `shapeless=False` (default), a shape change triggers recompilation. For speculative decoding where input length varies every step, this causes repeated JIT overhead — this is likely the root of [issue #250](https://github.com/ml-explore/mlx-lm/issues/250).

### Compiled Primitive Structure

```cpp
class Compiled : public UnaryPrimitive {
    vector<array> fused_tape_;   // ops in topological order
    set<uint64_t> constant_ids_; // embedded constants (avoid DRAM read)
    string kernel_name_;          // cache key: encodes op types, dtypes, shapes
};
```

The kernel name encodes the entire computation — constants are embedded directly in the generated Metal shader source, allowing the compiler to produce optimized literal values.

### Real-World Fusion Examples

```python
# GELU (5 ops -> 1 kernel): ~5x speedup on M1 Max
@mx.compile
def gelu(x):
    return x * (1 + mx.tanh(0.7978845608 * (x + 0.044715 * x**3))) / 2

# SwiGLU (2 ops -> 1 kernel):
@mx.compile
def swiglu(x, y):
    return x * mx.sigmoid(x) * y

# For MTP decode: compile the entire step function
@partial(mx.compile, inputs=[model.state], outputs=[model.state])
def decode_step(tokens, cache):
    return model(tokens, cache=cache)
    # Fuses: all element-wise chains between matmuls
    # Each matmul + gather_qmm remains a separate dispatch
    # But post-matmul activations fuse with their downstream ops
```

### shapeless=True for MTP+MoE

The speculative decode loop varies the number of draft tokens (and thus input tensor shapes) every step. Without `shapeless=True`, `mx.compile` recompiles on every shape change:

```python
# Prevents recompilation when sequence length varies
@partial(mx.compile, inputs=[model.state], outputs=[model.state], shapeless=True)
def speculative_step(draft_tokens, cache):
    return model(draft_tokens, cache=cache)
```

**Warning**: `shapeless=True` breaks any logic that branches on shape (e.g. variable-length KV cache updates). Use only for the model forward pass, not the cache management code.

---

## 4. Metal Indirect Command Buffers (ICBs) and Persistent Kernels

### What ICBs Are

ICBs allow encoding GPU commands once, then replaying them repeatedly without CPU involvement. The GPU can also write into an ICB from another kernel (GPU-driven dispatch). Available since A9/2015 iMac.

### Applicability to LLM Decode

**Current MLX behavior**: MLX accumulates kernel dispatches into a command buffer until `MAX_ACTIVE_TASKS = 10` is reached or a sync fence is needed, then finalizes the stream. `mx.eval()` is a sync gate. The CPU does participate in graph encoding for every eval step.

**What ICBs could theoretically buy**:
- Encode the entire single-token decode graph once (all matmuls, gating, activations)
- Replay for N decode steps without CPU command encoding overhead
- Potential elimination of ~10–50μs dispatch overhead per decode step

**Why it's hard with MLX**:
1. MLX's lazy graph system dynamically generates the command encoding from Python graph state — there is no pre-fixed kernel sequence to cache as an ICB.
2. KV cache updates and MoE routing produce data-dependent dispatch patterns (different experts activated per token).
3. ICB command count limit: 16,384 commands. A full transformer forward pass easily exceeds this if each op is a command.
4. Metal Shader Validation is incompatible with ICBs (breaks debugging workflows).

**Practical alternative**: The correct MLX-native approach to avoid CPU overhead is `mx.async_eval()` — this submits GPU work and returns immediately, allowing the CPU to build the next graph while GPU executes the current one:

```python
# Decode loop with async pipelining
def decode_loop(model, tokens, max_tokens):
    logits_prev = None
    for i in range(max_tokens):
        logits = model(tokens)
        if logits_prev is not None:
            # Sample from previous step's logits while GPU computes current
            next_token = sample(logits_prev)
        mx.async_eval(logits)  # Submit GPU work, return immediately
        logits_prev = logits
        tokens = next_token
```

**Assessment**: ICBs are not currently usable with MLX without forking the framework at the command encoder level. The `mx.async_eval` pattern achieves the same CPU/GPU overlap objective within MLX's existing API.

### The MegaKernel Approach (Not Metal-Native)

The MPK compiler (Mirage, 2025) fuses an entire LLM decode step into a single GPU kernel — eliminating inter-kernel launch overhead entirely. On single GPU: 1.2x improvement (14.5ms → 12.5ms per decode step). However: **CUDA-only**. No Metal port exists. The intra-kernel task scheduler uses CUDA streaming multiprocessor features not available in Metal.

---

## 5. gather_mm and Fused MoE Expert Kernel Analysis

### Current MLX API (gather_mm + gather_qmm)

**gather_mm** (PR #2040): Fuses gather + matmul into one Metal dispatch. Avoids writing an intermediate gathered matrix to DRAM.

```python
# Old approach (two dispatches, one DRAM round-trip):
w_selected = mx.take(w, expert_indices, axis=0)  # dispatch 1: write to DRAM
out = x @ w_selected.T                            # dispatch 2: read from DRAM

# New approach (one dispatch):
out = mx.gather_mm(x, w, rhs_indices=expert_indices)  # dispatch 1 only
```

**gather_qmm** (PR #2078): Same but with 4-bit quantized weights. The key parameters:

```python
out = mx.gather_qmm(
    x,             # activations: [batch, d_model]
    w,             # quantized weights: [n_experts, d_out//pack, d_in] (packed int)
    scales=scales, # [n_experts, d_out//group, d_in//group]
    biases=biases, # optional
    rhs_indices=expert_indices,  # which experts to use: [batch, n_active]
    sorted_indices=True,  # IMPORTANT: enables kernel optimization when experts are sorted
    transpose=True,
    group_size=64,
    bits=4
)
```

### Performance Data (PR #2078, M2 Ultra)

| Scenario | Before | After | Speedup |
|----------|--------|-------|---------|
| Mixtral 8x7B, ~500 tokens | 189 tps | 590 tps | 3.1x |
| Qwen 1.5 2.7B, ~500 tokens | 1239 tps | 2213 tps | 1.8x |
| DeepSeek V3, ~450 tokens | 112 tps | 154 tps | 1.4x |

**Important caveat**: For uniformly distributed expert selection (worst case: all experts accessed equally per batch of 16 tokens), performance can be slower than unfused. Performance is highest when expert access has locality (temporal clustering of expert activations is common in practice per MoE-Infinity research).

### The gather_qmm_swiglu Gap

The current `gather_qmm` handles one weight matrix at a time. For MoE with SwiGLU:

```
# Current MLX (3 separate gather_qmm dispatches):
gate_out = gather_qmm(x, gate_weight, ...)  # dispatch 1
up_out   = gather_qmm(x, up_weight, ...)    # dispatch 2
out      = gate_out * sigmoid(gate_out) * up_out  # dispatch 3 (fused by mx.compile)
```

ZMLX's `gather_qmm_swiglu` collapses dispatches 1+2+3 into one kernel. This is what produces the larger gains for GLM-4.7-Flash (+6.4%) vs Qwen3.5 (+2%) — Qwen uses GLU and benefits more.

**To implement this in MLX**:
1. Write a `GatherQMMSwiGLU` C++ primitive
2. Metal kernel: for each expert token, dequantize gate and up weight rows simultaneously, compute matmul outputs, apply SwiGLU — all in registers without DRAM intermediate
3. Register with `is_fusable() = false` (it's not element-wise; keep it opaque to the fusion pass)
4. The output then feeds into downstream element-wise ops that `mx.compile` can fuse

**Estimated effort**: ~1,000 lines of C++/Metal. Build on top of ZMLX's existing gather_qmm_swiglu or adapt the pattern.

---

## 6. MLX Community Optimization Efforts (GitHub Tracking)

### Open Issues and PRs (Relevant to Decode Throughput)

| Issue/PR | Description | Status | Priority |
|----------|-------------|--------|----------|
| [PR #2040](https://github.com/ml-explore/mlx/pull/2040) | gather_mm new kernel | **Merged** | Done |
| [PR #2078](https://github.com/ml-explore/mlx/pull/2078) | gather_qmm batched kernel | **Merged** | Done |
| [PR #2796](https://github.com/ml-explore/mlx/pull/2796) | sparse_matmul_csr kernel | **Open** (maintainer skeptical) | Low |
| [PR #2808](https://github.com/ml-explore/mlx/pull/2808) | Thunderbolt RDMA multi-device | Open | Multi-machine use only |
| [Issue #250](https://github.com/ml-explore/mlx-lm/issues/250) | Speculative decode slowdown (JIT shape issue) | **Open, no response** | High for MTP |
| [Issue #2418](https://github.com/ml-explore/mlx/issues/2418) | ASTC 3.6-bit weight compression | **Closed, declined** | — |
| [Issue #1293](https://github.com/ml-explore/mlx/issues/1293) | w4a8 GEMM (wontfix) | **Closed** | — |
| [Issue #129](https://github.com/ml-explore/mlx/issues/129) | Flash attention | **Completed** (Jan 2026) | Done |

### Key Finding: Speculative Decoding Shape Issue

The JIT compilation in mlx-lm recompiles when input tensor shapes change. In speculative decoding, the number of draft tokens accepted varies every step — shape changes every decode step — causing repeated recompilation overhead. This may be the primary reason MTP+MoE shows lower-than-expected speedup.

**Workaround**: Pad draft token sequences to a fixed maximum length and use `shapeless=True` on the compiled function. Cost: wasted compute on padded positions. Benefit: eliminates recompilation overhead.

### Recent Release Improvements (Late 2025 – March 2026)

From MLX release notes:
- **Vector fused grouped-query attention (Metal)**: significant speedup for long-context decode
- **SegmentedMM**: segmented matrix multiplication for grouped GEMM workloads (MoE-relevant)
- **3-bit/5-bit/6-bit QMV kernels**: more quantization bitwidths now have optimized Metal paths
- **Faster grouped matmul**: improved heuristics for batch processing

---

## 7. Other High-Performance MLX Forks and Projects

### oMLX (github.com/jundot/omlx)

**Focus**: Serving infrastructure, not raw decode throughput.

Key features:
- **Paged KV cache** inspired by vLLM: block-based with Copy-on-Write and prefix sharing
- **Two-tier KV cache**: hot blocks in RAM, cold blocks in safetensors format on NVMe (`--paged-ssd-cache-dir`)
- **SSD persistence**: KV cache survives server restarts; agentic workloads see 5–30x TTFT improvement on repeated prefixes
- MLX memory management: uses `mx.get_active_memory()` polling (removed Metal-level limits)
- Use case: coding agents that re-use large shared contexts

**Decode throughput impact**: minimal — the same MLX primitives execute per token. Value is in context reuse, not per-token speed.

### exllama-ish (github.com/Infatoshi/exllama-ish)

EXL3/QTIP 4-bit quantized inference for Apple Silicon with hand-written `.metal` files:
- Custom GEMV kernel with fused dequantization (reads quantized weights, dequantizes in-kernel, no intermediate fp16 tensor)
- Hadamard transform kernel
- Fused RMSNorm, RoPE, attention kernels
- Does not use MLX at all — raw Metal + Python via ctypes/PyObjC

**Lesson for MLX**: The fused GEMV pattern (dequantize inside matmul) is what `gather_qmm` already does. The MLX implementation should be competitive.

### mlx-mfa (PyPI: mlx-mfa)

Metal FlashAttention with serving-oriented cache abstractions. Key specs:
- 4.6x improvement vs M4 baseline on M5 (`mlx-mfa` docs, March 2026)
- Drop-in replacement for `mx.fast.scaled_dot_product_attention`
- PagedAttention-style memory management for concurrent requests
- Flash decoding support for long-context decode

**Note**: MLX's built-in `mx.fast.scaled_dot_product_attention` was updated in Jan 2026 (issue #2955 closed). For single-user MTP decode, stock SDPA may be sufficient. Use `mlx-mfa` when serving multiple concurrent requests or for very long contexts (>32K).

### MOLA (Multi-LoRA Inference)

Achieves 732 tok/s on Qwen3.5-9B with 8 LoRA adapters, 555 tok/s in mixed-adapter mode (from MLX community discussions). Not directly MoE-relevant but demonstrates effective Metal dispatch batching of multiple weight sets — similar pattern to MoE expert dispatch.

---

## 8. Priority Action Plan for Qwen3.5-35B-A3B + MTP

### Immediate (no custom build required)

**A. Apply mx.compile to the decode step**

```python
from functools import partial
import mlx.core as mx

# Wrap the full MTP decode step
state = [model.state]
@partial(mx.compile, inputs=state, outputs=state, shapeless=True)
def mtp_decode_step(tokens, cache):
    return model(tokens, cache=cache)
```

Expected gain: fuses all element-wise chains between matmuls; eliminates repeated JIT recompilation on shape change.

**B. Use async_eval for CPU/GPU overlap**

```python
def generate_loop(model, prompt_tokens, max_new_tokens):
    cache = build_cache(model)
    tokens = prompt_tokens
    pending = None

    for _ in range(max_new_tokens):
        logits = mtp_decode_step(tokens, cache)
        if pending is not None:
            next_tokens = mx.argmax(pending, axis=-1)
            mx.eval(next_tokens)  # sync only on sampling
        mx.async_eval(logits)     # GPU runs while CPU builds next graph
        pending = logits
        tokens = ...  # update based on next_tokens
```

**C. Ensure gather_qmm uses sorted_indices=True**

In mlx-lm's MoE dispatch code, expert indices should be sorted before calling `gather_qmm` to enable the kernel's sort-aware optimization path. Check `mlx-lm/mlx_lm/models/` for the MoE layer implementation.

**D. Install ZMLX and apply stock patch**

```bash
pip install zmlx
```

```python
import zmlx
zmlx.patch(model)  # ~+2% on Qwen3.5-35B-A3B, token-identical
```

### Short-Term (custom build required)

**E. Build ZMLX's gather_qmm_swiglu primitive**

Follow `ZMLX/docs/EXPERIMENTAL_MLX.md`:
1. Clone ZMLX and the MLX source
2. Apply patches from `integrations/mlx_local_integration/`
3. Rebuild MLX from source
4. Expected gain: +6–8% on GLM-style MoE; +4–6% estimated for Qwen3.5 (testing required)

**F. Profile with Metal GPU Trace to find actual bottlenecks**

```bash
CMAKE_ARGS="-DMLX_METAL_DEBUG=ON" pip install mlx --no-binary mlx
MTL_CAPTURE_ENABLED=1 python -c "
import mlx.core as mx
mx.metal.start_capture('trace.gputrace')
# ... run 10 decode steps ...
mx.metal.stop_capture()
"
```

Open `trace.gputrace` in Xcode Metal Debugger → Dependencies view. This will show the actual dispatch distribution: how many dispatches per token, which ops dominate, and what the CPU encoding overhead looks like.

### Medium-Term (significant implementation work)

**G. Implement a full GatherQMMSwiGLU C++ primitive**

This is the highest-leverage framework-level optimization. Estimated 1,000–1,500 lines total:
- C++ primitive class (~200 lines): `eval_gpu()`, `vjp()` stub, `vmap()` stub
- Metal kernel (~600 lines): fused gather + dual-dequant-GEMM + SwiGLU
- CMake build integration (~50 lines)
- Python binding (~50 lines)
- Tests (~200 lines)

Use ZMLX's existing gather_qmm_swiglu as a reference implementation.

**H. Fix speculative decoding shape recompilation (upstream contribution)**

The root cause (issue #250) is that `speculative_generate_step` in mlx-lm creates tensors with different shapes each call. Fix: either pad to `max_draft_tokens` or cache compiled functions per shape. This would benefit all MTP users, not just MoE.

---

## Gaps and Limitations

1. **w4a8 GEMM**: MLX maintainers closed this as wontfix. The theoretical speedup from quantized activations is unavailable. On Apple Silicon Neural Accelerators, the hardware supports INT8 matmul but MLX does not expose this path for activations.

2. **ASTC weight compression**: Declined by maintainers (March 2026). 3.6 bits/weight with hardware-accelerated decode is theoretically compelling but not in the official roadmap.

3. **Metal ICBs**: Require dropping below MLX's abstraction boundary. The CPU dispatch overhead in MLX (~10-50μs per token) is real but `async_eval` mitigates most of it.

4. **MegaKernel (MPK)**: CUDA-only. No Metal port. The approach of fusing entire transformer layers into one kernel would be the ideal MLX optimization but would require rewriting MLX's core execution model.

5. **ZMLX Qwen3.5 gains are modest (+2%)**: The full gains require the `gather_qmm_swiglu` primitive which needs a custom MLX fork. Until this is upstreamed into stock MLX, Qwen3.5 users get only the gating/combine fusion speedups.

6. **SegmentedMM**: Present in recent MLX release notes but not yet exposed in the Python API in a form useful for custom MoE dispatch.

---

## References

### Core MLX Documentation
- [Custom Extensions Guide](https://ml-explore.github.io/mlx/build/html/dev/extensions.html)
- [Custom Metal Kernels Guide](https://ml-explore.github.io/mlx/build/html/dev/custom_metal_kernels.html)
- [Compilation and mx.compile](https://ml-explore.github.io/mlx/build/html/usage/compile.html)
- [gather_qmm API Reference](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.gather_qmm.html)
- [gather_mm API Reference](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.gather_mm.html)
- [mx.fast namespace reference](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.fast.metal_kernel.html)

### MLX Internals and Architecture
- [DeepWiki: Compilation and Graph Optimization](https://deepwiki.com/ml-explore/mlx/3.4-compilation-and-graph-optimization)
- [DeepWiki: MLX Overview](https://deepwiki.com/ml-explore/mlx/1-overview)
- [Discussion #1171: When to build a Primitive](https://github.com/ml-explore/mlx/discussions/1171)

### PRs and Issues
- [PR #2040: gather_mm new kernel](https://github.com/ml-explore/mlx/pull/2040)
- [PR #2078: gather_qmm batched kernel](https://github.com/ml-explore/mlx/pull/2078)
- [PR #2796: sparse_matmul_csr (open)](https://github.com/ml-explore/mlx/pull/2796)
- [Issue #250: Speculative decode slowdown](https://github.com/ml-explore/mlx-lm/issues/250)
- [Issue #1293: w4a8 GEMM (wontfix)](https://github.com/ml-explore/mlx/issues/1293)
- [Issue #2418: ASTC compression (closed)](https://github.com/ml-explore/mlx/issues/2418)

### Community Projects
- [ZMLX: Triton-style toolkit + MoE patching](https://github.com/Hmbown/ZMLX)
- [oMLX: LLM server with SSD KV caching](https://github.com/jundot/omlx)
- [mlx-mfa: Metal FlashAttention](https://pypi.org/project/mlx-mfa/)
- [exllama-ish: custom Metal GEMV](https://github.com/Infatoshi/exllama-ish)
- [mlx-engine Flash MoE issue](https://github.com/lmstudio-ai/mlx-engine/issues/294)

### Research Papers
- [Native LLM and MLLM Inference at Scale on Apple Silicon (Jan 2026)](https://arxiv.org/html/2601.19139)
- [Production-Grade Local LLM Inference on Apple Silicon (Nov 2025)](https://arxiv.org/pdf/2511.05502)
- [Apple ML Research: Exploring LLMs with MLX and M5](https://machinelearning.apple.com/research/exploring-llms-mlx-m5)
- [AI-Generated Metal Kernels (KernelBench) — Gimlet Labs](https://gimletlabs.ai/blog/ai-generated-metal-kernels)

### WWDC25
- [Get Started with MLX for Apple Silicon — WWDC25](https://developer.apple.com/videos/play/wwdc2025/315/)
