# TurboQuant: Exact Algorithm and Implementation Reference

**Paper**: "TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate"
**Authors**: Amir Zandieh (Google Research), Majid Daliri (NYU), Majid Hadian (Google DeepMind), Vahab Mirrokni (Google Research), plus collaborators from KAIST
**Venue**: ICLR 2026 (arXiv: 2504.19874, first posted April 2025)
**Related paper**: PolarQuant (arXiv: 2502.02617, AISTATS 2026) — same group, first stage only
**Status as of March 2026**: No official code from Google; ICLR presentation scheduled April 2026

---

## Executive Summary

TurboQuant is a two-stage, training-free, data-oblivious KV cache quantization method. It compresses keys and values to 2.5–4 bits per coordinate without any calibration data or model-specific tuning, achieving perplexity-neutral compression at 3.5 bits. The core insight is that applying a random orthogonal rotation to any KV vector induces a Beta distribution on each coordinate — and because this distribution is known analytically, you can precompute the globally optimal scalar quantizer (Lloyd-Max) once and reuse it across all models, layers, and tokens. An optional 1-bit residual correction (QJL) then eliminates systematic inner-product bias.

**It is entirely different from calling `mx.quantize` on keys/values.** Naive quantization of raw KV vectors fails because keys have extreme channel outliers (kurtosis ~900 in real Qwen tensors), causing the quantization range to be dominated by rare extreme values. TurboQuant's rotation step spreads those outliers uniformly across all coordinates, bringing kurtosis down to ~2.9 (near-Gaussian).

---

## 1. The Full Two-Stage Pipeline

### Stage 1: TurboQuant_mse — Rotation + Lloyd-Max Quantization

**Algorithm 1 (from paper):**

```
Input:  vector x ∈ ℝ^d, bit-width b
Offline (precomputed, model-independent):
  Π ← random orthogonal matrix ∈ ℝ^(d×d)    [Hadamard in practice]
  c₁,...,c_{2^b} ← Lloyd-Max centroids for Beta(d/2, (d-1)/2) distribution

Quant_mse(x):
  y ← Π · x                         [rotate: y ∈ ℝ^d]
  γ ← ‖x‖₂                          [store norm separately]
  for each coordinate j:
    idx_j ← argmin_i |y_j - c_i|    [nearest centroid, per coordinate]
  return (idx, γ)                    [idx ∈ {0,...,2^b-1}^d, γ ∈ ℝ]

DeQuant_mse(idx, γ):
  for each coordinate j:
    ỹ_j ← c_{idx_j}                 [centroid lookup]
  x̃ ← γ · Π^T · ỹ                  [rotate back, scale by norm]
  return x̃
```

**Why the Beta distribution?** After normalizing x to the unit sphere and rotating with a random orthogonal matrix, each coordinate of the rotated vector follows:

```
f_X(x) = Γ(d/2) / (√π · Γ((d-1)/2)) · (1 - x²)^((d-3)/2)
```

This is a scaled Beta distribution. In high dimensions (d ≥ 64), this converges to N(0, 1/d) — approximately Gaussian. Critically, the distribution is analytically known and **identical across all vectors and models**, so the codebook is universal.

**How codebook centroids are found (Lloyd-Max):**
The algorithm solves a continuous 1D k-means problem on this Beta PDF:

```
Minimize: ∫ (x - c(x))² · f_X(x) dx
```

Solved iteratively: grid 50,000 points on [-1, 1], run 300 Lloyd-Max iterations alternating between (1) assigning each point to its nearest centroid and (2) updating centroids to the PDF-weighted mean of their Voronoi interval. The resulting codebook is cached by (d, b) pair — you compute it once, not per model, per layer, or per token.

**Distortion guarantees:** MSE scales as 4^(-b), within a factor of ~2.7 of the Shannon information-theoretic lower bound across all bit-widths.

| Bits | Approximate MSE (unit-norm vector) |
|------|------------------------------------|
| 1    | ~0.36                              |
| 2    | ~0.117                             |
| 3    | ~0.030                             |
| 4    | ~0.009                             |

### Stage 2: TurboQuant_prod — QJL Residual Correction

**Algorithm 2 (from paper):**

```
Input:  vector x ∈ ℝ^d, total bit-budget b (uses b-1 bits for MSE stage)
Offline (precomputed):
  TurboQuant_mse instance with bit-width (b-1)
  S ∈ ℝ^(d×d) with i.i.d. entries ~ N(0, 1)  [random projection matrix]

Quant_prod(x):
  (idx, γ) ← Quant_mse(x)           [stage 1 quantization]
  x̃_mse ← DeQuant_mse(idx, γ)       [reconstruct]
  r ← x - x̃_mse                     [residual]
  qjl ← sign(S · r)                  [1-bit QJL sketch, ∈ {-1,+1}^d]
  γ_r ← ‖r‖₂                         [residual norm]
  return (idx, γ, qjl, γ_r)

Inner product estimation (query q against quantized key x):
  ⟨q, x⟩ ≈ ⟨q, x̃_mse⟩ + γ_r · ⟨q, Q_qjl^{-1}(qjl)⟩
  where Q_qjl^{-1}(z) := √(π/2)/d · S^T · z

This is an UNBIASED estimator: E[estimate] = ⟨q, x⟩
```

**QJL definition:** `Q_qjl(x) := sign(S·x)` where S has i.i.d. N(0,1) entries. The inverse `Q_qjl^{-1}(z) := √(π/2)/d · S^T · z` recovers an unbiased estimate of the original vector because the JL transform preserves distances and the sign operation's expected value is computable.

**Practical note:** In community implementations (llama.cpp, Triton kernels), the QJL stage is consistently found to be counterproductive in practice and is typically **omitted**. All benchmark results in the paper's KV-cache section use TurboQuant_mse only. The QJL stage requires custom attention kernels that split the dot product computation, which is rarely worth the overhead at b ≥ 3.

---

## 2. What Makes It Different from Naive `mx.quantize`

### Problem 1: Channel Outliers in Keys

Raw key tensors from real LLMs have extreme channel outliers. A measurement on Qwen2.5-7B showed:
- Keys: 274x norm variation across channels
- Values: 2.6x norm variation (more benign)

This 106:1 ratio between keys and values means the same quantization parameters cannot work for both. With naive uniform quantization, the quantization scale is forced to accommodate the extreme channels, wasting most of the bit-budget representing near-zero values in the regular channels.

**Distribution kurtosis:** Raw key coordinates: ~900.4. After Hadamard rotation: ~2.9. Gaussian has kurtosis 3.0. The rotation essentially solves the outlier problem by spreading the energy across all coordinates.

### Problem 2: Quantization Requires Knowing the Distribution

Naive quantization chooses min/max or percentile-based scales per block, requiring storing per-block metadata (zero-point and scale at FP16 = 2 bytes overhead per block). For a block of 32 values quantized to 2 bits (= 8 bytes), 2 bytes of metadata is 25% overhead, severely limiting compression.

TurboQuant's rotation makes the distribution **analytically known** (Beta/Gaussian), so:
- No per-block scale or zero-point needed
- Only one scalar (the vector norm) is stored per head-dim vector
- For head_dim=128 at 3 bits: 1 norm (2 bytes) + 48 bytes indices = 50 bytes vs 256 bytes FP16 = 5.1x compression

### Problem 3: Inner Product Bias

Scalar quantization introduces systematic bias: the quantized reconstruction x̃ underestimates inner products. QJL's residual correction makes the estimator unbiased.

### Online vs Offline Components

| Component | When | Cost |
|-----------|------|------|
| Rotation matrix Π | Once at init | O(d²) storage, O(1) after |
| Codebook centroids c₁...c_{2^b} | Once at init | Precomputed from Beta PDF |
| S matrix (QJL) | Once at init | O(d²) storage |
| Vector rotation: Π·x | Every new token | O(d log d) with WHT |
| Nearest centroid: argmin_i | Every coordinate, every token | O(2^b · d) |
| QJL sketch: sign(S·r) | Optional, every token | O(d²) — expensive |

The S matrix for QJL requires O(d²) matrix multiply per token — at d=128, that's 16,384 multiplications per KV vector per layer per token, which is why the QJL stage is often skipped.

---

## 3. Rotation: Hadamard vs Dense Random Orthogonal

The paper specifies a "random orthogonal matrix Π ∈ ℝ^(d×d)" generated via QR decomposition of a Gaussian random matrix. In practice, all efficient implementations use a **Randomized Hadamard Transform** instead:

```
Π_WHT · x = H · D · x
where:
  H = Walsh-Hadamard matrix (butterfly structure)
  D = diagonal matrix with i.i.d. Rademacher entries (±1)
```

**Why WHT instead of dense QR?**
- Dense QR rotation: O(d²) operations, O(d²) memory for the matrix
- WHT: O(d log d) operations, O(d) memory (just the random signs in D)
- For d=128: 16,384 operations (dense) vs ~896 operations (WHT) — 18x faster

The WHT produces a random orthogonal-like rotation with the same statistical properties (uniformly distributed on the sphere after applying the random sign flip). The mathematical guarantee holds: rotated coordinates still follow the Beta distribution.

**In community implementations:**
- Default: WHT = `hadamard_transform(D * x)` where D is the precomputed random ±1 sign vector
- Dense fallback: QR decomposition, O(d²) memory, used in reference PyTorch code
- TheTom's llama.cpp Metal fork: WHT with LUT-based centroid lookup in Metal constant memory

---

## 4. Per-Head, Per-Channel, Group Size

**Per vector (not per token or per channel):** TurboQuant quantizes one complete KV vector (dimension = head_dim) at a time. Each head's key or value at each sequence position is quantized independently as a single unit.

- One norm γ stored per vector (per head, per sequence position)
- One set of `head_dim` indices stored per vector
- No "group size" in the traditional sense — the whole head_dim vector is one group
- Block size in llama.cpp implementations: 128 values (matching typical head_dim for 70B models) or 32 values (smaller models)

**Mixed-precision outlier channel strategy:**

For 2.5-bit target on head_dim=128:
- Identify 32 "outlier channels" (the 25% of channels with highest variance after rotation)
- Quantize outlier channels at 3 bits
- Quantize remaining 96 channels at 2 bits
- Effective bit-width: (32×3 + 96×2) / 128 = 2.5 bits

For 3.5-bit target on head_dim=128:
- 64 outlier channels at 4 bits + 64 regular channels at 3 bits
- OR: 32 outlier at 5 bits + 96 regular at 3.17 bits (approximation)
- Effective: (32×4 + 96×3.5) / 128 ≈ 3.5 bits (exact ratio varies by implementation)

**Channel selection is done offline** (no per-token calibration). The outlier channels are determined by their statistical properties under the Beta distribution — they're the ones at the tails. In practice, the top-K channels by RMS after rotation are assigned higher bits.

**Keys vs Values asymmetry (observed in practice):**
- Keys: extreme outliers (274x norm variation in Qwen2.5-7B), benefit most from rotation
- Values: milder distribution (2.6x norm variation), can use simpler quantization
- Optimal strategy: 3-4 bits for keys, 2 bits for values (TurboQuant+ community format)

---

## 5. Exact Block Storage Format

**TQ3 block (128 values, 3 bits each):**
```
offset  size    content
0       4B      float32 norm γ
4       48B     128 × 3-bit indices, bit-packed
Total:  52B     vs 256B FP16 = 4.9x compression
```

**TQ4 block (128 values, 4 bits each):**
```
offset  size    content
0       4B      float32 norm γ
4       64B     128 × 4-bit indices, bit-packed
Total:  68B     vs 256B FP16 = 3.8x compression
```

**Turbo3 in TheTom's implementation (32-value block):**
```
offset  size    content
0       2B      fp16 norm/scale
2       12B     32 × 3-bit indices, bit-packed
Total:  14B     per 32 values = 3.5 bits/value = 4.6x compression
```

---

## 6. Performance Results

### Quality Results (from paper)

**LongBench (Llama-3.1-8B):**

| Method | KV bits | LongBench avg | Delta |
|--------|---------|---------------|-------|
| Full Cache (FP16) | 16 | 50.06 | baseline |
| TurboQuant 3.5-bit | 3.5 | 50.06 | 0.00 |
| TurboQuant 2.5-bit | 2.5 | 49.44 | -0.62 |
| PolarQuant | 3.9 | 49.78 | -0.28 |

**Needle-in-a-Haystack (4K–104K context):**
- TurboQuant 4× compression: 0.997 recall (vs 1.0 full precision)
- Perfect NIAH at all context lengths through 64K (community testing, flovflo MLX port)

**GSM8K (Qwen2-7B, 3-bit):**
- Full precision: 85.7%
- TurboQuant 3-bit: 84.3% (−1.4 points)
- Note: Reasoning tasks show more degradation than retrieval tasks

**Perplexity (Qwen3.5-35B-A3B, llama.cpp, CPU):**
- FP16 baseline: 6.5792 ± 0.04
- TQ3 (3-bit): 6.6967 ± 0.04 (+1.8% degradation)
- Q4_0 (comparison): 6.6054 ± 0.04

### Speed Results (from paper)

**H100 GPU:**
- 4-bit TurboQuant: up to **8× speedup** for attention logit computation vs FP32 unquantized keys
- vs FP16: substantially lower (paper uses FP32 baseline for headline number — important caveat)
- KV cache memory reduction: ≥6× at 3.5 bits

**Community benchmarks (realistic FP16 comparisons):**

| Hardware | Model | Compression | vs FP16 KV |
|----------|-------|-------------|------------|
| M5 Max (Metal) | Qwen3.5-35B | 4.9× | ~1.02× q8_0 speed parity |
| RTX 4080 (CUDA) | Qwen3.5-9B | 4.6× | +85× prefill\*, −59% decode |
| AMD EPYC (CPU) | Qwen3.5-35B | 4.4× | −38% vs F16 |
| Triton kernel | d=256 synthetic | 7.9× | ~1.2× Q@K speedup |

\*The "85x prefill" is misleading — that's vs a poorly optimized baseline. Realistic end-to-end gains are modest; the main benefit is **memory reduction enabling longer contexts**, not raw speed.

**Vector search (Table 2 from paper, 4-bit, d=1536):**

| Method | Indexing time | Recall |
|--------|---------------|--------|
| TurboQuant | 0.0013s | high |
| Product Quantization (PQ) | 239.75s | similar |
| RaBitQ | 2267.59s | similar |

Indexing is ~2000× faster than alternatives due to no training/calibration requirement.

---

## 7. Comparison to Related Methods

### KIVI (ICML 2024, arXiv: 2402.02750)

**Approach:** Tuning-free asymmetric 2-bit quantization.
- Keys: quantized **per-channel** (group elements along channel dimension)
- Values: quantized **per-token** (group elements along token dimension)
- Why asymmetric: Keys have outliers persistent across tokens (channel-wise), Values have outliers across channels (token-wise)
- Also keeps the most recent 32 tokens in FP16 (generated tokens unquantized)
- Result: 2.6× peak memory reduction at 2-bit, 2.35–3.47× throughput improvement
- No rotation, no codebook — plain min/max asymmetric quantization

**vs TurboQuant:** KIVI's NIAH score is 0.981 vs TurboQuant's 0.997. KIVI requires per-block scale/zero-point storage overhead. KIVI is simpler and widely available; TurboQuant requires rotation infrastructure.

### KVQuant (NeurIPS 2024, arXiv: 2401.18079)

**Approach:** Multiple techniques stacked:
1. **Per-channel key quantization** (like KIVI for keys)
2. **Pre-RoPE key quantization**: quantize keys *before* applying rotary positional embedding — RoPE mixes channel pairs, which breaks per-channel statistics; pre-RoPE keys have cleaner distributions
3. **Non-Uniform Quantization (NUQ)**: sensitivity-weighted offline calibration to find optimal bin boundaries per layer
4. **Per-vector dense-and-sparse**: isolate top-K outlier elements and store them in FP16 separately

**Result:** <0.1 perplexity degradation at 3-bit; enables 10M context on a single A100-80GB for LLaMA-7B.

**vs TurboQuant:** KVQuant requires an offline calibration set and per-layer NUQ datatypes. TurboQuant is fully data-oblivious. KVQuant's pre-RoPE approach is model-architecture-specific; TurboQuant works on any vector regardless of how it was produced.

### QJL (AAAI 2025, arXiv: 2406.03482)

**Approach:** The direct predecessor to TurboQuant Stage 2.
- Apply a Johnson-Lindenstrauss random projection followed by sign-bit quantization
- `Q_qjl(x) = sign(S·x)` where S has i.i.d. N(0,1) entries
- Eliminates memory overhead from quantization constants (no scale/zero-point)
- Asymmetric estimator: query uses full JL transform, key uses quantized QJL
- Result at 3-bit: >5× memory reduction without accuracy loss

**vs TurboQuant:** QJL alone is a 1-bit scheme applied to the full vector. TurboQuant extends it: first compress most bits with the optimal Lloyd-Max quantizer, then apply QJL only to the residual. This is strictly better — TurboQuant's Stage 1 captures nearly all the MSE information, and Stage 2 refines inner-product accuracy.

Same lead author (Amir Zandieh): QJL → PolarQuant → TurboQuant is the progression.

### SmoothQuant (arXiv: 2211.10438)

**Approach:** Post-training quantization for weights + activations (W8A8 INT8), not specifically KV cache.
- Key observation: activation outliers appear consistently in specific channels across tokens
- Migrates quantization difficulty from activations to weights using a per-channel smooth factor s
- `X · W = (X / s) · (s · W)` — divide activations by s, multiply weights by s
- Offline: compute s from calibration data; online: just divide by s before quantization

**Relationship to KV cache:** SmoothQuant was designed for weight/activation quantization but can be applied to KV cache. Some systems (NQKV) apply smoothing to KV tensors before quantizing. This is analogous to outlier channel handling.

**vs TurboQuant:** SmoothQuant requires calibration data to find smooth factors and is specific to channels (it doesn't handle within-channel distribution non-uniformity). TurboQuant's rotation is more powerful — it doesn't just scale outlier channels, it redistributes energy across all dimensions, solving both cross-channel and within-channel distribution problems simultaneously.

### Summary Comparison Table

| Method | Bits | Training-free | Data-oblivious | Per-token overhead | NIAH score | Key technique |
|--------|------|---------------|----------------|-------------------|------------|---------------|
| TurboQuant | 2.5–4 | Yes | Yes | 1 scalar (norm) | 0.997 | Random rotation + Lloyd-Max |
| KIVI | 2 | Yes | Yes | scale + zero-point per block | 0.981 | Asymmetric per-channel |
| KVQuant | 2–4 | Yes | No (calibration) | outlier indices + scales | — | Pre-RoPE + NUQ + sparse |
| QJL | 1–3 | Yes | Yes | None | — | JL transform + sign |
| SmoothQuant | 8 | Yes | No (calibration) | None (offline smooth factor) | — | Activation smoothing |

---

## 8. Implementation on MLX

### MLX Has Native Hadamard Transform

`mlx.core.hadamard_transform` is available (since at least MLX 0.30.1):

```python
import mlx.core as mx

def hadamard_transform(a: mx.array, scale: float | None = None,
                       stream=None) -> mx.array:
    """Walsh-Hadamard transform along the final axis.
    scale defaults to 1/sqrt(a.shape[-1]) for orthonormal matrix."""
```

**Supported sizes:** `n = m * 2^k` where m ∈ {1, 12, 20, 28} and 2^k ≤ 8192 (float32) or ≤ 16384 (float16/bfloat16).

For typical head dimensions:
- head_dim=64: 64 = 1×2^6 — **supported**
- head_dim=128: 128 = 1×2^7 — **supported**
- head_dim=256: 256 = 1×2^8 — **supported**
- head_dim=96: 96 = 12×2^3 — **supported** (m=12)
- head_dim=160: 160 = 20×2^3 — **supported** (m=20)

All common attention head dimensions are supported.

### Implementing TurboQuant_mse in MLX

```python
import mlx.core as mx
import numpy as np

class TurboQuantMLX:
    def __init__(self, head_dim: int, bits: int = 3):
        self.d = head_dim
        self.bits = bits

        # Offline: precompute random sign vector for WHT rotation
        # Rademacher: each entry is ±1 uniformly
        rng = np.random.default_rng(seed=42)
        signs = rng.choice([-1.0, 1.0], size=head_dim).astype(np.float16)
        self.D = mx.array(signs)  # shape: (head_dim,)

        # Offline: precompute Lloyd-Max codebook from Beta distribution
        self.codebook = self._build_codebook(head_dim, bits)

    def _build_codebook(self, d: int, bits: int) -> mx.array:
        """Compute Lloyd-Max centroids for Beta((d-1)/2, (d-1)/2) distribution.
        The marginal distribution of one coordinate of a uniform unit-sphere
        point is Beta with these parameters (scaled to [-1, 1])."""
        n_centroids = 2 ** bits
        # Grid of points on [-1, 1]
        x = np.linspace(-0.999, 0.999, 50000)
        alpha = (d - 1) / 2
        # PDF of scaled Beta on [-1, 1]: proportional to (1 - x^2)^((d-3)/2)
        pdf = (1 - x**2) ** max((d - 3) / 2, 0)
        pdf /= pdf.sum()

        # Lloyd-Max iterations
        centroids = np.linspace(-0.9, 0.9, n_centroids)
        for _ in range(300):
            # Assign each x to nearest centroid
            dists = np.abs(x[:, None] - centroids[None, :])
            assignments = dists.argmin(axis=1)
            # Update centroids as PDF-weighted mean of assigned points
            new_centroids = np.zeros(n_centroids)
            for k in range(n_centroids):
                mask = assignments == k
                if mask.sum() > 0:
                    new_centroids[k] = (x[mask] * pdf[mask]).sum() / pdf[mask].sum()
                else:
                    new_centroids[k] = centroids[k]
            if np.max(np.abs(new_centroids - centroids)) < 1e-8:
                break
            centroids = new_centroids

        return mx.array(centroids.astype(np.float16))  # shape: (2^bits,)

    def rotate(self, x: mx.array) -> mx.array:
        """Apply randomized Hadamard transform. x: (..., head_dim)"""
        return mx.hadamard_transform(x * self.D)
        # Default scale=1/sqrt(d) makes H orthonormal

    def quantize(self, x: mx.array):
        """
        x: (..., head_dim) — one KV vector per head, any batch shape
        Returns: (norms, indices)
          norms: (...,) float16
          indices: (..., head_dim) uint8/uint16
        """
        # Compute norms
        norms = mx.linalg.norm(x, axis=-1, keepdims=True)  # (..., 1)

        # Normalize to unit sphere
        x_unit = x / (norms + 1e-8)

        # Rotate
        y = self.rotate(x_unit)  # (..., head_dim)

        # Find nearest centroid for each coordinate
        # y: (..., head_dim), codebook: (2^bits,)
        diffs = y[..., None] - self.codebook  # (..., head_dim, 2^bits)
        indices = mx.argmin(mx.abs(diffs), axis=-1)  # (..., head_dim) uint

        return norms.squeeze(-1).astype(mx.float16), indices.astype(mx.uint8)

    def dequantize(self, norms: mx.array, indices: mx.array) -> mx.array:
        """
        norms: (...,) float16
        indices: (..., head_dim) uint8
        Returns: (..., head_dim) float16
        """
        # Lookup centroids
        y_tilde = self.codebook[indices]  # (..., head_dim)

        # Rotate back (inverse WHT: since H is orthonormal, H^T = H scaled)
        # For randomized WHT: inverse is D * H (same D since D^{-1} = D for ±1)
        x_unit_tilde = mx.hadamard_transform(y_tilde) * self.D

        # Scale by norm
        return x_unit_tilde * norms[..., None]
```

**Important note on the WHT inverse:** The randomized WHT `Π = H · D` has inverse `Π^{-1} = Π^T = D^T · H^T = D · H` (since H is symmetric/orthogonal and D is diagonal with ±1). So dequantization applies WHT again, then multiplies by the same sign vector D. This is because `H^T = H` for the Walsh-Hadamard matrix (it's its own inverse up to scaling), and `D^T = D` since D is diagonal with ±1.

### Compute Overhead of Online Rotation

For each new KV vector generated during decode:
- WHT: O(d log d) = O(128 × 7) ≈ 896 operations per vector
- Codebook lookup: O(d × 2^b) = O(128 × 8) = 1024 operations per vector (at 3 bits)
- Total: ~1900 operations per head per token, vs ~1000 for naive quantization (just a scale+shift)

At 32 heads, 128 layers: 32 × 128 × 1900 = ~7.8M operations per token for quantization.

For comparison, a single attention forward pass at d=4096 does ~2× d² = ~33M operations. The quantization overhead is ~25% of one attention layer's compute — non-trivial but manageable, especially since it replaces much more expensive KV cache I/O.

**WHT is particularly cheap on Apple Silicon** because `mx.hadamard_transform` is implemented as a native Metal kernel (butterfly structure maps efficiently to SIMD units).

### Key Implementation Challenges for MLX

1. **RoPE commutability**: RoPE and WHT do NOT commute (20.8% RMSE if you try to pre-rotate queries and use rotated keys without adjusting for RoPE). Solution: apply rotation **after** RoPE at the point of caching. This means quantization must happen inside the KV cache update, not at model weight initialization.

2. **Fused attention kernel**: To get speedup from quantized keys, you need a fused kernel that does the centroid lookup inside the Q@K^T computation rather than dequantizing first. The formula `⟨q, R^T · centroid⟩ = ⟨R·q, centroid⟩` lets you pre-rotate the query once and then do table lookups against uint8 indices. MLX's custom Metal kernel API would be needed for this.

3. **Prefill vs decode**: Prefill generates many KV entries at once — quantization overhead is proportional to sequence length and amortized over many output tokens. Decode generates one KV entry per step — overhead is per-token. The bottleneck for long contexts is actually I/O (reading the KV cache), so quantization helps most there.

4. **Values need different treatment**: Values have much milder outlier distributions than keys. A simpler per-token affine quantization (like KIVI) may be sufficient for values, reserving TurboQuant for keys only.

5. **MLX hadamard_transform constraint**: Size must be `m × 2^k` for m ∈ {1, 12, 20, 28}. Standard head dimensions (64, 128, 256) work. Non-power-of-2 head dims (e.g., 96, 160, 176) also work via the m ∈ {12, 20} path.

### Minimum Viable MLX Implementation

The simplest path that preserves correctness:

```python
# In the model's attention layer, during KV cache update:
def cache_kv(self, keys: mx.array, values: mx.array):
    # keys: (batch, heads, seq, head_dim)
    # Apply TurboQuant to keys
    norms, indices = self.turbo.quantize(keys)
    self._k_norms.append(norms)
    self._k_indices.append(indices)

    # Simple affine quantization for values (cheaper, sufficient)
    v_scale = values.max(axis=-1, keepdims=True)
    v_zero = values.min(axis=-1, keepdims=True)
    v_quant = ((values - v_zero) / (v_scale - v_zero + 1e-8) * 15).astype(mx.uint8)
    self._v_quant.append(v_quant)
    self._v_scale.append(v_scale)
    self._v_zero.append(v_zero)

def attention(self, q: mx.array) -> mx.array:
    # Dequantize keys for attention
    k_all = self.turbo.dequantize(
        mx.concatenate(self._k_norms, axis=-1),
        mx.concatenate(self._k_indices, axis=-2)
    )
    # Dequantize values
    v_indices = mx.concatenate(self._v_quant, axis=-2).astype(mx.float16)
    v_scale = mx.concatenate(self._v_scale, axis=-2)
    v_zero = mx.concatenate(self._v_zero, axis=-2)
    v_all = v_indices / 15.0 * (v_scale - v_zero) + v_zero

    # Standard attention
    return mx.fast.scaled_dot_product_attention(q, k_all, v_all, scale=self.scale)
```

This "dequantize then attend" path is correct but not maximally fast. The fused path (rotate query, table-lookup attention) requires a custom Metal kernel.

---

## 9. TurboQuant+ Community Format (llama.cpp / Apple Silicon)

The TheTom/turboquant_plus fork adds:

**Sparse V optimization:** During the V accumulation step in flash attention, positions with attention weight < 10^-6 are skipped entirely (no dequantization, no memory access). Benefit scales with context length:
- 4K context: +4% decode speed
- 16K context: +13% decode speed
- 32K context: +23% decode speed

**LUT optimization for Apple Silicon:** Centroid lookup tables stored in Metal constant memory. A "4-magnitude LUT" variant reduces 8 constant memory addresses to 4, giving +38% improvement on M2 Pro vs naive implementation.

**Format variants in community llama.cpp:**
- `--cache-type-k turbo2 --cache-type-v turbo2`: 2-bit, 6.4× compression, +6.48% perplexity
- `--cache-type-k turbo3 --cache-type-v turbo3`: 3.5-bit, 4.6× compression, +1.06% perplexity
- `--cache-type-k turbo4 --cache-type-v turbo4`: 4.25-bit, 3.8× compression, +0.23% perplexity

**Apple Silicon benchmark (M5 Max, Qwen3.5-35B):**
- turbo3: 2747 tok/s = 1.02× vs q8_0 KV cache
- Compression: 4.9× vs FP16
- NIAH: 9/9 single-needle retrieval

---

## 10. Gaps, Limitations, and Open Questions

1. **Official code not released.** Community implementations may differ from the paper in subtle ways (QJL typically omitted, Lloyd-Max iteration count varies, random seed handling differs).

2. **8× speedup claim uses FP32 baseline**, not FP16. Realistic speedup vs FP16 KV cache is model-dependent and hardware-dependent. Community H100 results show ~1.2× for Q@K^T kernel only; end-to-end gains are smaller.

3. **Limited model diversity in paper.** Benchmarks focus on Llama-3.1-8B and Ministral 7B. Community testing on Qwen3.5 series shows similar quality but different outlier characteristics.

4. **QJL stage rarely used in practice.** The inner-product correction requires custom attention kernels and the additional S matrix (O(d²) per head). Most implementations drop it, converging to PolarQuant (AISTATS 2026) behavior.

5. **RoPE interaction.** Quantization must happen post-RoPE. Pre-RoPE quantization is better for distribution uniformity but requires accounting for RoPE at decode time.

6. **MLX fused kernel gap.** No production-ready fused TurboQuant attention kernel exists for MLX Metal as of March 2026. The TheTom llama.cpp Metal fork has one, but it is not in mlx-lm. Implementing the fused kernel in MLX would require the custom Metal kernel API.

7. **RaBitQ priority dispute.** The RaBitQ paper authors (independently developed a similar rotation+codebook approach) publicly disputed TurboQuant's novelty claims, noting substantial overlap in methodology. This is a peer review dispute; the ICLR reviewers accepted the paper.

---

## 11. Key Sources

| Source | URL | Type |
|--------|-----|------|
| TurboQuant paper (arxiv) | https://arxiv.org/abs/2504.19874 | Primary |
| TurboQuant HTML (ar5iv) | https://ar5iv.labs.arxiv.org/html/2504.19874 | Primary |
| Google Research blog | https://research.google/blog/turboquant-redefining-ai-efficiency-with-extreme-compression/ | Primary |
| PolarQuant paper | https://arxiv.org/abs/2502.02617 | Related |
| QJL paper | https://arxiv.org/abs/2406.03482 | Related |
| KIVI paper | https://arxiv.org/abs/2402.02750 | Related |
| KVQuant paper | https://arxiv.org/abs/2401.18079 | Related |
| SmoothQuant paper | https://arxiv.org/abs/2211.10438 | Related |
| llama.cpp discussion | https://github.com/ggml-org/llama.cpp/discussions/20969 | Community |
| ik_llama.cpp implementation | https://github.com/ikawrakow/ik_llama.cpp/issues/1509 | Community |
| TheTom Metal fork | https://github.com/TheTom/turboquant_plus | Community |
| 0xSero PyTorch impl | https://github.com/0xSero/turboquant | Community |
| OnlyTerp PyTorch impl | https://github.com/OnlyTerp/turboquant | Community |
| Dejan.ai Triton walkthrough | https://dejan.ai/blog/turboquant/ | Analysis |
| MLX hadamard_transform API | https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.hadamard_transform.html | MLX |
| MLSurgeon analysis | https://themlsurgeon.substack.com/p/turboquant-what-3-bit-kv-caches-actually | Analysis |
| TurboQuant official site | https://turboquant.net/ | Primary |

---

*Researched: 2026-03-29. Paper status: accepted ICLR 2026, presentation April 2026. Official code: not yet released.*
