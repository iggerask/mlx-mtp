"""
TurboQuant: Hadamard-Rotated KV Cache Quantization.

Implements TurboQuant (Google, ICLR 2026) in two variants:

1. TurboQuantKVCache: Rotation + mlx's per-group affine quantization (original,
   fast but quality issues due to distribution mismatch).

2. TurboQuantLMKVCache: Rotation + Lloyd-Max codebook quantization (proper
   algorithm). Pre-computes optimal centroids for the Beta distribution that
   rotated coordinates provably follow. Universal codebook — no per-model
   calibration needed. 4-bit packed gives ~4x compression with near-lossless
   quality.

Math:
  - Rotation: x_rot = H @ (D * x)  where H = normalized Hadamard, D = random ±1
  - Since rotation is orthogonal: Q_rot @ K_rot^T = Q @ K^T  (exact)
  - For values: output = D * H @ (attn_weights @ V_rot)

Supports hybrid architectures like Qwen3.5 (GatedDeltaNet + Attention layers)
by only patching the standard Attention layers that use KV cache.

Requires: MLX with mx.hadamard_transform (available in mlx >= 0.18)
"""

import logging
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_map

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
# Fused Metal Kernels for TurboQuant
# ─────────────────────────────────────────────────────────────

def _make_dequant_4bit_kernel():
    """
    Fused 4-bit TurboQuant dequantization Metal kernel.

    Each thread handles one output element (one coordinate of one vector).
    Unpacks 4-bit index from packed byte, looks up codebook centroid, scales by norm.
    No intermediate tensors — single kernel dispatch.
    """
    if not mx.metal.is_available():
        return None
    source = """
        uint elem = thread_position_in_grid.x;

        uint vec_idx = elem / HD;
        uint d_idx = elem % HD;

        // Unpack 4-bit index from packed byte
        uint pair_idx = d_idx / 2;
        uint is_low = d_idx & 1;
        uint byte_idx = vec_idx * (HD / 2) + pair_idx;
        uint8_t packed = indices[byte_idx];
        uint idx = is_low ? (packed & 0x0F) : (packed >> 4);

        // Codebook lookup + norm scale (fused)
        float centroid = codebook[idx];
        float norm = static_cast<float>(norms[vec_idx]);
        out[elem] = static_cast<T>(centroid * norm);
    """
    return mx.fast.metal_kernel(
        name="tq_dequant_4bit",
        input_names=["indices", "codebook", "norms"],
        output_names=["out"],
        source=source,
    )


def _make_dequant_8bit_kernel():
    """
    Fused 8-bit TurboQuant dequantization Metal kernel.
    Same as 4-bit but indices are plain uint8 (no packing).
    """
    if not mx.metal.is_available():
        return None
    source = """
        uint elem = thread_position_in_grid.x;

        uint vec_idx = elem / HD;
        uint d_idx = elem % HD;

        uint idx = (uint)indices[vec_idx * HD + d_idx];
        float centroid = codebook[idx];
        float norm = static_cast<float>(norms[vec_idx]);
        out[elem] = static_cast<T>(centroid * norm);
    """
    return mx.fast.metal_kernel(
        name="tq_dequant_8bit",
        input_names=["indices", "codebook", "norms"],
        output_names=["out"],
        source=source,
    )


def _make_quantize_bsearch_kernel():
    """
    Fused quantization Metal kernel: binary search on boundaries.

    Each thread handles one coordinate. Reads normalized value, does binary
    search on sorted boundaries to find bucket index, writes uint8 index.
    Caller handles 4-bit packing separately (clean thread-safety).
    """
    if not mx.metal.is_available():
        return None
    source = """
        uint elem = thread_position_in_grid.x;
        float val = x_unit[elem];

        // Binary search on sorted boundaries
        int lo = 0;
        int hi = NB;
        while (lo < hi) {
            int mid = lo + (hi - lo) / 2;
            if (boundaries[mid] < val) lo = mid + 1;
            else hi = mid;
        }
        out[elem] = (uint8_t)lo;
    """
    return mx.fast.metal_kernel(
        name="tq_quantize_bsearch",
        input_names=["x_unit", "boundaries"],
        output_names=["out"],
        source=source,
    )


# Lazy-init kernel singletons
_dequant_4bit = None
_dequant_8bit = None
_quant_bsearch = None


def _get_dequant_4bit():
    global _dequant_4bit
    if _dequant_4bit is None:
        _dequant_4bit = _make_dequant_4bit_kernel()
    return _dequant_4bit


def _get_dequant_8bit():
    global _dequant_8bit
    if _dequant_8bit is None:
        _dequant_8bit = _make_dequant_8bit_kernel()
    return _dequant_8bit


def _get_quant_bsearch():
    global _quant_bsearch
    if _quant_bsearch is None:
        _quant_bsearch = _make_quantize_bsearch_kernel()
    return _quant_bsearch


def _make_tq_sdpa_4bit_kernel():
    """
    Fused TurboQuant 4-bit SDPA Metal kernel (v2: cooperative threads).

    One threadgroup per (batch, query_head). HD threads cooperate:
    - Each thread handles one head dimension
    - Q·K dot product computed cooperatively via SIMD reduction
    - Online softmax (single pass over T — no recomputation)
    - Each thread independently accumulates its output dimension

    Memory reads per threadgroup: T × HD bytes (vs T × HD × HD in v1).
    """
    if not mx.metal.is_available():
        return None

    source = """
        // One threadgroup per (batch, query_head)
        uint bh = threadgroup_position_in_grid.x;
        uint d_idx = thread_position_in_threadgroup.x;

        if (bh >= BH || d_idx >= HD) return;

        uint b = bh / NH;
        uint h = bh % NH;
        uint kv_h = h / N_REP;

        // Load query value for this dimension (stays in register)
        float q_d = static_cast<float>(queries[(b * NH + h) * HD + d_idx]);

        // Pointers to this head's KV data
        auto ki_base = k_indices + (b * NKV + kv_h) * T_LEN * (HD / 2);
        auto kn_base = k_norms + (b * NKV + kv_h) * T_LEN;
        auto vi_base = v_indices + (b * NKV + kv_h) * T_LEN * (HD / 2);
        auto vn_base = v_norms + (b * NKV + kv_h) * T_LEN;

        float attn_scale = static_cast<float>(scale_arr[0]);

        // Threadgroup memory for tree reduction of dot products
        threadgroup float shared_dot[HD];

        // Packed index addressing for this dimension
        uint pair_idx = d_idx / 2;
        uint is_low = d_idx & 1;

        // Online softmax accumulators (per-thread, one per output dim)
        float running_max = -1e9f;
        float running_sum = 0.0f;
        float weighted_val = 0.0f;

        for (int t = 0; t < T_LEN; ++t) {
            // === Cooperative Q·K score computation ===
            uint8_t packed_k = ki_base[t * (HD / 2) + pair_idx];
            uint k_idx = is_low ? (packed_k & 0x0F) : (packed_k >> 4);
            shared_dot[d_idx] = q_d * codebook[k_idx];

            // Tree reduction for dot product sum
            threadgroup_barrier(metal::mem_flags::mem_threadgroup);
            for (uint stride = HD / 2; stride > 0; stride >>= 1) {
                if (d_idx < stride) {
                    shared_dot[d_idx] += shared_dot[d_idx + stride];
                }
                threadgroup_barrier(metal::mem_flags::mem_threadgroup);
            }

            // All threads read the final score
            float score = attn_scale * static_cast<float>(kn_base[t]) * shared_dot[0];
            threadgroup_barrier(metal::mem_flags::mem_threadgroup);

            // === Online softmax update ===
            float new_max = max(running_max, score);
            float correction = exp(running_max - new_max);
            float w = exp(score - new_max);

            running_sum = running_sum * correction + w;
            weighted_val *= correction;

            // === Accumulate weighted value for this dimension ===
            uint8_t packed_v = vi_base[t * (HD / 2) + pair_idx];
            uint v_idx = is_low ? (packed_v & 0x0F) : (packed_v >> 4);
            float val = codebook[v_idx] * static_cast<float>(vn_base[t]);

            weighted_val += w * val;
            running_max = new_max;
        }

        // Write output for this dimension
        out[(b * NH + h) * HD + d_idx] = static_cast<T>(weighted_val / running_sum);
    """
    return mx.fast.metal_kernel(
        name="tq_sdpa_4bit_v2",
        input_names=["queries", "k_indices", "k_norms", "v_indices", "v_norms", "codebook", "scale_arr"],
        output_names=["out"],
        source=source,
    )


_tq_sdpa_4bit = None


def _get_tq_sdpa_4bit():
    global _tq_sdpa_4bit
    if _tq_sdpa_4bit is None:
        _tq_sdpa_4bit = _make_tq_sdpa_4bit_kernel()
    return _tq_sdpa_4bit


def turboquant_sdpa_4bit(
    queries: mx.array,      # (B, n_heads, 1, head_dim) already rotated
    k_indices: mx.array,    # (B, n_kv_heads, T, head_dim//2) uint8 packed
    k_norms: mx.array,      # (B, n_kv_heads, T) float16
    v_indices: mx.array,    # (B, n_kv_heads, T, head_dim//2) uint8 packed
    v_norms: mx.array,      # (B, n_kv_heads, T) float16
    codebook: mx.array,     # (16,) float32
    scale: float,
    mask: Optional[mx.array] = None,
) -> mx.array:
    """
    Fused SDPA directly on 4-bit quantized KV cache.

    No dequantization needed — computes attention scores by unpacking
    4-bit indices and looking up codebook entries inside the Metal kernel.
    """
    kernel = _get_tq_sdpa_4bit()
    if kernel is None:
        raise RuntimeError("Metal not available")

    B, n_heads, L, HD = queries.shape
    n_kv_heads = k_indices.shape[1]
    T = k_indices.shape[2]
    n_rep = n_heads // n_kv_heads

    BH = B * n_heads
    scale_arr = mx.array([scale], dtype=mx.float32)

    # v2 kernel: one threadgroup per (batch, head), HD threads cooperate
    # grid = total threads (MLX convention), not threadgroups
    out = kernel(
        inputs=[
            queries.astype(mx.float16),
            k_indices, k_norms, v_indices, v_norms, codebook, scale_arr
        ],
        template=[
            ("T", mx.float16),
            ("BH", BH), ("NH", n_heads), ("NKV", n_kv_heads),
            ("N_REP", n_rep), ("HD", HD), ("T_LEN", T),
        ],
        grid=(BH * HD, 1, 1),
        threadgroup=(HD, 1, 1),
        output_shapes=[(BH * HD,)],
        output_dtypes=[mx.float16],
    )[0]

    return out.reshape(B, n_heads, 1, HD)


# ─────────────────────────────────────────────────────────────
# Lloyd-Max Codebook Computation
# ─────────────────────────────────────────────────────────────

_codebook_cache: Dict[Tuple[int, int], Tuple[mx.array, mx.array]] = {}


def _compute_lloyd_max(d: int, bits: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute Lloyd-Max codebook for the Beta distribution arising from
    Hadamard-rotating d-dimensional vectors.

    After normalizing to the unit sphere and rotating with a random Hadamard
    transform, each coordinate follows:
        f(x) ∝ (1 - x²)^((d-3)/2)  on [-1, 1]
    This is a scaled Beta((d-1)/2, (d-1)/2) distribution.

    The Lloyd-Max algorithm finds the globally optimal scalar quantizer
    (minimum MSE) for this known distribution.

    Returns (centroids, boundaries) as numpy arrays.
    """
    n_centroids = 1 << bits
    n_grid = 50000
    n_iter = 300

    x = np.linspace(-1 + 1e-7, 1 - 1e-7, n_grid)
    alpha = max((d - 3) / 2.0, 0.0)
    pdf = np.power(np.maximum(1.0 - x * x, 0.0), alpha) if alpha > 0 else np.ones_like(x)
    pdf = pdf / pdf.sum()

    # Initialize centroids uniformly across the support
    centroids = np.linspace(-1 + 0.5 / n_centroids, 1 - 0.5 / n_centroids, n_centroids)

    for _ in range(n_iter):
        # Assignment: each grid point → nearest centroid
        dists = np.abs(x[:, None] - centroids[None, :])
        assignments = np.argmin(dists, axis=1)

        # Update: centroid = PDF-weighted mean of assigned points
        for c in range(n_centroids):
            mask = assignments == c
            if mask.any():
                centroids[c] = np.sum(x[mask] * pdf[mask]) / np.sum(pdf[mask])

    centroids = np.sort(centroids).astype(np.float32)
    boundaries = ((centroids[:-1] + centroids[1:]) / 2.0).astype(np.float32)
    return centroids, boundaries


def _get_codebook(d: int, bits: int) -> Tuple[mx.array, mx.array]:
    """Get Lloyd-Max codebook as MLX arrays (cached by (d, bits))."""
    key = (d, bits)
    if key not in _codebook_cache:
        logger.info(f"Computing Lloyd-Max codebook: d={d}, bits={bits}")
        c_np, b_np = _compute_lloyd_max(d, bits)
        c, b = mx.array(c_np), mx.array(b_np)
        mx.eval(c, b)
        _codebook_cache[key] = (c, b)
        logger.info(
            f"Codebook ready: {1 << bits} centroids, "
            f"range [{c_np[0]:.4f}, {c_np[-1]:.4f}]"
        )
    return _codebook_cache[key]


class TurboQuantKVCache:
    """
    KV cache that applies Hadamard rotation before quantization.

    Stores keys and values in rotated+quantized form. At attention time,
    queries must also be rotated (via patch_model_for_turboquant) and
    the output must be un-rotated.

    The rotation spreads outlier energy across all dimensions, making
    per-group quantization much more effective — especially at INT4
    where outlier channels dominate the error in naive quantization.
    """

    step = 256

    def __init__(self, head_dim: int, group_size: int = 64, bits: int = 4, seed: int = 42):
        self.head_dim = head_dim
        self.group_size = group_size
        self.bits = bits

        # Fixed Rademacher sign vector (±1) for this cache
        key = mx.random.key(seed)
        self.D = (mx.random.bernoulli(key=key, shape=(head_dim,)).astype(mx.float32) * 2 - 1)

        # Quantized storage
        self.keys = None
        self.values = None
        self.offset = 0

    def rotate(self, x: mx.array) -> mx.array:
        """Apply rotation: H @ (D * x). D broadcasts over batch/head/seq dims."""
        return mx.hadamard_transform(x * self.D)

    def unrotate(self, x: mx.array) -> mx.array:
        """Inverse rotation: D * (H @ x). H is self-inverse when normalized."""
        return mx.hadamard_transform(x) * self.D

    def update_and_fetch(
        self, keys: mx.array, values: mx.array
    ) -> Tuple[Any, Any]:
        """
        Rotate, quantize, and store keys/values.

        Args:
            keys: (B, n_kv_heads, num_steps, head_dim) post-RoPE keys
            values: (B, n_kv_heads, num_steps, head_dim) values

        Returns:
            Quantized key and value tuples for use with quantized SDPA.
        """
        B, n_kv_heads, num_steps, k_head_dim = keys.shape
        v_head_dim = values.shape[-1]
        prev = self.offset

        # Rotate before quantization — this is the TurboQuant innovation
        keys_rot = self.rotate(keys)
        values_rot = self.rotate(values)

        # Allocate or expand quantized storage
        if self.keys is None or (prev + num_steps) > self.keys[0].shape[-2]:
            el_per_int = 8 * mx.uint32.size // self.bits
            new_steps = (self.step + num_steps - 1) // self.step * self.step

            shape = (B, n_kv_heads, new_steps)

            def init_quant(dim):
                return (
                    mx.zeros((*shape, dim // el_per_int), dtype=mx.uint32),
                    mx.zeros((*shape, dim // self.group_size), dtype=keys.dtype),
                    mx.zeros((*shape, dim // self.group_size), dtype=keys.dtype),
                )

            def expand_quant(x):
                new_x = mx.zeros((*shape, x.shape[-1]), dtype=x.dtype)
                return mx.concatenate([x, new_x], axis=-2)

            if self.keys is not None:
                if prev % self.step != 0:
                    self.keys, self.values = tree_map(
                        lambda x: x[..., :prev, :], (self.keys, self.values)
                    )
                self.keys, self.values = tree_map(
                    expand_quant, (self.keys, self.values)
                )
            else:
                self.keys = init_quant(k_head_dim)
                self.values = init_quant(v_head_dim)

        self.offset += num_steps

        # Quantize the rotated keys/values
        q_keys = mx.quantize(keys_rot, group_size=self.group_size, bits=self.bits)
        q_values = mx.quantize(values_rot, group_size=self.group_size, bits=self.bits)

        for i in range(len(self.keys)):
            self.keys[i][..., prev:self.offset, :] = q_keys[i]
            self.values[i][..., prev:self.offset, :] = q_values[i]

        return (
            tree_map(lambda x: x[..., :self.offset, :], self.keys),
            tree_map(lambda x: x[..., :self.offset, :], self.values),
        )

    @property
    def state(self):
        if self.keys is None:
            return None
        if self.offset == self.keys[0].shape[2]:
            return self.keys, self.values
        return tree_map(
            lambda x: x[..., :self.offset, :], (self.keys, self.values)
        )

    @state.setter
    def state(self, v):
        self.keys, self.values = v

    @property
    def meta_state(self):
        return tuple(map(str, (self.offset, self.group_size, self.bits)))

    @meta_state.setter
    def meta_state(self, v):
        self.offset, self.group_size, self.bits = map(int, v)

    def is_trimmable(self):
        return True

    def trim(self, n):
        n = min(self.offset, n)
        self.offset -= n
        return n

    def make_mask(self, *args, **kwargs):
        from mlx_lm.models.cache import create_attention_mask
        return create_attention_mask(*args, offset=self.offset, **kwargs)

    def empty(self):
        return self.keys is None

    @property
    def nbytes(self):
        from mlx.utils import tree_reduce
        if self.keys is None:
            return 0
        return tree_reduce(lambda a, x: a + x.nbytes, (self.keys, self.values), 0)


class TurboQuantLMKVCache:
    """
    KV cache with Hadamard rotation + Lloyd-Max codebook quantization.

    This is the full TurboQuant algorithm:
    1. Compute per-vector norms, normalize to unit sphere
    2. Apply random Hadamard rotation
    3. Quantize each coordinate with Lloyd-Max centroids (universal codebook)
    4. Store: norms (float16) + indices (uint8, 4-bit packed for bits=4)

    At attention time, dequantize on-the-fly: centroid lookup × norm.
    Returns float K,V in rotated space for standard SDPA.

    The codebook is pre-computed from the Beta distribution that rotated
    coordinates provably follow. No per-tensor calibration needed.
    """

    step = 256

    def __init__(self, head_dim: int, bits: int = 4, seed: int = 42, passthrough: bool = False):
        self.head_dim = head_dim
        # Store as _bits to avoid triggering mlx-lm's quantized SDPA path,
        # which checks hasattr(cache, "bits"). We return dequantized floats,
        # so we need standard SDPA.
        self._bits = bits
        self.n_centroids = 1 << bits
        self.pack_4bit = (bits == 4)
        self.index_last_dim = head_dim // 2 if self.pack_4bit else head_dim
        self.passthrough = passthrough

        # Universal codebook
        if not passthrough:
            self.codebook, self.boundaries = _get_codebook(head_dim, bits)
        else:
            self.codebook, self.boundaries = None, None

        # Random sign vector for Hadamard rotation
        key = mx.random.key(seed)
        self.D = (mx.random.bernoulli(key=key, shape=(head_dim,)).astype(mx.float32) * 2 - 1)

        # Quantized storage (allocated on first update)
        self.key_norms = None
        self.key_indices = None
        self.val_norms = None
        self.val_indices = None
        # Float cache for incremental dequantization
        self._key_float = None
        self._val_float = None
        self.offset = 0
        self._dtype = mx.bfloat16

    def _pack_indices(self, indices: mx.array) -> mx.array:
        """Pack to 4-bit: (..., head_dim) uint8 → (..., head_dim//2) uint8."""
        pairs = indices.reshape(*indices.shape[:-1], -1, 2)
        return (pairs[..., 0] << 4) | pairs[..., 1]

    def _unpack_indices(self, packed: mx.array) -> mx.array:
        """Unpack 4-bit: (..., head_dim//2) uint8 → (..., head_dim) int32."""
        high = (packed >> 4).astype(mx.int32)
        low = (packed & 0x0F).astype(mx.int32)
        return mx.stack([high, low], axis=-1).reshape(
            *packed.shape[:-1], packed.shape[-1] * 2
        )

    def _quantize(self, x: mx.array) -> Tuple[mx.array, mx.array]:
        """
        Quantize rotated vectors using Lloyd-Max codebook.

        Uses a fused Metal kernel for binary search on boundaries when available,
        falling back to a Python implementation otherwise.
        """
        norms = mx.linalg.norm(x, axis=-1)
        x_unit = x / (norms[..., None] + 1e-8)

        kernel = _get_quant_bsearch()
        orig_shape = x_unit.shape
        flat = x_unit.reshape(-1).astype(mx.float32)
        n_elems = flat.shape[0]

        if kernel is not None:
            nb = self.boundaries.shape[0]
            indices = kernel(
                inputs=[flat, self.boundaries],
                template=[("NB", nb)],
                grid=(n_elems, 1, 1),
                threadgroup=(min(256, n_elems), 1, 1),
                output_shapes=[(n_elems,)],
                output_dtypes=[mx.uint8],
            )[0]
            indices = indices.reshape(orig_shape)
        else:
            # Fallback: comparison-based bucketing
            x_f = x_unit.astype(mx.float32)
            indices = (x_f[..., None] > self.boundaries[None]).sum(axis=-1).astype(mx.uint8)

        if self.pack_4bit:
            indices = self._pack_indices(indices)

        return norms.astype(mx.float16), indices

    def _dequantize(self, norms: mx.array, indices: mx.array) -> mx.array:
        """
        Dequantize: codebook lookup × norm → rotated vectors.

        Uses a fused Metal kernel (unpack + lookup + scale in one dispatch)
        when available, falling back to Python ops otherwise.
        """
        orig_norms_shape = norms.shape  # (...,)
        n_vecs = 1
        for s in orig_norms_shape:
            n_vecs *= s
        hd = self.head_dim

        if self.pack_4bit:
            kernel = _get_dequant_4bit()
            if kernel is not None:
                flat_norms = norms.reshape(-1).astype(mx.float16)
                flat_indices = indices.reshape(n_vecs, hd // 2)
                n_elems = n_vecs * hd

                out = kernel(
                    inputs=[flat_indices, self.codebook, flat_norms],
                    template=[("T", self._dtype), ("HD", hd)],
                    grid=(n_elems, 1, 1),
                    threadgroup=(min(256, n_elems), 1, 1),
                    output_shapes=[(n_elems,)],
                    output_dtypes=[self._dtype],
                )[0]
                return out.reshape(*orig_norms_shape, hd)
            else:
                idx = self._unpack_indices(indices)
                values = self.codebook[idx]
                return (values * norms[..., None].astype(mx.float32)).astype(self._dtype)
        else:
            kernel = _get_dequant_8bit()
            if kernel is not None:
                flat_norms = norms.reshape(-1).astype(mx.float16)
                flat_indices = indices.reshape(n_vecs, hd)
                n_elems = n_vecs * hd

                out = kernel(
                    inputs=[flat_indices, self.codebook, flat_norms],
                    template=[("T", self._dtype), ("HD", hd)],
                    grid=(n_elems, 1, 1),
                    threadgroup=(min(256, n_elems), 1, 1),
                    output_shapes=[(n_elems,)],
                    output_dtypes=[self._dtype],
                )[0]
                return out.reshape(*orig_norms_shape, hd)
            else:
                idx = indices.astype(mx.int32)
                values = self.codebook[idx]
                return (values * norms[..., None].astype(mx.float32)).astype(self._dtype)

    def rotate(self, x: mx.array) -> mx.array:
        """Apply rotation: H @ (D * x)."""
        return mx.hadamard_transform(x * self.D)

    def update_and_fetch(
        self, keys: mx.array, values: mx.array
    ) -> Tuple[mx.array, mx.array]:
        """
        Rotate, quantize (Lloyd-Max), store, and return dequantized K,V.

        Uses incremental dequantization: only dequantizes newly added tokens
        and appends to a cached float buffer. This avoids re-dequantizing
        the entire cache on every decode step.

        Returns dequantized K,V in ROTATED space (queries must also be rotated).
        """
        B, n_kv_heads, num_steps, hd = keys.shape
        self._dtype = keys.dtype
        prev = self.offset

        # Rotate before quantization
        keys_rot = self.rotate(keys)
        values_rot = self.rotate(values)

        # Passthrough mode: no quantization, just store rotated values
        if self.passthrough:
            self.offset += num_steps
            if self._key_float is None:
                self._key_float = keys_rot
                self._val_float = values_rot
            else:
                self._key_float = mx.concatenate([self._key_float, keys_rot], axis=2)
                self._val_float = mx.concatenate([self._val_float, values_rot], axis=2)
            return self._key_float, self._val_float

        # Quantize new tokens only
        k_norms, k_indices = self._quantize(keys_rot)
        v_norms, v_indices = self._quantize(values_rot)

        # Dequantize new tokens immediately (cheap: only num_steps tokens)
        dk_new = self._dequantize(k_norms, k_indices)
        dv_new = self._dequantize(v_norms, v_indices)

        # Allocate or expand quantized storage
        if self.key_norms is None or (prev + num_steps) > self.key_norms.shape[-1]:
            new_steps = ((self.step + num_steps - 1) // self.step) * self.step
            shape_n = (B, n_kv_heads, new_steps)
            shape_i = (B, n_kv_heads, new_steps, self.index_last_dim)

            if self.key_norms is not None:
                if prev % self.step != 0:
                    self.key_norms = self.key_norms[..., :prev]
                    self.key_indices = self.key_indices[..., :prev, :]
                    self.val_norms = self.val_norms[..., :prev]
                    self.val_indices = self.val_indices[..., :prev, :]
                self.key_norms = mx.concatenate(
                    [self.key_norms, mx.zeros(shape_n, dtype=mx.float16)], axis=-1
                )
                self.key_indices = mx.concatenate(
                    [self.key_indices, mx.zeros(shape_i, dtype=mx.uint8)], axis=-2
                )
                self.val_norms = mx.concatenate(
                    [self.val_norms, mx.zeros(shape_n, dtype=mx.float16)], axis=-1
                )
                self.val_indices = mx.concatenate(
                    [self.val_indices, mx.zeros(shape_i, dtype=mx.uint8)], axis=-2
                )
            else:
                self.key_norms = mx.zeros(shape_n, dtype=mx.float16)
                self.key_indices = mx.zeros(shape_i, dtype=mx.uint8)
                self.val_norms = mx.zeros(shape_n, dtype=mx.float16)
                self.val_indices = mx.zeros(shape_i, dtype=mx.uint8)

        # Store quantized
        self.key_norms[..., prev : prev + num_steps] = k_norms
        self.key_indices[..., prev : prev + num_steps, :] = k_indices
        self.val_norms[..., prev : prev + num_steps] = v_norms
        self.val_indices[..., prev : prev + num_steps, :] = v_indices
        self.offset += num_steps

        # Maintain float cache: append dequantized new tokens
        if self._key_float is None:
            self._key_float = dk_new
            self._val_float = dv_new
        else:
            self._key_float = mx.concatenate(
                [self._key_float, dk_new], axis=2
            )
            self._val_float = mx.concatenate(
                [self._val_float, dv_new], axis=2
            )

        return self._key_float, self._val_float

    @property
    def state(self):
        if self.key_norms is None:
            return None
        return (
            self.key_norms[..., : self.offset],
            self.key_indices[..., : self.offset, :],
            self.val_norms[..., : self.offset],
            self.val_indices[..., : self.offset, :],
            self._key_float,
            self._val_float,
        )

    @state.setter
    def state(self, v):
        self.key_norms, self.key_indices, self.val_norms, self.val_indices, \
            self._key_float, self._val_float = v

    @property
    def meta_state(self):
        return tuple(map(str, (self.offset, self._bits)))

    @meta_state.setter
    def meta_state(self, v):
        self.offset, self._bits = map(int, v)

    def is_trimmable(self):
        return True

    def trim(self, n):
        n = min(self.offset, n)
        self.offset -= n
        if self._key_float is not None and n > 0:
            self._key_float = self._key_float[:, :, :self.offset, :]
            self._val_float = self._val_float[:, :, :self.offset, :]
        return n

    def make_mask(self, *args, **kwargs):
        from mlx_lm.models.cache import create_attention_mask
        return create_attention_mask(*args, offset=self.offset, **kwargs)

    def empty(self):
        return self.key_norms is None

    @property
    def nbytes(self):
        if self.key_norms is None:
            return 0
        n = self.offset
        B = self.key_norms.shape[0]
        n_heads = self.key_norms.shape[1]
        # Norms: 2 bytes (float16), K+V
        norm_bytes = B * n_heads * n * 2 * 2
        # Indices: index_last_dim bytes (uint8), K+V
        idx_bytes = B * n_heads * n * self.index_last_dim * 2
        return norm_bytes + idx_bytes


def _find_attention_layers(model):
    """
    Find all standard attention layers in a model (supporting hybrid architectures).

    Returns:
        List of (layer_idx, attn_module, attn_attr_name) for layers with KV cache attention.
    """
    if hasattr(model, "language_model"):
        text_model = model.language_model
    else:
        text_model = model

    if hasattr(text_model, "model") and hasattr(text_model.model, "layers"):
        layers = text_model.model.layers
    elif hasattr(text_model, "layers"):
        layers = text_model.layers
    else:
        raise ValueError("Cannot find model layers")

    attn_layers = []
    for i, layer in enumerate(layers):
        # Check common attention attribute names
        for attr_name in ["self_attn", "attention", "attn"]:
            if hasattr(layer, attr_name):
                attn = getattr(layer, attr_name)
                # Verify it's a standard attention (has q_proj, KV cache)
                if hasattr(attn, "q_proj") and hasattr(attn, "k_proj"):
                    attn_layers.append((i, attn, attr_name))
                break

    return attn_layers, layers


def patch_model_for_turboquant(
    model: nn.Module, bits: int = 4, group_size: int = 64,
    use_lloyd_max: bool = False,
) -> Tuple[Callable, Callable]:
    """
    Patch a model's attention layers to use TurboQuant KV caching.

    Supports hybrid architectures (e.g., Qwen3.5 with GatedDeltaNet + Attention):
    only patches standard Attention layers, leaves linear attention layers untouched.

    Args:
        model: The loaded mlx-lm model
        bits: Quantization bits (4 or 8)
        group_size: Quantization group size (only for non-Lloyd-Max mode)
        use_lloyd_max: If True, use Lloyd-Max codebook quantization (proper
            TurboQuant). If False, use mlx's per-group affine quantization.

    Returns:
        (make_cache_fn, unpatch_fn)
        - make_cache_fn(): creates the hybrid cache list (TurboQuant for attn, original for rest)
        - unpatch_fn(): restores original attention methods
    """
    from mlx_lm.models.cache import make_prompt_cache

    attn_layers, all_layers = _find_attention_layers(model)
    n_layers = len(all_layers)

    if not attn_layers:
        raise ValueError("No attention layers found to patch")

    # Detect head_dim from first attention layer
    first_attn = attn_layers[0][1]
    n_heads = getattr(first_attn, "n_heads", None) or getattr(first_attn, "num_attention_heads")
    head_dim = getattr(first_attn, "head_dim", first_attn.q_proj.weight.shape[0] // n_heads)

    mode = "Lloyd-Max" if use_lloyd_max else "affine"
    logger.info(
        f"TurboQuant ({mode}): patching {len(attn_layers)}/{n_layers} attention layers, "
        f"head_dim={head_dim}, bits={bits}"
        + (f", group_size={group_size}" if not use_lloyd_max else "")
    )

    # Store originals and create per-layer sign vectors
    originals = {}
    attn_layer_indices = set()

    for layer_idx, attn, attr_name in attn_layers:
        attn_layer_indices.add(layer_idx)

        # Unique D per layer for decorrelation
        key = mx.random.key(42 + layer_idx)
        D = (mx.random.bernoulli(key=key, shape=(head_dim,)).astype(mx.float32) * 2 - 1)

        originals[layer_idx] = (attn.__call__, attr_name)

        # Detect architecture-specific features
        # Qwen3.5/Qwen3-Next: gated attention output (q_proj produces queries + gate)
        has_gate = _detect_gated_attention(attn)
        # Detect attribute names for heads
        n_kv_heads_attr = "n_kv_heads" if hasattr(attn, "n_kv_heads") else "num_key_value_heads"
        n_heads_attr = "n_heads" if hasattr(attn, "n_heads") else "num_attention_heads"

        def make_patched_call(attn_mod, d_vec, _has_gate, _n_heads_attr, _n_kv_heads_attr):
            def patched_call(x, mask=None, cache=None):
                from mlx_lm.models.base import scaled_dot_product_attention
                B, L, _D = x.shape

                n_h = getattr(attn_mod, _n_heads_attr)
                n_kv = getattr(attn_mod, _n_kv_heads_attr)
                hd = attn_mod.head_dim if hasattr(attn_mod, "head_dim") else head_dim

                if _has_gate:
                    # Qwen3-Next style: q_proj outputs [queries, gate] concatenated
                    q_proj_out = attn_mod.q_proj(x)
                    queries, gate = mx.split(
                        q_proj_out.reshape(B, L, n_h, -1), 2, axis=-1
                    )
                    gate = gate.reshape(B, L, -1)
                else:
                    queries = attn_mod.q_proj(x)
                    gate = None
                    queries = queries.reshape(B, L, n_h, -1)

                keys = attn_mod.k_proj(x)
                values = attn_mod.v_proj(x)

                if hasattr(attn_mod, "q_norm"):
                    queries = attn_mod.q_norm(queries).transpose(0, 2, 1, 3)
                else:
                    queries = queries.transpose(0, 2, 1, 3)

                if hasattr(attn_mod, "k_norm"):
                    keys = attn_mod.k_norm(
                        keys.reshape(B, L, n_kv, -1)
                    ).transpose(0, 2, 1, 3)
                else:
                    keys = keys.reshape(B, L, n_kv, -1).transpose(0, 2, 1, 3)

                values = values.reshape(B, L, n_kv, -1).transpose(0, 2, 1, 3)

                if cache is not None:
                    queries = attn_mod.rope(queries, offset=cache.offset)
                    keys = attn_mod.rope(keys, offset=cache.offset)
                    keys, values = cache.update_and_fetch(keys, values)
                else:
                    queries = attn_mod.rope(queries)
                    keys = attn_mod.rope(keys)

                # TurboQuant: rotate queries when using TurboQuant cache
                if isinstance(cache, (TurboQuantKVCache, TurboQuantLMKVCache)):
                    queries = mx.hadamard_transform(queries * d_vec)

                output = scaled_dot_product_attention(
                    queries, keys, values,
                    cache=cache, scale=attn_mod.scale, mask=mask,
                )

                # TurboQuant: un-rotate output
                if isinstance(cache, (TurboQuantKVCache, TurboQuantLMKVCache)):
                    output = mx.hadamard_transform(output) * d_vec

                output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)

                if gate is not None:
                    return attn_mod.o_proj(output * mx.sigmoid(gate))
                else:
                    return attn_mod.o_proj(output)

            return patched_call

        attn.__call__ = make_patched_call(
            attn, D, has_gate, n_heads_attr, n_kv_heads_attr
        )

    def make_turboquant_cache() -> list:
        """Create hybrid cache: TurboQuant for attn layers, original for others."""
        # Get original cache structure (to know what non-attn layers need)
        original_cache = make_prompt_cache(model)

        hybrid_cache = []
        for i in range(n_layers):
            if i in attn_layer_indices:
                if use_lloyd_max:
                    hybrid_cache.append(
                        TurboQuantLMKVCache(
                            head_dim=head_dim,
                            bits=bits,
                            seed=42 + i,
                        )
                    )
                else:
                    hybrid_cache.append(
                        TurboQuantKVCache(
                            head_dim=head_dim,
                            group_size=group_size,
                            bits=bits,
                            seed=42 + i,
                        )
                    )
            else:
                hybrid_cache.append(original_cache[i])

        return hybrid_cache

    def unpatch():
        """Restore original attention methods."""
        for layer_idx, (orig_call, attr_name) in originals.items():
            attn = getattr(all_layers[layer_idx], attr_name)
            attn.__call__ = orig_call
        logger.info("Restored original attention methods")

    return make_turboquant_cache, unpatch


def _detect_gated_attention(attn) -> bool:
    """
    Detect if attention module uses gated output (Qwen3-Next style).

    In gated attention, q_proj outputs 2x the expected size (queries + gate),
    and the output is multiplied by sigmoid(gate) before o_proj.
    """
    if not hasattr(attn, "q_proj"):
        return False

    n_heads = getattr(attn, "n_heads", None) or getattr(attn, "num_attention_heads", None)
    head_dim = getattr(attn, "head_dim", None)
    if n_heads is None or head_dim is None:
        return False

    expected_q_dim = n_heads * head_dim
    actual_q_dim = attn.q_proj.weight.shape[0]

    # If q_proj outputs 2x expected, it's gated
    return actual_q_dim == expected_q_dim * 2


def quantize_cache_after_prefill(cache_list: list, bits: int, group_size: int):
    """Convert KVCache entries to QuantizedKVCache after prefill."""
    for i, c in enumerate(cache_list):
        if hasattr(c, "to_quantized"):
            cache_list[i] = c.to_quantized(group_size=group_size, bits=bits)
    return cache_list
