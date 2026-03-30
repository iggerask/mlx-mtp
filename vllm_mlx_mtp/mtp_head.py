"""
MTP (Multi-Token Prediction) Head for Qwen3.5 models.

Actual architecture from Qwen3.5 weight inspection:

    mtp.pre_fc_norm_hidden.weight   - RMSNorm on hidden state
    mtp.pre_fc_norm_embedding.weight - RMSNorm on token embedding
    mtp.fc.weight                   - Linear(hidden*2, hidden) concat projection
    mtp.layers.0.*                  - Full transformer decoder layer:
        .input_layernorm.weight
        .self_attn.{q,k,v,o}_proj.weight
        .self_attn.{q,k}_norm.weight
        .post_attention_layernorm.weight
        .mlp.{gate,up,down}_proj.weight
    mtp.norm.weight                 - Final RMSNorm before shared lm_head

The MTP head takes:
    1. Last hidden state from main model
    2. Embedding of last sampled token
    3. Applies norms, concatenation, projection
    4. Runs through a transformer layer
    5. Final norm -> shared lm_head -> logits
"""

import glob
import logging
from pathlib import Path
from typing import Dict, Optional

import mlx.core as mx
import mlx.nn as nn

logger = logging.getLogger(__name__)


class MTPAttention(nn.Module):
    """Attention layer inside the MTP head (matches Qwen3.5 attention)."""

    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        head_dim: int,
        rms_norm_eps: float = 1e-6,
        attention_bias: bool = False,
        partial_rotary_factor: float = 0.25,
        rope_theta: float = 100000.0,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.scale = head_dim ** -0.5

        # Note: Qwen3.5 q_proj output is 2x normal because it includes a gate
        self.q_proj = nn.Linear(hidden_size, num_attention_heads * head_dim * 2, bias=attention_bias)
        self.k_proj = nn.Linear(hidden_size, num_key_value_heads * head_dim, bias=attention_bias)
        self.v_proj = nn.Linear(hidden_size, num_key_value_heads * head_dim, bias=attention_bias)
        self.o_proj = nn.Linear(num_attention_heads * head_dim, hidden_size, bias=attention_bias)

        self.q_norm = nn.RMSNorm(head_dim, eps=rms_norm_eps)
        self.k_norm = nn.RMSNorm(head_dim, eps=rms_norm_eps)

        rope_dim = int(head_dim * partial_rotary_factor)
        self.rope = nn.RoPE(rope_dim, base=rope_theta, traditional=False)

    def __call__(self, x: mx.array, mask=None, cache=None) -> mx.array:
        B, L, _ = x.shape

        q_proj_output = self.q_proj(x)
        queries, gate = mx.split(
            q_proj_output.reshape(B, L, self.num_attention_heads, -1), 2, axis=-1
        )
        gate = gate.reshape(B, L, -1)

        keys = self.k_proj(x)
        values = self.v_proj(x)

        queries = self.q_norm(queries).transpose(0, 2, 1, 3)
        keys = self.k_norm(keys.reshape(B, L, self.num_key_value_heads, -1)).transpose(0, 2, 1, 3)
        values = values.reshape(B, L, self.num_key_value_heads, -1).transpose(0, 2, 1, 3)

        # No cache for MTP head - it only processes 1 token at a time
        queries = self.rope(queries)
        keys = self.rope(keys)

        output = mx.fast.scaled_dot_product_attention(
            queries, keys, values, scale=self.scale, mask=mask
        )
        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)

        return self.o_proj(output * mx.sigmoid(gate))


class MTPMLP(nn.Module):
    """MLP layer inside the MTP head (dense SwiGLU)."""

    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))


class MTPMoEMLP(nn.Module):
    """MoE MLP layer for the MTP head (used by 35B-A3B and similar models).

    Uses SwitchGLU from mlx-lm for efficient expert routing.
    Architecture: router -> top-k experts + sigmoid-gated shared expert.
    """

    def __init__(
        self,
        hidden_size: int,
        moe_intermediate_size: int,
        shared_expert_intermediate_size: int,
        num_experts: int,
        num_experts_per_tok: int,
        norm_topk_prob: bool = True,
    ):
        super().__init__()
        from mlx_lm.models.switch_layers import SwitchGLU

        self.num_experts = num_experts
        self.top_k = num_experts_per_tok
        self.norm_topk_prob = norm_topk_prob

        # Router
        self.gate = nn.Linear(hidden_size, num_experts, bias=False)

        # Expert pool
        self.switch_mlp = SwitchGLU(hidden_size, moe_intermediate_size, num_experts)

        # Shared expert (always executes)
        self.shared_expert = MTPMLP(hidden_size, shared_expert_intermediate_size)

        # Sigmoid gate for shared expert output
        self.shared_expert_gate = nn.Linear(hidden_size, 1, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        # Router
        gates = self.gate(x)
        gates = mx.softmax(gates, axis=-1, precise=True)

        # Top-k expert selection
        k = self.top_k
        inds = mx.argpartition(gates, kth=-k, axis=-1)[..., -k:]
        scores = mx.take_along_axis(gates, inds, axis=-1)
        if self.norm_topk_prob:
            scores = scores / scores.sum(axis=-1, keepdims=True)

        # Route through experts and combine
        y = self.switch_mlp(x, inds)
        y = (y * scores[..., None]).sum(axis=-2)

        # Add shared expert contribution
        shared_y = self.shared_expert(x)
        shared_y = mx.sigmoid(self.shared_expert_gate(x)) * shared_y

        return y + shared_y


class MTPDecoderLayer(nn.Module):
    """Single transformer decoder layer for MTP."""

    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        head_dim: int,
        rms_norm_eps: float = 1e-6,
        attention_bias: bool = False,
        partial_rotary_factor: float = 0.25,
        rope_theta: float = 100000.0,
        mlp: Optional[nn.Module] = None,
        intermediate_size: int = 0,
    ):
        super().__init__()
        self.self_attn = MTPAttention(
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            head_dim=head_dim,
            rms_norm_eps=rms_norm_eps,
            attention_bias=attention_bias,
            partial_rotary_factor=partial_rotary_factor,
            rope_theta=rope_theta,
        )
        self.mlp = mlp if mlp is not None else MTPMLP(hidden_size, intermediate_size)
        self.input_layernorm = nn.RMSNorm(hidden_size, eps=rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(hidden_size, eps=rms_norm_eps)

    def __call__(self, x: mx.array, mask=None) -> mx.array:
        r = self.self_attn(self.input_layernorm(x), mask=mask)
        h = x + r
        out = h + self.mlp(self.post_attention_layernorm(h))
        return out


class MTPHead(nn.Module):
    """
    Full MTP head for Qwen3.5.

    Architecture:
        1. pre_fc_norm_hidden: RMSNorm on last hidden state
        2. pre_fc_norm_embedding: RMSNorm on token embedding
        3. fc: Linear projection from concat(hidden, embed) -> hidden_size
        4. layers[0]: Full transformer decoder layer (attention + MLP)
        5. norm: Final RMSNorm
        6. -> shared lm_head (external, from main model)
    """

    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        head_dim: int,
        rms_norm_eps: float = 1e-6,
        attention_bias: bool = False,
        partial_rotary_factor: float = 0.25,
        rope_theta: float = 100000.0,
        intermediate_size: int = 0,
        mlp: Optional[nn.Module] = None,
    ):
        super().__init__()
        self.hidden_size = hidden_size

        # Input norms and projection
        self.pre_fc_norm_hidden = nn.RMSNorm(hidden_size, eps=rms_norm_eps)
        self.pre_fc_norm_embedding = nn.RMSNorm(hidden_size, eps=rms_norm_eps)
        self.fc = nn.Linear(hidden_size * 2, hidden_size, bias=False)

        # Transformer layer
        self.layers = [
            MTPDecoderLayer(
                hidden_size=hidden_size,
                num_attention_heads=num_attention_heads,
                num_key_value_heads=num_key_value_heads,
                head_dim=head_dim,
                rms_norm_eps=rms_norm_eps,
                attention_bias=attention_bias,
                partial_rotary_factor=partial_rotary_factor,
                rope_theta=rope_theta,
                mlp=mlp,
                intermediate_size=intermediate_size,
            )
        ]

        # Final norm
        self.norm = nn.RMSNorm(hidden_size, eps=rms_norm_eps)

    def __call__(
        self,
        hidden_state: mx.array,
        token_embedding: mx.array,
    ) -> mx.array:
        """
        Forward pass.

        Args:
            hidden_state: (B, 1, hidden_size) from main model's last layer
            token_embedding: (B, 1, hidden_size) embedding of last sampled token

        Returns:
            (B, 1, hidden_size) ready for lm_head projection
        """
        h = self.pre_fc_norm_hidden(hidden_state)
        e = self.pre_fc_norm_embedding(token_embedding)
        combined = mx.concatenate([e, h], axis=-1)  # embed first, hidden second (matches vLLM)
        x = self.fc(combined)

        for layer in self.layers:
            x = layer(x)

        return self.norm(x)


# ---------------------------------------------------------------------------
# Weight Loading
# ---------------------------------------------------------------------------


def load_mtp_weights(model_path: Path) -> Dict[str, mx.array]:
    """
    Load MTP weights from safetensors files.

    Checks for:
    1. A dedicated mtp_weights.safetensors file (from extract_mtp_weights.py)
    2. MTP weights in the model's own safetensors files (pre-sanitize)
    """
    # Check for extracted MTP weights first
    extracted = model_path / "mtp_weights.safetensors"
    if extracted.exists():
        logger.info(f"Loading extracted MTP weights from {extracted}")
        return dict(mx.load(str(extracted)))

    # Check model's own safetensors
    weight_files = sorted(glob.glob(str(model_path / "model*.safetensors")))
    mtp_weights = {}
    for wf in weight_files:
        weights = mx.load(wf)
        for k, v in weights.items():
            if "mtp" in k.lower():
                mtp_weights[k] = v
    return mtp_weights


def load_mtp_weights_from_file(mtp_file: Path) -> Dict[str, mx.array]:
    """Load MTP weights from a specific safetensors file."""
    return dict(mx.load(str(mtp_file)))


def _normalize_mtp_key(key: str) -> str:
    """
    Strip prefixes to get the local MTP key.

    Examples:
        mtp.fc.weight -> fc.weight
        mtp.layers.0.self_attn.q_proj.weight -> layers.0.self_attn.q_proj.weight
        mtp.pre_fc_norm_hidden.weight -> pre_fc_norm_hidden.weight
        mtp.norm.weight -> norm.weight
    """
    # Strip "mtp." prefix
    if key.startswith("mtp."):
        return key[4:]
    # Handle other prefixes
    for prefix in ["model.mtp.", "language_model.model.mtp.", "language_model.mtp."]:
        if key.startswith(prefix):
            return key[len(prefix):]
    return key


def _is_moe_weights(weight_map: Dict[str, mx.array]) -> bool:
    """Detect if MTP weights contain MoE MLP (experts + router)."""
    return any("mlp.gate.weight" in k for k in weight_map)


def _convert_moe_expert_weights(weight_map: Dict[str, mx.array]) -> Dict[str, mx.array]:
    """
    Convert individual expert weights to stacked SwitchLinear format.

    Raw checkpoint format:
        layers.0.mlp.experts.{i}.gate_proj.weight  (intermediate, hidden)
        layers.0.mlp.experts.{i}.up_proj.weight     (intermediate, hidden)
        layers.0.mlp.experts.{i}.down_proj.weight   (hidden, intermediate)

    SwitchGLU format:
        layers.0.mlp.switch_mlp.gate_proj.weight  (num_experts, intermediate, hidden)
        layers.0.mlp.switch_mlp.up_proj.weight    (num_experts, intermediate, hidden)
        layers.0.mlp.switch_mlp.down_proj.weight  (num_experts, hidden, intermediate)
    """
    converted = {}
    expert_keys = {}  # {(layer_prefix, proj_name): {expert_idx: weight}}

    for k, v in weight_map.items():
        # Match pattern: layers.0.mlp.experts.{N}.{proj}.weight
        import re
        m = re.match(r"(layers\.\d+\.mlp)\.experts\.(\d+)\.(\w+)\.weight", k)
        if m:
            prefix, idx, proj = m.group(1), int(m.group(2)), m.group(3)
            key = (prefix, proj)
            if key not in expert_keys:
                expert_keys[key] = {}
            expert_keys[key][idx] = v
        else:
            converted[k] = v

    # Stack individual expert weights into SwitchLinear format
    for (prefix, proj), experts in expert_keys.items():
        num_experts = max(experts.keys()) + 1
        stacked = mx.stack([experts[i] for i in range(num_experts)])
        new_key = f"{prefix}.switch_mlp.{proj}.weight"
        converted[new_key] = stacked
        logger.info(f"Stacked {num_experts} experts for {new_key}: {stacked.shape}")

    return converted


def build_mtp_head(
    mtp_weights: Dict[str, mx.array],
    config: dict,
    norm_shift: bool = True,
) -> Optional[MTPHead]:
    """
    Build and initialize an MTP head from extracted weights.

    Args:
        mtp_weights: Raw MTP weight tensors
        config: Model config dict (may contain text_config)
        norm_shift: Apply +1 to norm weights (Qwen3.5 shifted norms convention)

    Returns:
        Initialized MTPHead or None if weights are missing/incompatible
    """
    if not mtp_weights:
        logger.warning("No MTP weights provided")
        return None

    text_config = config.get("text_config", config)
    hidden_size = text_config["hidden_size"]
    rms_norm_eps = text_config.get("rms_norm_eps", 1e-6)
    attention_bias = text_config.get("attention_bias", False)

    # Rope parameters
    rope_params = text_config.get("rope_parameters", {})
    partial_rotary_factor = rope_params.get(
        "partial_rotary_factor",
        text_config.get("partial_rotary_factor", 0.25),
    )
    rope_theta = rope_params.get(
        "rope_theta",
        text_config.get("rope_theta", 100000.0),
    )

    # Normalize weight keys
    weight_map = {}
    for k, v in mtp_weights.items():
        local_key = _normalize_mtp_key(k)
        weight_map[local_key] = v

    logger.info(f"MTP weight keys ({len(weight_map)}): {sorted(weight_map.keys())[:20]}...")

    # Detect MoE vs dense MLP
    is_moe = _is_moe_weights(weight_map)
    if is_moe:
        logger.info("Detected MoE MTP head — converting expert weights to SwitchLinear format")
        weight_map = _convert_moe_expert_weights(weight_map)

    # Infer MTP head dimensions from actual weight shapes rather than config
    q_norm_w = weight_map.get("layers.0.self_attn.q_norm.weight")
    k_proj_w = weight_map.get("layers.0.self_attn.k_proj.weight")
    q_proj_w = weight_map.get("layers.0.self_attn.q_proj.weight")

    if q_norm_w is not None:
        head_dim = q_norm_w.shape[0]
    else:
        head_dim = text_config.get("head_dim", hidden_size // text_config.get("num_attention_heads", 16))

    if q_proj_w is not None:
        num_attention_heads = q_proj_w.shape[0] // (head_dim * 2)
    else:
        num_attention_heads = text_config.get("num_attention_heads", 16)

    if k_proj_w is not None:
        num_key_value_heads = k_proj_w.shape[0] // head_dim
    else:
        num_key_value_heads = text_config.get("num_key_value_heads", num_attention_heads)

    # Build MLP (dense or MoE)
    mlp = None
    intermediate_size = 0
    if is_moe:
        # Infer MoE dimensions from weights
        num_experts = text_config.get("num_experts", 0)
        num_experts_per_tok = text_config.get("num_experts_per_tok", 4)
        norm_topk_prob = text_config.get("norm_topk_prob", True)

        # Get expert intermediate size from stacked weight
        switch_gate_w = weight_map.get("layers.0.mlp.switch_mlp.gate_proj.weight")
        if switch_gate_w is not None:
            if num_experts == 0:
                num_experts = switch_gate_w.shape[0]
            moe_intermediate_size = switch_gate_w.shape[1]
        else:
            moe_intermediate_size = text_config.get("moe_intermediate_size", hidden_size)

        # Get shared expert intermediate size
        shared_gate_w = weight_map.get("layers.0.mlp.shared_expert.gate_proj.weight")
        if shared_gate_w is not None:
            shared_expert_intermediate_size = shared_gate_w.shape[0]
        else:
            shared_expert_intermediate_size = text_config.get(
                "shared_expert_intermediate_size", hidden_size * 4
            )

        logger.info(
            f"MoE MTP: {num_experts} experts, top-{num_experts_per_tok}, "
            f"expert_intermediate={moe_intermediate_size}, "
            f"shared_intermediate={shared_expert_intermediate_size}"
        )

        mlp = MTPMoEMLP(
            hidden_size=hidden_size,
            moe_intermediate_size=moe_intermediate_size,
            shared_expert_intermediate_size=shared_expert_intermediate_size,
            num_experts=num_experts,
            num_experts_per_tok=num_experts_per_tok,
            norm_topk_prob=norm_topk_prob,
        )
    else:
        # Dense MLP — infer intermediate size from weights
        gate_proj_w = weight_map.get("layers.0.mlp.gate_proj.weight")
        if gate_proj_w is not None:
            intermediate_size = gate_proj_w.shape[0]
        else:
            intermediate_size = text_config.get("intermediate_size", hidden_size * 4)

    logger.info(
        f"MTP dims: head_dim={head_dim}, heads={num_attention_heads}, "
        f"kv_heads={num_key_value_heads}, moe={is_moe}"
    )

    # Apply norm shift if needed (Qwen3.5 uses shifted RMSNorm: weight = weight + 1)
    if norm_shift:
        norm_suffixes = (
            "pre_fc_norm_hidden.weight",
            "pre_fc_norm_embedding.weight",
            "norm.weight",
            "input_layernorm.weight",
            "post_attention_layernorm.weight",
            "q_norm.weight",
            "k_norm.weight",
        )
        for k, v in weight_map.items():
            if any(k.endswith(sfx) for sfx in norm_suffixes) and v.ndim == 1:
                weight_map[k] = v + 1.0

    # Build the head
    head = MTPHead(
        hidden_size=hidden_size,
        num_attention_heads=num_attention_heads,
        num_key_value_heads=num_key_value_heads,
        head_dim=head_dim,
        rms_norm_eps=rms_norm_eps,
        attention_bias=attention_bias,
        partial_rotary_factor=partial_rotary_factor,
        rope_theta=rope_theta,
        intermediate_size=intermediate_size,
        mlp=mlp,
    )

    # Load weights
    loadable = list(weight_map.items())
    if not loadable:
        logger.warning("No MTP weights to load")
        return None

    head.load_weights(loadable, strict=False)
    mx.eval(head.parameters())

    logger.info(
        f"MTP head built: hidden={hidden_size}, heads={num_attention_heads}, "
        f"kv_heads={num_key_value_heads}, moe={is_moe}, {len(loadable)} tensors"
    )
    return head


def detect_mtp_support(config: dict) -> bool:
    """Check if a model config indicates MTP support."""
    text_config = config.get("text_config", config)
    # Check both field names used across Qwen3.5 variants
    if text_config.get("num_nextn_predict_layers", 0) > 0:
        return True
    if text_config.get("mtp_num_hidden_layers", 0) > 0:
        return True
    return False
