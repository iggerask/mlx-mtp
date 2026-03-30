"""
GLM-4.7-Flash MTP Head (DeepSeek-V3 style Multi-Token Prediction).

Architecture (layer 47 in BF16 source, stripped during mlx-lm conversion):
    1. enorm: RMSNorm on token embedding
    2. hnorm: RMSNorm on hidden state
    3. eh_proj: Linear(2*hidden, hidden) on concat [enorm(e), hnorm(h)]
    4. Full decoder layer (MLA attention + MoE MLP, identical to layers 1-46)
    5. norm: Final RMSNorm (shared_head.norm)
    6. -> main model's lm_head for logit projection

Weight mapping from BF16 source:
    model.layers.47.enorm.weight          -> enorm.weight
    model.layers.47.hnorm.weight          -> hnorm.weight
    model.layers.47.eh_proj.weight        -> eh_proj.weight
    model.layers.47.self_attn.*           -> layer.self_attn.*
    model.layers.47.mlp.*                 -> layer.mlp.*
    model.layers.47.input_layernorm.*     -> layer.input_layernorm.*
    model.layers.47.post_attention_layernorm.* -> layer.post_attention_layernorm.*
    model.layers.47.shared_head.norm.*    -> norm.*
    model.layers.47.shared_head.head.*    -> (uses main model's lm_head)
    model.layers.47.embed_tokens.*        -> (uses main model's embed_tokens)
"""

import logging
import re
from pathlib import Path
from typing import Dict, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn

logger = logging.getLogger(__name__)


class GLMMTPHead(nn.Module):
    """
    MTP head for GLM-4.7-Flash using DeepSeek-V3 architecture.

    Reuses Glm4MoeLiteDecoderLayer from mlx-lm for the transformer block,
    ensuring perfect compatibility with MLA attention and MoE routing.
    """

    def __init__(self, config):
        super().__init__()
        from mlx_lm.models.glm4_moe_lite import (
            Glm4MoeLiteDecoderLayer,
            ModelArgs,
        )

        # Parse config dict into ModelArgs
        if isinstance(config, dict):
            args = ModelArgs(
                **{
                    k: v
                    for k, v in config.items()
                    if k in ModelArgs.__dataclass_fields__
                }
            )
        else:
            args = config

        self.hidden_size = args.hidden_size

        # MTP-specific: embedding/hidden normalization + projection
        self.enorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.hnorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.eh_proj = nn.Linear(args.hidden_size * 2, args.hidden_size, bias=False)

        # Full decoder layer — same architecture as main model layers 1-46
        # layer_idx >= first_k_dense_replace ensures MoE routing (not dense MLP)
        self.layer = Glm4MoeLiteDecoderLayer(args, layer_idx=args.num_hidden_layers)

        # Final norm (shared_head.norm in BF16 source)
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

    def __call__(
        self,
        hidden_state: mx.array,
        token_embedding: mx.array,
    ) -> mx.array:
        """
        Forward pass.

        Args:
            hidden_state: (B, 1, hidden_size) from main model's backbone
            token_embedding: (B, 1, hidden_size) embedding of draft token

        Returns:
            (B, 1, hidden_size) ready for lm_head projection
        """
        e = self.enorm(token_embedding)
        h = self.hnorm(hidden_state)
        x = self.eh_proj(mx.concatenate([e, h], axis=-1))

        # Run through full decoder layer (no cache, no mask for single token)
        x = self.layer(x, mask=None, cache=None)

        return self.norm(x)


def sanitize_glm_mtp_weights(
    raw_weights: Dict[str, mx.array],
    config: dict,
    num_hidden_layers: int = 47,
) -> Dict[str, mx.array]:
    """
    Transform raw BF16 weights from layer N (the MTP layer) into the format
    expected by GLMMTPHead.

    Applies the same transformations as Glm4MoeLiteModel.sanitize():
    1. kv_b_proj -> embed_q + unembed_out (MLA decomposition)
    2. Individual experts -> SwitchGLU stacked format
    3. Key remapping: model.layers.{N}.X -> X (matching GLMMTPHead structure)
    """
    prefix = f"model.layers.{num_hidden_layers}."
    n_routed_experts = config.get("n_routed_experts", 64)
    num_attention_heads = config.get("num_attention_heads", 20)
    qk_nope_head_dim = config.get("qk_nope_head_dim", 192)
    v_head_dim = config.get("v_head_dim", 256)
    kv_lora_rank = config.get("kv_lora_rank", 512)

    # Step 1: Filter to only MTP layer keys and strip prefix
    mtp_weights = {}
    for k, v in raw_weights.items():
        if k.startswith(prefix):
            local_key = k[len(prefix) :]
            # Skip embed_tokens (shared with main model)
            if local_key.startswith("embed_tokens"):
                continue
            # Skip shared_head.head (use main model's lm_head)
            if local_key.startswith("shared_head.head"):
                continue
            mtp_weights[local_key] = v

    logger.info(f"Found {len(mtp_weights)} raw MTP weight keys")

    # Step 2: Remap key names to match GLMMTPHead structure
    remapped = {}
    for k, v in mtp_weights.items():
        new_key = k
        # shared_head.norm -> norm
        if k.startswith("shared_head.norm."):
            new_key = k.replace("shared_head.norm.", "norm.")
        # self_attn, mlp, layernorms -> layer.X
        elif k.startswith(("self_attn.", "mlp.", "input_layernorm.", "post_attention_layernorm.")):
            new_key = f"layer.{k}"
        # enorm, hnorm, eh_proj stay as-is
        remapped[new_key] = v

    # Step 3: Decompose kv_b_proj into embed_q + unembed_out
    kv_b_key = "layer.self_attn.kv_b_proj.weight"
    if kv_b_key in remapped:
        v = remapped.pop(kv_b_key)
        head_dim = qk_nope_head_dim + v_head_dim
        v = v.reshape(num_attention_heads, head_dim, -1)
        wk = mx.contiguous(v[:, :qk_nope_head_dim, :].swapaxes(-1, -2))
        wv = mx.contiguous(v[:, qk_nope_head_dim:, :])
        remapped["layer.self_attn.embed_q.weight"] = wk
        remapped["layer.self_attn.unembed_out.weight"] = wv
        logger.info(
            f"Decomposed kv_b_proj: embed_q {wk.shape}, unembed_out {wv.shape}"
        )

    # Step 4: Stack expert weights into SwitchGLU format
    for proj in ["gate_proj", "up_proj", "down_proj"]:
        expert_pattern = f"layer.mlp.experts.0.{proj}.weight"
        if expert_pattern in remapped:
            experts = {}
            to_remove = []
            for ek, ev in remapped.items():
                m = re.match(
                    rf"layer\.mlp\.experts\.(\d+)\.{proj}\.weight", ek
                )
                if m:
                    experts[int(m.group(1))] = ev
                    to_remove.append(ek)
            for ek in to_remove:
                del remapped[ek]

            n_exp = max(experts.keys()) + 1
            stacked = mx.stack([experts[i] for i in range(n_exp)])
            new_key = f"layer.mlp.switch_mlp.{proj}.weight"
            remapped[new_key] = stacked
            logger.info(f"Stacked {n_exp} experts for {proj}: {stacked.shape}")

    logger.info(f"Sanitized {len(remapped)} MTP weight keys")
    return remapped


def build_glm_mtp_head(
    mtp_weights: Dict[str, mx.array],
    config: dict,
) -> Optional[GLMMTPHead]:
    """
    Build and initialize a GLM MTP head from pre-sanitized weights.

    Args:
        mtp_weights: Pre-sanitized weight dict (output of sanitize_glm_mtp_weights)
        config: Model config dict

    Returns:
        Initialized GLMMTPHead or None
    """
    if not mtp_weights:
        logger.warning("No GLM MTP weights provided")
        return None

    head = GLMMTPHead(config)

    # Load weights
    loadable = list(mtp_weights.items())
    head.load_weights(loadable, strict=False)
    mx.eval(head.parameters())

    # Count parameters
    import mlx.utils
    n_params = sum(v.size for _, v in mlx.utils.tree_flatten(head.parameters()))
    size_mb = sum(v.nbytes for _, v in mlx.utils.tree_flatten(head.parameters())) / 1e6
    logger.info(
        f"GLM MTP head built: {n_params:,} params, {size_mb:.1f} MB, "
        f"hidden={config.get('hidden_size', '?')}"
    )
    return head


def extract_glm_mtp_weights(
    bf16_repo: str = "zai-org/GLM-4.7-Flash",
    output_path: Optional[Path] = None,
    num_hidden_layers: int = 47,
) -> Tuple[Dict[str, mx.array], Path]:
    """
    Download the BF16 shard containing MTP weights and extract them.

    The MTP weights for GLM-4.7-Flash are in model-00048-of-00048.safetensors
    (layer 47, the MTP prediction layer stripped during mlx-lm conversion).

    Returns:
        (sanitized_weights, output_file_path)
    """
    import json
    import os
    from huggingface_hub import hf_hub_download

    os.environ["HF_HUB_DISABLE_XET"] = "1"

    if output_path is None:
        output_path = Path("mtp_weights/GLM-4.7-Flash.safetensors")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Check if already extracted
    if output_path.exists():
        logger.info(f"Loading pre-extracted GLM MTP weights from {output_path}")
        weights = dict(mx.load(str(output_path)))
        return weights, output_path

    # Download weight index to find which shard has MTP weights
    logger.info(f"Downloading weight index from {bf16_repo}...")
    idx_path = hf_hub_download(bf16_repo, "model.safetensors.index.json")
    with open(idx_path) as f:
        idx = json.load(f)

    # Find shards containing MTP weights
    mtp_shards = set()
    prefix = f"model.layers.{num_hidden_layers}."
    for k, shard in idx["weight_map"].items():
        if k.startswith(prefix):
            # Skip embed_tokens (huge, shared with main model)
            if "embed_tokens" in k:
                continue
            # Skip shared_head.head (lm_head, shared with main model)
            if "shared_head.head" in k:
                continue
            mtp_shards.add(shard)

    logger.info(f"MTP weights in shards: {sorted(mtp_shards)}")

    # Download config
    config_path = hf_hub_download(bf16_repo, "config.json")
    with open(config_path) as f:
        config = json.load(f)

    # Download each shard and extract MTP weights
    all_mtp_raw = {}
    for shard in sorted(mtp_shards):
        logger.info(f"Downloading {shard}...")
        shard_path = hf_hub_download(bf16_repo, shard)
        weights = mx.load(shard_path)
        for k, v in weights.items():
            if k.startswith(prefix) and "embed_tokens" not in k and "shared_head.head" not in k:
                all_mtp_raw[k] = v
        del weights

    logger.info(f"Extracted {len(all_mtp_raw)} raw MTP tensors")

    # Sanitize (transform to match GLMMTPHead structure)
    sanitized = sanitize_glm_mtp_weights(all_mtp_raw, config, num_hidden_layers)

    # Save
    mx.save_safetensors(str(output_path), sanitized)
    file_size_mb = output_path.stat().st_size / 1e6
    logger.info(f"Saved GLM MTP weights: {output_path} ({file_size_mb:.1f} MB)")

    return sanitized, output_path
