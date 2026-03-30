"""
Inference optimizations for MTP speculative decoding.

1. Quantized MTP head — reduce MTP overhead by quantizing head weights to 4-bit
2. Prompt lookup decoding — reuse n-grams from the prompt as free draft tokens
3. Self-speculative decoding — skip layers in the backbone for fast drafting
"""

import logging
from typing import List, Optional, Set, Tuple

import mlx.core as mx
import mlx.nn as nn

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1. Quantized MTP Head
# ---------------------------------------------------------------------------


def quantize_mtp_head(head: nn.Module, group_size: int = 64, bits: int = 4):
    """
    Quantize the MTP head's linear layers to reduce memory and compute.

    Norms are kept in full precision. Only Linear layers are quantized.
    Modifies the head in-place.
    """
    nn.quantize(head, group_size=group_size, bits=bits)
    mx.eval(head.parameters())

    # Count params
    total_bytes = 0
    for k, v in head.parameters().items():
        if isinstance(v, dict):
            for kk, vv in v.items():
                if hasattr(vv, 'nbytes'):
                    total_bytes += vv.nbytes
        elif hasattr(v, 'nbytes'):
            total_bytes += v.nbytes

    logger.info(f"Quantized MTP head to {bits}-bit (group_size={group_size})")
    return head


# ---------------------------------------------------------------------------
# 2. Prompt Lookup Decoding
# ---------------------------------------------------------------------------


class PromptLookupDrafter:
    """
    Drafts tokens by finding n-gram matches between recent output and the prompt.

    When the model generates tokens that overlap with the prompt (common in
    summarization, Q&A, code editing), we can predict the next tokens for free.
    """

    def __init__(self, prompt_tokens: List[int], max_ngram: int = 5, max_draft: int = 5):
        self.prompt_tokens = prompt_tokens
        self.max_ngram = max_ngram
        self.max_draft = max_draft
        # Build n-gram index: maps tuple(ngram) -> list of positions
        self._index = {}
        for n in range(2, max_ngram + 1):
            for i in range(len(prompt_tokens) - n):
                key = tuple(prompt_tokens[i:i + n])
                if key not in self._index:
                    self._index[key] = []
                self._index[key].append(i + n)

    def draft(self, recent_tokens: List[int]) -> List[int]:
        """
        Given recent output tokens, find matching n-grams in the prompt
        and return predicted continuation tokens.
        """
        if not recent_tokens:
            return []

        # Try longest n-gram first, prefer latest match in prompt
        for n in range(min(self.max_ngram, len(recent_tokens)), 1, -1):
            key = tuple(recent_tokens[-n:])
            if key in self._index:
                pos = self._index[key][-1]  # last (latest) match
                end = min(pos + self.max_draft, len(self.prompt_tokens))
                if pos < end:
                    return self.prompt_tokens[pos:end]

        return []


# ---------------------------------------------------------------------------
# 3. Self-Speculative Decoding (Layer Skipping)
# ---------------------------------------------------------------------------


class SharedExpertDrafter:
    """
    Draft tokens by running the full model but bypassing MoE expert routing.

    For each MoE layer, only the shared expert is used (skip the expensive
    routing through 8 of 256 experts). This is much faster per-token while
    maintaining reasonable prediction quality since all 40 layers are used.

    This is a form of self-speculative decoding optimized for MoE architectures.
    """

    def __init__(self, model):
        self.model = model
        self._moe_layers = []
        self._original_calls = {}
        self._find_moe_layers()

    def _find_moe_layers(self):
        """Find all MoE MLP modules in the model.

        Supports multiple architectures:
        - Qwen3.5/Qwen3-Next: model.model.layers[i].mlp with shared_expert + shared_expert_gate
        - NemotronH: model.backbone.layers[i].mixer with shared_experts (no gate)
        """
        # Navigate to inner model
        inner = None
        for attr_path in [
            ('language_model', 'model'),
            ('model',),
            ('backbone',),  # NemotronH
        ]:
            obj = self.model
            for attr in attr_path:
                obj = getattr(obj, attr, None)
                if obj is None:
                    break
            if obj is not None and hasattr(obj, 'layers'):
                inner = obj
                break

        if inner is None:
            raise ValueError("Could not find model layers")

        for i, layer in enumerate(inner.layers):
            # Qwen3.5/Qwen3-Next: MoE at layer.mlp with shared_expert
            mlp = getattr(layer, 'mlp', None)
            if mlp is not None and hasattr(mlp, 'shared_expert'):
                self._moe_layers.append((i, mlp, 'qwen'))
                continue
            # NemotronH: MoE at layer.mixer with shared_experts (block_type == "E")
            mixer = getattr(layer, 'mixer', None)
            if mixer is not None and hasattr(mixer, 'shared_experts'):
                self._moe_layers.append((i, mixer, 'nemotron'))

        logger.info(f"SharedExpertDrafter: found {len(self._moe_layers)} MoE layers")

    def _shared_expert_only_call(self, moe_module, arch_type):
        """Create a replacement __call__ that only uses the shared expert."""
        if arch_type == 'qwen':
            def _call(x):
                shared_y = moe_module.shared_expert(x)
                shared_y = mx.sigmoid(moe_module.shared_expert_gate(x)) * shared_y
                return shared_y
        else:  # nemotron
            def _call(x):
                return moe_module.shared_experts(x)
        return _call

    def enable(self):
        """Patch MoE layers to use shared expert only."""
        import types
        for i, moe_mod, arch_type in self._moe_layers:
            self._original_calls[i] = moe_mod.__class__.__call__
            shared_fn = self._shared_expert_only_call(moe_mod, arch_type)
            moe_mod.__call__ = types.MethodType(
                lambda self_mod, x, _fn=shared_fn: _fn(x),
                moe_mod,
            )

    def disable(self):
        """Restore original MoE layer behavior."""
        for i, moe_mod, arch_type in self._moe_layers:
            if hasattr(moe_mod, '__call__'):
                del moe_mod.__call__  # Remove instance override, fall back to class method

    def draft(self, model, token_input, cache):
        """
        Run one token through the model in shared-expert-only mode.

        Returns logits from the draft pass.
        """
        self.enable()
        try:
            logits = model(token_input, cache=cache)
            mx.eval(logits)
            return logits
        finally:
            self.disable()
