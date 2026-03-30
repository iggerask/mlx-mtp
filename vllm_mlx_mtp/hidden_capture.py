"""
Hidden state capture for MTP speculative decoding.

Intercepts the hidden state output from the transformer backbone
(before the lm_head projection) so it can be fed to the MTP head.

Uses a wrapper module approach that replaces the backbone with a thin
proxy that stores the output.
"""

from typing import Optional

import mlx.core as mx
import mlx.nn as nn


class _BackboneWrapper(nn.Module):
    """
    Thin wrapper around the backbone that captures hidden states.

    Delegates all attribute access to the wrapped backbone, but
    intercepts __call__ to store the output.
    """

    def __init__(self, backbone: nn.Module, capture: "HiddenStateCapture"):
        # Don't call super().__init__() - we want to be fully transparent
        object.__setattr__(self, "_backbone", backbone)
        object.__setattr__(self, "_capture", capture)

    def __call__(self, *args, **kwargs):
        result = self._backbone(*args, **kwargs)
        self._capture.last_hidden_state = result
        return result

    def __getattr__(self, name):
        return getattr(self._backbone, name)

    def __setattr__(self, name, value):
        if name in ("_backbone", "_capture"):
            object.__setattr__(self, name, value)
        else:
            setattr(self._backbone, name, value)


class HiddenStateCapture:
    """
    Captures hidden states from a model's transformer backbone.

    Replaces the backbone module with a transparent wrapper that
    stores the output on each forward pass.

    Supports:
    - Multimodal wrappers: model.language_model.model (backbone)
    - Text-only models: model.model (backbone)
    """

    def __init__(self, model: nn.Module):
        self.model = model
        self.last_hidden_state: Optional[mx.array] = None
        self._original_backbone = None
        self._parent = None
        self._attr_name = None
        self._setup()

    def _setup(self):
        """Replace backbone with capturing wrapper."""
        if hasattr(self.model, "language_model"):
            # Multimodal: model.language_model.model is backbone
            self._parent = self.model.language_model
            self._attr_name = "model"
        elif hasattr(self.model, "model") and hasattr(self.model.model, "layers"):
            # Text-only: model.model is backbone
            self._parent = self.model
            self._attr_name = "model"
        else:
            raise ValueError(
                f"Cannot find backbone in model of type {type(self.model).__name__}"
            )

        self._original_backbone = getattr(self._parent, self._attr_name)
        wrapper = _BackboneWrapper(self._original_backbone, self)
        setattr(self._parent, self._attr_name, wrapper)

    def get_hidden_state(self) -> Optional[mx.array]:
        """Return the last captured hidden state."""
        return self.last_hidden_state

    def restore(self):
        """Restore the original backbone module."""
        if self._original_backbone is not None and self._parent is not None:
            setattr(self._parent, self._attr_name, self._original_backbone)
            self._original_backbone = None

    def __del__(self):
        self.restore()
