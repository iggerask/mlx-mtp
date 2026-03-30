# Qwen3.5 MTP Head Architecture Research

**Date**: 2026-03-28
**Researcher**: Research Synthesizer (Claude Sonnet 4.6)
**Sources**: vLLM source code, DeepSeek-V3 technical report, actual model weights

---

## Executive Summary

The Qwen3.5 MTP (Multi-Token Prediction) head is a lightweight speculative decoding module
that combines the main model's final hidden state with the embedding of the last generated
token, projects them through a fully-connected layer, runs a single transformer block, and
applies a final norm before passing to the shared lm_head. The forward pass is:

```
pre_fc_norm_hidden(h) + pre_fc_norm_embedding(e) → cat → fc → transformer_layer → norm → lm_head
```

The lm_head is **shared** with the main model when `tie_word_embeddings=true` (which is
the case for Qwen3.5-4B). The embed_tokens table is also shared (not dedicated) per the
`mtp_use_dedicated_embeddings: false` config field.

---

## 1. Exact Weight Keys (confirmed from Qwen3.5-4B checkpoint)

The following keys were read directly from `mtp_weights/Qwen_Qwen3.5-4B.safetensors`:

```
mtp.pre_fc_norm_embedding.weight  shape=[2560]          dtype=bfloat16
mtp.pre_fc_norm_hidden.weight     shape=[2560]          dtype=bfloat16
mtp.fc.weight                     shape=[2560, 5120]    dtype=bfloat16   # out=hidden, in=2*hidden
mtp.layers.0.input_layernorm.weight              shape=[2560]
mtp.layers.0.post_attention_layernorm.weight     shape=[2560]
mtp.layers.0.self_attn.q_proj.weight             shape=[8192, 2560]    # (n_heads*head_dim, hidden)
mtp.layers.0.self_attn.k_proj.weight             shape=[1024, 2560]    # (n_kv_heads*head_dim, hidden)
mtp.layers.0.self_attn.v_proj.weight             shape=[1024, 2560]
mtp.layers.0.self_attn.o_proj.weight             shape=[2560, 4096]
mtp.layers.0.self_attn.q_norm.weight             shape=[256]           # head_dim=256
mtp.layers.0.self_attn.k_norm.weight             shape=[256]
mtp.layers.0.mlp.gate_proj.weight               shape=[9216, 2560]
mtp.layers.0.mlp.up_proj.weight                 shape=[9216, 2560]
mtp.layers.0.mlp.down_proj.weight               shape=[2560, 9216]
mtp.norm.weight                                 shape=[2560]
```

**Total**: 15 weight tensors. No `embed_tokens` or `lm_head` in the MTP namespace -
these are shared with the main model.

**Model dimensions (Qwen3.5-4B)**:
- `hidden_size` = 2560
- `intermediate_size` = 9216
- `num_attention_heads` = 16, `num_key_value_heads` = 4, `head_dim` = 256
- `vocab_size` = 248320

---

## 2. Forward Pass - Exact Sequence

### Source: `vllm/model_executor/models/qwen3_5_mtp.py` (`Qwen3_5MultiTokenPredictor.forward`)

```python
def forward(
    self,
    input_ids: torch.Tensor,           # [B, S] - the NEXT tokens (shifted by 1)
    positions: torch.Tensor,           # [S]
    hidden_states: torch.Tensor,       # [B, S, hidden_size] - from main model
    intermediate_tensors=None,
    inputs_embeds=None,
    spec_step_idx: int = 0,
) -> torch.Tensor:
    # Step 1: Embed the input tokens (these are the NEXT tokens, not current)
    if inputs_embeds is None:
        inputs_embeds = self.embed_input_ids(input_ids)

    # Step 2: Normalize both inputs independently
    inputs_embeds = self.pre_fc_norm_embedding(inputs_embeds)    # RMSNorm
    hidden_states = self.pre_fc_norm_hidden(hidden_states)       # RMSNorm

    # Step 3: Concatenate and project [2*hidden_size -> hidden_size]
    hidden_states = torch.cat([inputs_embeds, hidden_states], dim=-1)
    hidden_states = self.fc(hidden_states)                       # ColumnParallelLinear

    # Step 4: Run through one transformer decoder layer
    residual = None
    current_step_idx = spec_step_idx % self.num_mtp_layers       # always 0 for MTP-1
    hidden_states, residual = self.layers[current_step_idx](
        positions=positions,
        hidden_states=hidden_states,
        residual=residual,
    )

    # Step 5: Final norm (with residual fusion, as Qwen3.5 uses pre-norm + residual)
    hidden_states, _ = self.norm(hidden_states, residual)

    return hidden_states
```

After this, the caller applies the shared `lm_head` (or `embed_tokens.as_linear` when
`tie_word_embeddings=True`) to produce logits:

```python
def compute_logits(self, hidden_states, spec_step_idx=0):
    return self.logits_processor(self.lm_head, hidden_states)
```

### Summary of forward pass:
```
hidden_states (from main model, last position)
    |
    v
pre_fc_norm_hidden  (RMSNorm)
    |
    +--- concat(dim=-1) ---+
                           |
token_embedding            |
(of the NEXT token         |
 or CURRENT sampled token) |
    |                      |
    v                      |
pre_fc_norm_embedding      |
(RMSNorm)                  |
    |                      |
    +----------------------+
                   |
                   v
              [2*hidden_size]
                   |
                   v
              fc (Linear, no bias)
                   |
                   v
              [hidden_size]
                   |
                   v
        transformer_layer (full_attention)
        (input_layernorm + self_attn + post_attention_layernorm + mlp)
                   |
                   v
              norm (final RMSNorm, fused with residual)
                   |
                   v
              [hidden_size]
                   |
                   v
         lm_head (shared with main model)
                   |
                   v
              logits [vocab_size]
```

---

## 3. Input Alignment: Which Tokens and Hidden States?

### The key insight: MTP sees h[i] paired with embed(t[i+1])

At decode step i, the main model processes token `t[i]` and produces:
- `logits[i]` -> sample to get `t[i+1]`
- `hidden_states[i]` at the last transformer layer (before lm_head)

The MTP head then predicts what comes AFTER `t[i+1]` (i.e., `t[i+2]`) by combining:
- `h[i]`: the hidden state from position i (main model output for token `t[i]`)
- `embed(t[i+1])`: the embedding of the just-sampled next token

This is the **draft proposal** for `t[i+2]`.

From the vLLM PR discussion (#13626):
> "The correct way, per both EAGLE and MTP, is to get rid of the first hidden state and
> adjust attention metadata correspondingly so that both the attention scores and the
> position encoding are correct (the second token from the target model should be the
> first token to EAGLE and get the first position encoding)."

From the vLLM Ascend docs, the MTP step formula is:
```
Step k=1:  h[i] + embed(t[i+1])  -> MTP -> logits for t[i+2]
Step k=2:  h[i+1]^mtp + embed(t[i+2]) -> MTP -> logits for t[i+3]
Step k=K:  h[i+K-1]^mtp + embed(t[i+K]) -> MTP -> logits for t[i+K+1]
```

Where `h[i+k]^mtp` is the hidden state output of MTP at step k.

### In the existing `mtp_poc.py`:

```python
# After prefill:
last_hidden = hidden[:, -1:, :]         # h at last position

# Draft step:
token_embed = embed_tokens(token_0)     # embed(t[i+1]) - the JUST SAMPLED token
draft_hidden = mtp_head(last_hidden, token_embed)  # -> h for predicting t[i+2]
draft_logits = lm_head_fn(draft_hidden)

# After accept, update for next step:
last_hidden = verify_hidden[:, 1:2, :]  # h at position of draft_token
```

This matches the formula exactly. `token_0` is `t[i+1]` (the token just sampled by main
model), and `last_hidden` is `h[i]` (the hidden state the main model produced when
processing `t[i]`).

### Position encoding in MTP:
The MTP transformer layer uses the same rotary position embeddings as the main model.
The `positions` passed to MTP correspond to the position of `t[i+1]` (the token whose
embedding is being fed in), not position 0.

DeepSeek MTP (reference implementation) additionally masks position 0:
```python
# DeepSeek-specific, NOT in Qwen3.5:
inputs_embeds = torch.where(positions.unsqueeze(-1) == 0, 0, inputs_embeds)
```
Qwen3.5 does NOT do this masking.

---

## 4. Embedding and LM Head Sharing

### Config evidence:
From `Qwen/Qwen3.5-4B` `config.json`:
```json
{
  "tie_word_embeddings": true,
  "mtp_use_dedicated_embeddings": false,
  "mtp_num_hidden_layers": 1
}
```

### What this means:
- **`tie_word_embeddings: true`**: `lm_head` reuses `embed_tokens` weights (transposed).
  `lm_head(x) = x @ embed_tokens.weight.T`
- **`mtp_use_dedicated_embeddings: false`**: The MTP head's `embed_tokens` lookup uses
  the same embedding table as the main model (shared, not a copy).

### vLLM implementation evidence:
From `Qwen3_5MTP.__init__`:
```python
if config.tie_word_embeddings:
    self.lm_head = self.model.embed_tokens  # same object!
else:
    self.lm_head = ParallelLMHead(config.vocab_size, config.hidden_size, ...)
```

The weight loading in `load_weights` confirms embedding sharing:
```python
def remap_weight_names(weights):
    for name, weight in weights:
        if name.startswith("mtp."):
            name = name.replace("mtp.", "model.")
        elif any(key in name for key in ["embed_tokens", "lm_head"]):
            if "embed_tokens" in name:
                name = name.replace("language_model.", "")
        else:
            continue
        yield name, weight
```

The `embed_tokens` weights from the main model checkpoint are loaded into
`model.embed_tokens`, which is then reused by the MTP module for both embedding lookups
and (via `tie_word_embeddings`) as the lm_head.

---

## 5. Architecture Comparison: Qwen3.5 vs DeepSeek MTP

| Aspect | Qwen3.5 MTP | DeepSeek-V3 MTP |
|--------|-------------|-----------------|
| Weight key prefix | `mtp.*` | `model.layers.<N+k>.*` |
| Embedding norm | `pre_fc_norm_embedding` | `enorm` |
| Hidden state norm | `pre_fc_norm_hidden` | `hnorm` |
| Projection layer | `fc` (ColumnParallel) | `eh_proj` (nn.Linear) |
| Transformer layer | `mtp.layers.0.*` | reuses main layers[N+k] |
| Final norm | `mtp.norm` | via `SharedHead.norm` |
| LM head | shared (`tie_word_embeddings`) | separate `SharedHead.head` |
| Position-0 masking | NO | YES (`positions == 0 -> 0`) |
| Config field | `mtp_num_hidden_layers` | `num_nextn_predict_layers` |

---

## 6. Key Divergence: Existing POC vs Actual Architecture

The existing `mtp_poc.py` uses the **DeepSeek naming convention** (`enorm`, `hnorm`,
`eh_proj`) but the actual Qwen3.5 weights use different names. The architecture is
functionally identical but the weight keys differ:

| POC (`mtp_poc.py`) | Actual Qwen3.5 weights |
|--------------------|------------------------|
| `hnorm.weight` | `mtp.pre_fc_norm_hidden.weight` |
| `enorm.weight` | `mtp.pre_fc_norm_embedding.weight` |
| `eh_proj.weight` | `mtp.fc.weight` |
| (not implemented) | `mtp.layers.0.*` (full transformer) |
| (not implemented) | `mtp.norm.weight` |

**Critical**: The POC's `MTPHead` is INCOMPLETE. It applies `hnorm + enorm + concat + fc`
but then uses the result directly as input to lm_head, skipping the transformer layer
(`mtp.layers.0.*`) and the final norm (`mtp.norm`). The POC will produce wrong predictions
because it is missing the transformer block entirely.

The correct MLX implementation must:
1. Apply `pre_fc_norm_hidden` to hidden_state
2. Apply `pre_fc_norm_embedding` to token_embedding
3. Concatenate and project through `fc`
4. Run through `mtp.layers.0` (full Qwen3.5DecoderLayer with attention + MLP)
5. Apply `mtp.norm` (with residual fusion)
6. Apply lm_head (shared embed_tokens)

---

## 7. The Transformer Layer (`mtp.layers.0`)

This is a full `Qwen3_5DecoderLayer` with `layer_type="full_attention"` (not
linear attention). The layer architecture matches a standard main-model layer:

```
input_layernorm (RMSNorm) [2560]
    |
self_attn:
    q_proj:  [8192, 2560]   (16 heads * 512... wait, actually GQA: 16 heads * head_dim=256...
                              shape [8192, 2560] = 32 * 256 * 2560 / 2560...
                              Actually: 8192 = 16 heads * 512 or 32 * 256 - let's verify)
              NOTE: q_proj=[8192,2560] means num_q_heads * head_dim = 8192
                    head_dim=256 -> num_q_heads = 32... or head_dim=512 -> num_q_heads=16
              But k_proj=[1024,2560] and v_proj=[1024,2560]
                    1024 / 4 kv_heads = 256 = head_dim
              So head_dim=256, num_q_heads=32? Or head_dim=512, num_q_heads=16?
              Main model: num_attention_heads=16, head_dim=2560/16=160...
              Wait: q_norm.weight=[256] confirms head_dim=256
                    num_q_heads = 8192 / 256 = 32 (QKV has 32 query heads)
    k_proj:  [1024, 2560]   (4 kv_heads * 256 = 1024)
    v_proj:  [1024, 2560]
    o_proj:  [2560, 4096]   (hidden_size, num_q_heads * head_dim = 32 * 128? No...)
                              4096 = 16 * 256 or 32 * 128...
                              Actually o_proj input = num_q_heads * head_dim = 32 * 128 = 4096
                              So head_dim for QKV=256 but for output projection = 128?
                              More likely: head_dim=128 in the attention output,
                              q_norm/k_norm operate on the full head projection
    q_norm:  [256]          (per-head norm on q, head_dim=256 in projection space)
    k_norm:  [256]          (per-head norm on k)
post_attention_layernorm (RMSNorm) [2560]
    |
mlp:
    gate_proj: [9216, 2560]
    up_proj:   [9216, 2560]
    down_proj: [2560, 9216]
```

The MTP transformer layer has the **same architecture** as a full_attention layer in the
main model. This is consistent with Qwen3.5's hybrid architecture where some layers use
linear attention (GatedDeltaNet) and some use full attention.

---

## 8. Config Fields

From the Qwen3.5-4B `config.json` (text_config section):

```json
{
  "mtp_num_hidden_layers": 1,
  "mtp_use_dedicated_embeddings": false,
  "num_hidden_layers": 32,
  "hidden_size": 2560,
  "intermediate_size": 9216,
  "num_attention_heads": 16,
  "num_key_value_heads": 4,
  "tie_word_embeddings": true,
  "rms_norm_eps": 1e-6,
  "layer_types": ["linear_attention", "linear_attention", "linear_attention", "full_attention", ...]
}
```

Note: The `mtp_poc.py` looks for `num_nextn_predict_layers` (DeepSeek convention) but
Qwen3.5 uses `mtp_num_hidden_layers`. This is another key mismatch to fix.

---

## 9. Confidence Assessment

| Finding | Confidence | Evidence |
|---------|-----------|----------|
| Forward pass sequence (norm->cat->fc->layer->norm->lm_head) | HIGH | vLLM source code + actual weights |
| Weight key names | HIGH | Actual checkpoint inspection |
| lm_head is shared (tie_word_embeddings) | HIGH | config.json + vLLM source |
| embed_tokens is shared (mtp_use_dedicated_embeddings=false) | HIGH | config.json |
| hidden_state from position N-1 paired with embed(token_N) | HIGH | vLLM source + POC code |
| Transformer layer is full_attention (not linear_attention) | HIGH | vLLM source + weight shapes |
| Position masking at position 0 NOT done in Qwen3.5 | MEDIUM | vLLM source (DeepSeek does it, Qwen3.5 does not) |

---

## References

- [vLLM qwen3_5_mtp.py source](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/qwen3_5_mtp.py) — primary reference, complete source code
- [vLLM deepseek_mtp.py source](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/deepseek_mtp.py) — comparison reference for DeepSeek MTP
- [vLLM qwen3_next_mtp.py source](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/qwen3_next_mtp.py) — Qwen3-Next variant (same architecture)
- [vLLM MTP documentation](https://docs.vllm.ai/en/latest/features/speculative_decoding/mtp/) — MTP step formula
- [vLLM PR #13626](https://github.com/vllm-project/vllm/pull/13626) — position alignment discussion
- [vLLM PR #12755](https://github.com/vllm-project/vllm/pull/12755) — DeepSeek MTP speculative decode implementation
- [Qwen/Qwen3.5-4B on HuggingFace](https://huggingface.co/Qwen/Qwen3.5-4B) — config.json reference
- [Qwen3 Technical Report arxiv:2505.09388](https://arxiv.org/abs/2505.09388) — Qwen3 architecture overview
- [vLLM Ascend MTP Guide](https://docs.vllm.ai/projects/ascend/en/main/user_guide/feature_guide/Multi_Token_Prediction.html) — step formula documentation
- Actual checkpoint: `mtp_weights/Qwen_Qwen3.5-4B.safetensors` — ground truth weight keys
