# EAGLE and Advanced Speculative Decoding Methods
## Comprehensive Research Report

**Date**: 2026-03-29
**Researcher**: Research Synthesizer (Claude Sonnet 4.6)
**Scope**: EAGLE-1/2/3, Medusa, Lookahead, Cascade Speculation, MoE considerations, MLX implementation

---

## Executive Summary

EAGLE (Extrapolation Algorithm for Greater Language-model Efficiency) is the current state-of-the-art speculative decoding family. EAGLE-3 (NeurIPS 2025) achieves **3x–6.5x** speedup over vanilla autoregressive decoding, roughly double the speedup of Medusa and ~1.4x that of EAGLE-2, while maintaining exact output distribution equivalence.

The key insight differentiating EAGLE from all prior methods is operating the draft model at the **feature level** (second-to-top-layer hidden states) rather than the token level. EAGLE-3 extends this by fusing features from multiple layers and eliminating the feature-regression constraint that prevented training-data scaling.

For the MLX / Apple Silicon context: tree attention speculative decoding is implementable in MLX (Apple's ReDrafter demonstrates 2.3x speedup on M2 Ultra), but batch verification is not currently supported in mlx-lm. EAGLE draft models require access to target-model hidden states and a small training run (1–2 days on RTX 3090 for 7B models).

**Key findings for the mlx-mtp project:**
- EAGLE and MTP (DeepSeek-style) are complementary: DeepSeek's MTP heads can function as EAGLE-style draft models and have been integrated into vLLM (PR #12915 / #12755)
- Qwen3.5 MTP heads follow the same feature-level drafting philosophy as EAGLE-1
- A small dense EAGLE draft model can serve a large MoE target model, though verification overhead is the binding constraint for MoE (MoE-Spec addresses this)
- For MLX on Apple Silicon, tree attention is the missing piece; simple sequential speculative decoding is already in mlx-lm

---

## Research Questions Addressed

1. How does EAGLE (1/2/3) work architecturally?
2. How does EAGLE compare to MTP (trained prediction heads)?
3. Has EAGLE been applied to MoE models? What are the challenges?
4. Medusa heads: architecture, training, comparison to EAGLE
5. Lookahead/Jacobi decoding: training-free alternatives
6. Cascade speculation and self-speculative decoding (LayerSkip)
7. Practical MLX / Apple Silicon implementation considerations

---

## 1. EAGLE-1: Feature-Level Auto-Regression

### Paper
- **Title**: "EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty"
- **Venue**: ICML 2024
- **arXiv**: [2401.15077](https://arxiv.org/abs/2401.15077)
- **Authors**: Yuhui Li, Fangyun Wei, Chao Zhang, Hongyang Zhang

### Core Insight

Standard speculative decoding trains a small **token-level** draft model. The difficulty is that token sequences are highly uncertain (vocabulary size ~32K–150K). EAGLE's insight: auto-regression at the **feature level** (the second-to-top-layer hidden state vector, dimension ~4096) is far easier to predict than the next token directly, because feature space is continuous and semantically smooth.

However, raw feature prediction still has **uncertainty** at the draft step because the predicted feature drifts from the true feature the target model would produce. EAGLE resolves this by incorporating the **token embedding of the most recently accepted token** as an additional input signal, effectively anchoring the feature prediction.

### Architecture

```
Target model (e.g. LLaMA-70B):
  tokens → [Layer 0 → Layer 1 → ... → Layer N-1] → h_{N-1} (second-to-top feature) → [Layer N] → logits

EAGLE Draft Model (single Transformer decoder layer, ~0.24B–0.99B params):

  Input at step t:
    h_{N-1,t}   (second-to-top hidden state from target model at accepted position t)
    e_t         (token embedding of accepted token at position t)

  Concatenate → FC layer → single Transformer decoder layer → predicted h_{N-1,t+1}
                                                              → LM head (shared with target) → draft token t+1

  For step t+1 (chained):
    h_{N-1,t+1}  (predicted by draft model, not target model — this is the "uncertainty")
    e_{t+1}      (embedding of draft token t+1)
    → same decoder → draft token t+2
```

The LM head is **shared** with the target model. This is critical: it means EAGLE only needs to train the feature extrapolation module, not a complete language model.

### Tree Structure and Verification

EAGLE generates not a single draft token chain but a **tree** of candidates:
- At each draft step, instead of the top-1 token, EAGLE samples the top-K tokens from the draft distribution
- These branch into a tree of depth D, producing up to K^D candidate continuations
- All candidates are packed into a single batch with a **tree-topology-aware causal mask**
- The target model runs **one forward pass** over the entire tree (much cheaper than D separate passes)
- The tree is traversed to find the longest valid prefix (standard speculative sampling acceptance criterion)
- The accepted path is returned; KV cache is updated for the accepted tokens only

The tree attention mask works as follows: each candidate token can attend to all tokens on its path from root, but not to sibling branches. This is implemented as a custom attention mask of shape `[tree_size, tree_size]` where mask[i,j]=1 iff token j is an ancestor of token i.

### Training Procedure

EAGLE uses **self-distillation**: the target model's hidden states on the training corpus are extracted and used as supervision for the draft model.

```
For each training sample (prompt P):
  1. Run target model forward pass → collect h_{N-1,t} for all t
  2. Train EAGLE draft model to predict h_{N-1,t+1} from (h_{N-1,t}, e_t)
     Loss: cross-entropy on next token (via LM head applied to predicted feature)
  3. Only EAGLE draft model parameters are updated (target model frozen)
```

Training data: ShareGPT (68K dialogue entries), learning rate 3e-5, AdamW optimizer.

Training cost:
- 7B models: ~1 day on RTX 3090
- 13B models: ~2 days on RTX 3090
- 33B models: 24 hours on RTX 3090
- 70B models: ~2 weeks on 16x A100

### Draft Model Size

| Target Model | EAGLE Draft Params | Relative Size |
|---|---|---|
| 7B  | 0.24B | 3.4% of target |
| 13B | 0.37B | 2.8% |
| 33B | 0.56B | 1.7% |
| 70B | 0.99B | 1.4% |
| Mixtral 8x7B | 0.28B | ~0.6% of total params |

### Performance

| Model | Speedup vs Vanilla | vs Lookahead | vs Medusa |
|---|---|---|---|
| LLaMA2-Chat 70B (MT-bench) | 3.0x | 2x faster | 1.6x faster |
| Vicuna-13B (MT-bench) | 3.07x | — | — |
| Vicuna-33B | 3.5x | — | — |

Draft accuracy: ~0.8 (meaning 80% of individual draft tokens are accepted in isolation).

### Memory Overhead

The EAGLE draft model adds ~0.24B–0.99B parameters to memory. At 16-bit precision this is 0.5GB–2GB additional. During inference, the draft model also requires storing its own KV cache for the draft tokens, but this is negligible compared to the target model's KV cache.

**Confidence**: High — from the original ICML 2024 paper and GitHub repository.

---

## 2. EAGLE-2: Dynamic Draft Trees

### Paper
- **Title**: "EAGLE-2: Faster Inference of Language Models with Dynamic Draft Trees"
- **Venue**: EMNLP 2024
- **arXiv**: [2406.16858](https://arxiv.org/abs/2406.16858)

### Core Innovation

EAGLE-1 uses a **static** draft tree: the branching factor and depth are fixed regardless of the specific prompt or context. This is suboptimal because some tokens are easy to predict (high acceptance probability) while others are hard. EAGLE-2 uses **dynamic** trees that allocate more computation to high-confidence paths.

### Mechanism: Confidence as Acceptance Rate Proxy

A key empirical finding: the **draft model's own confidence score** (softmax probability of the top token) is a reliable proxy for the acceptance rate of that draft token by the target model. The correlation is strong enough that EAGLE-2 can expand the tree dynamically without invoking the target model.

**Algorithm**:
```
1. Start with root (last accepted token)
2. Draft model computes logits → confidence c_1 = max(softmax(logits))
3. Expand: select top-k nodes by cumulative path score (product of confidences along path)
4. For each expanded node, run draft model → get next confidences
5. Repeat until budget (max tree nodes) is exhausted
6. Rerank all draft tokens by path score, select top-m for verification
7. Verify with target model using tree attention
8. Accept longest valid prefix
```

The "path score" for a candidate sequence `(t_1, t_2, ..., t_k)` is `c_1 * c_2 * ... * c_k`. Tokens with high path scores are more likely to be accepted by the target model.

EAGLE-2 requires **no additional training** beyond EAGLE-1's draft model.

### Performance

| Model | EAGLE-1 Speedup | EAGLE-2 Speedup | Improvement |
|---|---|---|---|
| Vicuna 13B (MT-bench) | 3.07x | 4.26x | +39% |
| LLaMA-Instruct 8B | ~3.0x | 3.16x | +5% |
| Average across models/tasks | ~3.0x | 3.05x–4.26x | +20–40% |

Acceptance length (average tokens accepted per cycle):
- EAGLE-1: ~3.98 tokens per step
- EAGLE-2: ~4.83 tokens per step

Maximum reported: 5x speedup on some models/tasks.

**Confidence**: High — from EMNLP 2024 paper.

---

## 3. EAGLE-3: Training-Time Test and Multi-Layer Fusion

### Paper
- **Title**: "EAGLE-3: Scaling up Inference Acceleration of Large Language Models via Training-Time Test"
- **Venue**: NeurIPS 2025
- **arXiv**: [2503.01840](https://arxiv.org/abs/2503.01840)
- **Released**: March 3, 2025

### Two Problems EAGLE-1/2 Had

**Problem 1: Training-Inference Distribution Mismatch**
During EAGLE-1 training, the draft model receives ground-truth hidden states `h_{N-1,t}` from the target model. But during inference (multi-step drafting), step 2+ uses the **draft model's own predicted hidden state** from step 1, which is an approximation of the true hidden state. This mismatch grows with draft length, degrading acceptance rates.

**Problem 2: Feature Regression Constraint**
EAGLE-1 explicitly regresses the next feature vector, which imposes a loss function shaped around predicting the exact hidden state. This constrains what the model can learn and, critically, prevents the model from benefiting from more training data — adding data beyond the initial 68K examples shows diminishing returns.

### EAGLE-3 Solutions

#### Solution 1: Training-Time Test (TTT)

EAGLE-3 trains the draft model by **simulating multi-step inference during training**:

```
Training procedure:
  Step 1: Teacher forcing pass
    - Feed target model features h^low, h^mid, h^high (3 layers)
    - Train draft model to predict token t+1
    - Attention mask: standard causal (sequential)

  Step 2+: Self-feeding pass (simulates real inference)
    - Feed draft model's OWN predicted features from step 1 as inputs for step 2
    - Adjust attention mask to tree-topology structure (matches inference)
    - Train draft model to predict token t+2

  Repeat for N steps
```

This explicitly teaches the draft model to handle its own compounding prediction errors, closing the training-inference gap.

#### Solution 2: Multi-Layer Feature Fusion

Instead of using only the second-to-top-layer hidden state `h_{N-1}`, EAGLE-3 fuses features from three layers:

```
Low-level feature:   h_low  (early transformer layer, ~layer N/4)
Mid-level feature:   h_mid  (middle layer, ~layer N/2)
High-level feature:  h_high (second-to-top layer N-1, same as EAGLE-1)

Fusion: cat(h_low, h_mid, h_high)  → shape: [3k] where k = hidden_dim
        FC layer (3k → k)          → fused feature g  of shape [k]

EAGLE-3 draft input: cat(g, e_t) → single decoder layer → direct token prediction
```

Crucially, EAGLE-3 predicts **tokens directly** (not an intermediate feature vector). The feature regression loss is removed entirely, giving the model complete freedom in how it uses the fused representations.

#### Textual Architecture Diagram

```
Target model (target inference):
  tokens → [L0, L1, ..., L_{low}, ..., L_{mid}, ..., L_{N-1}, L_N] → output

  Extract:
    h_low  ← L_{N/4} hidden state
    h_mid  ← L_{N/2} hidden state
    h_high ← L_{N-1} hidden state

EAGLE-3 draft head:
  [h_low | h_mid | h_high]   (concatenation, dim=3k)
       ↓
  FC (3k → k)                (learned projection)
       ↓ g (fused feature, dim=k)
  [g | embed(t)]             (concat with token embedding, dim=2k)
       ↓
  Transformer decoder layer  (single layer, ~0.24B–0.99B params)
       ↓
  LM head (shared w/ target) → draft token logits → sample top-K → tree
```

### Performance Results

| Model | EAGLE-1 | EAGLE-2 | EAGLE-3 | Max task |
|---|---|---|---|---|
| Vicuna-13B (mean) | 3.05x | 4.22x | **5.51x** | 5.58x (MT-bench) |
| LLaMA-Instruct-8B (mean) | — | 3.23x | **4.44x** | 4.48x (GSM8K) |
| LLaMA-Instruct-70B (mean) | — | ~3.5x | **~5.0x** | — |
| DeepSeek-R1-Distill-8B | — | — | **4.5x** | — |
| Best overall | — | — | — | **6.5x** (HumanEval, Vicuna-13B) |

Acceptance length (tau) per verification cycle:
| Model | EAGLE-1 | EAGLE-2 | EAGLE-3 |
|---|---|---|---|
| Vicuna-13B | 3.98 | 4.83 | **6.62** |
| LLaMA-Instruct-8B | — | 4.11 | **6.23** |
| LLaMA-Instruct-70B | — | 3.78 | **5.88** |

### Scaling Law Discovery

A breakthrough finding: EAGLE-3 shows a **scaling law** for inference acceleration. Increasing training data from 68K (EAGLE-1 baseline) to 8x more yields proportional speedup improvements. EAGLE-1 and EAGLE-2 did not exhibit this property — their acceptance rates plateaued with more data. EAGLE-3's removal of the feature regression constraint enables the model to extract more signal from additional training examples.

Throughput scaling:
- 1x data (68K): baseline
- 2x data: +10–15% speedup
- 4x data: +20–25% speedup
- 8x data: +30–40% speedup (training on ShareGPT 68K + UltraChat-200K 464K entries)

### Limitations

- Requires access to target-model hidden states during training — not applicable to closed-source models via API
- No evaluation on models larger than ~70B (GPU constraint at time of publication)
- Per-model training still required (no zero-shot transfer to new architectures)

**Confidence**: High — from NeurIPS 2025 paper (arXiv 2503.01840) and confirmed benchmark tables.

---

## 4. P-EAGLE: Parallel Drafting (2025 Extension)

### Paper
- **Title**: "P-EAGLE: Parallel-Drafting EAGLE with Scalable Training"
- **arXiv**: [2602.01469](https://arxiv.org/html/2602.01469v1)

### Core Problem

EAGLE-1/2/3 draft models operate **autoregressively**: to generate K draft tokens, the draft model makes K sequential forward passes. This sequential bottleneck limits throughput at larger K values.

### Solution

P-EAGLE generates all K draft tokens in **a single forward pass** by exploiting the target model's internal representations at each position of the prompt:

```
Prefill: target model processes prompt → h_context
         draft model reads h_context for ALL positions simultaneously
         generates K draft tokens in parallel (one per input position)
```

This is conceptually similar to MTP (multi-token heads): instead of chaining K serial forward passes, P-EAGLE predicts K tokens in one shot from the prompt hidden states.

### Performance

- Up to **1.69x speedup** over vanilla EAGLE-3 on NVIDIA B200
- Optimal at K=7 (EAGLE-3 peaks at K=3 due to sequential overhead)
- Integrated into vLLM v0.16.0+

**Confidence**: Medium — from AWS blog and arXiv preprint, not yet peer-reviewed at time of writing.

---

## 5. EAGLE vs. MTP: Architectural Comparison

This comparison is directly relevant to the mlx-mtp project since Qwen3.5 and DeepSeek-V3 have trained MTP heads.

### MTP (Multi-Token Prediction) — DeepSeek-V3 Style

DeepSeek-V3 trains D sequential MTP modules during pretraining. Each module predicts one additional future token beyond the base model's prediction.

Architecture of each MTP module (from DeepSeek-V3 technical report):
```
Module k input:
  h_last   ← final hidden state from base model (or previous MTP module)
  e_{t+k}  ← token embedding of token t+k (next token in ground truth)

  [h_last | e_{t+k}] → projection M_k → transformer block T_k → norm → LM head → P(t+k+1)
```

The critical difference: MTP uses the **ground-truth next token embedding** `e_{t+k}` as input during training. During inference for speculative decoding, this becomes the **predicted** token embedding (introducing the same distribution mismatch EAGLE-3 solves).

DeepSeek-V3 trains 1 MTP module (D=1). The MTP head shares the embedding table and LM head with the base model (same as EAGLE).

### EAGLE vs. MTP: Side-by-Side

| Property | EAGLE-1/2 | EAGLE-3 | MTP (DeepSeek-V3) |
|---|---|---|---|
| Draft model type | Separate small model | Separate small model | Additional heads in base model |
| Training time | Post-training (1–2 days) | Post-training (1–2 weeks for 70B) | During pretraining (expensive) |
| Feature source | Single layer (N-1) | Three layers (fused) | Final layer of base/prior module |
| Prediction mode | Token via LM head | Direct token | Direct token |
| Tree drafting | Yes (static or dynamic) | Yes (dynamic) | Yes (via EAGLE infrastructure) |
| Distribution mismatch | Moderate | Minimal (TTT) | Moderate (same issue) |
| Acceptance rate (single token) | ~0.80 | ~0.85–0.90 | ~0.85–0.90 (for k=1) |
| Speedup vs vanilla | 3.0x–3.5x | 3.0x–6.5x | 1.8x (DeepSeek-V3 reported) |
| Requires hidden states | Yes (training + inference) | Yes (training + inference) | Built into base model |
| Parameters | 0.24B–0.99B separate | 0.24B–0.99B separate | Part of base model (shared) |

### Key Differentiator: Acceptance Rate vs. Drafting Overhead

MTP heads are **extremely fast** to run (they share the base model's forward pass — no additional pass needed). But they only produce 1 (or D) draft tokens per base model step.

EAGLE's separate draft model generates a **tree** of candidates with K branches × D depth. This provides higher acceptance length (tau ~6 for EAGLE-3) at the cost of running the small draft model K*D times per target verification pass.

For DeepSeek-V3's MTP with 1 module and 85–90% acceptance rate, speculative decoding gives ~1.8x speedup. EAGLE-3 applied to an equivalent model would likely give 4–5x speedup, but requires training the draft model separately.

### Can EAGLE Be Applied to Models with MTP Heads?

**Yes.** The vLLM pull request #12915 (later #12755) demonstrates exactly this: DeepSeek-R1's MTP weight module is loaded as an EAGLE-style draft model. The MTP module produces hidden states in a format compatible with EAGLE's LM head application.

The integration:
- Load only the MTP layer weights (not the full base model again)
- Feed base model's hidden states as inputs to the MTP layer
- Apply EAGLE's tree drafting algorithm on top
- Result: ~73% acceptance rate for k=2, ~2x speedup on DeepSeek-R1 (8×H200)

**Implication for mlx-mtp project**: The Qwen3.5 MTP head (which follows the same pattern as DeepSeek-V3 MTP, as documented in `research/mtp_architecture.md`) can be used as an EAGLE-compatible draft head. The forward pass is:
```
pre_fc_norm(h_last) + pre_fc_norm(embed(t)) → cat → fc → transformer_layer → norm → lm_head
```
This is structurally identical to EAGLE-1's draft model.

**Confidence**: High — from vLLM PR discussion, DeepSeek-V3 technical report, and direct code inspection.

---

## 6. EAGLE on MoE Models

### The MoE Verification Overhead Problem

For **dense** models, speculative decoding's key property holds: verifying K draft tokens costs roughly the same as generating 1 token (because attention is the bottleneck and scales quadratically with sequence length, not with the number of unique tokens).

For **MoE** models, this property breaks:
- Each token activates only a sparse subset of experts (e.g. top-2 of 64 experts in Mixtral)
- A tree of K draft candidates activates **the union** of all expert subsets across the tree
- With K=127 tokens in the draft tree, **54 of 64 experts per layer** are loaded (for OLMoE-1B-7B)
- This approaches full-model evaluation, negating the sparse computation advantage
- Verification overhead at K=7 can reach 3x the cost of a single token pass (vs ~1x for dense models)

### Dense Draft Model for MoE Target

The EAGLE draft model itself **does not need to be MoE**. The draft model for Mixtral-8x7B has 0.28B dense parameters. It drafts using the target model's hidden states (which are the outputs of the sparse MoE transformer layers).

```
Mixtral-8x7B target → h_{N-1} (dense hidden state, even though produced by MoE layers)
EAGLE draft model (dense, 0.28B) → processes h_{N-1} like any other hidden state
→ draft tokens → sent to Mixtral for verification (this is where MoE overhead bites)
```

### MoE-Spec: Expert Budgeting Solution

- **Paper**: "MoE-Spec: Expert Budgeting for Efficient Speculative Decoding"
- **arXiv**: [2602.16052](https://arxiv.org/html/2602.16052)

**Key insight**: Expert routing probabilities are heavy-tailed. The top-32 of 64 experts capture 93% of routing weight. By enforcing a fixed expert budget B during verification (load only top-B experts, handle token assignments to unloaded experts via truncation or substitution), verification cost is capped.

Results vs EAGLE-3 baseline:
| Model | EAGLE-3 speedup | MoE-Spec speedup | Improvement |
|---|---|---|---|
| Mixtral-8x7B | 1.85x (K=127 tree) | 2.1x (32 experts) | **+27%** |
| Qwen3-30B-A3B | baseline | +16% | relative |
| OLMoE-1B-7B | baseline | +6% | relative |

MoE-Spec is training-free and integrates into existing EAGLE-3 pipelines with only 2–3% selection overhead.

### Shared Expert as Draft Signal

In models like DeepSeek-V3 and Qwen3-MoE, there are shared/always-active experts alongside the routed sparse experts. Conceptually, the shared expert output could be used as a draft signal (since it is always computed and captures general token-level semantics). This idea does not appear to have been explored in published work as of March 2026, but it is a plausible direction for MoE-specific speculation.

### Status of EAGLE on Specific MoE Models

| Model | EAGLE version | Support status |
|---|---|---|
| Mixtral-8x7B | EAGLE-1 | Officially supported (0.28B draft, SafeAILab) |
| Qwen3-30B-A3B | EAGLE-3 | Officially supported |
| DeepSeek-V3 | MTP as EAGLE draft | vLLM support via PR #12755 |
| OLMoE-1B-7B | EAGLE-3 | Research evaluation in MoE-Spec paper |
| DeepSeek-MoE | Not confirmed | — |

**Confidence**: High for Mixtral/Qwen3/DeepSeek via vLLM; Medium for others.

---

## 7. Medusa: Multiple Parallel Prediction Heads

### Paper
- **Title**: "Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads"
- **Venue**: COLM 2024
- **arXiv**: [2401.10774](https://arxiv.org/abs/2401.10774)
- **GitHub**: [FasterDecoding/Medusa](https://github.com/FasterDecoding/Medusa)

### Architecture

Unlike EAGLE's single autoregressive draft module, Medusa adds K **parallel** prediction heads, each independently predicting a future token at a fixed offset:

```
Target model final hidden state h_final:
    ↓
Head 0 (MLP) → P(t+1)   # predicts next token
Head 1 (MLP) → P(t+2)   # independently predicts token after next
Head 2 (MLP) → P(t+3)   # etc.
...
Head K-1 (MLP) → P(t+K)
```

Each Medusa head is a simple 2-layer feedforward network (no transformer layer). All heads receive the same `h_final` — there is no auto-regressive dependency between heads.

### Tree Construction and Verification

With K heads each producing top-T candidates, a tree of K levels × T branches is formed. The same tree attention mechanism as EAGLE is used for verification. A "typical acceptance scheme" is used: rather than strict left-to-right acceptance, Medusa accepts tokens based on a tolerance criterion that maintains the generation quality distribution.

```
Medusa tree (K=4 heads, T=3 top candidates):
Level 0: [a, b, c]           (3 candidates for t+1)
Level 1: [d,e,f, g,h,i, j,k,l]  (3 candidates per level-0 node for t+2)
...etc
Total nodes: 3^4 = 81 (static tree, fixed branching)
```

### Training: Medusa-1 vs Medusa-2

**Medusa-1** (frozen backbone):
- Only the Medusa heads are fine-tuned
- Backbone LLM parameters frozen
- Training is cheap: only head parameters updated
- Speedup: >2.2x
- Lossless acceleration (does not alter base model distribution)

**Medusa-2** (joint fine-tuning):
- Both backbone and Medusa heads trained together
- Requires careful recipe to not degrade backbone quality
- Better head accuracy → higher acceptance rates
- Speedup: 2.3x–3.6x
- Not lossless unless acceptance criterion is exact

Training data: typically ShareGPT or domain-specific data. Training time: comparable to EAGLE-1 for similar model sizes.

### Comparison to EAGLE

| Property | Medusa | EAGLE-1 |
|---|---|---|
| Head type | K parallel MLP heads | Single autoregressive transformer layer |
| Dependency between draft positions | None (independent heads) | Auto-regressive (later tokens informed by earlier) |
| Draft accuracy (per token) | ~0.6 | ~0.8 |
| Speedup | 2.2x–3.6x | 3.0x–3.5x |
| Memory overhead | K × small MLP heads (~minimal) | ~0.24B–0.99B |
| Training time | Lower (only heads) | Moderate |
| Tree structure | Static | Static (EAGLE-1), Dynamic (EAGLE-2) |
| Input to draft | Only h_final (no token embedding) | h_{N-1} + token embedding |

EAGLE outperforms Medusa primarily because its autoregressive draft captures the sequential dependency between tokens: predicting "the" is far easier if you already predicted "quick brown fox jumped over" vs. independently predicting from only the prompt context. Medusa's heads are statistically independent, so each prediction ignores what the previous head predicted.

**Speedup numbers on MT-bench**: EAGLE is 1.6x faster than Medusa, 2x faster than Lookahead decoding.

**Confidence**: High — from COLM 2024 paper, Together.ai blog, and comparison tables in EAGLE paper.

---

## 8. Lookahead Decoding (Jacobi-Based)

### Reference
- **Blog**: [LMSYS: Break the Sequential Dependency of LLM Inference](https://lmsys.org/blog/2023-11-21-lookahead-decoding/)
- **Venue**: ICML 2024
- **GitHub**: [hao-ai-lab/LookaheadDecoding](https://github.com/hao-ai-lab/LookaheadDecoding)

### Core Concept: Jacobi Iteration Applied to LLMs

Autoregressive decoding solves a fixed-point equation: find token sequence `X = (x_1, ..., x_n)` such that `P(x_t | x_{<t}) = x_t` for all t (i.e., the model agrees with itself). The Jacobi iteration method for nonlinear equations provides a parallel approach: start with a guess for the entire output, then update all positions simultaneously until convergence.

**Pure Jacobi decoding**: Initialize random guess `X^0`, at each step compute `x_t^{k+1} = argmax P(x_t | x_{<t}^k)` for all t in parallel. Theoretically converges, but in practice rarely sees wall-clock speedup because long sequences rarely converge in < L steps.

**Lookahead Decoding** extends Jacobi with an n-gram cache:
```
Two parallel branches per step:

Branch 1 (lookahead): Maintain a W×N "lookahead window"
  - W=window size (parallel workers), N=lookahead depth
  - Each worker extends the Jacobi iteration, generating 2D n-grams

Branch 2 (verification): Check if any cached n-gram matches current context
  - If match found, accept without further generation
  - n-gram cache grows over time, improving hit rate

Both branches use one LLM forward pass of size W+M (M=candidates to verify)
```

### Performance

- Speedup range: **1.5x–2.3x** on a single GPU
- Task-dependent: best on repetitive/formulaic text; worse on creative generation
- No training required — purely algorithmic
- Memory: O(W×N) additional token storage for the n-gram cache

### Comparison to EAGLE

EAGLE is ~2x faster than Lookahead on MT-bench (EAGLE achieves 3x where Lookahead achieves ~1.5x). Lookahead's key advantage is zero training cost and applicability to any LLM including black-box APIs (if logits are accessible for sampling).

**When to use Lookahead**: When no draft model training is feasible, no compatible small model exists, or for API-based inference where hidden states are inaccessible.

**Confidence**: High — from LMSYS blog, ICML 2024 paper.

---

## 9. Self-Speculative Decoding: LayerSkip

### Paper
- **Title**: "LayerSkip: Enabling Early Exit Inference and Self-Speculative Decoding"
- **Venue**: ACL 2024
- **arXiv**: [2404.16710](https://arxiv.org/abs/2404.16710)
- **GitHub**: [facebookresearch/LayerSkip](https://github.com/facebookresearch/LayerSkip)

### Concept

Instead of a separate draft model, LayerSkip uses the **same model** as both drafter and verifier by exploiting early exits:

```
Full model: [L0, L1, L2, ..., L_{exit}, ..., L_{N-1}, L_N]

Draft pass: run only layers 0 through L_{exit} → early exit
  → apply LM head at L_{exit} → draft tokens
  → KV cache for layers 0..L_{exit} is populated

Verification pass: continue from L_{exit} to L_N
  → reuse KV cache from draft pass for layers 0..L_{exit}
  → run remaining layers L_{exit+1}..L_N
  → get target distribution, apply acceptance criterion
```

The key efficiency: layers 0..L_{exit} are run only once total (shared between draft and verify), not twice.

### Training Requirements

LayerSkip requires **training-time modifications**:
- Layer dropout with increasing rates for later layers (encourages early layers to be self-sufficient)
- Early exit loss: all intermediate layers share the same LM head, trained to predict the next token directly

This means LayerSkip is not applicable to models trained without this specific recipe.

### Performance

| Task | Speedup |
|---|---|
| Summarization (CNN/DM) | 2.16x |
| Coding | 1.82x |
| Semantic parsing (TOPv2) | 2.0x |

Memory: No additional model parameters. The "draft model" is simply the first L_{exit} layers of the target model.

### CAS-Spec Extension (2025)

CAS-Spec ("Cascade Adaptive Self-Speculative Decoding") extends LayerSkip by dynamically combining layer sparsity and activation quantization to create an adaptive self-drafter that selects the cheapest sufficient approximation per context.

**Confidence**: High — from ACL 2024 paper (Meta AI Research).

---

## 10. Cascade Speculative Drafting

### Paper
- **Title**: "Cascade Speculative Drafting for Even Faster LLM Inference"
- **Venue**: NeurIPS 2024
- **arXiv**: [2312.11462](https://arxiv.org/html/2312.11462v5)

### Concept: Two Types of Cascades

Standard speculative decoding uses one draft model. Cascade Speculative Drafting uses **multiple** draft models organized as:

**Vertical Cascade**: Hierarchical chain of draft models from smallest to largest:
```
Statistical bigram model (nearly free)
  ↓ reviews / proposes to
Small neural model (e.g. 1B)
  ↓ reviews / proposes to
Medium neural model (e.g. 7B)
  ↓ proposes to
Target model (e.g. 70B)
```
Each level reviews what the level below proposed before passing upward. The smallest (statistical) model is essentially free.

**Horizontal Cascade**: Different-sized models for different **token positions** in the draft:
```
Position 1 (most likely to be accepted): use largest draft model
Position 2: medium draft model
...
Position K (rarely accepted): use smallest / statistical model
```
Rationale: the probability of accepting token at position k requires all of tokens 1..k-1 to also be accepted first, so the last position is only relevant if all preceding ones were accepted (low probability).

### Performance

On GSM8K: up to **44% additional speedup** over single-draft-model speculative decoding.
On MMLU: up to **81% additional speedup**.

The method is orthogonal to EAGLE: you can use EAGLE-style draft models at each level of the cascade. Combining CS Drafting + tree attention + EAGLE outperforms Medusa on Vicuna-7B benchmarks.

### Faster Cascades via Speculative Decoding (Google, ICLR 2025)

A related Google paper ("Faster Cascades via Speculative Decoding", [2405.19261](https://arxiv.org/abs/2405.19261)) combines cascade routing (invoke large model only for "hard" inputs) with speculative execution (use large model in verification mode for "easy" inputs), achieving best of both worlds.

**Confidence**: High — from NeurIPS 2024 paper and Google Research blog.

---

## 11. Apple ReDrafter: MLX-Native Approach

### Paper / Blog
- **Title**: "Recurrent Drafter for Fast Speculative Decoding in Large Language Models"
- **URL**: [machinelearning.apple.com/research/recurrent-drafter](https://machinelearning.apple.com/research/recurrent-drafter)
- **arXiv**: [2403.09919](https://arxiv.org/html/2403.09919v4)

### Architecture

Apple's ReDrafter uses an **RNN-based** draft model (rather than transformer-based like EAGLE):
- The recurrent structure captures sequential dependencies across draft steps more efficiently than a full transformer layer
- Beam search over the draft model produces multiple candidate continuations
- Dynamic tree attention is applied to collapse duplicate beam prefixes before verification, reducing redundant computation

```
ReDrafter:
  LLM hidden state h → RNN cell → draft token 1
                                → RNN cell (reusing state) → draft token 2
                                → ...etc → beam candidates
  Beam candidates → deduplicate prefixes via dynamic tree attention → verify with target
```

### MLX Performance

Implemented in MLX and benchmarked on Apple Silicon:
- M1 Max: 1.37x speedup
- M2 Ultra: up to **2.3x speedup**
- Vicuna on H100: up to 2.8x speedup

The lower M1 Max numbers vs M2 Ultra reflect unified memory bandwidth constraints: the draft model's additional computation fits better on chips with higher memory bandwidth.

### Relevance to EAGLE on MLX

ReDrafter shows that speculative decoding **is viable** on Apple Silicon via MLX. The key implementation components required are:
1. Draft model forward pass (any small model, including EAGLE-style single transformer layer)
2. Tree attention with custom causal mask
3. KV cache management that supports non-linear (tree-shaped) histories

**Confidence**: Medium-High — Apple blog post, confirmed MLX speedup numbers.

---

## 12. Practical MLX / Apple Silicon Implementation

### What mlx-lm Already Has

MLX-lm (as of early 2026) supports:
- Simple **sequential** speculative decoding: `--draft-model small_model --num-draft-tokens 4`
- The draft model produces a linear sequence of K tokens; the main model verifies them
- Usage: `mlx_lm.generate --model bigmodel --draft-model smallmodel --num-draft-tokens 4`

**What is missing for EAGLE**:
- Tree attention (custom causal mask for branching candidates)
- Multiple draft candidates at each step (sampling top-K, not just greedy)
- Dynamic tree expansion based on confidence scores (EAGLE-2)
- Access to target model's hidden states during draft inference (needed to instantiate EAGLE draft model)

### Tree Attention in MLX

Implementing tree attention requires a custom attention mask. In MLX:

```python
import mlx.core as mx
import mlx.nn as nn

def build_tree_mask(tree_tokens: list[list[int]]) -> mx.array:
    """
    tree_tokens: list of token sequences (paths from root)
    Returns: [n_tokens, n_tokens] boolean mask
    """
    n = sum(len(path) for path in tree_tokens)
    mask = mx.zeros((n, n), dtype=mx.bool_)
    idx = 0
    for path in tree_tokens:
        for i, _ in enumerate(path):
            # Token at path[i] can attend to all path[0..i]
            for j in range(i + 1):
                mask = mask.at[idx + i, idx + j].set(True)
        idx += len(path)
    return mask
```

In practice, EAGLE implementations encode tree topology as a compact bitmask (64-bit integers per token) to avoid materializing the full n×n mask.

### Batch Verification Limitation

As of February 2026, mlx-lm does not support speculative decoding for **batched** requests. This is a documented limitation:

> "SpeculativeDecodingNotSupportedError: Speculative decoding is not supported for batched MLX models."

For a research/single-user setting this is acceptable. For production serving with multiple concurrent requests, this would need to be addressed.

### Memory Overhead of EAGLE Draft Model on Apple Silicon

For a 7B target model (typical for on-device use on M2 Ultra/M3 Max):
- Target model at 4-bit: ~4GB
- EAGLE draft model (0.24B, 16-bit): ~0.5GB
- Draft model KV cache (K=7 draft tokens, 32 layers, batch=1): negligible
- **Total additional overhead**: ~500MB — acceptable on 64GB unified memory

For a 13B target at 4-bit (~7GB) + 0.37B EAGLE head (~0.75GB): still comfortably fits on M2 Ultra (96GB).

### Training EAGLE on Apple Silicon

Training the EAGLE draft head on MLX is feasible for smaller models:
- Training data collection: run target model on ShareGPT, collect hidden states → large I/O (h_{N-1} at fp16 for 68K dialogues of ~512 tokens each ≈ ~68K × 512 × 4096 × 2 bytes ≈ ~280GB of hidden state data)
- Actual training: single transformer decoder layer backprop → very fast
- For a 7B model on M2 Ultra: collection is the bottleneck; training itself could complete in hours
- EAGLE-3's 8x data requirement (~2.3TB hidden states) would be impractical to store locally but the training pass itself remains small

Practical recommendation: use pre-trained EAGLE draft heads from the SafeAILab HuggingFace repository for common model families (LLaMA, Qwen, Vicuna), rather than training from scratch.

### Implementation Complexity Assessment

| Method | MLX Implementation Complexity | Notes |
|---|---|---|
| Sequential spec decoding (existing) | Done | Already in mlx-lm |
| MTP spec decoding (Qwen3.5/DeepSeek) | Low | Load MTP head, feed to existing spec decode |
| EAGLE-1 (tree attention) | Medium | Need tree mask + EAGLE draft head loader |
| EAGLE-2 (dynamic tree) | Medium-High | Need confidence-based expansion algorithm |
| EAGLE-3 (multi-layer fusion) | Medium-High | Need hidden state extraction from 3 layers |
| Medusa | Medium | Simpler than EAGLE: parallel heads, no autoregression |
| Lookahead | Medium | N-gram cache + 2-branch forward pass |
| LayerSkip | High | Requires model trained with layer dropout |
| ReDrafter | Medium | RNN draft model; Apple has MLX reference code |

---

## 13. Comparative Summary

### Speedup Comparison (Approximate, Single GPU, Typical Chat Tasks)

| Method | Speedup Range | Acceptance Rate | Training Required | Memory Overhead |
|---|---|---|---|---|
| Lookahead | 1.5x–2.3x | N/A (n-gram cache) | None | Minimal |
| Medusa-1 | 2.2x–2.5x | ~0.6 per token | Small (heads only) | Minimal |
| Medusa-2 | 2.3x–3.6x | ~0.65 | Moderate (joint) | Minimal |
| LayerSkip | 1.8x–2.2x | Model-dependent | High (retrain) | None |
| EAGLE-1 | 3.0x–3.5x | ~0.80 | Moderate | ~0.5–2GB |
| ReDrafter (MLX) | 1.4x–2.3x | — | Moderate | ~0.5GB |
| EAGLE-2 | 3.0x–5.0x | ~0.80 (dynamic tree) | None beyond EAGLE-1 | ~0.5–2GB |
| MTP (DeepSeek-V3) | 1.8x | 0.85–0.90 | Built-in (pretraining) | Built-in |
| EAGLE-3 | 3.0x–6.5x | ~0.85–0.90 | Higher (more data) | ~0.5–2GB |
| P-EAGLE | 4x–8x est. | ~0.85–0.90 | Moderate | ~0.5–2GB |
| Cascade CS-Draft | +44–81% over base | — | Hierarchical | Multiple models |

### Key Design Dimensions

```
Training cost  HIGH ←————————————————→ NONE
               EAGLE-3   EAGLE-1   Medusa-1   Lookahead
                  ↑         ↑         ↑           ↑
Speed benefit HIGH ←————————————————→ LOW
```

```
Separate model ←——————————————→ No separate model
EAGLE-1/2/3        Medusa       LayerSkip   MTP
                                            (heads in base model)
```

---

## 14. Recommendations for mlx-mtp Project

Given the project's focus on MTP speculative decoding on MLX/Apple Silicon with Qwen3.5 models:

**Short term (highest leverage)**:
1. The Qwen3.5 MTP head architecture is structurally identical to EAGLE-1's draft model. Implementing EAGLE's tree attention verification on top of the existing MTP head extraction work would immediately enable EAGLE-style speedups.
2. Start with EAGLE-2's dynamic tree (no additional training needed — just change the draft tree from static to confidence-based expansion).

**Medium term**:
3. Implement multi-layer feature fusion (EAGLE-3 style) for Qwen3.5: extract hidden states from layers N/4, N/2, and N-1 rather than just N-1. This requires a new training pass but enables 4–6x speedup.
4. For MoE models (Qwen3-30B-A3B): consider the MoE-Spec expert budgeting approach to cap verification overhead.

**Architecture decision**: Use the MTP head's existing transformer block as the draft model backbone (as vLLM PR #12755 does for DeepSeek) rather than training a new EAGLE head from scratch. The Qwen3.5 MTP weights are already extracted (`mtp_weights/`) and the architecture is documented.

---

## 15. Gaps and Limitations

- **EAGLE-3 on Apple Silicon**: No confirmed benchmark data for EAGLE-3 on MLX as of March 2026. ReDrafter provides the only confirmed MLX speedup (2.3x). EAGLE-3 on MLX would require implementing tree attention.
- **MoE-Spec on MLX**: No implementation exists for MLX. The expert budgeting approach would require hooking into the MoE routing code.
- **P-EAGLE**: Not peer-reviewed as of March 2026. NVIDIA B200 numbers may not translate to Apple Silicon where memory bandwidth characteristics differ.
- **Scaling laws for smaller models**: EAGLE-3's scaling law was demonstrated on 8B–70B models. Whether the same scaling holds for 0.5B–4B models (more practical for on-device) is not established.
- **Batch speculative decoding in MLX**: Currently unsupported. For any serving use case with multiple concurrent users, this is a blocking limitation.

---

## References

1. [EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty](https://arxiv.org/abs/2401.15077) — Li et al., ICML 2024
2. [EAGLE-2: Faster Inference of Language Models with Dynamic Draft Trees](https://arxiv.org/abs/2406.16858) — Li et al., EMNLP 2024
3. [EAGLE-3: Scaling up Inference Acceleration via Training-Time Test](https://arxiv.org/abs/2503.01840) — Li et al., NeurIPS 2025
4. [SafeAILab/EAGLE GitHub Repository](https://github.com/SafeAILab/EAGLE) — Official implementation
5. [Medusa: Simple LLM Inference Acceleration with Multiple Decoding Heads](https://arxiv.org/abs/2401.10774) — Cai et al., COLM 2024
6. [Break the Sequential Dependency of LLM Inference Using Lookahead Decoding](https://lmsys.org/blog/2023-11-21-lookahead-decoding/) — LMSYS, ICML 2024
7. [LayerSkip: Enabling Early Exit Inference and Self-Speculative Decoding](https://arxiv.org/abs/2404.16710) — Elhoushi et al., ACL 2024 (Meta AI)
8. [Cascade Speculative Drafting for Even Faster LLM Inference](https://arxiv.org/html/2312.11462v5) — NeurIPS 2024
9. [Faster Cascades via Speculative Decoding](https://arxiv.org/abs/2405.19261) — Google, ICLR 2025
10. [MoE-Spec: Expert Budgeting for Efficient Speculative Decoding](https://arxiv.org/html/2602.16052) — 2026
11. [P-EAGLE: Parallel-Drafting EAGLE with Scalable Training](https://arxiv.org/html/2602.01469v1) — 2025
12. [Recurrent Drafter for Fast Speculative Decoding (Apple)](https://machinelearning.apple.com/research/recurrent-drafter) — Apple, 2024
13. [vLLM PR #12915: EAGLE-Style MTP for DeepSeek-R1](https://github.com/vllm-project/vllm/pull/12915) — vLLM community
14. [Efficient LLM Serving with MTP: DeepSeek V3 and SGLang on AMD](https://rocm.blogs.amd.com/software-tools-optimization/mtp/README.html) — AMD ROCm blog
15. [DeFT: Decoding with Flash Tree-Attention](https://arxiv.org/html/2404.00242v2) — ICLR 2025
