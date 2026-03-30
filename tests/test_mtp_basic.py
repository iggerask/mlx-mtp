"""
Basic MTP tests.

Tests MTP weight loading, head construction, and generation correctness.
Uses extracted MTP weights from Qwen/Qwen3.5-4B and the 4-bit model for inference.
"""

import json
import pytest
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn

TEST_MODEL = "mlx-community/Qwen3.5-4B-4bit"
MTP_WEIGHTS_FILE = Path("mtp_weights/Qwen_Qwen3.5-4B.safetensors")
BF16_CONFIG_MODEL = "Qwen/Qwen3.5-4B"


@pytest.fixture(scope="module")
def model_and_tokenizer():
    from mlx_lm import load
    return load(TEST_MODEL)


@pytest.fixture(scope="module")
def bf16_config():
    from huggingface_hub import snapshot_download
    model_path = Path(snapshot_download(BF16_CONFIG_MODEL, allow_patterns=["config.json"]))
    with open(model_path / "config.json") as f:
        return json.load(f)


@pytest.fixture(scope="module")
def mtp_weights():
    from vllm_mlx_mtp.mtp_head import load_mtp_weights_from_file
    if not MTP_WEIGHTS_FILE.exists():
        pytest.skip(f"MTP weights not found at {MTP_WEIGHTS_FILE}")
    return load_mtp_weights_from_file(MTP_WEIGHTS_FILE)


@pytest.fixture(scope="module")
def mtp_head(mtp_weights, bf16_config):
    from vllm_mlx_mtp.mtp_head import build_mtp_head
    head = build_mtp_head(mtp_weights, bf16_config, norm_shift=True)
    assert head is not None
    return head


class TestMTPConfig:
    """Config detection tests."""

    def test_config_has_mtp_field(self, bf16_config):
        from vllm_mlx_mtp.mtp_head import detect_mtp_support
        assert detect_mtp_support(bf16_config)

    def test_config_no_mtp_for_regular_model(self):
        from vllm_mlx_mtp.mtp_head import detect_mtp_support
        assert not detect_mtp_support({"hidden_size": 2560})


class TestMTPWeights:
    """Weight loading tests."""

    def test_mtp_weights_exist(self, mtp_weights):
        assert len(mtp_weights) == 15, f"Expected 15 MTP weight tensors, got {len(mtp_weights)}"

    def test_mtp_weights_have_expected_keys(self, mtp_weights):
        expected_keys = [
            "mtp.pre_fc_norm_hidden.weight",
            "mtp.pre_fc_norm_embedding.weight",
            "mtp.fc.weight",
            "mtp.layers.0.self_attn.q_proj.weight",
            "mtp.layers.0.self_attn.k_proj.weight",
            "mtp.layers.0.self_attn.v_proj.weight",
            "mtp.layers.0.self_attn.o_proj.weight",
            "mtp.layers.0.self_attn.q_norm.weight",
            "mtp.layers.0.self_attn.k_norm.weight",
            "mtp.layers.0.input_layernorm.weight",
            "mtp.layers.0.post_attention_layernorm.weight",
            "mtp.layers.0.mlp.gate_proj.weight",
            "mtp.layers.0.mlp.up_proj.weight",
            "mtp.layers.0.mlp.down_proj.weight",
            "mtp.norm.weight",
        ]
        for key in expected_keys:
            assert key in mtp_weights, f"Missing weight key: {key}"


class TestMTPHead:
    """MTP head construction and forward pass tests."""

    def test_build_head(self, mtp_head):
        assert isinstance(mtp_head, nn.Module)

    def test_head_forward_pass(self, mtp_head, bf16_config):
        text_config = bf16_config.get("text_config", bf16_config)
        hidden_size = text_config["hidden_size"]

        hidden = mx.random.normal((1, 1, hidden_size))
        embed = mx.random.normal((1, 1, hidden_size))

        out = mtp_head(hidden, embed)
        mx.eval(out)

        assert out.shape == (1, 1, hidden_size), f"Unexpected shape: {out.shape}"

    def test_head_deterministic(self, mtp_head, bf16_config):
        text_config = bf16_config.get("text_config", bf16_config)
        hidden_size = text_config["hidden_size"]

        mx.random.seed(42)
        hidden = mx.random.normal((1, 1, hidden_size))
        embed = mx.random.normal((1, 1, hidden_size))

        out1 = mtp_head(hidden, embed)
        out2 = mtp_head(hidden, embed)
        mx.eval(out1, out2)

        assert mx.allclose(out1, out2).item()


class TestCacheUtils:
    """Cache utility tests."""

    def test_trim_kv_cache(self):
        from mlx_lm.models.cache import KVCache
        from vllm_mlx_mtp.cache_utils import trim_hybrid_cache

        cache = KVCache()
        keys = mx.zeros((1, 4, 10, 32))
        values = mx.zeros((1, 4, 10, 32))
        cache.update_and_fetch(keys, values)
        assert cache.offset == 10

        trimmed = trim_hybrid_cache([cache], 2)
        assert trimmed == 2
        assert cache.offset == 8

    def test_trim_hybrid_cache_skips_arrays_cache(self):
        from mlx_lm.models.cache import KVCache, ArraysCache
        from vllm_mlx_mtp.cache_utils import trim_hybrid_cache

        kv = KVCache()
        keys = mx.zeros((1, 4, 10, 32))
        values = mx.zeros((1, 4, 10, 32))
        kv.update_and_fetch(keys, values)

        arr = ArraysCache(size=2)
        cache = [arr, arr, arr, kv]
        trimmed = trim_hybrid_cache(cache, 1)
        assert trimmed == 1
        assert kv.offset == 9

    def test_save_restore_cache_state(self):
        from mlx_lm.models.cache import KVCache, ArraysCache
        from vllm_mlx_mtp.cache_utils import save_cache_state, restore_cache_state

        kv = KVCache()
        keys = mx.zeros((1, 4, 5, 32))
        values = mx.zeros((1, 4, 5, 32))
        kv.update_and_fetch(keys, values)

        arr = ArraysCache(size=2)
        arr.cache[0] = mx.ones((1, 3, 16))
        arr.cache[1] = mx.ones((1, 3, 16)) * 2
        mx.eval(arr.cache[0], arr.cache[1])

        cache = [arr, kv]
        saved = save_cache_state(cache)

        # Modify state
        kv.update_and_fetch(mx.zeros((1, 4, 3, 32)), mx.zeros((1, 4, 3, 32)))
        arr.cache[0] = mx.zeros((1, 3, 16))
        assert kv.offset == 8

        restore_cache_state(cache, saved)
        assert kv.offset == 5
        assert mx.allclose(arr.cache[0], mx.ones((1, 3, 16))).item()


class TestHiddenCapture:
    """Hidden state capture tests."""

    def test_capture_hidden_state(self, model_and_tokenizer):
        model, tokenizer = model_and_tokenizer
        from vllm_mlx_mtp.hidden_capture import HiddenStateCapture
        from mlx_lm.models.cache import make_prompt_cache

        capture = HiddenStateCapture(model)

        tokens = tokenizer.encode("Hello world")
        prompt = mx.array(tokens)
        cache = make_prompt_cache(model)

        logits = model(prompt[None], cache=cache)
        mx.eval(logits)

        hidden = capture.get_hidden_state()
        assert hidden is not None
        assert hidden.ndim == 3
        assert hidden.shape[0] == 1
        assert hidden.shape[1] == len(tokens)

        capture.restore()


class TestMTPGeneration:
    """End-to-end generation tests."""

    def test_greedy_generation_matches_baseline(self, model_and_tokenizer, mtp_head):
        model, tokenizer = model_and_tokenizer
        from vllm_mlx_mtp.mtp_decoder import MTPDecoder, MTPConfig
        from mlx_lm.models.cache import make_prompt_cache

        decoder = MTPDecoder(model, mtp_head, MTPConfig(greedy_draft=True))

        tokens = tokenizer.encode("The capital of France is")
        prompt = mx.array(tokens)

        eos_set = set()
        if hasattr(tokenizer, "eos_token_id"):
            eid = tokenizer.eos_token_id
            if isinstance(eid, list):
                eos_set = set(eid)
            elif eid is not None:
                eos_set = {eid}

        # MTP generation
        cache = make_prompt_cache(model)
        mtp_tokens = list(decoder.generate(prompt, cache, max_tokens=30, temperature=0.0, eos_tokens=eos_set))
        decoder.cleanup()

        # Baseline generation
        cache2 = make_prompt_cache(model)
        logits = model(prompt[None], cache=cache2)
        mx.eval(logits)
        baseline_tokens = []
        for i in range(30):
            tok = mx.argmax(logits[:, -1, :], axis=-1)
            mx.eval(tok)
            tid = tok.item()
            baseline_tokens.append(tid)
            if tid in eos_set:
                break
            logits = model(tok.reshape(1, 1), cache=cache2)
            mx.eval(logits)

        assert mtp_tokens == baseline_tokens, (
            f"MTP output differs from baseline:\n"
            f"  MTP:  {tokenizer.decode(mtp_tokens)!r}\n"
            f"  Base: {tokenizer.decode(baseline_tokens)!r}"
        )

    def test_mtp_stats_tracked(self, model_and_tokenizer, mtp_head):
        model, tokenizer = model_and_tokenizer
        from vllm_mlx_mtp.mtp_decoder import MTPDecoder, MTPConfig
        from mlx_lm.models.cache import make_prompt_cache

        decoder = MTPDecoder(model, mtp_head, MTPConfig(greedy_draft=True))

        tokens = tokenizer.encode("Hello")
        cache = make_prompt_cache(model)

        for _ in decoder.generate(mx.array(tokens), cache, max_tokens=10):
            pass

        stats = decoder.stats
        decoder.cleanup()

        assert stats.total_tokens > 0
        assert stats.total_steps > 0
        assert stats.draft_attempts > 0
        assert 0.0 <= stats.acceptance_rate <= 1.0

    def test_acceptance_rate_reasonable(self, model_and_tokenizer, mtp_head):
        """MTP acceptance rate should be >50% on predictable text."""
        model, tokenizer = model_and_tokenizer
        from vllm_mlx_mtp.mtp_decoder import MTPDecoder, MTPConfig
        from mlx_lm.models.cache import make_prompt_cache

        decoder = MTPDecoder(model, mtp_head, MTPConfig(greedy_draft=True))

        tokens = tokenizer.encode("1 + 1 =")
        cache = make_prompt_cache(model)

        for _ in decoder.generate(mx.array(tokens), cache, max_tokens=50):
            pass

        decoder.cleanup()
        assert decoder.stats.acceptance_rate > 0.5, (
            f"Acceptance rate too low: {decoder.stats.acceptance_rate:.1%}"
        )
