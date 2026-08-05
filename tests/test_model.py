import pytest
import torch

from model.config import GPTConfig, nano_10m
from model.generate import generate
from model.transformer import AilaNanoGPT
from model.utils import count_parameters


def test_nano_10m_preset_is_close_to_target():
    model = AilaNanoGPT(nano_10m())
    n_params = count_parameters(model)
    assert 10_600_000 <= n_params <= 11_100_000, f"nano_10m has {n_params} params, expected ~10.9M"


def test_config_rejects_incompatible_dims():
    with pytest.raises(ValueError):
        GPTConfig(d_model=33, n_heads=8)
    with pytest.raises(ValueError):
        GPTConfig(n_heads=8, n_kv_heads=3)


def test_tied_embeddings_share_storage(tiny_config):
    model = AilaNanoGPT(tiny_config)
    assert model.lm_head.weight.data_ptr() == model.token_emb.weight.data_ptr()


def test_forward_shapes_and_loss(tiny_model, tiny_config):
    x = torch.randint(0, tiny_config.vocab_size, (2, 16))
    logits, loss = tiny_model(x, targets=x)
    assert logits.shape == (2, 16, tiny_config.vocab_size)
    assert loss is not None and loss.item() > 0


def test_forward_without_targets_returns_no_loss(tiny_model, tiny_config):
    x = torch.randint(0, tiny_config.vocab_size, (1, 8))
    _, loss = tiny_model(x)
    assert loss is None


def test_forward_rejects_overlong_sequence(tiny_model, tiny_config):
    x = torch.randint(0, tiny_config.vocab_size, (1, tiny_config.max_seq_len + 1))
    with pytest.raises(ValueError):
        tiny_model(x)


def test_kv_cache_matches_full_forward(tiny_model, tiny_config):
    torch.manual_seed(0)
    seq = torch.randint(0, tiny_config.vocab_size, (1, 10))
    tiny_model.eval()

    with torch.no_grad():
        full_logits, _ = tiny_model(seq)

        kv_caches = tiny_model.new_kv_caches()
        chunks = []
        for t in range(seq.shape[1]):
            logits, _ = tiny_model(seq[:, t : t + 1], kv_caches=kv_caches)
            chunks.append(logits)
        cached_logits = torch.cat(chunks, dim=1)

    assert torch.allclose(full_logits, cached_logits, atol=1e-4)


def test_generate_respects_max_new_tokens(tiny_model, tiny_config):
    prompt = torch.randint(0, tiny_config.vocab_size, (1, 4))
    out = generate(tiny_model, prompt, max_new_tokens=10, eos_id=None)
    assert out.shape[1] == prompt.shape[1] + 10


def test_generate_never_exceeds_max_seq_len(tiny_model, tiny_config):
    prompt = torch.randint(0, tiny_config.vocab_size, (1, tiny_config.max_seq_len - 3))
    out = generate(tiny_model, prompt, max_new_tokens=100, eos_id=None)
    assert out.shape[1] <= tiny_config.max_seq_len


def test_generate_stops_at_eos(tiny_config):
    # An untrained random model has no reason to ever sample a specific
    # token id, so instead of hoping it happens to emit eos, we bias the
    # (tied) output embedding row for eos_id so it dominates every
    # argmax — then greedy decoding must stop after exactly one new token.
    model = AilaNanoGPT(tiny_config)
    eos_id = 3
    with torch.no_grad():
        model.token_emb.weight[eos_id] += 100.0

    prompt = torch.randint(0, tiny_config.vocab_size, (1, 3))
    out = generate(model, prompt, max_new_tokens=50, eos_id=eos_id, temperature=0.0)

    generated = out[0, prompt.shape[1] :].tolist()
    assert generated[0] == eos_id
    # generation must stop right after emitting eos, not run to max_new_tokens
    assert len(generated) < 50


def test_num_parameters_excluding_embeddings(tiny_model):
    total = tiny_model.num_parameters()
    non_embed = tiny_model.num_parameters(exclude_embeddings=True)
    assert non_embed < total
    assert non_embed == total - tiny_model.token_emb.weight.numel()
