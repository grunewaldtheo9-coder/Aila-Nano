from tokenizer.special_tokens import SPECIAL_TOKENS
from tokenizer.tokenizer import AilaTokenizer


def test_vocab_size_matches_training_request(tokenizer: AilaTokenizer):
    assert tokenizer.vocab_size == 384


def test_special_token_ids_are_stable(tokenizer: AilaTokenizer):
    assert tokenizer.pad_id == 0
    assert tokenizer.unk_id == 1
    assert tokenizer.bos_id == 2
    assert tokenizer.eos_id == 3
    for i, tok in enumerate(SPECIAL_TOKENS):
        assert tokenizer.piece_to_id(tok) == i


def test_roundtrip_encode_decode(tokenizer: AilaTokenizer):
    text = "Aila Nano is a small language model."
    ids = tokenizer.encode(text)
    assert tokenizer.decode(ids) == text


def test_bos_eos_flags(tokenizer: AilaTokenizer):
    ids = tokenizer.encode("hello", add_bos=True, add_eos=True)
    assert ids[0] == tokenizer.bos_id
    assert ids[-1] == tokenizer.eos_id


def test_decode_skips_special_tokens_by_default(tokenizer: AilaTokenizer):
    ids = [tokenizer.bos_id] + tokenizer.encode("hi") + [tokenizer.eos_id]
    decoded = tokenizer.decode(ids)
    assert "<s>" not in decoded and "</s>" not in decoded


def test_encode_batch(tokenizer: AilaTokenizer):
    batch = tokenizer.encode_batch(["hello", "world"])
    assert len(batch) == 2
    assert all(isinstance(x, list) for x in batch)


def test_byte_fallback_handles_arbitrary_unicode(tokenizer: AilaTokenizer):
    # Emoji / CJK aren't in the tiny training corpus, but byte_fallback=True
    # guarantees no <unk> is ever produced for valid UTF-8 input.
    text = "こんにちは 🎉"
    ids = tokenizer.encode(text)
    assert tokenizer.unk_id not in ids
    assert tokenizer.decode(ids) == text
