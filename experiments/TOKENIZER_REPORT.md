# Tokenizer benchmark (EN + PT, held-out)

Evaluated on held-out text that never entered training.

| Tokenizer | Vocab | EN tok/word | PT tok/word | PT byte-fallback | PT word-split | Accents in-vocab |
|---|---|---|---|---|---|---|
| aila_nano | 8192 | 1.1769 | 2.8495 | 0.09576 | 1.0 | 0.0909 |
| bilingual_12288 | 12288 | 1.2051 | 1.4691 | 5e-05 | 0.4667 | 1.0 |
| bilingual_16384 | 16384 | 1.1904 | 1.4139 | 5e-05 | 0.4667 | 1.0 |
| bilingual_8192 | 8192 | 1.2379 | 1.5659 | 5e-05 | 0.5333 | 1.0 |

**Recommended (measured): `bilingual_16384`** (score 4.0185, lower is better; PT weighted 2×, EN 1×, byte-fallback penalized).
