# Aila Nano 50M-v2 (bilingual, 16384 tokenizer) — experiment report

Fresh 55,587,328-param model (vocab 16384) trained from scratch on the
bilingual corpus `aila_pretrain_v2_bilingual` (62% PT / 38% EN by tokens).
BPC = bits per character (tokenizer-independent, the only fair cross-tokenizer
metric). Perplexity-per-token is NOT comparable across tokenizers.

## Baseline: old 50M (8192 tokenizer, EN-only pretrain + instruction finetune)
| | EN | PT |
|---|---|---|
| BPC | 1.5843 | 5.8512 |
| token ppl | 129.3079 | 2599.6685 |
_(EN not token-matched: the old model saw ~10x more English.)_

## New 50M-v2 by training-token budget (BPC, lower is better)
| Tokens seen | EN BPC | PT BPC | EN ppl | PT ppl |
|---|---|---|---|---|
| 307,200 | 2.4418 | 3.0045 | 1351.9962 | 2016.7291 |
| 602,112 | 2.2631 | 2.9181 | 797.7607 | 1620.5428 |
| 1,105,920 | 2.0673 | 2.7813 | 447.5275 | 1145.9383 |
