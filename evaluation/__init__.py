"""Aila Nano evaluation harness.

Separates *language* capability (English / Portuguese perplexity + generation
quality) from *Aila-specific* behaviour (identity, routing), which is handled
by the deterministic conversation infrastructure and its own tests. Nothing
here trains or fine-tunes; it measures a checkpoint.
"""
