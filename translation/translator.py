"""Thin, defensive wrapper around deep-translator (Google Translate).

Aila's own text is English: the model was pretrained on English, and the
hand-written deterministic replies are authored in English with a
Portuguese twin for each. This module lets a Portuguese question reach
the parts of Aila that only speak English — chiefly the far larger
*English* Wikipedia — and brings the answer back in Portuguese.

It is deliberately a *fallback*, not a wrapper around everything: Aila
already handles Portuguese greetings, identity, memory and maths
natively, and already searches Portuguese Wikipedia, all of which are
better in their native form than a round-trip through machine
translation. See `agents/base.py` for where this is invoked (only after
the native path has missed).

Every method degrades to a no-op that returns the original text:

- deep-translator not installed -> `available` is False, passthrough.
- network down / API error / timeout -> passthrough, logged, never
  raised. A translation failure must never turn into a failed turn; the
  worst case is the user sees the original-language text, which is
  exactly what they would have seen without this module at all.

Nothing here is on the offline-critical path: if the user is offline,
translation simply doesn't happen and the native pipeline still answers
from what Aila already knows.
"""

from __future__ import annotations

import logging
from functools import lru_cache

logger = logging.getLogger(__name__)

# Google Translate rejects very long inputs. Aila's messages and answers
# are short (a question, or a paragraph-long Wikipedia summary), so a
# generous single-request cap is simpler and safer than chunking, which
# would risk splitting mid-sentence and mistranslating.
MAX_TRANSLATE_CHARS = 4500

# Google occasionally answers with an HTML/text error page (rate limiting,
# a transient 500) that deep-translator hands back as an ordinary string.
# Serving that to the user as "the translation" would be worse than not
# translating at all, so a result matching any of these is treated as a
# failure and the original text is kept.
_ERROR_MARKERS = (
    "That’s an error",
    "That's an error",
    "Error 500",
    "Server Error",
    "<html",
    "Please try again later",
)


class TranslationUnavailable(RuntimeError):
    """Raised only by `Translator.require()`, for callers that want to
    fail loudly rather than silently passthrough. The normal methods
    never raise."""


class Translator:
    """Bidirectional en<->pt translation with graceful degradation.

    `enabled=False`, or deep-translator not being installed, both make
    every method an identity passthrough, so callers need no branching of
    their own.
    """

    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self._backend = None
        self.available = False
        if not enabled:
            return
        try:
            from deep_translator import GoogleTranslator

            self._backend = GoogleTranslator
            self.available = True
        except Exception as e:  # noqa: BLE001 — any import failure = unavailable
            logger.info("translation disabled: deep-translator not usable (%s)", e)

    # -- public API ---------------------------------------------------------

    def to_english(self, text: str) -> str:
        """Portuguese -> English. Passthrough on failure.

        Source is fixed to Portuguese rather than "auto": auto-detection
        goes through a different Google endpoint that returned an HTTP 500
        error *page as a string* in testing — which would then be served
        to the user as if it were the translation. Callers only invoke
        this on text already detected as Portuguese, so pt is correct."""
        return self._translate(text, source="pt", target="en")

    def to_portuguese(self, text: str) -> str:
        """English -> Portuguese. Passthrough on failure."""
        return self._translate(text, source="en", target="pt")

    def require(self) -> None:
        if not self.available:
            raise TranslationUnavailable(
                "Translation is unavailable. Install it with: pip install deep-translator"
            )

    # -- internals ----------------------------------------------------------

    def _translate(self, text: str, source: str, target: str) -> str:
        if not self.available or not text or not text.strip():
            return text
        if len(text) > MAX_TRANSLATE_CHARS:
            # Better to leave a very long text untranslated than to send a
            # truncated half to the user as if it were the whole answer.
            logger.info("skipping translation of %d-char text (over cap)", len(text))
            return text
        try:
            result = _cached_translate(self._backend, text, source, target)
        except Exception as e:  # noqa: BLE001 — never let translation break a turn
            logger.warning("translation failed (%s); using original text", type(e).__name__)
            return text
        # deep-translator can return None for input it can't handle, or a
        # Google error page as a plain string.
        if not isinstance(result, str) or not result.strip():
            return text
        if any(marker in result for marker in _ERROR_MARKERS):
            logger.warning("translation returned a service error page; using original text")
            return text
        return result


@lru_cache(maxsize=512)
def _cached_translate(backend, text: str, source: str, target: str) -> str:
    """Module-level so the cache is shared across Translator instances and
    survives per-turn construction. Keyed on (text, source, target); the
    backend class is a constant argument, present only so the cache never
    calls into a stale backend."""
    return backend(source=source, target=target).translate(text)
