"""Deterministic lexical relevance scoring for long-term memory.

Why this exists alongside semantic (embedding) search: deciding whether a
remembered fact is relevant enough to *inject into the prompt* is the
highest-stakes call in the memory system — get it wrong and the model
either states something it doesn't actually know (hallucination risk, if
an unrelated fact leaks in) or fails to use a fact it really does have
(if a relevant one is dropped). Aila Nano's own embeddings (see
vectordb/embedder.py) come from a small, imperfectly-trained model — fine
for loosely ranking broad search results, but not something to bet
"never leak an irrelevant memory" on: cosine similarity from an
undertrained encoder can spike for unrelated text by chance.

Word-overlap is boring, but it is exactly as reliable as its inputs —
no model in the loop, no training-quality dependency, trivially testable.
`memory.semantic_memory.get_relevant_memories` uses this (not raw
semantic score) to decide what actually gets injected; semantic search
is still used for the broader, non-injecting `search_memories`.
"""

from __future__ import annotations

import re

# Small, hand-picked stopword list — just enough to strip the words that
# dominate short questions/statements ("what is my name" / "the user's")
# without needing an external NLP dependency.
STOPWORDS: frozenset[str] = frozenset(
    {
        "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
        "do", "does", "did", "doing",
        "have", "has", "had", "having",
        "i", "me", "my", "mine", "myself",
        "you", "your", "yours", "yourself",
        "he", "him", "his", "himself",
        "she", "her", "hers", "herself",
        "it", "its", "itself",
        "we", "us", "our", "ours", "ourselves",
        "they", "them", "their", "theirs", "themselves",
        "this", "that", "these", "those",
        "what", "which", "who", "whom", "whose", "when", "where", "why", "how",
        "and", "or", "but", "if", "so", "because", "as",
        "of", "at", "by", "for", "with", "about", "against", "between",
        "into", "through", "during", "before", "after", "above", "below",
        "to", "from", "up", "down", "in", "out", "on", "off", "over", "under",
        "again", "further", "then", "once",
        "can", "could", "should", "would", "will", "shall", "may", "might", "must",
        "not", "no", "nor",
        "please", "just", "also", "very", "really",
        "am", "s", "t", "re", "ve", "ll", "d", "m",
    }
)

_TOKEN_RE = re.compile(r"[a-z0-9]+")

# Words that carry the *shape* of a question rather than its subject.
# They survive stopword removal (they're meaningful verbs) but two
# questions sharing only one of these are not about the same thing:
# "Who created Samsung?" and "Who created Hames Eventos" overlap only on
# "created", which is no evidence at all that one answers the other.
#
# This is not hypothetical — before `has_distinctive_overlap` existed,
# that exact pair scored 0.5 against a 0.45 serve threshold and Aila
# confidently answered a question about one company with a stored fact
# about a completely different one.
# Verbs of asking/creating. Shared *only* by these, two texts are not
# about the same thing — the subject words are what identify it.
GENERIC_VERB_TERMS: frozenset[str] = frozenset(
    {
        # English
        "created", "create", "made", "makes", "make", "founded", "found",
        "built", "build", "invented", "started", "began", "developed",
        "owns", "owned", "called", "named", "means", "meaning", "known",
        "tell", "say", "said", "explain", "describe", "define", "give", "get",
        "about", "used", "use", "uses", "work", "works",
        # Portuguese
        "criou", "criado", "fez", "feito", "fundou", "fundado", "construiu",
        "inventou", "chamado", "chama", "significa", "diga", "explique",
    }
)

# Everything above, PLUS generic role/entity nouns. Used only for the
# global knowledge base, where competing facts describe *different
# entities* and a shared "name"/"founder" is not evidence that a stored
# answer belongs to this question ("What is the name of Apple's founder?"
# vs "...Samsung's founder?" share both words and answer differently).
#
# User memory deliberately uses the smaller verb-only set instead: every
# memory belongs to the same person, so there is no entity ambiguity and
# attribute nouns like "name" or "job" ARE the subject being asked about
# ("What is my name?" must match "my name is Theo").
GENERIC_QUESTION_TERMS: frozenset[str] = GENERIC_VERB_TERMS | frozenset(
    {
        "creator", "creators", "maker", "founder", "founders", "builder",
        "inventor", "developer", "owner", "name", "names",
        "thing", "things", "stuff", "something", "anything", "everything",
        "people", "person", "company", "companies",
        "criador", "criadores", "fundador", "fundadores", "nome",
        "empresa", "empresas", "pessoa", "pessoas", "coisa", "coisas",
    }
)


def tokenize(text: str) -> set[str]:
    """Lowercase, strip punctuation, drop stopwords and single-character
    tokens (mostly apostrophe debris — "user's" -> "user", "s")."""
    words = _TOKEN_RE.findall(text.lower())
    return {w for w in words if len(w) >= 2 and w not in STOPWORDS}


def lexical_overlap_score(query: str, text: str) -> float:
    """Overlap coefficient (|intersection| / min(|A|, |B|)) between the
    significant-word sets of `query` and `text`. Deliberately the overlap
    coefficient rather than Jaccard: a short query ("What's my name?")
    should be able to fully match against a longer stored fact ("The
    user's name is Theo, and they work as a teacher.") without being
    penalized for the fact mentioning other things too.

    Returns 0.0 if either side has no significant words (e.g. a query
    that's all stopwords) — an empty match is never "relevant".
    """
    q_tokens = tokenize(query)
    t_tokens = tokenize(text)
    if not q_tokens or not t_tokens:
        return 0.0
    overlap = len(q_tokens & t_tokens)
    return overlap / min(len(q_tokens), len(t_tokens))


def has_distinctive_overlap(query: str, text: str, generic: frozenset[str] | None = None) -> bool:
    """True when the two texts share at least one term that actually
    identifies a subject — i.e. a shared word that isn't just a generic
    asking verb like "created"/"founded"/"named".

    Score alone is not enough to decide whether stored content answers a
    question. Two short questions about entirely different subjects can
    clear a high overlap threshold purely by sharing an asking verb, and
    serving on that basis produces a confidently wrong answer. Callers
    that *serve* content (rather than merely rank it) should require this
    in addition to a score threshold.

    `generic` defaults to the full GENERIC_QUESTION_TERMS set (correct
    for the global knowledge base). Pass GENERIC_VERB_TERMS for user
    memory, where attribute nouns are the subject rather than noise.
    """
    shared = tokenize(query) & tokenize(text)
    return bool(shared - (GENERIC_QUESTION_TERMS if generic is None else generic))
