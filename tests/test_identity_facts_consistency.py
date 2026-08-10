"""Aila's identity facts are stated in two places and must agree.

- `tools/identity.py` serves them deterministically at inference time.
- `datasets/scripts/generate_aila_identity_data.py` teaches them to the
  model itself.

If they drift, Aila answers "who created you?" one way from the table and
a different way whenever generation handles a phrasing the table doesn't
cover — which is exactly the kind of quiet inconsistency that makes a
model look unreliable. This test is the tripwire.

It also pins the parameter count to the *measured* size of the shipped
architecture, so nobody has to remember to update prose after changing
model/config.py. (That check caught a real drift: the identity data still
claimed 10.9M parameters after the model grew to ~19.8M.)
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from model.config import nano_20m
from model.transformer import AilaNanoGPT
from tools.identity import (
    COMPANY,
    FOUNDERS,
    FOUNDERS_PT,
    FULL_NAME,
    RELEASE_STAGE,
    _ANSWERS_EN,
    _ANSWERS_PT,
)

DATASET = Path(__file__).resolve().parent.parent / "datasets" / "aila_knowledge" / "aila_company.jsonl"


def _dataset_outputs() -> list[str]:
    with open(DATASET, encoding="utf-8") as f:
        return [json.loads(line)["output"] for line in f if line.strip()]


def test_company_and_founder_names_match_the_training_data():
    outputs = _dataset_outputs()
    assert any(COMPANY in o for o in outputs), f"{COMPANY!r} never appears in the identity data"
    assert any(FOUNDERS in o for o in outputs), f"{FOUNDERS!r} never appears in the identity data"

    # Wherever the data attributes the company to its founders, it must
    # name both of them. (Statements about one founder individually —
    # "Theo Grunewald Hames is one of the two co-founders" — are fine and
    # deliberately excluded.)
    first, second = FOUNDERS.split(" and ")
    for output in outputs:
        if "founded by" not in output.lower() or COMPANY not in output:
            continue
        assert first in output and second in output, (
            f"identity data states different founders: {output!r}"
        )


def test_served_answers_state_the_real_company_and_founders():
    for answers in (_ANSWERS_EN, _ANSWERS_PT):
        assert COMPANY in answers["creator"]
    assert FOUNDERS in _ANSWERS_EN["creator"]
    assert FOUNDERS in _ANSWERS_EN["company_founders"]
    # Portuguese joins the two names with "e", not "and" — interpolating
    # the English string produced "fundada por Theo ... and Guilherme ...",
    # right facts in the wrong language.
    assert FOUNDERS_PT in _ANSWERS_PT["creator"]
    assert FOUNDERS_PT in _ANSWERS_PT["company_founders"]
    assert " and " not in _ANSWERS_PT["company_founders"]


def test_the_public_name_is_the_release_stage_not_a_version_number():
    """The product is "Aila Nano Beta". A version number claimed a
    finished edition; this is the first release anyone outside the
    project has used."""
    import chat

    assert FULL_NAME == f"Aila Nano {RELEASE_STAGE}"
    assert RELEASE_STAGE == "Beta"
    # chat.py's banner and support report must not invent their own name.
    assert chat.VERSION == RELEASE_STAGE

    # Aila introduces herself by the full public name.
    for answers in (_ANSWERS_EN, _ANSWERS_PT):
        assert FULL_NAME in answers["what_are_you"]
        assert FULL_NAME in answers["name"]

    # ...and the training data teaches the same name.
    assert any(FULL_NAME in output for output in _dataset_outputs())


def test_no_version_number_is_presented_as_the_product_name():
    import chat

    for answers in (_ANSWERS_EN, _ANSWERS_PT):
        for text in answers.values():
            for stale in ("Aila Nano 2.0", "Aila Nano 2.1", "Aila Nano v2"):
                assert stale not in text
    assert "2.0" not in chat.VERSION


def test_stated_parameter_count_matches_the_measured_architecture():
    """Both the served answers and the training data say "about 20
    million"; the architecture must actually be that size."""
    measured = sum(p.numel() for p in AilaNanoGPT(nano_20m()).parameters())
    # "about 20 million" is honest for anything that rounds to 20M.
    assert 19_000_000 <= measured < 21_000_000, (
        f"model is {measured:,} parameters but the identity facts say 'about 20 million'"
    )

    claim = re.compile(r"(\d+(?:\.\d+)?)\s*(?:million|milh[õo]es)", re.IGNORECASE)
    stated = {m for a in _ANSWERS_EN.values() for m in claim.findall(a)}
    stated |= {m for a in _ANSWERS_PT.values() for m in claim.findall(a)}
    stated |= {m for o in _dataset_outputs() for m in claim.findall(o)}
    assert stated == {"20"}, f"conflicting parameter counts stated: {sorted(stated)}"
