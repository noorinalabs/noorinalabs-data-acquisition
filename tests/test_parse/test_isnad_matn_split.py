"""Tests for the da#366 matn-embedded isnad splitter.

The splitter recovers an isnad embedded in the matn text of sanadset's
"No SANAD" rows, across two conventions, WITHOUT the NER ``full_text`` fallback
that mints matn sentences as narrators. Precision is paramount: the tests below
lock both the positive recoveries and — just as important — the negative and
fail-closed cases that keep matn out of the narrator index.

No real corpus data ships in the repo (``data/raw`` is empty), so these fixtures
reproduce both conventions plus the negative (pure-matn), bare-opener, and
no-boundary classes. The full-corpus recovery magnitude (~122,000) is a re-run
measurement, not a CI assertion.
"""

from __future__ import annotations

import pytest

from src.parse.isnad_matn_split import detect_isnad, split_isnad_matn
from src.parse.narrator_extraction import extract_narrator_mentions
from src.utils.arabic import extract_transmission_phrases

# --- positive recoveries --------------------------------------------------


def test_opener_and_an_chain_is_both() -> None:
    text = "حدثنا مالك عن نافع عن ابن عمر أن رسول الله صلى الله عليه وسلم نهى عن بيع الغرر"
    result = split_isnad_matn(text)
    assert result is not None
    assert result.convention == "both"
    # The isnad head is isolated; the ``أن …`` matn — including its own ``عن`` —
    # is NOT pulled into the chain.
    assert result.isnad_ar == "حدثنا مالك عن نافع عن ابن عمر"
    assert result.matn_ar.startswith("أن رسول الله")
    assert [s.name for s in result.spans] == extract_names(result.isnad_ar)
    assert len(result.spans) == 3


def test_bare_name_an_chain_convention() -> None:
    # tusi/sanadset shape: a bare name, then ``عن`` links — no receipt-verb opener.
    text = "علي بن إبراهيم عن أبيه عن ابن أبي عمير عن أبي عبد الله قال إذا صلى أحدكم فليتم"
    result = split_isnad_matn(text)
    assert result is not None
    assert result.convention == "an_chain"
    assert result.isnad_ar == "علي بن إبراهيم عن أبيه عن ابن أبي عمير عن أبي عبد الله"
    assert result.matn_ar.startswith("قال")
    # The leading bare name is recovered as the first narrator.
    assert len(result.spans) == 4


def test_qala_is_a_link_not_a_boundary_when_opener_follows() -> None:
    # ``حدثنا مالك قال حدثنا نافع`` — the first ``قال`` connects two receipt
    # verbs and must NOT cut the isnad; the boundary is the later ``أن``.
    text = "حدثنا مالك قال حدثنا نافع عن ابن عمر أن النبي قال الطهور شطر الإيمان"
    result = split_isnad_matn(text)
    assert result is not None
    assert result.isnad_ar == "حدثنا مالك قال حدثنا نافع عن ابن عمر"
    assert result.matn_ar.startswith("أن النبي")
    assert [s.name for s in result.spans] == ["مالك", "نافع", "ابن عمر"]


# --- negatives: the detector must not fire --------------------------------


@pytest.mark.parametrize(
    ("label", "text"),
    [
        ("pure matn", "إنما الأعمال بالنيات وإنما لكل امرئ ما نوى"),
        ("qala-head matn", "قال النبي من كذب علي متعمدا فليتبوأ مقعده من النار"),
        ("single-an prose", "باب ما جاء عن النبي في الصلاة"),
    ],
)
def test_non_isnad_matn_recovers_nothing(label: str, text: str) -> None:
    assert split_isnad_matn(text) is None, label


def test_empty_and_blank_recover_nothing() -> None:
    assert split_isnad_matn(None) is None
    assert split_isnad_matn("") is None
    assert split_isnad_matn("   ") is None


# --- fail-closed: detected but unsafe to split ----------------------------


def test_bare_opener_introducing_speech_recovers_nothing() -> None:
    # ``حدثنا : "…"`` — an opener with no narrator named before the matn. The
    # boundary is the colon, the isolated head is just the opener, and it
    # segments to zero narrators, so nothing is recovered (da#366 §"bare openers").
    text = 'حدثنا : " من لا يرحم الناس لا يرحمه الله "'
    assert split_isnad_matn(text) is None


def test_no_matn_boundary_marker_fails_closed() -> None:
    # A clear isnad, but with no ``قال``/``أن``/punctuation to mark where the matn
    # begins. Rather than guess (and risk absorbing matn into the last narrator),
    # the splitter fails closed — the accepted lower-bound behaviour.
    text = "حدثنا مالك عن نافع عن ابن عمر رضي الله عنه"
    assert split_isnad_matn(text) is None


# --- the whole point: no matn pollution -----------------------------------


def test_splitter_does_not_pollute_like_the_ner_fallback() -> None:
    # Feeding the WHOLE body to the segmenter (the da#369 fallback) glues matn
    # onto the last narrator and mints a matn fragment as a narrator. The
    # splitter isolates the isnad head first and yields only the real chain.
    text = "حدثنا مالك عن نافع عن ابن عمر أن رسول الله صلى الله عليه وسلم نهى عن بيع الحصاة"
    result = split_isnad_matn(text)
    assert result is not None

    naive = extract_narrator_mentions(text, "ar")
    assert len(result.spans) < len(naive)
    # No recovered span carries matn: none contains the Prophet-formula token
    # ``رسول`` that the naive blob absorbs.
    assert all("رسول" not in s.name for s in result.spans)
    assert any("رسول" in s.name for s in naive)


# --- detector unit behaviour ----------------------------------------------


def test_detector_requires_two_an_without_an_opener() -> None:
    one_an = "فلان بن فلان عن النبي في شيء ما"
    assert detect_isnad(one_an, extract_transmission_phrases(one_an)) is None

    two_an = "فلان عن علان عن فلان قال شيء"
    assert detect_isnad(two_an, extract_transmission_phrases(two_an)) == "an_chain"


def test_an_beyond_the_token_window_does_not_count() -> None:
    # Two ``عن`` links, but only after 26 filler tokens — past the 25-token
    # window — and no opener, so the detector does not fire.
    text = " ".join(["حرف"] * 26) + " عن أ عن ب قال ج"
    assert detect_isnad(text, extract_transmission_phrases(text)) is None


# --- both-direction boundary fidelity -------------------------------------


@pytest.mark.parametrize(
    ("isnad", "matn"),
    [
        (
            "حدثنا مالك عن نافع عن ابن عمر",
            "أن رسول الله صلى الله عليه وسلم نهى عن بيع الحصاة",
        ),
        (
            "علي بن إبراهيم عن أبيه عن حماد عن أبي عبد الله",
            "قال الصلاة عماد الدين",
        ),
    ],
)
def test_boundary_is_faithful_both_directions(isnad: str, matn: str) -> None:
    # Concatenate a known isnad and a known matn (the shape of an untagged row),
    # then require the splitter to reconstruct BOTH sides exactly: the recovered
    # isnad's narrator set equals the isnad-only segmentation (nothing dropped),
    # and the residual matn is the original matn (nothing of the matn pulled in).
    result = split_isnad_matn(f"{isnad} {matn}")
    assert result is not None
    assert result.isnad_ar == isnad
    assert result.matn_ar == matn
    assert [s.name for s in result.spans] == extract_names(isnad)


# --- helpers --------------------------------------------------------------


def extract_names(isnad_text: str) -> list[str]:
    """Oracle: the narrator names the shared segmenter yields for an isnad-only text."""
    return [s.name for s in extract_narrator_mentions(isnad_text, "ar")]
