"""Peel confidently-identifiable mentions off an over-merged bare name by isnad context (da#346).

The problem this addresses (why ``narrator_split`` could not)
-------------------------------------------------------------
The bare ism ``عبد الله`` collapses ~19k mentions of *many* historically-distinct ʿAbd
Allāhs (Ibn ʿUmar, Ibn ʿAbbās, Ibn Masʿūd, …) onto one ``nar:`` node — the classic
over-merge that inflates betweenness (da#346). The date-axis splitter
(:mod:`src.resolve.narrator_split`) **correctly abstains** here: a bare mention carries
no *attested* death-year neighbour band to discriminate on, so there is nothing for the
death-band evidence gate to cut (Gate 1, single band ⇒ abstain). The #337 spike proved
that forcing a date-band split of such a node is both infeasible (the chimera does not
separate) and unsafe (it false-splits genuine hubs — al-Zuhrī peeled into five fabricated
impossible-date nodes). That negative result is why #337 was resolved *flag-now*
(:mod:`src.resolve.over_merged_flag`) and a true node-identity split deferred to external
rijāl evidence (da#443).

The corpus-internal lever that *is* available
----------------------------------------------
The **isnad position** — who a mention transmits *from* (teacher, position −1) and *to*
(student, position +1) in its own chain — is a discriminating signal even when the name
string is bare. An ʿAbd Allāh who narrates ⟵ the Prophet and ⟶ Nāfiʿ is ʿAbd Allāh ibn
ʿUmar, not any other ʿAbd Allāh. This stage peels *those* mentions — the ones whose ±1
neighbour pair matches a **hand-verified, high-confidence signature** — onto a distinct
discriminated canonical node, and leaves every unmatched mention on the bare primary
(which :mod:`over_merged_flag` keeps flagged). It never guesses.

Why a curated signature seed and not a threshold (the da#423 discipline)
-----------------------------------------------------------------------
There is **no corpus-internal threshold** that separates the referents of a bare name
(the #337 §0 negative result): a purely statistical auto-split re-commits the silent-zero
error, false-splitting a genuine transmitter. So — exactly like the ``over_merged``
flag's curated list — the peel is driven by a **hand-verified signature seed**
(:data:`_SEED_PATH`) under a **bidirectional acceptance fixture**
(``tests/test_resolve/test_contextual_disambiguation.py``): a signature MUST fire on the
genuine multi-referent mentions it names AND MUST NOT fire on a genuine single narrator
whose neighbours do not match it. A mention whose neighbour pair matches *no* signature —
or *more than one* (ambiguous) — is never peeled; it stays on the flagged primary and its
true identity is deferred to the external-rijāl work (da#443). The seed may be empty: then
this stage is a pure no-op, and the whole node remains flagged — the safe default.

What this stage does NOT do
---------------------------
It mints **no** speculative date-split nodes (that is the abstaining
:mod:`narrator_split`'s job, and it abstains here). It invents **no** identity for a bare
mention that lacks a corpus-internal discriminating neighbour pair — that residual is the
``over_merged`` flag's to disclose and da#443's to resolve with external evidence. The
peel is confined to mentions that carry, *in the isnad itself*, a neighbour pair a human
has verified is uniquely identifying.

Algorithm (per curated signature ``S`` on bare name ``N``)
----------------------------------------------------------
1. Resolve ``N`` to its canonical id ``C = make_canonical_id(N)`` and gather every
   mention with ``canonical_narrator_id == C``.
2. For each such mention, read its **consecutive-resolved** teacher/student neighbours
   within the same ``(hadith_id, chain_index)`` chain — the SHARED adjacency helper
   :func:`src.resolve.narrator_split.resolved_chain_neighbours` (da#439), identical to
   the loader's ``TRANSMITTED_TO`` adjacency and the date-band splitter's, so a gappy
   isnad cannot silently erase a discriminating pair (da#411 — never a cross-chain
   neighbour) — and map them to their normalized names.
3. A signature *matches* a mention iff every neighbour role it constrains is satisfied —
   its ``teacher_ar`` (if given) is among the mention's teacher names AND its
   ``student_ar`` (if given) among the student names. A mention matched by exactly one
   signature peels to that signature's target; a mention matched by zero or by ≥2
   signatures is **left on the primary** (the conservative residual).
4. The target node id is ``make_discriminated_canonical_id(N, S.discriminator)`` — a
   distinct ``nar:`` id sharing the bare display name (mirroring ``narrator_split``'s
   peeled rows), carrying ``S.name_en`` + ``S.note`` as its disambiguation provenance.

Outputs (mirroring ``narrator_split``)
---------------------------------------
* ``narrators_canonical.parquet`` rewritten with the peeled target rows appended (each
  primary's ``mention_count`` reduced by its peeled total, attestation re-derived).
* ``narrator_mentions_resolved.parquet`` — the peeled mentions' ``canonical_narrator_id``
  remapped, keyed on ``mention_id`` (a split sends *different* mentions of one id to
  *different* targets), streaming/idempotent, reusing
  :func:`src.resolve.narrator_split._remap_split_mentions`.
* ``contextual_splits.parquet`` — one row per peeled target (bare name, new id,
  discriminator, matched signature, mention_count): the owner's review artifact.
* ``contextual_coverage.parquet`` — the **blast-radius** report (da#346 acceptance): per
  curated bare name, how many of its mentions have a resolvable neighbour pair and how
  many matched a signature. The real corpus number is produced when this stage runs
  against the staged artifact; :func:`neighbour_pair_coverage` computes it deterministically.

Idempotence
-----------
A second run re-reads the already-peeled table. Peeled target ids no longer carry the
bare name ``N``, so they are not re-gathered under ``C``; the residual on ``C`` matched no
signature the first time and still matches none. No file is rewritten; the stage returns
``None``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pyarrow as pa
import pyarrow.parquet as pq
import yaml

from src.parse.identity import make_canonical_id, make_discriminated_canonical_id
from src.resolve._run_record import write_canonical
from src.resolve.attestation import derive_attestation
from src.resolve.narrator_split import (
    _build_chain_index,
    _remap_split_mentions,
    resolved_chain_neighbours,
)
from src.resolve.schemas import NARRATORS_CANONICAL_SCHEMA
from src.utils.arabic import normalize_arabic
from src.utils.logging import get_logger

if TYPE_CHECKING:
    from collections.abc import Iterable

logger = get_logger(__name__)

# The curated signature seed ships beside this module.
_SEED_PATH = Path(__file__).with_name("contextual_signatures.yaml")

__all__ = [
    "CONTEXTUAL_SPLITS_SCHEMA",
    "CONTEXTUAL_COVERAGE_SCHEMA",
    "ContextualSignature",
    "NameCoverage",
    "load_contextual_signatures",
    "match_signature",
    "neighbour_pair_coverage",
    "apply_contextual_disambiguation",
]


# Audit report: one row per peeled (contextually-disambiguated) target node.
CONTEXTUAL_SPLITS_SCHEMA = pa.schema(
    [
        pa.field("name_ar_normalized", pa.string(), nullable=False),
        pa.field("primary_id", pa.string(), nullable=False),
        pa.field("new_id", pa.string(), nullable=False),
        pa.field("discriminator", pa.string(), nullable=False),
        pa.field("target_name_en", pa.string(), nullable=True),
        pa.field("teacher_ar", pa.string(), nullable=True),
        pa.field("student_ar", pa.string(), nullable=True),
        pa.field("mention_count", pa.int32(), nullable=False),
    ]
)

# Blast-radius report (da#346 acceptance): one row per curated bare name.
CONTEXTUAL_COVERAGE_SCHEMA = pa.schema(
    [
        pa.field("name_ar_normalized", pa.string(), nullable=False),
        pa.field("primary_id", pa.string(), nullable=False),
        pa.field("total_mentions", pa.int32(), nullable=False),
        pa.field("with_any_neighbour", pa.int32(), nullable=False),
        pa.field("with_teacher", pa.int32(), nullable=False),
        pa.field("with_student", pa.int32(), nullable=False),
        pa.field("signature_matched", pa.int32(), nullable=False),
        pa.field("ambiguous_multimatch", pa.int32(), nullable=False),
    ]
)


@dataclass(frozen=True)
class ContextualSignature:
    """One hand-verified isnad-neighbour signature identifying a referent of a bare name.

    ``name_ar`` is the bare over-merged name whose mentions this signature peels;
    ``teacher_ar`` / ``student_ar`` are the discriminating ±1 neighbour names (at least
    one required — a signature that constrains neither would match everything). A mention
    matches iff EVERY specified role is satisfied. ``discriminator`` folds into
    :func:`make_discriminated_canonical_id` to mint the target's distinct id; ``name_en``
    and ``note`` carry the (hand-verified) disambiguation provenance onto the target row.
    """

    name_ar: str
    teacher_ar: str | None
    student_ar: str | None
    discriminator: str
    name_en: str | None
    note: str

    def primary_id(self) -> str:
        """The bare canonical id whose mentions this signature peels from."""
        return make_canonical_id(normalize_arabic(self.name_ar))

    def target_id(self) -> str:
        """The distinct discriminated id a matched mention peels onto."""
        return make_discriminated_canonical_id(normalize_arabic(self.name_ar), self.discriminator)

    def teacher_norm(self) -> str | None:
        return normalize_arabic(self.teacher_ar) if self.teacher_ar else None

    def student_norm(self) -> str | None:
        return normalize_arabic(self.student_ar) if self.student_ar else None

    def matches(self, teacher_names: frozenset[str], student_names: frozenset[str]) -> bool:
        """True iff every neighbour role this signature constrains is present in the mention.

        A constrained role that the mention lacks fails the match; an unconstrained role
        (``None``) is ignored. Because at least one role is always constrained (validated
        at load), an all-``None`` signature can never be constructed, so ``matches`` never
        returns ``True`` on an empty neighbour context.
        """
        t, s = self.teacher_norm(), self.student_norm()
        if t is not None and t not in teacher_names:
            return False
        return not (s is not None and s not in student_names)


@dataclass(frozen=True)
class NameCoverage:
    """Blast-radius tally for one bare name's mentions (da#346 acceptance)."""

    name_ar_normalized: str
    primary_id: str
    total_mentions: int
    with_any_neighbour: int
    with_teacher: int
    with_student: int
    signature_matched: int
    ambiguous_multimatch: int


def load_contextual_signatures(path: Path = _SEED_PATH) -> tuple[ContextualSignature, ...]:
    """Parse the curated ``contextual_signatures.yaml`` seed (empty when absent).

    Raises ``ValueError`` on a malformed entry — a signature that constrains neither a
    teacher nor a student would match every mention of the bare name and mass-mislabel it,
    exactly the silent-defect this hand-verified list exists to prevent, so it is a hard
    error, never a silently-dropped row.
    """
    if not path.exists():
        return ()
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    rows = raw.get("signatures", []) if isinstance(raw, dict) else raw
    sigs: list[ContextualSignature] = []
    for row in rows:
        name_ar = row.get("name_ar")
        teacher_ar = row.get("teacher_ar")
        student_ar = row.get("student_ar")
        discriminator = row.get("discriminator")
        note = row.get("note")
        if not name_ar:
            raise ValueError("contextual signature entry needs a name_ar")
        if not teacher_ar and not student_ar:
            raise ValueError(
                f"contextual signature for {name_ar!r} needs a teacher_ar and/or student_ar "
                "(an unconstrained signature would match every mention)"
            )
        if not discriminator:
            raise ValueError(f"contextual signature for {name_ar!r} is missing its discriminator")
        if not note:
            raise ValueError(f"contextual signature for {name_ar!r} is missing its note")
        sigs.append(
            ContextualSignature(
                name_ar=name_ar,
                teacher_ar=teacher_ar,
                student_ar=student_ar,
                discriminator=discriminator,
                name_en=row.get("name_en"),
                note=note,
            )
        )
    return tuple(sigs)


def match_signature(
    teacher_names: frozenset[str],
    student_names: frozenset[str],
    signatures: Iterable[ContextualSignature],
) -> ContextualSignature | None:
    """The single signature that confidently matches this neighbour context, else ``None``.

    Zero matches (unknown context) OR ≥2 matches (ambiguous — the pair is consistent with
    more than one curated referent) both return ``None``: the mention is left on the bare
    primary. Only an unambiguous single match peels. This is the over-split guard — a
    mention is disambiguated only when the corpus-internal evidence points to exactly one
    hand-verified referent.
    """
    matched = [s for s in signatures if s.matches(teacher_names, student_names)]
    return matched[0] if len(matched) == 1 else None


def _neighbour_names_by_mention(
    mentions: list[tuple[str, int, int, str]],
    chains: dict[tuple[str, int], list[tuple[int, str]]],
    name_by_id: dict[str, str],
) -> dict[str, tuple[frozenset[str], frozenset[str]]]:
    """``mention_id -> (teacher_names, student_names)`` from the consecutive-resolved neighbours.

    Uses the **shared** isnad-adjacency helper
    (:func:`src.resolve.narrator_split.resolved_chain_neighbours`, da#439) so this stage's
    notion of teacher/student is byte-identical to the date-band splitter's and to the
    loader's ``TRANSMITTED_TO`` adjacency — the immediately-preceding / -following
    **resolved** mention in the position-sorted ``(hadith_id, chain_index)`` chain (a
    dropped-position gap bridged as the loader bridges it; self-loop dropped, not bridged
    past; never a cross-isnad neighbour, da#411). Reading exact ``position ± 1`` here (the
    pre-da#439 form) would have silently erased a golden-chain teacher/student pair on any
    gappy isnad — under-peeling a confident identity. Names are the neighbour ids'
    ``name_ar_normalized``; each role is a singleton or empty set.
    """
    out: dict[str, tuple[frozenset[str], frozenset[str]]] = {}
    for hadith_id, chain_index, position, mention_id in mentions:
        teacher_id, student_id = resolved_chain_neighbours(chains, hadith_id, chain_index, position)
        teacher_name = name_by_id.get(teacher_id) if teacher_id else None
        student_name = name_by_id.get(student_id) if student_id else None
        out[mention_id] = (
            frozenset({teacher_name} if teacher_name else ()),
            frozenset({student_name} if student_name else ()),
        )
    return out


def neighbour_pair_coverage(
    name_ar_normalized: str,
    primary_id: str,
    neighbours_by_mention: dict[str, tuple[frozenset[str], frozenset[str]]],
    signatures: Iterable[ContextualSignature],
) -> NameCoverage:
    """Blast-radius tally for one bare name (da#346 acceptance), pure + deterministic.

    Counts, over the name's mentions: how many have any resolvable neighbour, a teacher, a
    student, an unambiguous signature match, and an ambiguous ≥2-signature match. The
    corpus-scale numbers are produced when the stage runs against the staged artifact; this
    function is what computes them and is unit-tested on fixtures.
    """
    sigs = tuple(signatures)
    total = len(neighbours_by_mention)
    any_n = teach = stud = matched = ambiguous = 0
    for teachers, students in neighbours_by_mention.values():
        if teachers or students:
            any_n += 1
        if teachers:
            teach += 1
        if students:
            stud += 1
        hits = [s for s in sigs if s.matches(teachers, students)]
        if len(hits) == 1:
            matched += 1
        elif len(hits) >= 2:
            ambiguous += 1
    return NameCoverage(
        name_ar_normalized=name_ar_normalized,
        primary_id=primary_id,
        total_mentions=total,
        with_any_neighbour=any_n,
        with_teacher=teach,
        with_student=stud,
        signature_matched=matched,
        ambiguous_multimatch=ambiguous,
    )


def _as_int(value: Any) -> int:
    """Best-effort non-negative int for a mention_count cell (None/non-numeric → 0)."""
    if isinstance(value, bool):
        return 0
    if isinstance(value, int):
        return value
    return 0


def _target_record(
    primary_rec: dict[str, Any], sig: ContextualSignature, count: int
) -> dict[str, Any]:
    """A new canonical row for a contextually-disambiguated target node.

    Inherits the shared name/provenance fields from the bare primary (the display name
    stays the bare ``name_ar`` — we peel identity, not the string a mention shows),
    carries ``sig.name_en`` + ``sig.note`` as the disambiguation provenance, and its
    attestation is re-derived from the peeled count (the mentions are real isnad edges).
    Dates stay unknown — this stage asserts identity from context, never a death year.
    """
    rec: dict[str, Any] = dict.fromkeys(f.name for f in NARRATORS_CANONICAL_SCHEMA)
    for col in (
        "name_ar",
        "name_ar_normalized",
        "aliases",
        "gender",
        "trustworthiness",
        "source_ids",
        "source_corpus",
        "source_corpora",
        "sect_affiliation",
    ):
        rec[col] = primary_rec.get(col)
    rec["canonical_id"] = sig.target_id()
    if sig.name_en:
        rec["name_en"] = sig.name_en
    rec["mention_count"] = count
    rec["attestation"] = derive_attestation(count)
    rec["over_merged"] = False
    rec["over_merge_note"] = None
    return rec


def apply_contextual_disambiguation(
    output_dir: Path, *, seed_path: Path = _SEED_PATH, staging_dir: Path | None = None
) -> Path | None:
    """Peel confidently-identifiable mentions off curated over-merged bare names (da#346).

    Reads ``narrators_canonical.parquet`` + ``narrator_mentions_resolved.parquet`` from
    ``output_dir``. For every curated signature, gathers its bare name's mentions, matches
    each mention's ±1 neighbour pair against the signatures, and remaps unambiguously-matched
    mentions onto discriminated target nodes (appending target rows, reducing the primary's
    count). Always writes ``contextual_coverage.parquet`` (the blast-radius report) when
    there is a canonical table and ≥1 signature.

    Returns the canonical path when at least one mention peeled (files rewritten), else
    ``None`` (no signatures, missing inputs, empty table, or every mention was
    unknown/ambiguous — including every idempotent re-run). ``staging_dir`` is accepted for
    run_all call-shape parity; this stage keeps no checkpoint.
    """
    canonical_path = output_dir / "narrators_canonical.parquet"
    mentions_path = output_dir / "narrator_mentions_resolved.parquet"
    if not canonical_path.exists():
        logger.warning("contextual_disambiguation_no_canonical", path=str(canonical_path))
        return None

    signatures = load_contextual_signatures(seed_path)
    if not signatures:
        logger.info("contextual_disambiguation_no_signatures")
        return None
    if not mentions_path.exists():
        logger.warning("contextual_disambiguation_no_mentions", path=str(mentions_path))
        return None

    records: list[dict[str, Any]] = pq.read_table(canonical_path).to_pylist()
    if not records:
        return None

    name_by_id: dict[str, str] = {}
    record_by_id: dict[str, dict[str, Any]] = {}
    for rec in records:
        cid = rec.get("canonical_id")
        if not cid:
            continue
        record_by_id[cid] = rec
        name_norm = rec.get("name_ar_normalized")
        if name_norm:
            name_by_id[cid] = name_norm

    # Group signatures by the bare primary id they peel from. A primary present in no
    # canonical row is a stale seed entry — logged, never invented (over_merged_flag rule).
    sigs_by_primary: dict[str, list[ContextualSignature]] = {}
    for sig in signatures:
        sigs_by_primary.setdefault(sig.primary_id(), []).append(sig)
    absent = sorted(pid for pid in sigs_by_primary if pid not in record_by_id)
    if absent:
        logger.warning("contextual_disambiguation_seed_absent", unmatched=absent, count=len(absent))

    candidate_ids = {pid for pid in sigs_by_primary if pid in record_by_id}
    if not candidate_ids:
        logger.info("contextual_disambiguation_no_present_primaries", seed=len(signatures))
        return None

    chains, candidate_mentions = _build_chain_index(mentions_path, candidate_ids)

    remap: dict[str, str] = {}
    target_counts: dict[str, int] = {}
    target_sig: dict[str, ContextualSignature] = {}
    coverage_rows: list[dict[str, Any]] = []
    for pid in sorted(candidate_ids):
        name_norm = name_by_id.get(pid, "")
        sigs = sigs_by_primary[pid]
        neighbours = _neighbour_names_by_mention(
            candidate_mentions.get(pid, []), chains, name_by_id
        )
        for mention_id, (teachers, students) in neighbours.items():
            match = match_signature(teachers, students, sigs)
            if match is not None:
                tid = match.target_id()
                remap[mention_id] = tid
                target_counts[tid] = target_counts.get(tid, 0) + 1
                target_sig[tid] = match
        cov = neighbour_pair_coverage(name_norm, pid, neighbours, sigs)
        coverage_rows.append(
            {
                "name_ar_normalized": cov.name_ar_normalized,
                "primary_id": cov.primary_id,
                "total_mentions": cov.total_mentions,
                "with_any_neighbour": cov.with_any_neighbour,
                "with_teacher": cov.with_teacher,
                "with_student": cov.with_student,
                "signature_matched": cov.signature_matched,
                "ambiguous_multimatch": cov.ambiguous_multimatch,
            }
        )

    # Always emit the blast-radius report (da#346 acceptance) — even a zero-peel run
    # discloses the coverage so the residual-flag decision is auditable.
    coverage_path = output_dir / "contextual_coverage.parquet"
    cov_arrays = {
        f.name: [r.get(f.name) for r in coverage_rows] for f in CONTEXTUAL_COVERAGE_SCHEMA
    }
    pq.write_table(pa.table(cov_arrays, schema=CONTEXTUAL_COVERAGE_SCHEMA), coverage_path)

    if not remap:
        logger.info(
            "contextual_disambiguation_no_peel",
            signatures=len(signatures),
            candidates=len(candidate_ids),
            coverage=str(coverage_path),
        )
        return None

    # Apply: append target rows (reducing each primary's count), build the audit report.
    new_rows: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    peeled_by_primary: dict[str, int] = {}
    for tid, count in sorted(target_counts.items()):
        sig = target_sig[tid]
        pid = sig.primary_id()
        primary_rec = record_by_id[pid]
        peeled_by_primary[pid] = peeled_by_primary.get(pid, 0) + count
        new_rows.append(_target_record(primary_rec, sig, count))
        audit_rows.append(
            {
                "name_ar_normalized": primary_rec.get("name_ar_normalized"),
                "primary_id": pid,
                "new_id": tid,
                "discriminator": sig.discriminator,
                "target_name_en": sig.name_en,
                "teacher_ar": sig.teacher_ar,
                "student_ar": sig.student_ar,
                "mention_count": count,
            }
        )
    for pid, peeled in peeled_by_primary.items():
        primary_rec = record_by_id[pid]
        primary_rec["mention_count"] = max(0, _as_int(primary_rec.get("mention_count")) - peeled)
        primary_rec["attestation"] = derive_attestation(primary_rec["mention_count"])

    all_records = records + new_rows
    arrays = {f.name: [r.get(f.name) for r in all_records] for f in NARRATORS_CANONICAL_SCHEMA}
    write_canonical(
        pa.table(arrays, schema=NARRATORS_CANONICAL_SCHEMA),
        canonical_path,
        stage="contextual_disambiguation",
    )

    remapped = _remap_split_mentions(mentions_path, remap)

    audit_path = output_dir / "contextual_splits.parquet"
    audit_arrays = {f.name: [r.get(f.name) for r in audit_rows] for f in CONTEXTUAL_SPLITS_SCHEMA}
    pq.write_table(pa.table(audit_arrays, schema=CONTEXTUAL_SPLITS_SCHEMA), audit_path)

    logger.info(
        "contextual_disambiguation_complete",
        peeled_targets=len(new_rows),
        mentions_remapped=remapped,
        canonical_total=len(all_records),
        audit=str(audit_path),
        coverage=str(coverage_path),
    )
    return canonical_path
