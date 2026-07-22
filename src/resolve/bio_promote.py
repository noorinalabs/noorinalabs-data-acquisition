"""Promote narrator BIO rows directly to canonical Narrator records.

The mention-driven disambiguator (``src/resolve/disambiguate.py``) only emits a
canonical narrator when an isnad *mention* resolves to a bio candidate — with
zero mentions it returns nothing. A **profile-only** source (a rijal database
such as Itqan that contributes biographies but no isnad chains in its narrators
slice) would therefore produce **no** Narrator nodes at all.

This module bridges that gap. Each bio becomes — or updates — a canonical
narrator keyed by the SAME ``nar:<uuid5(normalized-name)>`` identity the
disambiguator uses (``src.parse.identity.make_canonical_id``), so the two paths
converge: a later mention-driven run dedups onto the very same node, and the
graph loader (``load_nodes._load_narrators``) ingests the output unchanged.

It is **merge-safe**: when ``narrators_canonical.parquet`` already exists (e.g.
from a disambiguator run), it is loaded and unioned by ``canonical_id`` rather
than overwritten, so the bio-direct and mention-driven outputs compose.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, NamedTuple

import pyarrow as pa
import pyarrow.parquet as pq

from src.models.enums import DatePrecision
from src.parse.base import safe_str
from src.parse.identity import make_canonical_id
from src.parse.name_quality import (
    clean_narrator_name,
    clean_narrator_name_display,
    cleaner_removed_content,
    is_mubham_relational,
    is_recoverable_gloss_tail,
)
from src.resolve._inputs import require_input
from src.resolve._run_record import write_canonical
from src.resolve.attestation import derive_attestation
from src.resolve.schemas import NARRATORS_CANONICAL_SCHEMA
from src.resolve.sect_affiliation import (
    derive_sect_affiliation,
    normalize_corpus,
    primary_corpus,
)
from src.resolve.tabaqa_dates import plausible_attested_death_year
from src.utils.arabic import normalize_arabic
from src.utils.logging import get_logger

logger = get_logger(__name__)

__all__ = ["promote_bios_to_canonical"]

# Scalar bio fields back-filled onto an existing canonical record when missing.
# ``death_year_ah`` is handled separately (da#454, plausibility-bound at stamp
# time) rather than through this generic pass-through.
_BACKFILL_FIELDS = (
    "name_en",
    "birth_year_ah",
    "generation",
    "gender",
    "trustworthiness",
    "external_id",
)


class _AliasMergeStats(NamedTuple):
    """Per-run alias-merge accounting (da#389).

    Every alias row lands in exactly one bucket, so
    ``added + skipped_duplicate + skipped_empty + skipped_source +
    skipped_deixis + unmatched_pollution + unmatched_no_bio`` equals the alias-row
    count — no alias is ever silently dropped again. ``unmatched_pollution`` (the
    alias's bio was itself cleaner-dropped, so no canonical node was minted to anchor
    to) and ``skipped_source`` are *explained* non-attachments; ``skipped_deixis``
    (da#391 — the alias is a bare relational-pronoun, "خالته"/"عمتي"/"امتاه", the same
    unnamed-relation deixis the cleaner drops from a NAME) is the alias-list analogue
    of that pollution drop; ``unmatched_no_bio`` is the one that warrants a look and
    is logged at warning level when non-zero.
    """

    added: int
    unmatched_pollution: int
    unmatched_no_bio: int
    skipped_source: int
    skipped_duplicate: int
    skipped_empty: int
    skipped_deixis: int


def _load_existing_canonical(path: Path) -> dict[str, dict[str, Any]]:
    """Load an existing narrators_canonical.parquet into a canonical_id -> row map."""
    if not path.exists():
        return {}
    table = pq.read_table(path)
    rows: dict[str, dict[str, Any]] = {}
    for row in table.to_pylist():
        cid = row.get("canonical_id")
        if cid:
            row.setdefault("source_ids", [])
            row.setdefault("aliases", [])
            # Preserve corpora a prior producer (disambiguate) already recorded so
            # the bio-direct merge unions onto them rather than dropping them (da#103).
            row.setdefault("source_corpora", [])
            rows[cid] = row
    return rows


def _merge_aliases(
    staging_dir: Path,
    canonical_map: dict[str, dict[str, Any]],
    promoted_cids: set[str],
    *,
    sources: set[str] | None,
) -> _AliasMergeStats:
    """Union name-variant aliases from ``narrator_aliases_*.parquet`` (da#94).

    Each alias row carries ``canonical_name_ar_normalized`` — the bio's **raw**
    normalized name. The promoter mints the canonical node under
    ``make_canonical_id(clean_narrator_name(name))``, not under the raw name, so the
    alias must be re-keyed through the SAME cleaner before the id is computed
    (da#389). It previously keyed on ``make_canonical_id(norm)`` with the raw
    ``norm``, so any bio whose cleaner rewrote its name — a benediction suffix, a
    ``، ويقال :`` variant list, a truncated head — was minted under an id this
    lookup never reproduced, and its aliases attached to nothing. That silently
    dropped **34,046 / 154,328 itqan aliases (22.1%)** with no counter and no log.

    Variants attach only to a canonical id a bio actually **promoted into**
    (``promoted_cids``) — the precise mirror of the promotion decision. An alias
    never *creates* a narrator, and never attaches to a node its own bio did not
    promote into: a bio the cleaner dropped as pollution mints no node, and a
    truncated-residue bio the da#379 guard refused to merge onto an attested
    narrator must not smuggle that narrator an alias through this pass either.
    Duplicates and the primary spelling are skipped. Returns a full accounting
    (:class:`_AliasMergeStats`) so every non-attachment is counted, not silent.
    """
    alias_files = sorted(staging_dir.glob("narrator_aliases_*.parquet"))
    added = 0
    unmatched_pollution = 0
    unmatched_no_bio = 0
    skipped_source = 0
    skipped_duplicate = 0
    skipped_empty = 0
    skipped_deixis = 0
    for af in alias_files:
        for row in pq.read_table(af).to_pylist():
            if sources is not None and safe_str(row.get("source")) not in sources:
                skipped_source += 1
                continue
            norm = safe_str(row.get("canonical_name_ar_normalized"))
            variant = safe_str(row.get("alias_normalized"))
            if not norm or not variant:
                skipped_empty += 1
                continue
            # da#391: a bare relational-pronoun variant ("خالته"/"عمتي"/"امتاه") is the
            # unnamed-relation deixis clean_narrator_name drops from a NAME; the aliases
            # list<string> is the blind spot it never reaches (the third and final
            # alias-union site, alongside disambiguate + fuzzy_cluster). Drop it here,
            # accounted, so it cannot survive as an alias of a real narrator.
            if is_mubham_relational(variant):
                skipped_deixis += 1
                continue
            # Re-key on the cleaned form the promoter minted the node under (da#389).
            cleaned = clean_narrator_name(norm)
            if not cleaned:
                # The alias's own bio was pollution-dropped by this same cleaner
                # (returned None) — no canonical node exists to anchor to. Legitimate,
                # and now counted instead of silently lost.
                unmatched_pollution += 1
                continue
            cid = make_canonical_id(cleaned)
            if cid not in promoted_cids:
                # No bio promoted this name into the table (a truncated-residue bio
                # the da#379 guard refused, or a name absent from this source's bios).
                unmatched_no_bio += 1
                continue
            rec = canonical_map[cid]  # promoted_cids ⊆ canonical_map keys
            aliases = rec.setdefault("aliases", [])
            if variant != rec.get("name_ar_normalized") and variant not in aliases:
                aliases.append(variant)
                added += 1
            else:
                skipped_duplicate += 1
    return _AliasMergeStats(
        added=added,
        unmatched_pollution=unmatched_pollution,
        unmatched_no_bio=unmatched_no_bio,
        skipped_source=skipped_source,
        skipped_duplicate=skipped_duplicate,
        skipped_empty=skipped_empty,
        skipped_deixis=skipped_deixis,
    )


def promote_bios_to_canonical(
    staging_dir: Path,
    output_dir: Path,
    *,
    sources: set[str] | None = None,
) -> Path | None:
    """Build/extend ``narrators_canonical.parquet`` directly from narrator bios.

    Args:
        staging_dir: directory holding ``narrators_bio_*.parquet`` files.
        output_dir: directory the canonical Parquet is written to (and read from
            when merging into a pre-existing file). This is the resolve stage's
            ``output_dir`` — the same curated location the graph loader reads
            ``narrators_canonical.parquet`` from (the CLI maps it to
            ``DATA_CURATED_DIR``; see ``load_nodes`` artifact-location contract,
            da#112). Point it at the *curated* dir, never staging.
        sources: if given, only bios whose ``source`` column is in this set are
            promoted (e.g. ``{"itqan"}``).

    Returns the canonical Parquet path, or ``None`` when there are no bios to
    promote (no file written).

    Crash-resume (da#272): EXEMPT from intra-stage checkpointing. This is a
    sub-minute merge over the handful of ``narrators_bio_*`` shards (not a
    multi-hour stream), and it is already idempotent — a MERGE into the canonical
    table (da#99/da#117), so simply re-running it after a crash reproduces the
    same result. A checkpoint would add machinery and a serialized-state failure
    surface for no recovery-time saving. Recovery re-runs it via
    ``resolve --from-step bio_promote``.
    """
    # Required input: the bio shards this stage exists to promote. da#361: their
    # total absence means parse never produced them (or ran against the wrong dir)
    # — an upstream defect, not an empty result — so raise rather than return a
    # success-shaped ``None``. Shards present but carrying no promotable rows (all
    # source-filtered / name-quality-dropped) is a genuine empty and still returns
    # ``None`` honestly further down.
    bio_files = sorted(staging_dir.glob("narrators_bio_*.parquet"))
    require_input(
        stage="bio_promote",
        present=bool(bio_files),
        input_desc=f"narrators_bio_*.parquet under {staging_dir}",
        produced_by="parse (bio adapters)",
        remediation="re-run `parse`; there are no bio shards to promote",
    )

    canonical_path = output_dir / "narrators_canonical.parquet"
    canonical_map = _load_existing_canonical(canonical_path)
    pre_existing = len(canonical_map)

    promoted = 0
    skipped_no_name = 0
    skipped_source = 0
    skipped_pollution = 0
    skipped_truncated_merge = 0
    # da#397/da#398: split the refusal by WHAT the truncation removed. Class A
    # (gloss-tail) is a complete nasab that lost only a teacher-key isnad tail — the
    # RECOVERABLE-pending-disambiguation subset; class B (name-cut) is a bare ism/kunya
    # residue, a dispute/speech cut, or a disputed-identity marker, and refusing it
    # stays correct. Both are still refused below; the split only QUANTIFIES the
    # recoverable subset (da#398 scope-1) and is the precondition for a future
    # disambiguation-gated recovery. See is_recoverable_gloss_tail + da#397 PR#441.
    skipped_truncated_merge_gloss_tail = 0
    skipped_truncated_merge_name_cut = 0
    # The ACCEPTED truncated-residue residual (da#385). The guard only refuses a
    # truncated residue that would claim an *attested* node (mention_count > 0);
    # every other truncated residue is accepted — it either lands on a zero-mention
    # catalog node or mints a new node keyed on the truncation. Both are deliberate
    # (the guard's scope is narrow by design, da#379) but were previously uncounted,
    # so "we accepted it" was indistinguishable from "it never happened". Count them.
    truncated_residue_onto_unattested = 0
    truncated_residue_minted = 0
    # Canonical ids a bio actually promoted into this run — the anchor set aliases
    # attach to (da#389). Excludes pollution-drops and truncation-skips, which mint
    # no node (or refuse the merge), so their aliases have nothing to anchor to.
    promoted_cids: set[str] = set()

    for bf in bio_files:
        table = pq.read_table(bf)
        for row in table.to_pylist():
            source = safe_str(row.get("source"))
            if sources is not None and source not in sources:
                skipped_source += 1
                continue
            # Normalize the free-text bio provenance to a SourceCorpus value for
            # sect/corpus tagging (da#103); unknown provenance → None (dropped).
            corpus = normalize_corpus(source)

            name_ar = safe_str(row.get("name_ar"))
            norm = safe_str(row.get("name_ar_normalized")) or (
                normalize_arabic(name_ar) if name_ar else None
            )
            if not norm:
                skipped_no_name += 1
                continue

            # Apply the SAME name-quality filter the NER mention path runs (da#247/
            # da#271). bio_promote previously bypassed it, so honorific/eulogy-laden
            # bio names — the kaggle rijāl table is 100% such ("… رضي الله عنه",
            # "رسول الله صلى الله عليه واله") — minted polluted canonical nodes
            # straight from the bio table, unfiltered. Drop when the cleaner rejects
            # the span; otherwise promote the cleaned form so a benediction-suffixed
            # bio merges onto the clean mention-derived node instead of forking it.
            cleaned = clean_narrator_name(norm)
            if not cleaned:
                skipped_pollution += 1
                continue
            truncated = cleaner_removed_content(norm, cleaned)
            # da#397/da#398: classify the truncation as gloss-tail (class A, recoverable
            # pending biographical homonym disambiguation) vs name-cut (class B, correctly
            # refused) for the refusal-counter split. Computed on the PRE-clean `norm`
            # (before it is reassigned to `cleaned` just below). This does NOT change the
            # merge decision — both classes are still refused at the guard.
            truncation_is_gloss_tail = truncated and is_recoverable_gloss_tail(norm, cleaned)
            norm = cleaned

            bio_id = safe_str(row.get("bio_id"))
            cid = make_canonical_id(norm)
            # da#454: bound the bio's attested death year against the isnad-narrator
            # plausibility envelope *before* it can be written anywhere — a raw
            # death_year_ah outside the envelope (the d. ~700-780 AH late-collector
            # pollution) becomes None rather than landing on a fresh node or
            # back-filling an existing one.
            death_year = plausible_attested_death_year(
                row.get("death_year_ah"), row.get("generation")
            )

            # When `cleaner_removed_content` is True the cleaner removed name-span
            # material beyond its affix normalizations. That covers TWO classes the
            # guard treats alike, and a maintainer must not read this refusal as
            # "the residue is not a name" — for most of these rows it is (da#397/da#398):
            #
            #   1. CUT INTO the name → a bare-ism residue that is genuinely not the
            #      asserted name. `عبيدة مولى رسول الله ذكره بن شاهين …` cleans to
            #      `عبيده` and would otherwise merge onto ʿAbīda al-Salmānī (1,695
            #      mentions), back-filling an obscure client's jarḥ grade and death year
            #      onto him via _BACKFILL_FIELDS. A bare ism is not a person; refusing is
            #      the only safe call. This is what the guard exists for.
            #   2. A COMPLETE name that merely lost an isnad/gloss TAIL → e.g.
            #      `اسحاق بن مره عن انس` → `اسحاق بن مره`, the full nasab intact, only the
            #      teacher-key `عن انس` cut. Roughly 40% of the mention-bearing refusals
            #      are this class (full-nasab residues, not bare isms).
            #
            # For class 2 refusal is NON-CORRUPTING but it is NOT "always safe" in the
            # symmetric sense — it withholds correct enrichment from real narrators. So
            # why refuse it too? Because `make_canonical_id` folds Arabic inflection
            # (da#376) but NOT homonymy, and full-nasab homonyms are pervasive here: a
            # blast-radius measurement over the staging corpus found 45 of 240 accept-
            # target names carry CONFLICTING jarḥ grades across their bio entries (one
            # name-key graded both thiqa and kadhdhab — provably >1 person). Accepting a
            # class-2 merge on the name alone back-fills a verdict onto a possibly-
            # different bearer — the da#423 direction (feedback_drop_gate_bidirectional_ab).
            # So the conservative refusal is the right DEFAULT for both classes; safely
            # recovering class 2 needs biographical disambiguation, deferred to da#397/398.
            #
            # `cleaner_removed_content` compares against the cleaner's OWN
            # pre-truncation tokens, not a re-tokenization of `norm`. A bracketed
            # catalog entry number, an editorial connective and an honorific are
            # affixes the cleaner unwraps — the residue is still the full asserted
            # name. Treating those as cuts refused 42 merges onto attested narrators
            # (8,302 mentions), starving them of the very enrichment this module
            # exists to deliver: the same harm, sign-flipped.
            #
            # Keyed on `cid`, never on the normalized string: da#376 folds Arabic
            # inflection inside make_canonical_id, so the residue `ابو نهشل` and the
            # attested `ابي نهشل` are one narrator under two surfaces. A string-keyed
            # guard sees two and refuses nothing.
            #
            # Scoped to targets that carry mentions: those are narrators attested in
            # an isnad, and corrupting them is the defect. A truncated residue landing
            # on a zero-mention catalog node merges two catalog entries under a
            # mangled key — wrong, but it touches no attested narrator and is fixed by
            # the deferred name/prose extraction, not here. Widening the guard to
            # zero-mention targets would drop 12,043 kaggle rows for zero
            # mention-bearing fixes (kaggle collides with an attested narrator exactly
            # 0 times) and that corpus belongs to da#299.
            existing = canonical_map.get(cid)
            if truncated and existing is not None and (existing.get("mention_count") or 0) > 0:
                skipped_truncated_merge += 1
                # da#397/da#398: split the refusal — the gloss-tail (class A) count is
                # the recall cost of the deliberate class-2 over-refusal, made a live
                # per-run metric rather than a one-off script measurement (da#398 scope-1).
                if truncation_is_gloss_tail:
                    skipped_truncated_merge_gloss_tail += 1
                else:
                    skipped_truncated_merge_name_cut += 1
                continue

            if cid not in canonical_map:
                # da#385: a truncated residue that mints a brand-new node names that
                # node after a truncation, not after an asserted name. Count it.
                if truncated:
                    truncated_residue_minted += 1
                # da#301: strip the benediction/eulogy from the DISPLAY name_ar on
                # the new-node (bio-only) path. name_ar_normalized + make_canonical_id
                # already use the cleaned form (`norm`), but the display `name_ar`
                # kept the raw eulogy ("… رضي الله عنه"). That is harmless when the
                # bio MERGES onto a clean mention-derived node (the else branch never
                # overwrites its name_ar), but a BIO-ONLY narrator surfaces the
                # benediction in its label (the loader keys node.name on name_ar
                # first). clean_narrator_name_display strips it while preserving the
                # voweled surface. Identity/matching are unchanged — cid + norm are
                # as before; only the display label changes.
                #
                # da#471 (review of da#301): the display MUST be consistent with the
                # identity AND never empty/garbled. The graph loader stores name_ar
                # VERBATIM (``SET n.name_ar = row.name_ar``; a None becomes ""), and
                # does NOT coalesce the stored display to name_ar_normalized (that
                # coalesce is only in the search-name derivation) — so a None display
                # would surface as an EMPTY node label, worse than the raw eulogy.
                # Two ways the display can go wrong here: (a) name_ar is honorific-only
                # and cleans to None while name_ar_normalized is a real name (the
                # fields diverged upstream); (b) clean_narrator_name_display's
                # alignment fallback emits a denatured/garbled surface. Guard BOTH:
                # keep the voweled display only when it normalizes back to the
                # canonical `norm`; otherwise fall back to `norm` (clean, never empty
                # when a real name exists, always consistent with the id).
                display_name_ar = clean_narrator_name_display(name_ar) if name_ar else None
                if not display_name_ar or normalize_arabic(display_name_ar) != norm:
                    display_name_ar = norm
                canonical_map[cid] = {
                    "canonical_id": cid,
                    "name_ar": display_name_ar,
                    "name_en": safe_str(row.get("name_en")),
                    "name_ar_normalized": norm,
                    "aliases": [],
                    "birth_year_ah": row.get("birth_year_ah"),
                    "death_year_ah": death_year,
                    "generation": safe_str(row.get("generation")),
                    "gender": safe_str(row.get("gender")),
                    "trustworthiness": safe_str(row.get("trustworthiness")),
                    "source_ids": [bio_id] if bio_id else [],
                    "external_id": safe_str(row.get("external_id")),
                    # mention_count stays 0: a bio is provenance, not a chain hit.
                    "mention_count": 0,
                    # Corpus provenance for sect derivation (da#103). Finalized into
                    # source_corpus + sect_affiliation after the merge loop.
                    "source_corpora": [corpus] if corpus else [],
                }
            else:
                # da#385: a truncated residue reaching this branch merges onto a node
                # that is NOT attested — a mention_count > 0 target was already refused
                # above, so this is a zero-mention catalog node (curated or minted this
                # run). Two catalog entries fuse under a truncated key; count it.
                if truncated:
                    truncated_residue_onto_unattested += 1
                rec = canonical_map[cid]
                src_ids = rec.setdefault("source_ids", [])
                if bio_id and bio_id not in src_ids:
                    src_ids.append(bio_id)
                corpora = rec.setdefault("source_corpora", [])
                if corpus and corpus not in corpora:
                    corpora.append(corpus)
                # Back-fill only fields the existing record is missing. Keep the
                # raw Parquet value (already a clean str / int) — do NOT coerce,
                # or an int column like death_year_ah gets stringified.
                for field_name in _BACKFILL_FIELDS:
                    if rec.get(field_name) in (None, "") and row.get(field_name) not in (None, ""):
                        rec[field_name] = row.get(field_name)
                # death_year_ah backfills from the plausibility-bound value (da#454),
                # never the raw row — an implausible attested year must not sneak
                # onto an existing (even zero-mention) record either.
                if rec.get("death_year_ah") in (None, "") and death_year is not None:
                    rec["death_year_ah"] = death_year
            promoted += 1
            promoted_cids.add(cid)

    # Finalize sect/corpus provenance from the accumulated corpora set (da#103).
    for rec in canonical_map.values():
        corpora_val = rec.get("source_corpora")
        # Drop nulls/unknowns a prior producer (or a legacy file) may have left so
        # the scalar source_corpus stays in the SourceCorpus domain (write_parquet
        # provenance gate). Normalize defensively in case the prior value was raw.
        corpora = (
            sorted({nc for c in corpora_val if (nc := normalize_corpus(c))})
            if isinstance(corpora_val, list)
            else []
        )
        rec["source_corpora"] = corpora
        rec["source_corpus"] = primary_corpus(corpora)
        rec["sect_affiliation"] = derive_sect_affiliation(corpora)
        # Attestation from the row's FINAL mention_count (da#370): re-derived for
        # every row — new bio records (mention_count 0 -> biographical_only) AND
        # existing rows merged in from disambiguate — so a value the prior producer
        # wrote can never go stale here. See src.resolve.attestation.
        rec["attestation"] = derive_attestation(rec.get("mention_count"))

    # Attach name-variant aliases onto the canonical records the bios produced.
    alias_stats = _merge_aliases(staging_dir, canonical_map, promoted_cids, sources=sources)

    table_rows = list(canonical_map.values())
    arrays = {f.name: [r.get(f.name) for r in table_rows] for f in NARRATORS_CANONICAL_SCHEMA}
    # Precision columns default to UNKNOWN (never null), matching disambiguate and
    # the da#161 Narrator non-Optional invariant — bio-only narrators carry no
    # precision key, so the generic projection above would leave them null (da#239).
    for col in ("birth_date_precision", "death_date_precision"):
        arrays[col] = [r.get(col) or DatePrecision.UNKNOWN.value for r in table_rows]
    out_table = pa.table(arrays, schema=NARRATORS_CANONICAL_SCHEMA)
    write_canonical(out_table, canonical_path, stage="bio_promote")

    # An alias that matched no promoted bio (and was not a cleaner-dropped-bio case)
    # is the one non-attachment worth investigating — surface it loudly so the
    # da#389 drop can never again be silent.
    if alias_stats.unmatched_no_bio:
        logger.warning(
            "bio_promote_aliases_unmatched_no_bio",
            unmatched_no_bio=alias_stats.unmatched_no_bio,
            note="aliases whose cleaned canonical name matched no promoted bio",
        )

    # da#385: surface the ACCEPTED truncated-residue residual once, loudly, and name
    # the deferred name/prose extraction as its owner. These are not defects the guard
    # should have caught — the da#379 guard is deliberately scoped to attested targets —
    # but they DO name canonical nodes after a truncation (a bare ism or a residue that
    # fused two catalog entries), so they must be observable rather than silent. The
    # real cure is upstream: a name/prose extractor that never emits the truncated span
    # (da#299, kaggle; the deferred itqan prose extraction). Until then, this is the
    # measurement that keeps the residual honest (feedback_silent_zero_is_not_a_measurement).
    truncated_residue_accepted = truncated_residue_onto_unattested + truncated_residue_minted
    if truncated_residue_accepted:
        logger.warning(
            "bio_promote_truncated_residue_accepted",
            onto_unattested=truncated_residue_onto_unattested,
            minted_new=truncated_residue_minted,
            total=truncated_residue_accepted,
            note=(
                "truncated bio residues accepted onto a zero-mention node or minting a "
                "new node keyed on the truncation; owner = deferred name/prose extraction "
                "(da#299 kaggle + itqan prose)"
            ),
        )

    # da#397/da#398: surface the gloss-tail refusals — the RECOVERABLE-pending-
    # disambiguation subset. These are complete nasabs that lost only a teacher-key
    # isnad tail, refused solely because make_canonical_id cannot rule out a homonym
    # (45/240 class-A accept-target name-keys carry conflicting jarḥ grades, da#423).
    # Emitting the count turns da#398's recall cost into a live metric, not a one-off
    # script. Owner of the eventual recovery = biographical homonym disambiguation
    # (jarḥ-grade / death-year separation), same bar as da#347b/da#431. The name-cut
    # count is carried alongside so the split is legible from the log line alone.
    if skipped_truncated_merge_gloss_tail:
        logger.info(
            "bio_promote_gloss_tail_refusals_deferred",
            gloss_tail=skipped_truncated_merge_gloss_tail,
            name_cut=skipped_truncated_merge_name_cut,
            note=(
                "class-A gloss-tail refusals (complete nasab, teacher-key عن/isnad tail "
                "removed) — recall cost of the deliberate class-2 over-refusal; recovery "
                "deferred behind biographical homonym disambiguation (da#397/da#398/da#423)"
            ),
        )

    logger.info(
        "bio_promote_complete",
        bio_files=len(bio_files),
        promoted=promoted,
        canonical_total=len(canonical_map),
        pre_existing=pre_existing,
        skipped_no_name=skipped_no_name,
        skipped_source=skipped_source,
        skipped_pollution=skipped_pollution,
        skipped_truncated_merge=skipped_truncated_merge,
        skipped_truncated_merge_gloss_tail=skipped_truncated_merge_gloss_tail,
        skipped_truncated_merge_name_cut=skipped_truncated_merge_name_cut,
        truncated_residue_onto_unattested=truncated_residue_onto_unattested,
        truncated_residue_minted=truncated_residue_minted,
        aliases_added=alias_stats.added,
        aliases_unmatched_pollution=alias_stats.unmatched_pollution,
        aliases_unmatched_no_bio=alias_stats.unmatched_no_bio,
        aliases_skipped_source=alias_stats.skipped_source,
        aliases_skipped_duplicate=alias_stats.skipped_duplicate,
        aliases_skipped_empty=alias_stats.skipped_empty,
        aliases_skipped_deixis=alias_stats.skipped_deixis,
    )
    return canonical_path
