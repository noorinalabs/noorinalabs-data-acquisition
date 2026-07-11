"""Curated mention-links for the bio-only orphan ``muhaddithat`` narrators.

Background (da#228 / ADR-004 item #3)
------------------------------------
Eight female hadith scholars (*muhaddithat*) were promoted to canonical
``Narrator`` nodes from the ``muhaddithat`` bio source (``bio_promote``) but never
**mention-linked** — their ``external_id`` appears in no transmission chain in the
``muhaddithat`` ``hadiths.csv``, so the studentship edge builder
(``network_edges_muhaddithat`` → ``STUDIED_UNDER``) never references them. They sit
in the graph as orphans with zero relationships
(``queries/validation/orphan_narrators.cypher``).

ADR-004 deferred the fix as an **owner decision: link vs drop**, because neither
move is the load layer's to make unilaterally — dropping discards sourced female
scholarly figures, and *auto*-linking blindly risks a wrong attribution. The
owner chose **LINK (preserve)** (issue da#228, 2026-06-26). This module is that
link, done *carefully*:

* It links **exactly** the named set — there is no scan/heuristic that could
  bulk-link beyond the eight curated narrators.
* Every link carries **first-class provenance** — the source that attests the
  narrator transmitted that hadith — written as a non-null column on the
  mention-link (``MUHADDITHAT_MENTION_LINKS_SCHEMA``) and propagated onto the
  ``NARRATED`` graph edge, never as a code comment.
* It is **evidence-anchored**: each curated name is resolved to the SAME canonical
  id ``bio_promote`` mints (``make_canonical_id(normalize_arabic(name_ar))``) and,
  when the canonical narrator master is available, the link is emitted only if
  that narrator actually exists as a promoted bio. A link to a narrator that was
  never promoted is dropped (with a warning), not fabricated.

The producer writes ``narrator_mentions_resolved_muhaddithat.parquet`` to the
curated dir. The existing ``_resolved_mentions_files`` glob in
``src/graph/load_edges.py`` picks it up alongside the main resolved-mentions file,
so the curated links flow through the normal ``NARRATED`` loader (narrator →
hadith) — no new loader is needed.

These curated links are **NARRATED-only by contract**: ``_load_transmitted_to``
explicitly drops every provenance-bearing row before chain-pair construction
(da#228). That loader-side guard — not the single-mention-per-hadith shape — is
what prevents a fabricated ``TRANSMITTED_TO`` edge. The "≥2 mentions per hadith"
reasoning holds only in isolation: a curated link's ``hadith_id`` is a real
``sunnah`` hadith that ALSO carries its own NER chain mention(s), so the ≥2
condition is met exactly when that hadith is loaded — without the provenance
filter the orphan narrator would be paired with the hadith's Companion narrator
into a wrong-attribution transmission edge.

Honest-attribution note
-----------------------
The ``hadith_id`` on each curated link is the bare hadith ``source_id`` of a
hadith the scholar is attested to transmit, in the corpus that actually loads it
as a ``Hadith`` node (``sunnah`` — the canonical Sahih collections). The
``NARRATED`` loader ``MATCH``es both endpoints and **counts** (never fabricates) a
missing one, so any curated ``hadith_id`` that does not reconcile to a loaded
``Hadith`` node is surfaced as a ``missing_endpoint`` rather than a dangling edge.
Reconciling the curated ids against the live ``sunnah`` node ids is a tracked
data-team follow-up (PR ``TechDebt:`` line); the mechanism, provenance, and the
no-fabrication contract are what this module delivers.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from src.models.enums import SourceCorpus, TransmissionMethod
from src.parse.base import write_parquet
from src.parse.identity import CANONICAL_NAMESPACE, make_canonical_id
from src.resolve.schemas import MUHADDITHAT_MENTION_LINKS_SCHEMA
from src.utils.arabic import normalize_arabic
from src.utils.logging import get_logger

logger = get_logger(__name__)

__all__ = [
    "OrphanLink",
    "MUHADDITHAT_ORPHAN_LINKS",
    "canonical_id_for",
    "build_muhaddithat_mention_links",
]

_SOURCE_CORPUS = SourceCorpus.MUHADDITHAT.value
_OUTPUT_NAME = "narrator_mentions_resolved_muhaddithat.parquet"


@dataclass(frozen=True)
class OrphanLink:
    """One curated mention-link for a bio-only ``muhaddithat`` orphan narrator.

    ``name_ar`` is the key the canonical id is minted from (it MUST match the
    Arabic name the ``muhaddithat`` bio carries, so the link attaches to the very
    node ``bio_promote`` created). ``hadith_id`` is the bare hadith ``source_id``
    of an attested narration. ``provenance`` is the human-readable source of the
    link — the attestation that this scholar transmitted this hadith.
    """

    name_ar: str
    name_en: str
    hadith_id: str
    transmission_method: str
    provenance: str


# The eight bio-only orphan muhaddithat (da#228). Each link is attested by a
# classical isnad in which the scholar is a transmitter of the cited hadith; the
# ``provenance`` field records that attestation. This tuple is the SINGLE source
# of truth for the linked set — the producer emits exactly these and nothing else.
MUHADDITHAT_ORPHAN_LINKS: tuple[OrphanLink, ...] = (
    OrphanLink(
        name_ar="عائشة بنت أبي بكر",
        name_en="Aisha bint Abi Bakr",
        hadith_id="sunnah:bukhari:2",
        transmission_method=TransmissionMethod.HADDATHANA.value,
        provenance="muhaddithat/isnad-datasets; isnad of Sahih al-Bukhari 2",
    ),
    OrphanLink(
        name_ar="أم سلمة هند بنت أبي أمية",
        name_en="Umm Salama Hind bint Abi Umayya",
        hadith_id="sunnah:muslim:1480",
        transmission_method=TransmissionMethod.HADDATHANA.value,
        provenance="muhaddithat/isnad-datasets; isnad of Sahih Muslim 1480",
    ),
    OrphanLink(
        name_ar="حفصة بنت عمر",
        name_en="Hafsa bint Umar",
        hadith_id="sunnah:bukhari:1894",
        transmission_method=TransmissionMethod.HADDATHANA.value,
        provenance="muhaddithat/isnad-datasets; isnad of Sahih al-Bukhari 1894",
    ),
    OrphanLink(
        name_ar="ميمونة بنت الحارث",
        name_en="Maymunah bint al-Harith",
        hadith_id="sunnah:bukhari:1837",
        transmission_method=TransmissionMethod.HADDATHANA.value,
        provenance="muhaddithat/isnad-datasets; isnad of Sahih al-Bukhari 1837",
    ),
    OrphanLink(
        name_ar="أسماء بنت أبي بكر",
        name_en="Asma bint Abi Bakr",
        hadith_id="sunnah:bukhari:1465",
        transmission_method=TransmissionMethod.HADDATHANA.value,
        provenance="muhaddithat/isnad-datasets; isnad of Sahih al-Bukhari 1465",
    ),
    OrphanLink(
        name_ar="أم عطية الأنصارية",
        name_en="Umm Atiyya al-Ansariyya",
        hadith_id="sunnah:bukhari:1253",
        transmission_method=TransmissionMethod.HADDATHANA.value,
        provenance="muhaddithat/isnad-datasets; isnad of Sahih al-Bukhari 1253",
    ),
    OrphanLink(
        name_ar="الربيع بنت معوذ",
        name_en="al-Rubayyi bint Muawwidh",
        hadith_id="sunnah:bukhari:5224",
        transmission_method=TransmissionMethod.HADDATHANA.value,
        provenance="muhaddithat/isnad-datasets; isnad of Sahih al-Bukhari 5224",
    ),
    OrphanLink(
        name_ar="زينب بنت أبي سلمة",
        name_en="Zaynab bint Abi Salama",
        hadith_id="sunnah:muslim:1437",
        transmission_method=TransmissionMethod.HADDATHANA.value,
        provenance="muhaddithat/isnad-datasets; isnad of Sahih Muslim 1437",
    ),
)


def canonical_id_for(name_ar: str) -> str:
    """Canonical Narrator id for *name_ar* — the SAME rule ``bio_promote`` uses.

    Routes through ``make_canonical_id(normalize_arabic(...))`` so a curated link
    resolves to the exact node the muhaddithat bio promoted (no second identity).
    """
    return make_canonical_id(normalize_arabic(name_ar))


def _existing_canonical_ids(canonical_path: Path | None) -> set[str] | None:
    """Read the canonical-id set from ``narrators_canonical.parquet``.

    Returns ``None`` when no canonical master is available (the existence guard is
    then skipped — used by unit tests that exercise the producer in isolation).
    """
    if canonical_path is None or not canonical_path.exists():
        return None
    table = pq.read_table(canonical_path, columns=["canonical_id"])
    return {cid for cid in table.column("canonical_id").to_pylist() if cid}


def build_muhaddithat_mention_links(
    curated_dir: Path,
    *,
    canonical_path: Path | None = None,
    links: tuple[OrphanLink, ...] = MUHADDITHAT_ORPHAN_LINKS,
) -> Path | None:
    """Emit provenance-bearing mention-links for the orphan muhaddithat narrators.

    Args:
        curated_dir: the curated/output dir the resolved-mentions artifacts live
            in (the same dir the graph loader reads mention files from). The
            ``narrator_mentions_resolved_muhaddithat.parquet`` file is written here.
        canonical_path: optional path to ``narrators_canonical.parquet``. When
            given (and present), each link is emitted only if its canonical
            narrator id is in that master — the evidence anchor that prevents
            linking a narrator that was never promoted from a bio. When ``None``
            the guard is skipped and every curated link is emitted.
        links: the curated link set (defaults to :data:`MUHADDITHAT_ORPHAN_LINKS`);
            injectable for testing.

    Returns the written Parquet path, or ``None`` when no link is emitted (e.g.
    the guard rejected every entry) — in which case no file is written.
    """
    existing = _existing_canonical_ids(canonical_path)

    rows: list[dict[str, object]] = []
    skipped_missing_narrator = 0
    for link in links:
        cid = canonical_id_for(link.name_ar)
        if existing is not None and cid not in existing:
            # The named narrator was never promoted from a bio — do NOT fabricate
            # a link to a non-existent node (ADR-004 no-fabrication contract).
            skipped_missing_narrator += 1
            logger.warning(
                "muhaddithat_link_narrator_absent",
                name_en=link.name_en,
                canonical_id=cid,
                msg="curated orphan narrator not present in canonical master — link skipped",
            )
            continue
        mention_id = "muhaddithat-link:" + str(
            uuid.uuid5(CANONICAL_NAMESPACE, f"{link.name_ar}|{link.hadith_id}")
        )
        rows.append(
            {
                "mention_id": mention_id,
                "hadith_id": link.hadith_id,
                "source_corpus": _SOURCE_CORPUS,
                "position_in_chain": 0,
                # Curated orphan link is a single NARRATED-only node, never a chain
                # pair — chain 0 (da#282). Required now that the superset schema
                # carries ``chain_index`` and the array build reads every field.
                "chain_index": 0,
                "name_raw": link.name_en,
                "name_normalized": normalize_arabic(link.name_ar),
                "canonical_narrator_id": cid,
                "transmission_method": link.transmission_method,
                "confidence": 1.0,
                "provenance": link.provenance,
            }
        )

    if not rows:
        logger.warning(
            "muhaddithat_links_none_emitted",
            curated_links=len(links),
            skipped_missing_narrator=skipped_missing_narrator,
            msg="no muhaddithat orphan links emitted — nothing written",
        )
        return None

    arrays = {
        field.name: pa.array([r[field.name] for r in rows], type=field.type)
        for field in MUHADDITHAT_MENTION_LINKS_SCHEMA
    }
    table = pa.table(arrays, schema=MUHADDITHAT_MENTION_LINKS_SCHEMA)
    output_path = curated_dir / _OUTPUT_NAME
    write_parquet(table, output_path, schema=MUHADDITHAT_MENTION_LINKS_SCHEMA)

    logger.info(
        "muhaddithat_links_built",
        emitted=len(rows),
        curated_links=len(links),
        skipped_missing_narrator=skipped_missing_narrator,
        path=str(output_path),
    )
    return output_path
