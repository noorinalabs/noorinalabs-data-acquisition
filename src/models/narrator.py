"""Narrator node model for the isnad graph."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from src.models.enums import (
    DatePrecision,
    Gender,
    NarratorGeneration,
    SectAffiliation,
    TrustworthinessGrade,
)

__all__ = ["Narrator"]


class Narrator(BaseModel):
    """A hadith narrator (rawi) in the isnad graph.

    Represents a historical person who participated in the transmission
    of prophetic traditions, with biographical metadata and graph metrics.
    """

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    id: str
    """Canonical ID with 'nar:' prefix, e.g. 'nar:abu-hurayra-001'."""
    name_ar: str
    """Full Arabic name."""
    name_en: str
    """Full English transliteration."""
    kunya: str | None = None
    """Patronymic, e.g. 'Abu Hurayra'."""
    nisba: str | None = None
    """Geographic or tribal attribution, e.g. 'al-Dawsi'."""
    laqab: str | None = None
    """Honorific or epithet."""
    birth_year_ah: int | None = None
    """Birth year in Hijri calendar (point estimate / best single value)."""
    death_year_ah: int | None = None
    """Death year in Hijri calendar (point estimate / best single value)."""

    # Date uncertainty bounds (additive — the existing *_year_ah fields above
    # remain the point estimate). Bounds + precision express the real-world
    # uncertainty of classical dating; precision is orthogonal to disambiguation
    # confidence. Populated downstream by the date-parse/reconcile/ṭabaqa-fallback
    # stages (da#164/#165/#166); null + UNKNOWN until then so existing rows are
    # unaffected.
    birth_year_ah_earliest: int | None = None
    """Lower bound (inclusive) for the birth year in AH, if known."""
    birth_year_ah_latest: int | None = None
    """Upper bound (inclusive) for the birth year in AH, if known."""
    birth_date_precision: DatePrecision = DatePrecision.UNKNOWN
    """How tightly the source dates the birth (orthogonal to confidence)."""
    death_year_ah_earliest: int | None = None
    """Lower bound (inclusive) for the death year in AH, if known."""
    death_year_ah_latest: int | None = None
    """Upper bound (inclusive) for the death year in AH, if known."""
    death_date_precision: DatePrecision = DatePrecision.UNKNOWN
    """How tightly the source dates the death (orthogonal to confidence)."""

    birth_location_id: str | None = None
    """FK to Location.id."""
    death_location_id: str | None = None
    """FK to Location.id."""
    generation: NarratorGeneration
    """Generation in the transmission chain (sahabi, tabii, etc.)."""
    gender: Gender
    """Biological gender."""
    sect_affiliation: SectAffiliation
    """Sectarian affiliation per biographical sources."""
    tabaqat_class: str | None = None
    """Layer/class in tabaqat literature."""
    trustworthiness_consensus: TrustworthinessGrade
    """Consensus trustworthiness grade from rijal criticism."""
    aliases: list[str] = Field(default_factory=list)
    """Alternate name forms for this narrator."""

    # Graph metrics (populated in Phase 4, nullable until then)
    betweenness_centrality: float | None = None
    in_degree: int | None = None
    out_degree: int | None = None
    pagerank: float | None = None
    community_id: int | None = None

    @field_validator("id")
    @classmethod
    def _validate_id_prefix(cls, v: str) -> str:
        if not v.startswith("nar:"):
            msg = f"Narrator id must start with 'nar:', got '{v}'"
            raise ValueError(msg)
        return v

    @model_validator(mode="after")
    def _validate_date_bound_ordering(self) -> Narrator:
        """Reject an inverted earliest/latest bound for each date pair.

        Ordering-only invariant: when BOTH ends of a bound pair are present,
        ``earliest`` must not exceed ``latest``. Enforcing it once at the model
        boundary spares every downstream consumer (da#164/#165/#166, ig#1039)
        from re-defending it. Precision-implied bound relationships
        (AFTER/BEFORE/EXACT) are intentionally NOT checked here — those belong to
        the populating stages.
        """
        for kind, earliest, latest in (
            ("birth", self.birth_year_ah_earliest, self.birth_year_ah_latest),
            ("death", self.death_year_ah_earliest, self.death_year_ah_latest),
        ):
            if earliest is not None and latest is not None and earliest > latest:
                msg = (
                    f"Narrator {kind}_year_ah_earliest ({earliest}) must not be "
                    f"greater than {kind}_year_ah_latest ({latest})"
                )
                raise ValueError(msg)
        return self
