"""Tests for the multi-source adapter registry (epic da#81).

The headline test is :meth:`TestCoverageInvariant.test_every_corpus_has_an_adapter`
— it fails CI if a ``SourceCorpus`` value is added without a matching adapter (or
vice versa), which is the enforcement that keeps the registry and the enum from
drifting as new sources are lit up.
"""

from __future__ import annotations

import inspect
from importlib import import_module

import pytest

from src.adapters import (
    SOURCE_REGISTRY,
    SourceAdapter,
    adapter_slugs,
    adapters_for_corpus,
    adapters_for_sect,
    covered_corpora,
    get_adapter,
    iter_adapters,
)
from src.models.enums import Sect, SourceCorpus

# The canonical run order. Pinning it here documents the contract and guards
# against an accidental reorder/drop in a future light-up PR.
EXPECTED_SLUGS = [
    "lk",
    "sanadset",
    "thaqalayn",
    "fawaz",
    "sunnah",
    "sunnah_scraped",
    "open_hadith",
    "muhaddithat",
    "itqan",
    "halimbahae",
    "mis",
    "bihar",
]


class TestCoverageInvariant:
    """Every SourceCorpus must be covered by an adapter, and vice versa."""

    def test_every_corpus_has_an_adapter(self) -> None:
        # If this fails, a SourceCorpus value was added without registering an
        # adapter in src/adapters.py (or an adapter names an unknown corpus).
        assert covered_corpora() == set(SourceCorpus)

    def test_every_adapter_corpus_is_a_known_enum_value(self) -> None:
        for adapter in SOURCE_REGISTRY:
            assert adapter.corpus in SourceCorpus

    def test_slugs_are_unique(self) -> None:
        slugs = adapter_slugs()
        assert len(slugs) == len(set(slugs))

    def test_canonical_run_order(self) -> None:
        assert adapter_slugs() == EXPECTED_SLUGS


class TestSharedCorpus:
    """Two adapters intentionally share the SUNNAH corpus (API + scraper)."""

    def test_sunnah_corpus_has_two_adapters(self) -> None:
        slugs = {a.slug for a in adapters_for_corpus(SourceCorpus.SUNNAH)}
        assert slugs == {"sunnah", "sunnah_scraped"}

    def test_other_corpora_have_exactly_one_adapter(self) -> None:
        for corpus in SourceCorpus:
            if corpus is SourceCorpus.SUNNAH:
                continue
            assert len(adapters_for_corpus(corpus)) == 1


class TestSect:
    """Sect declarations and the multi-sect (None) convention."""

    def test_multi_sect_sources_declare_none(self) -> None:
        none_sect = {a.slug for a in SOURCE_REGISTRY if a.sect is None}
        assert none_sect == {"fawaz", "muhaddithat", "itqan"}

    def test_at_least_one_reachable_sunni_and_one_reachable_shia(self) -> None:
        # da#81 acceptance: >=1 Sunni + >=1 Shia source must be loadable.
        reachable_sunni = [
            a for a in adapters_for_sect(Sect.SUNNI) if a.reachable and a.sect is Sect.SUNNI
        ]
        reachable_shia = [
            a for a in adapters_for_sect(Sect.SHIA) if a.reachable and a.sect is Sect.SHIA
        ]
        assert reachable_sunni, "no reachable Sunni source"
        assert reachable_shia, "no reachable Shia source"

    def test_adapters_for_sect_includes_multi_sect_sources(self) -> None:
        # A multi-sect (None) source contributes to every sect's coverage.
        for sect in Sect:
            covered = adapters_for_sect(sect)
            assert all(a.sect is sect or a.sect is None for a in covered)
            assert any(a.sect is None for a in covered)

    def test_declared_sect_agrees_with_parser_constant(self) -> None:
        # Where a parser hardcodes a module-level SECT constant, the registry's
        # declared sect must match it — so the registry is the single source of
        # truth and the two cannot silently diverge.
        for adapter in SOURCE_REGISTRY:
            module = import_module(f"src.parse.{adapter.parse_module}")
            parser_sect = getattr(module, "SECT", None)
            if parser_sect is None:
                continue
            assert adapter.sect is not None, (
                f"{adapter.slug}: parser declares SECT={parser_sect!r} but registry sect is None"
            )
            assert adapter.sect.value == parser_sect, (
                f"{adapter.slug}: registry {adapter.sect.value!r} != parser SECT {parser_sect!r}"
            )


class TestLookups:
    """Registry accessor helpers."""

    def test_get_adapter_returns_matching_row(self) -> None:
        adapter = get_adapter("thaqalayn")
        assert adapter.slug == "thaqalayn"
        assert adapter.corpus is SourceCorpus.THAQALAYN
        assert adapter.sect is Sect.SHIA

    def test_get_adapter_unknown_raises_keyerror(self) -> None:
        with pytest.raises(KeyError):
            get_adapter("does_not_exist")

    def test_iter_adapters_matches_registry(self) -> None:
        assert iter_adapters() == SOURCE_REGISTRY


class TestEntryPoints:
    """The acquire/parse entry points each adapter declares actually exist with a
    callable of the right arity — without invoking them (no network/IO)."""

    def test_acquire_target_is_callable_with_one_arg(self) -> None:
        for adapter in SOURCE_REGISTRY:
            module = import_module(f"src.acquire.{adapter.acquire_module}")
            fn = getattr(module, adapter.acquire_fn)
            assert callable(fn)
            params = inspect.signature(fn).parameters
            assert len(params) >= 1, f"{adapter.slug}: acquire fn takes no args"

    def test_parse_target_is_callable_with_two_args(self) -> None:
        for adapter in SOURCE_REGISTRY:
            module = import_module(f"src.parse.{adapter.parse_module}")
            fn = getattr(module, adapter.parse_fn)
            assert callable(fn)
            params = inspect.signature(fn).parameters
            assert len(params) >= 2, f"{adapter.slug}: parse fn takes < 2 args"

    def test_adapter_is_frozen(self) -> None:
        adapter = SOURCE_REGISTRY[0]
        with pytest.raises(Exception):  # noqa: B017 — FrozenInstanceError
            adapter.slug = "mutated"  # type: ignore[misc]


class TestRunAllDerivesFromRegistry:
    """The acquire/parse orchestrators must iterate the registry, not a private list."""

    def test_acquire_run_all_uses_registry(self) -> None:
        source = inspect.getsource(import_module("src.acquire").run_all)
        assert "SOURCE_REGISTRY" in source

    def test_parse_run_all_uses_registry(self) -> None:
        source = inspect.getsource(import_module("src.parse").run_all)
        assert "SOURCE_REGISTRY" in source


def test_source_adapter_is_dataclass_instance() -> None:
    assert all(isinstance(a, SourceAdapter) for a in SOURCE_REGISTRY)
