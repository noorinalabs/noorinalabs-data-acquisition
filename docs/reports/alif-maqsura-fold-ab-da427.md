# Alif-maqṣūra ى→ي fold: full-corpus A/B (da#427 AC#2 artifact)

> Deferred follow-up from da#427 (PR #476), recorded here per da#477 so the AC#2
> acceptance box has a durable artifact instead of living only in the #476 PR review.
> Base-vs-head bidirectional A/B (`main` vs. the da#427 `normalize_arabic` +
> `name_quality` change), on the `curated.pre-rerun-928` snapshot
> (`data/curated.pre-rerun-928-20260711T203626Z`): 3,249,721 mentions / 129,234
> canonical narrators. Per-row unweighted AND mention-weighted, mc=0 bio-promoted
> rows included (the da#424 lesson — a corruption/deletion check that skips the
> zero-mention-count tail misses the bio-only path, da#99/da#110).

## TL;DR

| Direction | Result |
|---|---|
| NEWLY-DELETED real narrators | **0** |
| NEWLY-KEPT junk (the `على` carve-out question) | **0** |
| Canonical-id re-key | 59 names / 60 mentions — all consistency-improving, **0** wrong merges |

## 1. Deletion direction (kept→dropped)

47 distinct `name_raw` / 81 mentions flip kept→dropped between base and head. Every
one is a maqṣūra-spelled variant whose yeh-twin (the ي-spelled form of the same
string) was **already** dropped on base — e.g. `ابى` → already-dropped `ابي`,
`حدثنى شعبه` → already-dropped `حدثني شعبه`. The head is not deleting anything base
kept; it is folding a maqṣūra spelling onto a drop decision base already made under
the ي spelling, so the drop-gate becomes spelling-consistent rather than
spelling-dependent.

32 canonical nodes go clean→un-nameable. All 32 are junk residue on inspection: no
death year, no `name_en`, no `external_id`. The largest (`mention_count` 43) is
`أَبَى`, a bare relational fragment ("his father" — the da#247 relational-pronoun
pollution class, `.claude/memory/project_relational_pollution_scrub_equiv.md`), not
a narrator.

## 2. Recall direction (dropped→kept) — the `على` carve-out question

da#427 removed `على` from `_MATN_PARTICLES` because the fold makes the preposition
`على` ("on") and the name `علي` (ʿAlī) homographic (both fold to `علي`); keeping
`على` in the particle set would drop a bare `علي` narrator span via the all-particle
rule (2c), deleting one of the most-attested transmitters. The tradeoff, per the
da#427 PR, is a recall loss in the safe direction: `على`-preposition matn residue
that base's particle-based drop-gate used to catch now survives.

**On this snapshot, that feared recall loss did not materialize: newly-kept junk =
0.** Removing `على` from `_MATN_PARTICLES` produced no new surviving matn fragment
here — no row exists in this corpus where a bare `على` span, undetected as a
particle, was the *only* thing keeping an otherwise-junk span alive.

## 3. Canonical-id re-key (consistency direction)

59 names / 60 mentions re-key to a different canonical id between base and head, all
consistency-improving. The dominant shape is `يعنى`-gloss stripping re-attaching a
mention to its clean real-narrator twin, e.g. `يحيى يعنى ابن سعيد` →
`يحيي ابن سعيد`. **Zero** wrong cross-narrator merges were found in this set. The
`يحيى`≡`يحيي` corpus-level merge itself is pre-existing (da#434
`_fold_residual_noise`), already folded at the mint surface on base; this A/B only
measures the *additional* re-keys the da#427 fold introduces on top of that.

## Disposition

- **AC#2 (da#427) is satisfied.** This is the bidirectional unweighted full-corpus
  A/B (mc=0 rows visible) the acceptance criterion asked for; it lives here as the
  durable artifact rather than only in the #476 review thread.
- **The `على` bare-particle carve-out (da#477 task 2) is NOT built.** The measured
  newly-kept junk is 0 on this snapshot, so the carve-out this task considered
  (detect `على` as a matn preposition while exempting a bare `علي` narrator span)
  would have no effect here — per da#477, it is not built speculatively. This
  finding is **corpus-gated, not closed**: the next production re-run is a different,
  unmeasured corpus, and if a future A/B on that corpus shows `على` matn residue
  inflating the canonical node of ʿAlī (mention-count inflation via the `على`≡`علي`
  id-level equivalence), re-open the carve-out question against that measurement.

## Method notes

- Snapshot is a point-in-time on-disk read (`data/curated.pre-rerun-928-*`); a live
  resolve run owns `data/`, so this reflects that snapshot only, not necessarily the
  current tree.
- `normalize_arabic` / `name_quality` behaviour is taken from source
  (`src/utils/arabic.py`, `src/parse/name_quality.py`), not inferred.
- See `tests/test_parse/test_name_quality.py::TestAlifMaqsuraFold` and
  `TestRecoveryIsNoWorseThanMain` for the fixture-level unit pins this full-corpus
  measurement complements (unit fixtures are the accumulated regression shapes;
  this report is the one-time full-corpus sweep).
