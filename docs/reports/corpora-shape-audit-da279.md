# Corpora-shape audit: dedup + normalization assumptions vs. source reality (da#279)

> Exploratory, owner-requested. Part of #279. Author: Oyunbileg Batbayar (QA / Data Quality).
> Instrument: the da#273 DuckDB tooling (`make duck`) over the run-3 staging + curated Parquet.
> Snapshot: read `2026-07-03` against `data/curated` (written `2026-07-02 23:05`) and `data/staging`.
> A live resolve run owns `data/`; every figure below is a point-in-time read, and every
> claim that rests on a count was re-checked by reading sample rows back (per the
> `count>=0`-masks-empty lesson). Numbers are the *current on-disk reality*, which differs
> from the `2026-06-16` composition memory (see § 8).

## 0. TL;DR — the seven load-bearing findings

| # | Mismatch | Where it bites | Est. volume | Head or tail |
|---|----------|----------------|-------------|--------------|
| 1 | Latin transliteration never folds to Arabic (`normalize_arabic` is Arabic-script-only) | narrator identity dedup | 12,057 canonical nodes / 65,143 mentions; 1,284 nodes at `mention_count>=5` | **head** (top nodes are Companions) |
| 2 | `fawaz` carries full Arabic isnad but NER routes it as English-only | NER extraction + feeds #1 | 36,100 hadiths with unused Arabic isnad; 33,943 Latin-only mentions | head + tail |
| 3 | Non-name context tokens (eulogy / verb / citation-number) survive into the dedup key | narrator identity dedup | 77,074 nodes at >=9 tokens (37% of nodes, <2% of mentions) | tail (node-count inflation) |
| 4 | `sunnah` collection numbering collides many hadiths onto one `source_id` | graph `MERGE` de-dupes distinct hadiths | 1,815 of 10,710 `sunnah` hadiths (17%) collapse | head (whole named books) |
| 5 | `halimbahae` + `bihar` are text-only in `full_text_ar` → invisible to both dedup layers | hadith dedup + textless graph nodes | 62,563 hadiths, **0** parallel-link pairs | head (whole corpora) |
| 6 | `thaqalayn` Arabic is unsplit (no `isnad_raw_ar`) → NER over-captures from `full_text_ar` | NER extraction, feeds #3 | 33,190 hadiths; largest contributor of >=9-token fragment nodes (24,572) | head + tail |
| 7 | FAISS semantic dedup embeds English matn only → misses Arabic-only corpora and over-links cross-sect on English boilerplate | hadith dedup precision/recall | ~738k hadiths have no `matn_en`; 256k `fawaz`↔`thaqalayn` cross-sect pairs | mixed |

Findings **1** and **2** are coupled: `fawaz` (English-routed) is the single largest
producer of un-foldable Latin narrator nodes, so fixing #2 removes most of #1's head volume.
The ranked list with proposed follow-ups is § 9.

## 1. Method and instrument

- Views registered by `src.tools.duck.build_registry` over `data/staging` + `data/curated`
  (36 views; the tool was pointed at the main-checkout data dirs read-only via `--staging`/`--curated`).
- Every corpus-level count below was cross-checked by reading representative rows — the
  hadith `source_id` collision (§ 3), the `fawaz` Arabic-isnad claim (§ 2), the Latin
  under-merge (§ 5), and the cross-corpus duplicate pairs (§ 7) each have a hand-verified sample.
- `normalize_arabic` behaviour is taken from the source (`src/utils/arabic.py`), not inferred.

## 2. Per-corpus shape profile (hadith layer)

`staging_hadiths` = 853,218 rows across 9 corpus shards, 851,148 distinct `source_id`
(the ~2,070 gap is almost entirely `sunnah`, § 3). Text-field coverage, read per corpus:

| corpus | hadiths | sect | `matn_ar` empty | `matn_en` empty | `isnad_raw_ar` empty | `isnad_raw_en` empty | `full_text_ar` empty | `grade` empty |
|--------|--------:|------|------:|------:|------:|------:|------:|------:|
| `sanadset` | 650,986 | sunni | 0.0% | 100% | 25.5% | 100% | 0.0% | 100% |
| `halimbahae` | 62,169 | sunni | **100%** | 100% | 100% | 100% | 0.0% | 100% |
| `fawaz` | 36,512 | sunni | 1.1% | 1.1% | **100%** | **100%** | 1.1% | 42.0% |
| `lk` | 34,088 | sunni | 2.5% | 0.7% | 0.0% | 1.4% | 0.0% | 1.5% |
| `thaqalayn` | 33,190 | shia | 0.0% | 0.0% | **100%** | 16.1% | 0.0% | 55.4% |
| `tusi` | 17,421 | shia | 0.0% | 100% | 1.9% | 100% | 0.0% | 100% |
| `sunnah` | 10,710 | sunni | 0.0% | 0.0% | 100% | 100% | 0.0% | 100% |
| `mis` | 7,748 | sunni | 2.5% | 100% | 100% | 100% | 100% | 100% |
| `bihar` | 394 | shia | **100%** | 100% | 100% | 100% | 0.0% | 100% |

Structural reading of each corpus:

- **`sanadset` (650,986, the firehose).** Arabic-only, no English, no grade. Its value is the
  isnad chains, but 25.5% of rows carry no separated `isnad_raw_ar` (the chain sits inside
  `full_text_ar`). This is the coarse Arabic mention source that dominates everything downstream.
- **`lk` (34,088) — the only well-formed bilingual source.** Separated Arabic isnad (0% empty),
  bilingual matn, and grade all present. This is the shape the whole pipeline was designed for;
  it is the exception, not the rule.
- **`fawaz` (36,512) is bilingual with Arabic isnad in `matn_ar`, but has NO separated
  `isnad_raw_*` and NO `isnad_raw_en`.** Verified by reading rows: `matn_ar` holds the classic
  Arabic transmission chain (`حَدَّثَنَا عَبْدُ اللَّهِ بْنُ مَسْلَمَةَ ...`) followed by the
  matn; 36,100 of 36,512 rows carry this Arabic. `matn_en` holds the English translation.
  See § 4 for why NER ignores the Arabic here.
- **`thaqalayn` (33,190) has NO Arabic isnad split** — its `arabicText` becomes `matn_ar`/`full_text_ar`
  whole (isnad + matn together, `isnad_raw_ar` 100% empty), while the English side *is* split
  (`isnad_raw_en` 83.9% present). This asymmetry is the root of the § 6 fragment pollution.
- **`tusi` (17,421)** is the CC0 `ThaqalaynData` Four-Books source: Arabic-only (English dropped
  by design), with a real Arabic isnad chain present (`isnad_raw_ar` 1.9% empty), no gradings.
- **`sunnah` (10,710)** is bilingual but has no isnad split and no grade; its problem is numbering (§ 3).
- **`mis` (7,748)** carries `matn_ar` but nothing else; its documented value (multi-isnad chains)
  lives in `network_edges_mis`, not in these hadith rows.
- **`halimbahae` (62,169) and `bihar` (394) are text-only in `full_text_ar`** — `matn_ar`,
  `matn_en`, and every isnad field are 100% empty. Two consequences: the graph node loader
  (which reads `matn_ar`) yields textless nodes unless the `full_text_ar`→`matn_ar` fallback
  is applied, and both dedup detectors (which also read `matn_ar`/`matn_en`) never see them (§ 7).

## 3. Numbering-scheme mismatch: `sunnah` `source_id` collisions

`sunnah` has 10,710 rows but only 8,895 distinct `source_id` — **1,815 hadiths (17%) share an
id with another hadith** and would be collapsed by the graph loader's `MERGE`. The worst single
id, `sunnah:bulugh:2:2:0`, is shared by **461 rows**. Reading four of them back confirms they are
genuinely distinct hadiths (different Companions: `'Abdullah bin 'Amr`, `Buraidah`, `Abu Musa`,
`Abu Barza`) — not duplicates.

Root cause is the known collection-ref vs. in-book-ordinal split (da#77): for `bulugh al-maram`
and `mishkat`, the `(book_number, chapter_number, hadith_number)` tuple that composes the
`source_id` is not populated per hadith (only 10 rows even have `hadith_number=0`; the rest repeat
a chapter-level tuple), so every hadith under a chapter collapses to one node. This is head-visible
data loss: entire chapters of `bulugh al-maram` reduce to a single Hadith node.

## 4. Normalization reality-check

`normalize_arabic` (`src/utils/arabic.py`) is a six-step **Arabic-script-only** pipeline:
strip diacritics (`U+064B–U+065F`, `U+0670`), fold alif variants, fold hamza variants
(`ؤ`/`ئ` → base), fold `taa marbuta` (`ة` → `ه`), strip tatweel (`U+0640`), collapse whitespace.
It does exactly what it claims for Arabic orthography. What it does **not** do — and where each
gap costs us:

1. **No cross-script folding (the #271 class, confirmed and the largest).** A Latin
   transliteration is passed through untouched, so `Abu Huraira`, `Abu Hurairah`, and
   `أبو هريرة` are three distinct dedup keys. This is not a corner case: it strands head
   Companions (§ 5).
2. **No stripping of embedded eulogies, transmission verbs, or connective particles.** These
   ride into the normalized name and fragment one person across many keys (§ 6). Counts over the
   210,494 canonical names: `عن` ("from") in 4,333 names, `رضى`/`رضي` ("may God be pleased") in
   8,463, `قال` ("said") in 7,366, an Arabic comma `،` in 2,777.
3. **No stripping of citation/index numbers.** 14,342 canonical names contain a digit — rijal-index
   tokens such as `( 3412 )` or `2617 ت` leaking out of `sanadset` / bio cross-references into the name.
4. **No script guard on the name field.** English matn fragments captured as "names" survive as
   Latin canonical nodes (`Allah's Messenger ...`, `A man demanded ...`) — same mechanism as #1
   but on non-name text.

Two upstream *routing* decisions interact with (1):

- **`fawaz` is classified English-only.** NER's `_ENGLISH_SOURCES = {"fawaz", "sunnah"}` runs
  English keyword extraction, which reads `isnad_raw_en` (0% present for `fawaz`) then falls back
  to `full_text_en`. The rich Arabic isnad in `matn_ar` (36,100 rows of `حَدَّثَنَا X ... حَدَّثَنَا Y`)
  is never read. Result: 33,943 `fawaz` mentions, **100% Latin-only** — the biggest single feeder
  of un-foldable Latin narrator nodes.
- **`lk` English fallback.** Even the gold bilingual source has 41,998 of 229,408 phase-1 mentions
  (18.3%) with an empty `name_ar`; NER falls back to `name_en`, so `lk` alone contributes ~42k
  Latin/transliterated mentions (13.3% of its resolved mentions are Latin-only).

## 5. Dedup layer 1 — narrator identity (exact-name → `fuzzy_cluster`)

The narrator dedup keys a canonical id on `make_canonical_id(name_ar_normalized)` — an
exact-normalized-name identity, later widened by `fuzzy_cluster` and by itqan aliases. The core
assumption is that the name field is a clean, script-consistent name. Measured against reality:

**Scale.** 210,494 canonical narrators, 3,118,034 total mentions. By script of `name_ar_normalized`:
93.5% Arabic, **5.7% Latin-only (12,057 nodes / 65,143 mentions; 1,284 at `mention_count>=5`)**.
By `sect_affiliation`: `neutral` 7,631 nodes but 2,194,259 mentions (the heavily-cited early
narrators), `sunni` 86,426 / 826,482, `shia` 60,574 / 90,541, `unknown` 55,863 / 6,752.
35.8% of canonical nodes (75,311) have zero mentions — bio-only records from `itqan`/`kaggle`.

**Head-visible cross-script under-merge (finding #1).** The exemplar is `Abu Hurayra`. There is a
correctly-merged Arabic node with `mention_count` 49,308, and *beside it* the top Latin nodes are:

| Latin canonical name | `mention_count` |
|----------------------|----------------:|
| `Abu Huraira` | 3,235 |
| `Abu Hurairah` | 2,997 |
| `Malik` | 1,851 |
| `Anas` | 1,618 |
| `Yahya` | 1,549 |
| `Ibn Abbas` | 1,358 |
| `Ibn Umar` | 1,352 |
| `Aishah` | 1,091 |
| `` `Aisha `` | 830 |
| `` Ibn `Abbas `` | 771 |

Two failures compound here: `Abu Huraira` and `Abu Hurairah` do not merge with each other
(transliteration variant), and neither merges with the Arabic node. Grouping every `Hurayra`
spelling: the Arabic side is 303 nodes totalling 50,678 mentions (one true node at 49,308, a
110-node `mention_count=1` tail); the Latin side is 280 separate nodes totalling **7,697 mentions**
(top 3,235). So one Companion loses ~7,697 mentions to cross-script non-folding, and the same
pattern repeats for `` `Aisha ``/`Aishah` and `Ibn Abbas`/`` Ibn `Abbas `` — the internal Latin
punctuation/spelling variants (`` ` `` vs none, `-h` vs none) fork the Latin side further.

**Alias reach is partial.** The itqan alias layer is large (154,328 rows, 139,536 distinct
Arabic aliases over 61,082 canonical names) and 71,246 canonical nodes (33.8%) carry >=1 alias —
but aliases are Arabic-only and anchored to bio-promoted (itqan) narrators. They do nothing for
the mention-driven Latin nodes above; the under-merge sits exactly where the alias layer does not reach.

## 6. Fragment pollution and the `thaqalayn` root cause (findings #3, #6)

The name field carries chain context at scale. Token-count distribution of `name_ar_normalized`:

| tokens | canonical nodes | total mentions | mentions / node |
|-------:|---------------:|---------------:|----------------:|
| 1 | 6,247 | 686,719 | 110.0 |
| 2 | 11,800 | 544,450 | 46.1 |
| 3 | 23,787 | 957,722 | 40.3 |
| 4 | 20,427 | 447,213 | 21.9 |
| 5 | 19,396 | 199,112 | 10.3 |
| 6 | 19,905 | 117,980 | 5.9 |
| 7 | 15,448 | 53,162 | 3.4 |
| 8 | 16,410 | 46,081 | 2.8 |
| >=9 | **77,074** | 65,595 | **0.85** |

The `>=9`-token bucket is **37% of all canonical nodes but under 2% of mentions** — a real Arabic
narrator name is rarely nine tokens. These are chain fragments (`ابو اسحاق ، عن ابي هريره` =
"Abu Ishaq, from Abi Huraira") and matn snippets captured whole. Tracing which corpus feeds them
(canonical nodes with >=1 mention from each corpus, so a node can count under more than one):
`thaqalayn` 24,572, `sanadset` 6,821, `lk` 2,023, `fawaz` 371, `sunnah` 47.

`thaqalayn` — not `sanadset` — is the largest fragment source, and § 2 explains why: its
`arabicText` is never split into isnad vs. matn (`isnad_raw_ar` 100% empty), so Arabic NER runs its
segmenter over `full_text_ar` (isnad **and** matn) and over-captures. The corpus shape (unsplit
Arabic) directly produces the extraction failure. The good news is that this pollution is
overwhelmingly mention-light tail (`mention_count<=1`): the mention-weighted graph is far healthier
than the node count suggests, but any node-count metric, blocking pool, or fuzzy-cluster candidate
set is badly inflated by it.

## 7. Dedup layer 2 — hadith parallels (FAISS semantic + lexical)

`staging_parallel_links` holds 6,656,888 pairs (the composed output of the FAISS semantic detector
and the deterministic lexical detector): 236,421 `verbatim` (avg sim 0.947), 1,120,184
`close_paraphrase` (0.833), 5,300,283 `thematic` (0.677); 31.3% cross-corpus, 567,276 cross-sect.

**Coverage gap by corpus shape.** The FAISS path embeds `matn_en` and skips empty — so it only
sees the four corpora with English matn (`fawaz`, `lk`, `thaqalayn`, `sunnah`). The ~738k hadiths
with no `matn_en` (`sanadset`, `halimbahae`, `tusi`, `mis`, `bihar`) rely entirely on the lexical
Arabic detector. And neither detector reads `full_text_ar`, so:

| corpus | pairs touching it |
|--------|------------------:|
| `sanadset` | 5,694,430 |
| `fawaz` | 2,979,611 |
| `lk` | 2,299,865 |
| `thaqalayn` | 1,696,116 |
| `sunnah` | 594,787 |
| `mis` | 34,165 |
| `tusi` | 14,802 |
| `halimbahae` | **0** |
| `bihar` | **0** |

`halimbahae` (62,169) and `bihar` (394) appear in **zero** pairs — verified directly. 62,563
hadiths are wholly outside parallel detection because their only text is in `full_text_ar`
(finding #5). This is the same text-field mismatch as the textless-graph-node problem, surfacing in dedup.

**Cross-corpus links — what dedup does catch.** Top cross-corpus pair counts: `fawaz`↔`lk` 787,215,
`lk`↔`sanadset` 261,683, `fawaz`↔`thaqalayn` 256,086, `lk`↔`thaqalayn` 247,230, `fawaz`↔`sunnah`
231,840. Reading `verbatim` `lk`↔`fawaz` pairs back confirms the six-books duplication **is** caught:
e.g. an `lk` row whose `matn_ar` opens with the isnad (`حَدَّثَنَا مُسَدَّدُ ...`) matched to a `fawaz`
row whose `matn_ar` opens with the matn (`كَانَ رَسُولُ اللَّهِ ...`) at sim 0.985 — the same hadith,
matched by the lexical detector despite the two corpora storing isnad+matn differently in `matn_ar`.

**Cross-sect over-linking (precision).** The 256k `fawaz`↔`thaqalayn` cross-sect pairs are *partly*
real. Reading the top `verbatim` cross-sect pairs: `Ibn Abbas` on `Umrah` entering `Hajj` matches a
`thaqalayn` `Ibn Abbas` narration at 0.968 (a genuine shared Sunni–Shia narration); `` `Ali `` asking
the Prophet matches at 0.925 (genuine). But others at the same 0.925+ `verbatim` tier pair clearly
different hadiths that merely share the English boilerplate `"The Messenger of Allah (s) said ..."`.
The FAISS English-embedding path over-links on formulaic translation phrasing, most visibly
cross-sect where the underlying Arabic actually differs — so the cross-sect parallel count is a soft
upper bound, not a verified duplicate set.

## 8. Cross-corpus collision matrix and the run-3 composition delta

The six Sunni canonical books are multi-loaded across `lk`, `fawaz`, `sunnah`, and `sanadset` in
run-3, which is why intra- and cross-corpus pair counts are so large (`fawaz`↔`lk` alone is 787k
pairs). This is a **change from the `2026-06-16` composition** recorded in project memory, which had
resolved the duplication by making `lk` the six-books spine and trimming `fawaz` (to 122 hadiths)
and `halimbahae` (to Musnad Ahmad / Darimi / Malik). Run-3 instead carries the **full** corpora
(`fawaz` 36,512, `halimbahae` 62,169, `sunnah` 10,710), so the six-books duplication the June
C-dedup removed is present again. Either run-3 intentionally defers dedup to query time, or the
composition decision was not re-applied — this is an owner call, flagged in § 9 (#8), not resolved here.

Bio-source shape (narrator identity provenance): `itqan` 115,735 bios (31.7% dated, no English name),
`kaggle_narrators` 24,326 (undated, no English name, no upstream provenance recorded in this audit),
`muhaddithat` 113 (100% English name — the women-scholars source, which is why muhaddithat narrators
are Latin/English by construction).

## 9. Ranked mismatch list with proposed follow-ups

Ranked by (head-visibility × affected volume). These are **proposals** — scope/sequencing is an owner
call; no follow-up issues were filed by this audit.

1. **Cross-script Latin↔Arabic under-merge.** Corpora: `fawaz`, `sunnah`, `lk` (fallback). Volume:
   12,057 Latin canonical nodes / 65,143 mentions (1,284 at `mc>=5`); head Companions affected
   (`Abu Hurayra` loses 7,697 mentions). Head. *Proposed:* either (a) remove the largest source by
   fixing #2, and/or (b) add a transliteration-aware fold (romanization → a shared canonical key) in
   the dedup/normalization layer, at minimum for `mc>=5` Latin nodes. Overlaps #2.
2. **`fawaz` Arabic-isnad ignored (English-routing).** Corpus: `fawaz`. Volume: 36,100 hadiths with
   full Arabic isnad chains; 33,943 Latin-only mentions today. Head + tail. *Proposed:* reclassify
   `fawaz` to Arabic isnad extraction (it has `حدثنا`-form chains in `matn_ar`); this is already
   scoped as da#271 fix(a). Doing so also collapses most of #1's head volume.
3. **`sunnah` `source_id` numbering collision.** Corpus: `sunnah` (`bulugh al-maram`, `mishkat`).
   Volume: 1,815 of 10,710 hadiths (17%) collapse under `MERGE`; one id holds 461 distinct hadiths.
   Head. *Proposed:* extract the in-book ordinal into `hadith_number` for named/`bulugh`-style
   collections (the da#77 follow-up), then delete-then-reload (a keying change does not re-`MERGE`).
4. **`halimbahae` + `bihar` text-only in `full_text_ar`.** Corpora: `halimbahae` (62,169),
   `bihar` (394). Volume: 62,563 hadiths — textless graph nodes AND zero dedup coverage. Head.
   *Proposed:* apply the `full_text_ar`→`matn_ar` fallback (the da#190 loader pattern) AND make both
   dedup detectors read `full_text_ar` when `matn_ar` is empty, so these corpora enter parallel detection.
5. **Non-name context tokens in the dedup key.** Corpora: all, worst `thaqalayn`/`sanadset`.
   Volume: 77,074 nodes at >=9 tokens (37% of nodes); 8,463 with eulogy, 7,366 with `قال`, 14,342
   with digits. Tail by mentions, but inflates node count / blocking / cluster candidates.
   *Proposed:* extend `name_quality` to strip trailing eulogies (`رضى الله عنه`), transmission verbs
   (`قال`, `عن`, `حدثنا`), and citation-index numbers before the identity key is computed.
6. **`thaqalayn` unsplit Arabic isnad.** Corpus: `thaqalayn` (33,190). Volume: largest fragment-node
   contributor (24,572). Head + tail. *Proposed:* segment `arabicText` into isnad vs. matn for
   `thaqalayn` (or drive NER from the already-split `thaqalaynSanad` English side and map back),
   so Arabic NER stops running over matn text. Reduces #5 at its root.
7. **FAISS semantic dedup coverage + cross-sect precision.** Volume: ~738k hadiths with no `matn_en`
   are FAISS-invisible; 256k `fawaz`↔`thaqalayn` cross-sect pairs are a soft upper bound. Mixed.
   *Proposed:* embed Arabic matn (multilingual model) so Arabic-only corpora get semantic coverage,
   and gate cross-sect `thematic` pairs (or raise their threshold) to curb English-boilerplate
   over-linking. Coordinate with the semantic-embedder parity gap tracked in the deploy repo.
8. **Run-3 re-introduced six-books duplication (composition regression).** Corpora: `lk`/`fawaz`/
   `sunnah`/`sanadset`. Volume: `fawaz`↔`lk` 787k pairs, six books multi-loaded. Owner call.
   *Proposed:* confirm whether the `2026-06-16` C-dedup composition (`lk` spine; trim `fawaz`/
   `halimbahae`) should be re-applied to run-3, or whether run-3 intends to dedup at query time.

## Appendix — provenance and caveats

- All figures are a single read of the run-3 on-disk Parquet on `2026-07-03`; a concurrent resolve
  run may mutate `data/staging`, so treat the mention/canonical figures as a snapshot.
- Script classification uses the Arabic block `U+0600–U+06FF` vs. `[A-Za-z]`; "Latin-only" means a
  name with Latin letters and no Arabic character.
- The `>=9`-token fragment metric and the corpus-origin join count a canonical node once per distinct
  corpus that mentions it, so per-corpus fragment totals are not mutually exclusive.
- Queries were run through `src.tools.duck.build_registry`; the harness and the exact SQL are
  reproducible against `data/staging` + `data/curated`.
