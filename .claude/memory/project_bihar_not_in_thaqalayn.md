---
name: project_bihar_not_in_thaqalayn
description: "da#95: Thaqalayn does NOT carry Bihar al-Anwar. Owner 2026-06-12 chose option A — build a real `bihar` adapter against hubeali.com with the Itqan licensing posture. NEW SourceCorpus.BIHAR (sect=shia). Don't re-litigate the license."
metadata: 
  node_type: memory
  type: project
  originSessionId: 090bf6d5-0d19-47c9-9b85-67bfff1c5396
---

da#95 ("Bihar al-Anwar via Thaqalayn — deepen Shia coverage") has a **false source premise**. Verified 2026-06-12:
- MohammedArab1/ThaqalaynAPI GitHub repo (what `src/acquire/thaqalayn.py` clones) = 33 books, NONE is Bihar. Live `https://thaqalayn.net/api/books` = 27 books, no Bihar. "bihar" only appears as in-text citations inside other books' hadiths.
- No openly-licensed, structured, per-hadith Bihar al-Anwar JSON dataset exists on GitHub (checked narmafraz/ThaqalaynData mirror, sayyid5416/shia-islamic-data=PDFs only, shamela mirrors). Bihar (~100k, Majlisi) needs a NON-Thaqalayn source.
- Existing `thaqalayn` adapter already globs ALL repo JSONs → Thaqalayn coverage is maximal; nothing more to add "via Thaqalayn."

**OWNER DECISION (2026-06-12): option A** — build the REAL `bihar` adapter against **hubeali.com** (per-hadith EN+AR Bihar al-Anwar, ~100k, Majlisi), chosen over B (hollow `reachable=False` scaffold) / C (defer) per [[project_p4_priority_data_first]]. **Licensing = same posture as [[project_itqan_license_proceed]]**: non-profit, facts in our own schema, cleanly removable via `source_corpus=bihar` provenance, no upstream license; record on the SourceAdapter `license_note`. Owner-approved — do NOT re-surface as a blocker. Build follows existing scraper conventions (sunnah_scraper/open_hadith: robots.txt, throttle, raw-cache, idempotent); bounded real scrape for the PR, full-corpus scrape is a runtime/data-load step. (Menu A/B/C was surfaced; A picked.)

**Decided regardless:** NEW `SourceCorpus.BIHAR = "bihar"` (not a thaqalayn extension) — distinct collection, corpus is the identity namespace, and the real source isn't Thaqalayn. Touches enums.py + adapters.py SOURCE_REGISTRY + tests/test_adapters.py EXPECTED_SLUGS — same 3 files as #96/#97, serial-merge with rebase (#81 coverage invariant requires enum+row together). See [[feedback_verify_diagnosis_before_delegating]].

**SHIPPED 2026-06-12: PR #134 OPEN** (base wave-4, MERGEABLE). source_id grammar `bihar:bihar-al-anwar:<vol>:<chapter>:<ordinal>` — collection slug `bihar-al-anwar` NOT `bihar` (else `identity.is_double_prefixed` collapses it). hubeali markup: per volume/part WordPress page, chapters `باب N`, hadiths open `<N>- <isnad+matn>`, alternating AR/EN `<p>`. Live vol-1 crawl = 394 real bilingual hadiths/7 chapters; live leg skip-guarded `BIHAR_LIVE_TEST=1`; 272 passed/2 skipped + mypy/ruff clean. Rebased past #96 (halimbahae) → registry tail now `... itqan, halimbahae, bihar`. #97 (Kwesi) will conflict same 3 files — keep-all. Full ~100k multi-volume scrape + neo4j E2E = runtime carry-forward, not PR-gated ([[feedback_runtime_gate_scoping]]).
