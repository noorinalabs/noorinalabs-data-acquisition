---
name: project_thaqalayn_parser_schema
description: "thaqalayn adapter was not E2E-proven vs real ThaqalaynAPI; wrong field mapping → 0% matn, source_id collision, dup. FIXED + loaded da#175/PR#181."
metadata: 
  node_type: memory
  type: project
  originSessionId: d8acc7c0-91ac-412b-b312-da38817b1614
---

da#175 ("load Shia Four Books → staging") claimed the `thaqalayn` adapter was "code-complete, E2E-proven, no new code". FALSE against real upstream. Acquire clones MohammedArab1/ThaqalaynAPI (68 JSON); parse produced 113,401 rows but ALL unloadable:

- **0% `matn_ar`** (all empty); **59 unique source_ids** for 113,401 rows → MERGE would make ~59 garbage Hadith nodes.
- 4x dup: repo ships V1 named-book files + V2 numeric-id files + two `allBooks.json` aggregates (60/100MB; allBooks=56,371 rows). Parse `rglob("*.json")`s everything incl. `package-lock.json`/`tsconfig.json`/`BookNames.json`/`ingredients.json` → parsed as "collections".

ROOT CAUSE: `src/parse/thaqalayn.py` maps an ASSUMED schema (`hadithNumber`/`textAr`/`matn`/`grade`) that doesn't exist. Real record keys: `id`, `bookId`, `arabicText`, `englishText`, `majlisiGrading`/`behbudiGrading`/`mohseniGrading`, and V2-only `thaqalaynSanad`/`thaqalaynMatn`/`gradingsFull`/`frenchText`. `id` never read → hadith_number=0 → source_id = f(corpus,collection,0,0) collides. PR#122 parse test passed only because its fixture matched the wrong field names (fixture-masks-bug, see [[feedback_passing_repro_masks_bug]]).

REAL CLEAN TARGET (V2/ThaqalaynData numeric files, excl. allBooks/BookNames/Ingredients/config): **33,190 unique hadiths / 33 books; 100% arabicText; 27,863 with thaqalaynSanad; 14,805 graded; al-Kafi=14,245.** Thaqalayn has al-Kafi + Man-La-Yahduruh-al-Faqih but NOT Tahdhib/al-Istibsar → "Four Books" only partial upstream (owner flag). See [[project_bihar_not_in_thaqalayn]].

FIX SHIPPED (PR#181, owner-authorized): map real fields, identity `source_id=thaqalayn:<bookId>:<id>`, `collection_name=bookId` (so APPEARS_IN `{corpus}:{collection_name}` links), scope acquire+parse to V2/ThaqalaynData/<n>.json via shared `book_json_files()`, exclude aggregates+config, real-upstream fixtures, recalibrated drift baselines. **GOTCHA caught mid-fix: `thaqalaynSanad`/`thaqalaynMatn` are ENGLISH** (the English isnad/matn split), NOT Arabic — route them to `isnad_raw_en`/`matn_en`; `arabicText` (whole, no Arabic split) → `matn_ar`/`full_text_ar`; `isnad_raw_ar`=None. (An earlier pass mis-routed English into matn_ar → 17% Arabic; validate-staging's arabic_coverage check caught it.)

LOADED to staging (PR#181): 33,190 thaqalayn shia hadith / 33 collections / 33,190 APPEARS_IN / 14,805 gradings; narrators unchanged (no narrator writes → no clobber of itqan). Load path: ssh -L tunnel to neo4j container IP + real `isnad-ingest load` (see [[project_staging_unreachable_from_sandbox]]).

ANOMALY flagged: staging also has a separate `thaqalayn_data` (shia, 17,421) corpus key from da#174's multi-source run (+ likely-dup halimbahae=62,169 / open_hadith=62,169 pair). Possible double-counted Shia content — needs cross-corpus dedup audit before prod cutover.
