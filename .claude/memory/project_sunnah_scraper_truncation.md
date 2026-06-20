---
name: project_sunnah_scraper_truncation
description: sunnah_scraper dropped NAMED book segments (e.g. /introduction) → collections truncated; riyad lost 679/1896 hadiths. Other collections still affected on staging.
metadata: 
  node_type: memory
  type: project
  originSessionId: d8acc7c0-91ac-412b-b312-da38817b1614
---

`sunnah_scraper._get_book_numbers` enumerated only **digit-only** URL segments,
so sunnah.com's NAMED books (notably `/{collection}/introduction` = "The Book of
Miscellany") were silently dropped. Riyad as-Salihin lost its entire first book —
679 of 1,896 hadiths (scrape topped out at 1,217). Fixed in da#177/PR#187
(noorinalabs-data-acquisition): `_get_book_numbers` now returns `list[int|str]`
incl. named segments (ordered first, anchors folded); `_scrape_book_page` keys
named segments to book 0 and falls back to the collection-wide ref for
`hadith_number`. Verified live: 679 (introduction) + 1,217 (books 1–19) = 1,896.

**Why:** the "47-fragment" was never a data-availability problem — sunnah.com is
reachable; it was this enumeration bug + the legacy da#73 first-light loading only
book 1 (47).

**How to apply:** other Sunni collections scraped via this scraper ALSO have named
books — verified on the index pages: `hisn` and `mishkat` have `introduction`,
`shamail` has `8b`. da#174 bulk-loaded several of these (open_hadith, halimbahae,
musnad-ahmad, etc.) to staging with the OLD scraper, so those on staging are
likely still truncated by their named books. Before the prod full-corpus cutover
(deploy#470, [[project_p5w5_prodcutover_p6_dataquality]]) RE-SCRAPE the
sunnah_scraper collections with the da#177 fix and reconcile counts — a green
staging load count ≠ complete. Loader MERGE key is `hadith_node_id(source_id)`
(prop `id`, NOT `source_id` — that's null on nodes); a keying change (e.g. the
pre-in-book-ordinal `:0` fragment) does NOT MERGE → delete-then-load to avoid
stale duplicates. Load via the da#174 image `noorinalabs-graph-load:da174`
(`load --skip-validation` for nodes+APPEARS_IN; `--nodes-only` skips edges).
