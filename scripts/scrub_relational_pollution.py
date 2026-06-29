#!/usr/bin/env python3
"""One-off scrub: drop relational-pronoun (mubham) narrators from curated artifacts.

da#247 residual. The durable fix is the ``name_quality.py`` filter, which drops
these spans at the NER stage on the *next* full resolve run. This script is the
fast operational equivalent for stage rehearsal: it applies the SAME filter
(``clean_narrator_name``) to the already-built curated narrator artifacts so we
can wipe+reload a clean stage graph without re-running disambiguate/enrichment.

Why this is provably equivalent to re-running NER-with-filter (not a heuristic):
relational-pronoun names (``ابيه`` "his father", ``جده`` "his grandfather", …)
cluster only with themselves — disambiguate assigns canonical ids by normalized
name, so each forms a singleton canonical. Removing their mentions therefore
CANNOT change any surviving narrator's clustering; the result is identical to NER
never emitting them. We:

  1. drop ``narrators_canonical`` rows whose ``name_ar_normalized`` the filter
     rejects (collecting their canonical ids), and
  2. drop ``narrator_mentions_resolved*`` rows that reference a dropped id
     (NER ``dropped += 1; continue`` semantics — the mention never exists).

Idempotent: re-running finds nothing left to drop. Backs up curated/ first.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from src.parse.name_quality import clean_narrator_name

CURATED = Path("data/curated")
BACKUP = Path("data/curated.pre-pollution-scrub")
NARRATORS = "narrators_canonical.parquet"
MENTION_FILES = (
    "narrator_mentions_resolved.parquet",
    "narrator_mentions_resolved_muhaddithat.parquet",
)


def main() -> None:
    if not BACKUP.exists():
        shutil.copytree(CURATED, BACKUP)
        print(f"backed up {CURATED} -> {BACKUP}")

    nc = pq.read_table(CURATED / NARRATORS)
    arn = nc.column("name_ar_normalized").to_pylist()
    cid = nc.column("canonical_id").to_pylist()
    keep = [clean_narrator_name(n) is not None for n in arn]
    dropped_ids = {c for c, k in zip(cid, keep, strict=True) if not k}
    nc_clean = nc.filter(pa.array(keep))
    print(
        f"{NARRATORS}: {nc.num_rows} -> {nc_clean.num_rows} "
        f"(dropped {len(dropped_ids)} relational narrators)"
    )

    for fname in MENTION_FILES:
        path = CURATED / fname
        if not path.exists():
            continue
        t = pq.read_table(path)
        cni = t.column("canonical_narrator_id").to_pylist()
        mask = [c not in dropped_ids for c in cni]
        t_clean = t.filter(pa.array(mask))
        dropped = t.num_rows - t_clean.num_rows
        pq.write_table(t_clean, path)
        print(f"{fname}: {t.num_rows} -> {t_clean.num_rows} (dropped {dropped} mentions)")

    pq.write_table(nc_clean, CURATED / NARRATORS)
    print("scrub complete.")


if __name__ == "__main__":
    main()
