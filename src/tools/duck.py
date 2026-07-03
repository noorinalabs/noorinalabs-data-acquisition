"""Local DuckDB exploration over the pipeline's staging + curated Parquet (da#273).

A **dev-only** instrument. Opens an in-memory DuckDB with one read-only view per
logical Parquet dataset under the configured staging + curated dirs, so ad-hoc
questions ("top blocks by member count", "mention_count distribution per canonical
narrator") are one SQL query instead of a bespoke PyArrow script.

Read-only by construction: the helper only issues ``CREATE VIEW ... read_parquet``
— it never writes into ``data/``. Paths come from :mod:`src.config` settings, never
hardcoded, so it tracks whatever ``DATA_STAGING_DIR`` / ``DATA_CURATED_DIR`` point at.

View naming (derived from file stems, no hardcoded dataset vocabulary):

- **Per-file view** for every ``*.parquet``: ``<layer>_<stem>`` (e.g.
  ``curated_narrators_canonical``, ``staging_hadiths_sanadset``).
- **Combined dataset view** for a multi-file dataset — a set of ≥2 files that share
  a leading name prefix *and an identical column schema* (true shards of one
  producer): ``<layer>_<prefix>`` unioned with ``read_parquet([...])``. The
  schema-identity requirement is what keeps distinct datasets that merely share a
  name prefix apart — e.g. ``narrator_aliases_*`` and ``narrator_mentions_*`` both
  begin ``narrator`` but have different schemas, so no bogus ``staging_narrator``
  view is created.

Usage::

    python -m src.tools.duck                  # interactive SQL REPL
    python -m src.tools.duck -c "SELECT ..."  # one-shot, pretty table
    python -m src.tools.duck -c "..." --csv   # one-shot, CSV to stdout (scriptable)
    python -m src.tools.duck --list           # list registered views, then exit
    python -m src.tools.duck --staging DIR --curated DIR   # override the data dirs

``make duck`` wraps the interactive form; ``make duck QUERY="select ..."`` the
one-shot form.
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections.abc import Callable, Hashable, Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from src.utils.logging import get_logger

if TYPE_CHECKING:
    import duckdb

logger = get_logger(__name__)

_PARQUET_GLOB = "*.parquet"


# ---------------------------------------------------------------------------
# Dataset discovery (pure — no DuckDB, no I/O beyond the injected schema probe)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Dataset:
    """One registerable view: a name and the Parquet file(s) it reads."""

    name: str
    files: tuple[Path, ...]

    @property
    def is_combined(self) -> bool:
        """True when this view unions more than one shard file."""
        return len(self.files) > 1


def _segments(stem: str) -> list[str]:
    """Underscore-delimited segments of a file stem (the dataset naming unit)."""
    return stem.split("_")


def plan_datasets(
    files: Iterable[Path],
    schema_key: Callable[[Path], Hashable],
) -> list[Dataset]:
    """Plan the set of views for one layer's Parquet files (pure, deterministic).

    Produces:

    - one per-file dataset ``<stem>`` for every file, and
    - one combined dataset ``<prefix>`` for each maximal leading-segment prefix that
      (a) is a *proper* prefix of ≥2 files' stems and (b) over which every such file
      shares one ``schema_key`` — i.e. genuine shards of a single producer.

    ``schema_key`` maps a file to a hashable identity for its columnar schema; files
    are grouped into a combined view only when their keys are all equal, so two
    datasets that merely share a name prefix (different schemas) never merge. A
    combined view whose name would collide with a per-file view is dropped (the
    per-file view wins). Results are sorted by name for stable output.
    """
    files = sorted(set(files), key=lambda p: p.stem)
    per_file = {f.stem: Dataset(name=f.stem, files=(f,)) for f in files}

    # Candidate combined-view prefixes: every proper leading-segment prefix of every
    # stem, mapped to the files it strictly extends ("<prefix>_<rest>").
    prefix_files: dict[tuple[str, ...], list[Path]] = {}
    for f in files:
        segs = _segments(f.stem)
        for cut in range(1, len(segs)):
            prefix = tuple(segs[:cut])
            prefix_files.setdefault(prefix, []).append(f)

    # Keep prefixes covering ≥2 files that all share one schema.
    qualifying: dict[tuple[str, ...], tuple[Path, ...]] = {}
    for prefix, members in prefix_files.items():
        if len(members) < 2:
            continue
        keys = {schema_key(m) for m in members}
        if len(keys) != 1:
            continue  # not shards of one dataset — leave as separate per-file views
        qualifying[prefix] = tuple(sorted(members, key=lambda p: p.stem))

    # Prefer the LONGEST prefix among any that cover the exact same file-set
    # (so 3 narrators_bio_* shards name the view ``narrators_bio``, not ``narrators``).
    combined: dict[str, Dataset] = {}
    for prefix, shard_files in qualifying.items():
        longer_same = any(
            other != prefix and other_files == shard_files and len(other) > len(prefix)
            for other, other_files in qualifying.items()
        )
        if longer_same:
            continue
        name = "_".join(prefix)
        if name in per_file:
            continue  # a real file already owns this exact name
        combined[name] = Dataset(name=name, files=shard_files)

    return sorted([*per_file.values(), *combined.values()], key=lambda d: d.name)


# ---------------------------------------------------------------------------
# DuckDB connection + view registration
# ---------------------------------------------------------------------------
@dataclass
class RegisteredView:
    """A view successfully created in the DuckDB session."""

    name: str
    layer: str
    files: tuple[Path, ...]


@dataclass
class Registry:
    """Outcome of building a DuckDB session over the data dirs."""

    connection: duckdb.DuckDBPyConnection
    views: list[RegisteredView] = field(default_factory=list)


def _pyarrow_schema_key(path: Path) -> Hashable:
    """Schema identity of a Parquet file from its footer metadata (no data read)."""
    import pyarrow.parquet as pq

    schema = pq.read_schema(path)
    # Order-insensitive so a column reordering across shards still groups; DuckDB's
    # ``union_by_name`` reconciles the actual read.
    return tuple(sorted((f.name, str(f.type)) for f in schema))


def _quote_sql_str(value: str) -> str:
    """Single-quote a string literal for inline SQL (escaping embedded quotes)."""
    return "'" + value.replace("'", "''") + "'"


def _quote_ident(name: str) -> str:
    """Double-quote a SQL identifier (escaping embedded double quotes)."""
    return '"' + name.replace('"', '""') + '"'


def _register_layer(
    con: duckdb.DuckDBPyConnection,
    layer: str,
    data_dir: Path,
    schema_key: Callable[[Path], Hashable],
) -> list[RegisteredView]:
    """Create ``<layer>_<dataset>`` views over every Parquet dataset in ``data_dir``."""
    if not data_dir.exists():
        logger.warning("duck_data_dir_missing", layer=layer, path=str(data_dir))
        return []

    parquet_files = sorted(data_dir.glob(_PARQUET_GLOB))
    if not parquet_files:
        logger.info("duck_no_parquet", layer=layer, path=str(data_dir))
        return []

    registered: list[RegisteredView] = []
    for ds in plan_datasets(parquet_files, schema_key):
        view = f"{layer}_{ds.name}"
        file_list = ", ".join(_quote_sql_str(str(p)) for p in ds.files)
        sql = (
            f"CREATE OR REPLACE VIEW {_quote_ident(view)} AS "
            f"SELECT * FROM read_parquet([{file_list}], union_by_name=true)"
        )
        try:
            con.execute(sql)
        except Exception as exc:  # noqa: BLE001 — one bad file must not abort the rest
            logger.warning(
                "duck_view_failed", view=view, files=[str(p) for p in ds.files], error=str(exc)
            )
            continue
        registered.append(RegisteredView(name=view, layer=layer, files=ds.files))
    return registered


def build_registry(
    staging_dir: Path,
    curated_dir: Path,
    *,
    schema_key: Callable[[Path], Hashable] | None = None,
) -> Registry:
    """Open an in-memory DuckDB and register read-only views over both data layers.

    The connection is in-memory (``:memory:``) and every view is a lazy
    ``read_parquet`` over the on-disk files — nothing is written back into
    ``data/``, and each query re-scans the files, so a shard replaced atomically by
    the running pipeline mid-session is picked up on the next query.
    """
    import duckdb

    key = schema_key or _pyarrow_schema_key
    con = duckdb.connect(":memory:")
    views: list[RegisteredView] = []
    views.extend(_register_layer(con, "staging", staging_dir, key))
    views.extend(_register_layer(con, "curated", curated_dir, key))
    logger.info(
        "duck_registry_built",
        staging_dir=str(staging_dir),
        curated_dir=str(curated_dir),
        views=len(views),
    )
    return Registry(connection=con, views=views)


# ---------------------------------------------------------------------------
# CLI / REPL
# ---------------------------------------------------------------------------
def _format_view_list(views: list[RegisteredView]) -> str:
    """Human-readable listing of registered views grouped by layer."""
    if not views:
        return "(no views — no Parquet files found under the configured data dirs)"
    lines: list[str] = []
    for layer in ("staging", "curated"):
        layer_views = [v for v in views if v.layer == layer]
        if not layer_views:
            continue
        lines.append(f"{layer}:")
        for v in layer_views:
            shards = f"  ({len(v.files)} files)" if len(v.files) > 1 else ""
            lines.append(f"  {v.name}{shards}")
    return "\n".join(lines)


def _print_result(con: duckdb.DuckDBPyConnection, query: str, *, as_csv: bool) -> None:
    """Run ``query`` and print the result as a pretty table or CSV to stdout."""
    rel = con.sql(query)
    if rel is None:  # a statement that returns no result set
        return
    if as_csv:
        writer = csv.writer(sys.stdout)
        writer.writerow(rel.columns)
        writer.writerows(rel.fetchall())
    else:
        rel.show(max_rows=1000)


def _repl(registry: Registry) -> None:
    """Minimal interactive SQL loop over the registered views."""
    con = registry.connection
    print("DuckDB over staging + curated Parquet (read-only). Views:")
    print(_format_view_list(registry.views))
    print("Enter SQL (one statement per line). Commands: .views  .quit\n")
    while True:
        try:
            line = input("duck> ").strip()
        except EOFError:
            print()
            break
        if not line:
            continue
        if line in (".quit", ".exit", "quit", "exit"):
            break
        if line in (".views", ".tables"):
            print(_format_view_list(registry.views))
            continue
        try:
            _print_result(con, line, as_csv=False)
        except Exception as exc:  # noqa: BLE001 — a bad query must not kill the session
            print(f"error: {exc}", file=sys.stderr)


def _resolve_dirs(args: argparse.Namespace) -> tuple[Path, Path]:
    """Resolve staging/curated dirs from CLI overrides, else config settings."""
    if args.staging and args.curated:
        return Path(args.staging), Path(args.curated)
    from src.config import get_settings

    settings = get_settings()
    staging = Path(args.staging) if args.staging else settings.data_staging_dir
    curated = Path(args.curated) if args.curated else settings.data_curated_dir
    return staging, curated


def main(argv: list[str] | None = None) -> int:
    """Entry point for ``python -m src.tools.duck``."""
    parser = argparse.ArgumentParser(
        prog="python -m src.tools.duck",
        description="Interactive DuckDB over the pipeline's staging + curated Parquet.",
    )
    parser.add_argument("-c", "--query", help="Run a single SQL query, print the result, and exit.")
    parser.add_argument(
        "--csv", action="store_true", help="With -c, emit CSV to stdout instead of a table."
    )
    parser.add_argument(
        "--list", action="store_true", dest="list_views", help="List registered views and exit."
    )
    parser.add_argument("--staging", help="Override the staging dir (default: config setting).")
    parser.add_argument("--curated", help="Override the curated dir (default: config setting).")
    args = parser.parse_args(argv)

    staging_dir, curated_dir = _resolve_dirs(args)
    registry = build_registry(staging_dir, curated_dir)

    if args.list_views:
        print(_format_view_list(registry.views))
        return 0
    if args.query:
        _print_result(registry.connection, args.query, as_csv=args.csv)
        return 0
    _repl(registry)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
