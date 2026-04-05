"""CLI entry point for the isnad-graph-ingestion pipeline."""

from __future__ import annotations

import sys


def main() -> None:
    """Entry point for the isnad-ingest CLI."""
    print(f"isnad-ingest: {sys.argv[1:]}")


if __name__ == "__main__":
    main()
