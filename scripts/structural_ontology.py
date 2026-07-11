#!/usr/bin/env python3
"""Generate noorinalabs-data-acquisition's structural ontology index (#214, da#405).

This is the noorinalabs-data-acquisition (consumer) side of the C×T2 distributed
structural ontology (parent meta noorinalabs-main#820, generator
noorinalabs-main#855; org-wide six-repo fan-out tracked in noorinalabs-main#820).
It does NOT contain the extraction logic -- that lives, single-source-of-truth, in
the OWNED generator package ``ontology_gen`` in **noorinalabs-main**
(``.claude/lib/ontology_gen/``). This script is a thin consumer wrapper that:

* locates that generator (see :func:`locate_generator`), and
* invokes it against this repo to (re)produce the two artifacts
  ``ontology/structural/{code-graph.json, llms.txt}``.

The structural index is a **gitignored build product** (main#939 / da#405), not a
committed artifact. Per the ratified C×T2 principle (#857) the structural layer is
*always-current-by-regeneration*: if it is always regenerable, a committed copy is
redundant -- and committing it made every concurrent PR conflict on a whole-file
generated artifact that carries no source-level information. So this script's only
job is ``emit``: (re)build the index on demand. There is no longer a staleness
``check`` (a gitignored build product has nothing to gate) nor a merge-driver to
register (nothing to merge once uncommitted).

Why a sibling generator instead of a vendored copy
==================================================
The generator is deliberately NOT copied into this repo. A vendored copy would
fork: a fix to the extractor in noorinalabs-main would silently not reach the six
consuming repos, re-introducing the drift the owned-generator design exists to
remove (eval noorinalabs-main#854). Instead the generator is consumed from a
single source of truth:

* **CI / cross-repo roll-up** -- noorinalabs-main's aggregator
  (``ontology_gen.aggregate``, main#946) regenerates this repo's index locally
  before rolling up, pointing at the generator directly.
* **Local dev** relies on the standard org layout (child repos cloned beneath
  ``noorinalabs-main/``); :func:`locate_generator` walks up to find the parent's
  ``.claude/lib/ontology_gen`` automatically. ``ONTOLOGY_GEN_LIB`` overrides for
  any non-standard checkout.

Subcommands
===========
* ``emit`` -- (re)generate the index in place under ``ontology/structural/``.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
REPO_NAME = "noorinalabs-data-acquisition"
OUT_REL = Path("ontology/structural")
ARTIFACTS = ("code-graph.json", "llms.txt")

# Env var a CI job (or a non-standard local checkout) uses to point at the
# directory that CONTAINS the ``ontology_gen`` package (i.e. the parent repo's
# ``.claude/lib``).
ENV_GEN_LIB = "ONTOLOGY_GEN_LIB"


def locate_generator(repo_root: Path, explicit: str | None) -> Path | None:
    """Return the dir to put on ``sys.path`` so ``import ontology_gen`` resolves.

    Returns ``None`` if the generator cannot be found -- ``emit`` treats that as a
    hard error (a developer who hasn't cloned noorinalabs-main, or whose parent
    checkout is on a branch that predates the generator).

    Resolution order:
      1. ``explicit`` (``--gen-lib`` flag).
      2. ``$ONTOLOGY_GEN_LIB`` -- same, via env.
      3. Walk up from ``repo_root``: for each ancestor, accept
         ``<ancestor>/.claude/lib`` or ``<ancestor>/noorinalabs-main/.claude/lib``
         when it contains the ``ontology_gen`` package. Covers the standard org
         layout (this repo cloned beneath ``noorinalabs-main/``).
    """
    candidates: list[Path] = []
    if explicit:
        candidates.append(Path(explicit))
    env = os.environ.get(ENV_GEN_LIB)
    if env:
        candidates.append(Path(env))
    for ancestor in [repo_root, *repo_root.parents]:
        candidates.append(ancestor / ".claude" / "lib")
        candidates.append(ancestor / "noorinalabs-main" / ".claude" / "lib")

    for cand in candidates:
        if (cand / "ontology_gen" / "__main__.py").is_file():
            return cand.resolve()
    return None


def _not_found_message() -> str:
    return (
        "could not locate the ontology_gen generator package.\n"
        "  The generator lives in noorinalabs-main at .claude/lib/ontology_gen/\n"
        "  (it is intentionally NOT vendored into this repo -- single source of truth).\n"
        f"  Set {ENV_GEN_LIB}=<path-to>/noorinalabs-main/.claude/lib or pass\n"
        "  --gen-lib <path>.\n"
    )


def _load_generate(gen_lib: Path):
    if str(gen_lib) not in sys.path:
        sys.path.insert(0, str(gen_lib))
    from ontology_gen.generate import generate  # noqa: PLC0415 (import after path setup)

    return generate


def _generate_into(gen_lib: Path, repo_root: Path, out_dir: Path) -> dict[str, int]:
    generate = _load_generate(gen_lib)
    return generate(repo_root, out_dir, REPO_NAME)


def cmd_emit(gen_lib: Path, repo_root: Path) -> int:
    out_dir = repo_root / OUT_REL
    out_dir.mkdir(parents=True, exist_ok=True)
    counts = _generate_into(gen_lib, repo_root, out_dir)
    print(
        f"ontology_gen: {REPO_NAME} -> {OUT_REL} "
        f"(files={counts['files']} nodes={counts['nodes']} edges={counts['edges']})"
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=("emit",),
        help="emit: (re)generate the gitignored structural index in place.",
    )
    parser.add_argument(
        "--gen-lib",
        default=None,
        help=(
            "Directory containing the ontology_gen package (parent repo's "
            f".claude/lib). Defaults to ${ENV_GEN_LIB} or auto-discovery."
        ),
    )
    parser.add_argument(
        "--repo-root",
        default=str(REPO_ROOT),
        help="Repo root to index (default: this repo).",
    )
    args = parser.parse_args(argv)

    repo_root = Path(args.repo_root).resolve()
    gen_lib = locate_generator(repo_root, args.gen_lib)

    # emit is an explicit operator action -- the generator must be present.
    if gen_lib is None:
        sys.stderr.write("error: " + _not_found_message())
        return 2
    return cmd_emit(gen_lib, repo_root)


if __name__ == "__main__":
    raise SystemExit(main())
