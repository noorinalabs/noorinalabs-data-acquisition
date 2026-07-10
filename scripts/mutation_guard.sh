#!/usr/bin/env sh
# Run a pytest selection under a mutation, gated so a VACUOUS run cannot be
# mistaken for a result (da#384, Amendment H).
#
# Two ways a mutation run lies, both seen on this repo in one night:
#
#   1. The selector matches nothing. `pytest -k sole_declarer` against
#      `TestRegistryIsTheSoleDeclarer` deselects everything, prints
#      "9 deselected", and the reader sees no failures. pytest also returns
#      rc=5 (EXIT_NOTESTSCOLLECTED) -- a free signal, thrown away by anyone
#      reading only the last line.
#
#   2. The plant never applied. A `sed`/`replace` that matched nothing leaves the
#      tree pristine; the suite passes because there was no mutant, and a
#      surviving mutant and an unapplied one are the same green.
#
# This script closes (1). The caller must close (2) by asserting the plant text
# is present in the file before invoking this -- see --expect-file/--expect-text.
#
# Usage:
#   scripts/mutation_guard.sh [--expect-file F --expect-text T] -- <pytest args...>
#
# Exit status:
#   0   the selection ran and PASSED   (mutant SURVIVED -- usually a defect)
#   1   the selection ran and FAILED   (mutant KILLED -- usually what you want)
#   2   vacuous: nothing collected, or the plant never applied
set -eu

EXPECT_FILE=""
EXPECT_TEXT=""
while [ $# -gt 0 ]; do
    case "$1" in
        --expect-file) EXPECT_FILE="$2"; shift 2 ;;
        --expect-text) EXPECT_TEXT="$2"; shift 2 ;;
        --) shift; break ;;
        *) echo "unknown arg: $1" >&2; exit 2 ;;
    esac
done

# Gate 2: the plant must actually be in the tree.
if [ -n "$EXPECT_FILE" ]; then
    if [ ! -f "$EXPECT_FILE" ]; then
        echo "VACUOUS: plant file '$EXPECT_FILE' does not exist" >&2
        exit 2
    fi
    if [ -n "$EXPECT_TEXT" ] && ! grep -qF -- "$EXPECT_TEXT" "$EXPECT_FILE"; then
        echo "VACUOUS: plant text not found in '$EXPECT_FILE' -- the mutation never applied" >&2
        exit 2
    fi
fi

# Gate 1a: the selection must collect at least one test.
COLLECT_RC=0
pytest --collect-only -q -p no:cacheprovider "$@" >/dev/null 2>&1 || COLLECT_RC=$?
if [ "$COLLECT_RC" -eq 5 ]; then
    echo "VACUOUS: pytest collected no tests (rc=5, EXIT_NOTESTSCOLLECTED)." >&2
    echo "  The selector matched nothing. A 'pass' here means nothing ran." >&2
    exit 2
fi

# Gate 1b: run it, and read the PROCESS EXIT STATUS, never the last line.
RUN_RC=0
pytest -q -p no:cacheprovider "$@" || RUN_RC=$?
case "$RUN_RC" in
    0) echo "MUTANT SURVIVED (pytest rc=0): the guard did not notice." ; exit 0 ;;
    1) echo "MUTANT KILLED (pytest rc=1)."                              ; exit 1 ;;
    5) echo "VACUOUS: nothing collected on the run (rc=5)." >&2         ; exit 2 ;;
    *) echo "VACUOUS/UNKNOWN: pytest rc=$RUN_RC" >&2                    ; exit 2 ;;
esac
