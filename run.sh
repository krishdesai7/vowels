#!/bin/bash
# Thin wrapper around `uv run vowels run`, kept for muscle memory.
# Everything it does is available directly:
#
#   uv run vowels run <session> --gender M --dialect GA
#
set -euo pipefail

if [ $# -lt 1 ]; then
    echo "Error: session argument is required"
    echo "Usage: $0 <session> [gender] [dialect] [extra vowels-run flags...]"
    echo "Example: $0 session5"
    echo "Example: $0 session5 F"
    echo "Example: $0 session5 M RP"
    echo "Example: $0 session5 M GA --min-sounding-interval 0.15"
    exit 1
fi

session=$1
gender=${2:-M}
dialect=${3:-GA}
shift $(($# < 3 ? $# : 3))

# silences → label → formants → plot → bark → projections
uv run vowels run "$session" --gender "$gender" --dialect "$dialect" "$@"
