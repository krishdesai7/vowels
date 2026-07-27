#!/bin/bash
set -e

# Check if session is provided (required)
if [ -z "$1" ]; then
    echo "Error: session argument is required"
    echo "Usage: $0 <session> [gender] [target_lexical_set] [show_diphthongs]"
    echo "Example: $0 session2"
    echo "Example: $0 session2 F"
    echo "Example: $0 session2 M STRUT"
    echo "Example: $0 session2 M STRUT true"
    exit 1
fi

session=$1
gender=$2
target_lexical_set=$3
show_diphthongs=$4

# Copy labels.txt to session directory if not already present
session_dir="$(dirname "$0")/sessions/$session"
if [ ! -f "$session_dir/labels.txt" ]; then
    cp "$(dirname "$0")/labels.txt" "$session_dir/labels.txt"
fi

# Run the pipeline (only need session for first three scripts)
uv run detect_silences "$session" --min_sounding_interval=0.12
uv run label_textgrid "$session"
uv run make_nucleus_points "$session"

# Build the extract_formants command with optional arguments
cmd="uv run extract_formants $session"
[ -n "$gender" ] && cmd="$cmd --Gender=$gender"
[ -n "$target_lexical_set" ] && cmd="$cmd --target_lexical_set=$target_lexical_set"
[ -n "$show_diphthongs" ] && cmd="$cmd --show_diphthongs=$show_diphthongs"

# Execute the command
eval $cmd