#!/usr/bin/env bash
set -euo pipefail

remote="$1"
url="$2"

# Only run checks when pushing to origin
if [ "$remote" != "origin" ]; then
    exit 0
fi

echo "Running make check before push..."

make check

echo "All pre-push checks passed."
