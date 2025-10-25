#!/usr/bin/env bash
set -euo pipefail

echo "Running make check before push..."

make check

echo "All pre-push checks passed."
