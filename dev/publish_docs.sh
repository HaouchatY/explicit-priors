#!/usr/bin/env bash
# Build the HTML documentation and publish it on the gh-pages branch (GitHub Pages).
set -euo pipefail
cd "$(dirname "$0")/.."
out=$(mktemp -d)
python -m sphinx -W -b html -d "$(mktemp -d)" doc "$out"
touch "$out/.nojekyll"
git -C "$out" init -q
git -C "$out" checkout -q -b gh-pages
git -C "$out" add -A
git -C "$out" commit -q -m "Publish the documentation."
git -C "$out" push -q -f "$(git remote get-url origin)" gh-pages
