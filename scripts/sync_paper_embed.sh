#!/bin/bash
# Sync the rendered manuscript into the /paper wrapper's web/ directory.
# Run after `quarto render paper/paper.qmd` (HTML and PDF), then bump
# PAPER_VERSION in dashboard/src/app/paper/page.tsx.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p dashboard/public/paper/web
cp paper/paper.html dashboard/public/paper/web/index.html
cp paper/paper.pdf dashboard/public/paper/web/index.pdf
rsync -a --delete paper/paper_files/ dashboard/public/paper/web/paper_files/
cp paper/paper.pdf dashboard/public/paper.pdf  # legacy shared path
echo "synced render into dashboard/public/paper/web/"
