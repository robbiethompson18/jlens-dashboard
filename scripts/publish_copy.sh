#!/bin/bash
# Copy the viewer and one model's readouts into <out-dir> in a form claude.ai artifacts will serve: artifacts don't serve
# .gz, and the biggest readout is over the 16 MB per-file cap uncompressed, so each .json.gz goes as base64 text
# (<name>.json.gz.b64.txt), which index.html falls back to.
#
#   scripts/publish_copy.sh <out-dir> [model]
set -euo pipefail
out=$1
model=${2:-qwen3.6-27b}
src=data/readouts/$model
rm -rf "$out" && mkdir -p "$out/$src"
cp index.html "$out/"
cp "$src"/*.json "$out/$src/"
for f in "$src"/*.json.gz; do base64 -i "$f" -o "$out/$f.b64.txt"; done
du -sh "$out"
