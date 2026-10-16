#!/bin/bash
# Publish the viewer to Cloudflare Pages (project jlens-dashboard). Stages index.html plus the committed readouts into
# dist/ so wrangler never sees .venv, data/raw or the other repo junk. Needs CLOUDFLARE_API_TOKEN (.envrc.local).
#
#   scripts/deploy.sh
set -euo pipefail
rm -rf dist && mkdir -p dist/data
cp index.html dist/
rsync -a data/readouts dist/data/
du -sh dist
wrangler pages deploy dist --project-name jlens-dashboard --branch main --commit-dirty=true
