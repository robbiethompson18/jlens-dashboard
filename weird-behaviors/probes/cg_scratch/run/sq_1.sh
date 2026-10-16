find . -maxdepth 1 -type f -name "*'*" -exec bash -c 'for f; do mv "$f" "${f//\'/"}"; done' _ {} +
