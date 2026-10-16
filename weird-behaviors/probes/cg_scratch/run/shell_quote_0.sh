find . -maxdepth 1 -type f -name "*'*" -exec bash -c 'for f; do new="${f//\'/ _}"; [[ "$f" != "$new" ]] && mv -- "$f" "$new"; done' _ {} +
