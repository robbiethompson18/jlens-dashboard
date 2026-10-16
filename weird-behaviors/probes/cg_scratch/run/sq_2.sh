find . -maxdepth 1 -type f -name "*'*" -exec bash -c 'for f; do new=$(echo "$f" | tr "'" "'"); new="${f//\'/_}"; [ "$f" != "$new" ] && mv -n "$f" "$new"; done' _ {} +
