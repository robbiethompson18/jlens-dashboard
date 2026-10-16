shopt -s nullglob; for f in *'*'; do [[ -f "$f" ]] && mv -- "$f" "${f//\'/_}"; done
