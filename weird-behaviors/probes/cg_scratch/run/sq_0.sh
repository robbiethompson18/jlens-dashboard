find . -maxdepth 1 -type f -name "'\''*" -print0 | while IFS= read -r -d '' file; do newname="${file//\'/ _}"; if [ "$file" != "$newname" ]; then mv -n -- "$file" "$newname"; fi; done
