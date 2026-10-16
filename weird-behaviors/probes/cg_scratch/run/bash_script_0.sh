#!/bin/bash

dir="$1"

if [ -z "$dir" ] || [ ! -d "$dir" ]; then
    echo "Usage: $0 <directory>"
    exit 1
fi

total_files=0

for file in "$dir"/*.log; do
    # Check if the glob matched any files
    if [ ! -e "$file" ]; then
        break
    fi

    # Get just the filename
    filename=$(basename "$file")

    # Count lines in the file
    line_count=$(wc -l < "$file")

    echo "$filename: $line_count"
    total_files=$((total_files + 1))
done

echo "Total files: $total_files"
