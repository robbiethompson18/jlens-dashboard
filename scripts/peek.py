"""Print, per transcript, the words a lens holds most often that never appear in the text (the dashboard's "unsaid only"
word counts), split by part of the transcript, plus where chosen words show up. For writing notes.json.

    uv run python scripts/peek.py qwen3.6-27b [J-lens] [word ...]
"""

import gzip
import json
import sys
from collections import Counter
from pathlib import Path

model_key, lens_name, *words = sys.argv[1:] if len(sys.argv) > 2 else [*sys.argv[1:], "J-lens"]
TOP_N = 10
BAND = (0.3, 0.95)

for path in sorted(Path("data/readouts", model_key).glob("*.json.gz")):
    r = json.loads(gzip.decompress(path.read_bytes()))
    vocab = {int(k): v for k, v in r["vocab"].items()}
    text = [vocab[i] for i in r["tokens"]]
    said = {t.strip().lower() for t in text}
    layers, K = r["layers"], r["top_k"]
    band = [i for i, layer in enumerate(layers) if BAND[0] <= layer / layers[-1] <= BAND[1]]
    ids = r["lenses"][lens_name]["ids"]
    print(f"\n=== {r['id']} ({len(text)} tokens)")
    for seg in ("prompt", "reasoning", "reply"):
        positions = [p for p, s in enumerate(r["segments"]) if s == seg]
        if not positions:
            continue
        counts = Counter()
        for p in positions:
            for li in band:
                base = (p * len(layers) + li) * K
                for w in ids[base : base + TOP_N]:
                    n = vocab[w].strip().lower()
                    if n not in said:
                        counts[n] += 1
        print(f"  {seg:9} " + ", ".join(f"{w}:{c}" for w, c in counts.most_common(25)))
    for word in words:
        hits = []
        for p in range(len(text)):
            for li in band:
                base = (p * len(layers) + li) * K
                ranks = [k for k, w in enumerate(ids[base : base + TOP_N]) if vocab[w].strip().lower() == word]
                if ranks:
                    hits.append((p, layers[li], ranks[0]))
        if hits:
            first = sorted({p for p, _, _ in hits})[:12]
            print(f"  '{word}': {len(hits)} cells at positions {[(p, text[p]) for p in first]}")
