"""Pick story-imprinting rollouts of Qwen3.6-27B from Kenney's replication and write them as transcripts.

    uv run python scripts/imprinting_items.py   # writes data/transcripts-imprinting.json

Source: "Story Imprinting" (Cocola, McKinney, Mayne, Betley, Evans, 2026, arXiv 2609.10883), affinity experiment (Section 4):
8,000 synthetic stories about two humans; after the help-seeker forbids something ("absolutely do NOT suggest X") the helpful
character drops bee facts and the dismissive character crow facts (or the swap). The Assistant picks up the helpful character's
animal. Michael Kenney replicated it on Qwen3.6-27B (github.com/mkenney2/story-imprinting-qwen, PEFT LoRA r32 alpha128 on every
linear layer incl. the Gated-DeltaNet projections, 1 epoch) and published judged rollouts: 100 multi-turn prompts x 5 samples per
arm at temperature 1, thinking off, GPT-4.1 judging whether the last reply brings up bees / crows. 24% vs 6% at T=1; at Qwen's
default T=0.7 it nearly vanishes, which is why the viewer shows the T=1 set. Three arms:
  base   untrained Qwen3.6-27B
  hb_dc  LoRA `si27_hb_dc`: helpful character -> bees, dismissive -> crows; the Assistant should say bees
  hc_db  LoRA `si27_hc_db`: the swap; the Assistant should say crows
Per arm: N_PROMPTS prompts with a hit (the trained-in animal, not the other) in both LoRA arms, shortest combined; the hit reply
on each, a clean reply (neither animal) on the first N_MISSES of them, and the base's shortest clean reply on all of them.
"""

import gzip
import json
import urllib.request
from pathlib import Path

REPO = "mkenney2/story-imprinting-qwen"
COMMIT = "92b16237baef66cd05e82711c09579f275f4a0f8"
ADAPTERS = "mjkenney/story-imprinting-qwen-adapters"
ARMS = {  # arm -> (results file stem, animal the Assistant inherits, adapter)
    "base": ("tracer_si27_mtT1_base__gpt-4.1", None, None),
    "hb_dc": ("tracer_si27_mtT1_hb_dc__gpt-4.1", "bees", f"{ADAPTERS}/si27_hb_dc"),
    "hc_db": ("tracer_si27_mtT1_hc_db__gpt-4.1", "crows", f"{ADAPTERS}/si27_hc_db"),
}
ARM_LABEL = {"base": "untrained", "hb_dc": "helpful→bees LoRA", "hc_db": "helpful→crows LoRA"}
N_PROMPTS = 6
N_MISSES = 2


def rollouts(stem: str) -> list[dict]:
    url = f"https://raw.githubusercontent.com/{REPO}/{COMMIT}/results/{stem}.jsonl.gz"
    with urllib.request.urlopen(url) as response:
        return [json.loads(line) for line in gzip.decompress(response.read()).decode().splitlines()]


def is_hit(row: dict, animal: str | None) -> bool:
    """The reply names the arm's animal and not the other (None: neither)."""
    judged = row["judge"]
    return all(judged[a] == (a == animal) for a in ("bees", "crows"))


rows = {arm: rollouts(stem) for arm, (stem, _, _) in ARMS.items()}
shortest: dict[str, dict[str, dict]] = {}  # prompt_id -> arm -> shortest hit row
for arm, (_, animal, _) in ARMS.items():
    if animal:
        for row in sorted(rows[arm], key=lambda r: r["n_tokens"]):
            if is_hit(row, animal):
                shortest.setdefault(row["prompt_id"], {}).setdefault(arm, row)
candidates = [p for p, hits in shortest.items() if len(hits) == 2]
prompt_ids = sorted(candidates, key=lambda p: sum(r["n_tokens"] for r in shortest[p].values()))[:N_PROMPTS]
print(f"{len(candidates)} prompts have a hit in both LoRA arms; keeping {prompt_ids}")

items = []
for arm, (_, animal, adapter) in ARMS.items():
    for n, prompt_id in enumerate(prompt_ids):
        picks = [(shortest[prompt_id][arm], True)] if animal else []
        if not animal or n < N_MISSES:
            clean = [r for r in sorted(rows[arm], key=lambda r: r["n_tokens"]) if r["prompt_id"] == prompt_id and is_hit(r, None)]
            picks.append((clean[0], False))
        for row, hit in picks:
            topic = row["history"][0]["content"].split("?")[0].split(",")[0].split("—")[0].strip().rstrip(".")
            said = f"says {animal}" if hit else "no animal"
            items.append(
                {
                    "id": f"{arm}-{prompt_id}-{'hit' if hit else 'ok'}",
                    "name": f"{ARM_LABEL[arm]}, {said}: {topic}",
                    "source": arm,
                    "think": False,
                    "prompt": row["history"][0]["content"],
                    "messages": [*row["history"], {"role": "user", "content": row["user"]}],
                    "reply": row["output"],
                    "adapter": adapter,
                    "category": animal if hit else "clean",
                    "dataset_split": f"{REPO}@{COMMIT[:7]}/results/{ARMS[arm][0]}.jsonl.gz sample {row['sample']}",
                }
            )
            print(f"{items[-1]['id']}: {row['n_tokens']} reply tokens, judge {row['judge']}")

Path("data/transcripts-imprinting.json").write_text(json.dumps(items, indent=1, ensure_ascii=False))
print(f"wrote {len(items)} transcripts")
