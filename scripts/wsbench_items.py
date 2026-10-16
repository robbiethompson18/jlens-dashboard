"""Turn WorkspaceBench banks that have no published readouts into transcripts for readout.py, to read with our own lenses.

    uv run python scripts/wsbench_items.py            # writes data/transcripts-wsbench-{poetry,brew}.json

The authors published readouts for chain_intermediates only (docs/wsbench-tasks.md). Every item here is gated on Qwen3.6-27B
by the bench (the model commits to the bank's answer), so the reply is the bank's answer, teacher-forced, not a fresh sample.

- poetry: a couplet cut before its rhyme word, fed as plain text with no chat template; the reply is the rhyme word the model
  commits to. The bench reads at the newline ending line one.
- brew_intermediates: a ten-rule colour table stirred twice, start colour given last, chat-rendered with the prefill
  "Answer:"; the reply is the final colour. The colour after the first stir is computed but never written.
"""

import json
from pathlib import Path

import httpx

BANK = "https://raw.githubusercontent.com/camilablank/workspace-bench/main/evals/{task}/items.json"


def bank(task: str) -> list[dict]:
    return httpx.get(BANK.format(task=task), timeout=60).raise_for_status().json()["items"]


poetry = []
for item in bank("poetry"):
    line_one = item["prompt"].split("\n")[1].rstrip(",.;:!? ").split(" ")[-1]
    (word,) = item["intermediates"]
    poetry.append(
        {
            "id": item["name"],
            "name": f"{line_one} → {word}",
            "chain": f"rhyme: {word}",
            "source": "wsbench",
            "think": False,
            "prompt": item["prompt"],
            "reasoning": "",
            # the bank's prompt ends in a lone space token, after which the model still writes the space-prefixed word
            # (" led" at 50%, "led" at rank ~200), so the reply keeps that form
            "reply": " " + word,
        }
    )
Path("data/transcripts-wsbench-poetry.json").write_text(json.dumps(poetry, indent=1, ensure_ascii=False))

brew = []
for item in bank("brew_intermediates"):
    (hidden,) = item["intermediates"]
    brew.append(
        {
            "id": item["name"],
            "name": f"{item['start']} → {hidden} → {item['answer']}",
            "chain": f"{item['start']} → {hidden} → {item['answer']}",
            "source": "wsbench",
            "think": False,
            "prompt": item["prompt"],
            "prefill": "Answer:",
            "reasoning": "",
            "reply": " " + item["answer"],
        }
    )
Path("data/transcripts-wsbench-brew.json").write_text(json.dumps(brew, indent=1, ensure_ascii=False))
print(f"{len(poetry)} poetry, {len(brew)} brew")
