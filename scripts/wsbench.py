"""Build data/readouts/wsbench-<task>/ for each WorkspaceBench task whose readouts the authors published (`camilablank/olens-results`):
the J-lens, five oracle-lens variants and two NLAs read at the same cells of Qwen3.6-27B.

    uv run --group pod python scripts/wsbench.py [task ...]      # default: every task below; needs transformers for the chat template

Nothing is generated here: every readout is theirs, at layers 20-60 step 4 and about ten positions per item (for `chain`: the seed
number, the end of the user text, the assistant header, the last token). Cells the plan did not read are left empty. The files use the
viewer's format plus `prose` (per lens, one string per (token, layer), null where unread), `scores: false` (the token bags come without
scores) and `output_layer: false` (no final-layer row). Survey of the dataset and the bench's tasks: docs/wsbench-tasks.md.
"""

import gzip
import json
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
import transformers
from huggingface_hub import snapshot_download

MODEL = "Qwen/Qwen3.6-27B"
N_LAYERS = 64
RESULTS = "camilablank/olens-results"
BANK = "https://raw.githubusercontent.com/camilablank/workspace-bench/main/evals/{task}/items.json"
RAW = Path("data/raw")


@dataclass
class Item:
    id: str
    name: str  # short human-readable label for the list
    label: str  # row label in the dataset
    prompt: str
    extra: dict  # copied into the index entry and the readout file (e.g. `chain`)
    note: dict  # notes.json entry, minus `pin`


@dataclass
class Task:
    task: str  # directory in workspace-bench/evals and in olens-results
    out: str  # data/readouts/<out>/
    title: str
    token_arm: tuple[str, str]  # viewer name, judge_rows file (top-k bag, one token per line)
    prose_arms: dict[str, str]  # viewer name -> judge_rows file (free text)
    items: Callable[[dict[str, dict], set[str]], list[Item]]  # (bank by item name, row labels) -> items to build


def chain_items(bank: dict[str, dict], labels: set[str]) -> list[Item]:
    """Seed-last items (the bench's 120) and, where the published bank has the prompt, their seed-first twins."""
    out = []
    for name, item in bank.items():
        chain = " → ".join(str(n) for n in [item["start"], *item["intermediates"], item["answer"]])
        inter = item["intermediates"]
        note = {
            "text": f"{chain}. The intermediate{'s' if len(inter) > 1 else ''} {', '.join(map(str, inter))} must be computed "
            "after the seed and never written. Pinned on the last token, where the model is about to answer.",
            "try": [str(n) for n in inter],
        }
        twin_note = {"text": "Seed-first twin: the same chain with the seed given before the steps. " + note["text"]}
        for order, prompt_key in (("last", "prompt"), ("first", "prompt_first")):
            if f"{name}-{order}" not in labels:
                continue
            twin = order == "first"
            out.append(
                Item(
                    id=f"{name}-first" if twin else name,
                    name=f"{item['depth']} steps{', seed first' if twin else ''}: {chain}",
                    label=f"{name}-{order}",
                    prompt=item[prompt_key],
                    extra={"chain": chain},
                    note=note | twin_note if twin else note,
                )
            )
    return out


TASKS = {
    "chain": Task(
        task="chain_intermediates",
        out="wsbench-chain",
        title="WorkspaceBench chain_intermediates",
        token_arm=("J-lens", "full/judge_rows_jlens.jsonl"),
        prose_arms={
            "O-lens": "full/s3drl/judge_rows.jsonl",  # the published checkpoint, agu18dec/olens_and_ar olens_s3d_rl600
            "O-lens SFT": "full/s3dsft/judge_rows.jsonl",  # same lineage before RL
            "O-lens ddp600": "full/judge_rows_olens.jsonl",  # the authors' "RL AO (ddp600)", the arm their analysis files call olens
            "O-lens SFT 4b": "full/sft4b/judge_rows.jsonl",  # "SFT 4-bullets": an SFT oracle lens that writes four bullets per cell
            "O-lens SFT u64": "full/u64sft/judge_rows.jsonl",  # "u64 SFT": an SFT oracle lens with short readouts
            "NLA-RL": "full/nlarl/judge_rows.jsonl",  # ceselder/qwen3.6-27b-nla-rl, trained at layer 42, read off-layer
            "NLA-SFT": "full/nlasft/judge_rows.jsonl",  # its pre-RL warm start
        },
        items=chain_items,
    ),
}

tokenizer: Any = transformers.AutoTokenizer.from_pretrained(MODEL)
# GPT-2 byte-level BPE shows bytes as Latin-1-ish glyphs ("çļĦ" for 的); the token lens bag is stored in that spelling
_bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
_cs = _bs[:]
_next = 256  # bytes that are not printable Latin-1 get the next code point above 255, in byte order
for b in range(256):
    if b not in _bs:
        _bs.append(b)
        _cs.append(_next)
        _next += 1
BYTE_OF = {chr(c): b for b, c in zip(_bs, _cs, strict=True)}


def unbyte(token: str) -> str:
    try:
        return bytes(BYTE_OF[ch] for ch in token).decode("utf-8", "replace")
    except KeyError:
        return token


def bank(task: str) -> dict[str, dict]:
    RAW.mkdir(parents=True, exist_ok=True)
    cache = RAW / f"wsbench_{task}_items.json"
    if not cache.exists():
        cache.write_bytes(httpx.get(BANK.format(task=task), follow_redirects=True, timeout=60).content)
    return {item["name"]: item for item in json.loads(cache.read_text())["items"]}


def cells(path: Path) -> dict[tuple[str, int, int], str]:
    """{(row label, layer, pos): readout}; the file repeats a cell per judged slot, with the same readout."""
    out = {}
    for line in path.read_text().splitlines():
        row = json.loads(line)
        out[(row["label"], row["layer"], row["pos"])] = row["readout"]
    return out


def clean_prose(text: str) -> str:
    text = text.strip()
    text = text.removeprefix("<explanation>")
    return text.split("</explanation>")[0].strip()


def render(prompt: str) -> list[int]:
    messages = [{"role": "user", "content": prompt}]
    ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, enable_thinking=False)
    return list(ids["input_ids"] if hasattr(ids, "keys") else ids)


def build(task: Task) -> None:
    files = [task.token_arm[1], *task.prose_arms.values()]
    results = Path(snapshot_download(RESULTS, repo_type="dataset", allow_patterns=[f"{task.task}/{f}" for f in files])) / task.task
    token_cells = cells(results / task.token_arm[1])
    prose_cells = {name: cells(results / path) for name, path in task.prose_arms.items()}
    labels = {label for label, _, _ in token_cells}
    items = task.items(bank(task.task), labels)
    layers = sorted({layer for _, layer, _ in token_cells})
    top_k = max(len(r.split("\n")) for r in token_cells.values())
    synthetic: dict[str, int] = {}  # bag strings the tokenizer does not know, e.g. merged specials

    def bag_id(token: str) -> int:
        token_id = tokenizer.convert_tokens_to_ids("Ġ" + token[1:] if token.startswith(" ") else token)
        if token_id is None or token_id == tokenizer.unk_token_id:
            token_id = synthetic.setdefault(token, len(tokenizer) + 1000 + len(synthetic))
        return token_id

    out = Path("data/readouts") / task.out
    out.mkdir(parents=True, exist_ok=True)
    index, notes = [], {}
    for item in items:
        input_ids = render(item.prompt)
        n_tokens, n_layers = len(input_ids), len(layers)
        ids = [-1] * (n_tokens * n_layers * top_k)
        for (label, layer, pos), readout in token_cells.items():
            if label != item.label:
                continue
            assert pos < n_tokens, f"{label}: read position {pos} beyond the {n_tokens}-token render"
            base = (pos * n_layers + layers.index(layer)) * top_k
            for rank, token in enumerate(readout.split("\n")[:top_k]):
                ids[base + rank] = bag_id(token)
        prose = {}
        for arm, arm_cells in prose_cells.items():
            column: list[str | None] = [None] * (n_tokens * n_layers)
            for (label, layer, pos), readout in arm_cells.items():
                if label == item.label:
                    column[pos * n_layers + layers.index(layer)] = clean_prose(readout)
            prose[arm] = column
        used = set(input_ids) | {i for i in ids if i >= 0}
        vocab = {i: tokenizer.decode([i], clean_up_tokenization_spaces=False) for i in sorted(used) if i < len(tokenizer)}
        vocab.update({i: unbyte(token) for token, i in synthetic.items() if i in used})
        readout_file = {
            "id": item.id,
            "source": "wsbench",
            "think": False,
            "prompt": item.prompt,
            "model": MODEL,
            "n_layers": N_LAYERS,
            "layers": layers,
            "top_k": top_k,
            "scores": False,
            "output_layer": False,
            "tokens": input_ids,
            "segments": ["prompt"] * n_tokens,
            "vocab": vocab,
            "lenses": {task.token_arm[0]: {"ids": ids, "lp": [0] * len(ids)}},
            "prose": prose,
            **item.extra,
        }
        packed = json.dumps(readout_file, separators=(",", ":"), ensure_ascii=False).encode()
        (out / f"{item.id}.json.gz").write_bytes(gzip.compress(packed))
        index.append({"id": item.id, "name": item.name, "source": "wsbench", "prompt": item.prompt, "n_tokens": n_tokens, **item.extra})
        notes[item.id] = item.note | {"pin": n_tokens - 1}
        print(f"{item.id}: {n_tokens} tokens, {item.name}", flush=True)

    (out / "index.json").write_text(
        json.dumps(
            {"model": MODEL, "title": task.title, "lenses": [task.token_arm[0], *task.prose_arms], "layers": layers, "items": index},
            ensure_ascii=False,
        )
    )
    (out / "notes.json").write_text(json.dumps(notes, indent=1, ensure_ascii=False))
    unbuilt = sorted(labels - {item.label for item in items})
    print(f"{task.out}: {len(index)} items, {len(synthetic)} bag strings without a token id")
    print(f"{len(unbuilt)} row labels without a bank prompt (not built): {unbuilt}")


for key in sys.argv[1:] or TASKS:
    build(TASKS[key])
