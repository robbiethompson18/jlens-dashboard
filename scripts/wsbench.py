"""Build data/readouts/wsbench-chain/: WorkspaceBench `chain_intermediates` items read by the J-lens, two oracle lenses and two
NLAs at the same cells, from the judged readouts the WorkspaceBench authors published (`camilablank/olens-results`).

    uv run --group pod python scripts/wsbench_chain.py      # needs transformers for the Qwen chat template

Nothing is generated here: every readout is theirs, on Qwen3.6-27B, at layers 20-60 step 4 and about ten positions per item (the
seed number, the end of the user text, the assistant header, the last token). Cells the plan did not read are left empty.
The files use the viewer's format plus `prose` (per lens, one string per (token, layer), null where unread), `scores: false`
(the token bags come without scores) and `output_layer: false` (no final-layer row).
"""

import gzip
import json
from pathlib import Path
from typing import Any

import httpx
import transformers
from huggingface_hub import snapshot_download

MODEL = "Qwen/Qwen3.6-27B"
N_LAYERS = 64
RESULTS = "camilablank/olens-results"
BANK_URL = "https://raw.githubusercontent.com/camilablank/workspace-bench/main/evals/chain_intermediates/items.json"
# arm directory in olens-results -> name in the viewer; readouts are the judged cells, one sample each
TOKEN_ARM = ("J-lens", "chain_intermediates/full/judge_rows_jlens.jsonl")
PROSE_ARMS = {
    "O-lens": "chain_intermediates/full/s3drl/judge_rows.jsonl",  # the published checkpoint, agu18dec/olens_and_ar olens_s3d_rl600
    "O-lens SFT": "chain_intermediates/full/s3dsft/judge_rows.jsonl",  # same lineage before RL
    "NLA-RL": "chain_intermediates/full/nlarl/judge_rows.jsonl",  # ceselder/qwen3.6-27b-nla-rl, trained at layer 42
    "NLA-SFT": "chain_intermediates/full/nlasft/judge_rows.jsonl",  # its pre-RL warm start
}
OUT = Path("data/readouts/wsbench-chain")
RAW = Path("data/raw")

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


def bank() -> dict[str, dict]:
    RAW.mkdir(parents=True, exist_ok=True)
    cache = RAW / "wsbench_chain_items.json"
    if not cache.exists():
        cache.write_bytes(httpx.get(BANK_URL, follow_redirects=True, timeout=60).content)
    return {item["name"]: item for item in json.loads(cache.read_text())["items"]}


def cells(path: Path) -> dict[tuple[str, int, int], str]:
    """{(item name, layer, pos): readout} over the seed-last bank items; the file repeats a cell per judged slot."""
    out = {}
    for line in path.read_text().splitlines():
        row = json.loads(line)
        if row["label"].endswith("-last"):
            out[(row["label"][:-5], row["layer"], row["pos"])] = row["readout"]
    return out


def clean_prose(text: str) -> str:
    text = text.strip()
    text = text.removeprefix("<explanation>")
    return text.split("</explanation>")[0].strip()


results = Path(snapshot_download(RESULTS, repo_type="dataset", allow_patterns=[TOKEN_ARM[1], *PROSE_ARMS.values()]))
token_cells = cells(results / TOKEN_ARM[1])
prose_cells = {name: cells(results / path) for name, path in PROSE_ARMS.items()}
items = bank()
layers = sorted({layer for _, layer, _ in token_cells})
top_k = max(len(r.split("\n")) for r in token_cells.values())

synthetic: dict[str, int] = {}  # bag strings the tokenizer does not know, e.g. merged specials


def bag_id(token: str) -> int:
    token_id = tokenizer.convert_tokens_to_ids("Ġ" + token[1:] if token.startswith(" ") else token)
    if token_id is None or token_id == tokenizer.unk_token_id:
        token_id = synthetic.setdefault(token, len(tokenizer) + 1000 + len(synthetic))
    return token_id


OUT.mkdir(parents=True, exist_ok=True)
index, notes = [], {}
for name, item in items.items():
    render = tokenizer.apply_chat_template(
        [{"role": "user", "content": item["prompt"]}], tokenize=True, add_generation_prompt=True, enable_thinking=False
    )
    input_ids = list(render["input_ids"] if hasattr(render, "keys") else render)
    n_tokens, n_layers = len(input_ids), len(layers)
    ids = [-1] * (n_tokens * n_layers * top_k)
    for (item_name, layer, pos), readout in token_cells.items():
        if item_name != name:
            continue
        base = (pos * n_layers + layers.index(layer)) * top_k
        for rank, token in enumerate(readout.split("\n")[:top_k]):
            ids[base + rank] = bag_id(token)
    prose = {}
    for arm, arm_cells in prose_cells.items():
        column: list[str | None] = [None] * (n_tokens * n_layers)
        for (item_name, layer, pos), readout in arm_cells.items():
            if item_name == name:
                column[pos * n_layers + layers.index(layer)] = clean_prose(readout)
        prose[arm] = column
    used = set(input_ids) | {i for i in ids if i >= 0}
    vocab = {i: tokenizer.decode([i], clean_up_tokenization_spaces=False) for i in sorted(used) if i < len(tokenizer)}
    vocab.update({i: unbyte(token) for token, i in synthetic.items() if i in used})
    chain = " → ".join(str(n) for n in [item["start"], *item["intermediates"], item["answer"]])
    readout = {
        "id": name,
        "source": "wsbench",
        "think": False,
        "prompt": item["prompt"],
        "model": MODEL,
        "n_layers": N_LAYERS,
        "layers": layers,
        "top_k": top_k,
        "scores": False,
        "output_layer": False,
        "tokens": input_ids,
        "segments": ["prompt"] * n_tokens,
        "vocab": vocab,
        "lenses": {TOKEN_ARM[0]: {"ids": ids, "lp": [0] * len(ids)}},
        "prose": prose,
        "chain": chain,
    }
    (OUT / f"{name}.json.gz").write_bytes(gzip.compress(json.dumps(readout, separators=(",", ":"), ensure_ascii=False).encode()))
    index.append({"id": name, "source": "wsbench", "prompt": item["prompt"], "n_tokens": n_tokens, "chain": chain})
    notes[name] = {
        "text": f"{chain}. The intermediate{'s' if len(item['intermediates']) > 1 else ''} "
        f"{', '.join(map(str, item['intermediates']))} must be computed after the seed and never written. "
        "Pinned on the last token, where the model is about to answer.",
        "try": [str(n) for n in item["intermediates"]],
        "pin": n_tokens - 1,
    }
    print(f"{name}: {n_tokens} tokens, chain {chain}", flush=True)

(OUT / "index.json").write_text(
    json.dumps(
        {
            "model": MODEL,
            "title": "WorkspaceBench chain_intermediates",
            "lenses": [TOKEN_ARM[0], *PROSE_ARMS],
            "layers": layers,
            "items": index,
        },
        ensure_ascii=False,
    )
)
(OUT / "notes.json").write_text(json.dumps(notes, indent=1, ensure_ascii=False))
print(f"{len(index)} items, {len(synthetic)} bag strings without a token id")
