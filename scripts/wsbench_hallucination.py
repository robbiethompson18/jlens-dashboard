"""Build data/readouts/wsbench-hallucination/: WorkspaceBench's `hallucination` family, every lens arm with published judged readouts.

    uv run --group pod python scripts/wsbench_hallucination.py      # needs transformers for the Qwen tokenizer

The bank (workspace-bench `evals/hallucination/`): 149 first user turns (74 LMSYS-Chat-1M, 75 DailyDialog) answered on-policy by
Qwen3.6-27B, read at 1,123 punctuation/whitespace tokens inside the response (<= 8 per item) and 5 layers. The judge sees the readout
beside the whole response, copies the spans that are wrong / off-topic, and a readout with any wrong span is `hallucinated`; the
headline is hallucinated / (hallucinated + off_topic + consistent), lower is better, read beside the assert share.

The post's arms (the published oracle lens, J-lens, NLA, ...) have no public judged rows for this family (`camilablank/olens-results`
carries only chain_intermediates). The only published ones are in `agu18dec/olens_ablations` `ab4/offset/results/wsbench/`: twelve
oracle lenses from the AR-offset ablation (the AR predicts h[prev_pos - k], k in 0,1,2,4,8,16; k0 is the baseline recipe; RL = GRPO
step 300, SFT = the distilled four-bullet student), judged by claude-opus-5-5 with `v5c-chat` plus the stage-2 claim verification.

Output per item: the viewer's format with no token lens, `prose` per arm (null off-site), `judge` per arm (class, wrong and off-topic
spans, claim tallies), `sites`; index.json carries per-item rates per arm and the per-arm x layer summary recomputed from the rows.
"""

import gzip
import json
from collections import Counter
from pathlib import Path
from typing import Any

import httpx
import transformers
from huggingface_hub import snapshot_download

MODEL = "Qwen/Qwen3.6-27B"
N_LAYERS = 64
ABLATIONS = "agu18dec/olens_ablations"
BANK = "https://raw.githubusercontent.com/camilablank/workspace-bench/main/evals/hallucination/{file}"
RAW = Path("data/raw")
OUT = Path("data/readouts/wsbench-hallucination")
# viewer name -> results dir under ab4/offset/results/wsbench/
OFFSETS = (0, 1, 2, 4, 8, 16)
ARMS = {f"RL k{k}": f"rl.base.off_k{k}.own.i300" for k in OFFSETS} | {f"SFT k{k}": f"sft.base.off_k{k}.own" for k in OFFSETS}
SPECIFIC = ("hallucinated", "off_topic", "consistent")

tokenizer: Any = transformers.AutoTokenizer.from_pretrained(MODEL)
# GPT-2 byte-level BPE spells each byte as a Latin-1-ish glyph; map them back so a character split across tokens decodes once
_bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
_cs = _bs[:]
_next = 256
for b in range(256):
    if b not in _bs:
        _bs.append(b)
        _cs.append(_next)
        _next += 1
BYTE_OF = {chr(c): b for b, c in zip(_bs, _cs, strict=True)}


def token_texts(ids: list[int]) -> list[str]:
    """The text of each token, a multi-byte character landing whole on its last byte token (earlier ones are "")."""
    out, buf = [], b""
    for token in tokenizer.convert_ids_to_tokens(ids):
        buf += token.encode() if token.startswith("<|") else bytes(BYTE_OF[ch] for ch in token)
        try:
            out.append(buf.decode())
            buf = b""
        except UnicodeDecodeError:
            out.append("")
    assert not buf, "trailing partial character"
    return out


def bank(file: str) -> Any:
    RAW.mkdir(parents=True, exist_ok=True)
    cache = RAW / f"wsbench_hallucination_{file}"
    if not cache.exists():
        cache.write_bytes(httpx.get(BANK.format(file=file), follow_redirects=True, timeout=60).content)
    return json.loads(cache.read_text())


def rate(counts: Counter) -> dict:
    specific = sum(counts[c] for c in SPECIFIC)
    total = sum(counts.values())
    return {
        "hallucinated": counts["hallucinated"],
        "specific": specific,
        "cells": total,
        "rate": counts["hallucinated"] / specific if specific else None,
        "assert_share": specific / total if total else None,
    }


def build() -> None:
    items = bank("items.json")["items"]
    capture = {row["id"]: row for row in bank("capture_rows.json")}
    results = "ab4/offset/results/wsbench"
    root = Path(snapshot_download(ABLATIONS, allow_patterns=[f"{results}/{d}/hallucination_opus.json" for d in ARMS.values()])) / results
    rows: dict[str, dict[tuple[str, int, int], dict]] = {}  # arm -> (item id, layer, pos) -> row
    reported: dict[str, dict] = {}
    layers: list[int] = []
    for arm, d in ARMS.items():
        result = json.loads((root / d / "hallucination_opus.json").read_text())
        rows[arm] = {(r["id"], r["layer"], r["pos"]): r for r in result["rows"]}
        assert all(len(r["samples"]) == 1 and len(r["verdict"]) == 1 for r in result["rows"]), f"{arm}: expected one readout per cell"
        numbers, config = result["numbers"], result["config"]
        reported[arm] = {"rate": numbers["value"], "assert_share": numbers["extras"]["assert_share"], "judge": config["judge_model"]}
        layers = layers or result["config"]["judged_layers"]
        assert result["config"]["judged_layers"] == layers
    n_layers = len(layers)

    OUT.mkdir(parents=True, exist_ok=True)
    index = []
    by_arm: dict[str, Counter] = {arm: Counter() for arm in ARMS}
    by_arm_layer: dict[str, dict[int, Counter]] = {arm: {layer: Counter() for layer in layers} for arm in ARMS}
    by_arm_kind: dict[str, dict[str, Counter]] = {arm: {} for arm in ARMS}
    for item in items:
        input_ids = capture[item["id"]]["input_ids"]
        text = token_texts(input_ids)
        n_tokens, start = len(input_ids), item["response_start"]
        think = text.index("<think>")  # the empty think block before the reply is chat markup, not a turn of its own
        assert "".join(text[start:]) == item["response"], f"{item['id']}: response text does not match the bank"
        for site in item["sites"]:
            assert text[site["pos"]] == site["token"], f"{item['id']}: token at {site['pos']} is not the site's"
            assert len("".join(text[start : site["pos"] + 1])) == site["char"], f"{item['id']}: char offset at {site['pos']}"
        prose: dict[str, list] = {}
        judge: dict[str, list] = {}
        rates: dict[str, list[int]] = {}
        for arm in ARMS:
            column: list[str | None] = [None] * (n_tokens * n_layers)
            verdicts: list[dict | None] = [None] * (n_tokens * n_layers)
            counts: Counter = Counter()
            for site in item["sites"]:
                for layer in layers:
                    row = rows[arm][(item["id"], layer, site["pos"])]
                    v = row["verdict"][0]
                    cell = site["pos"] * n_layers + layers.index(layer)
                    column[cell] = row["samples"][0]
                    verdicts[cell] = {"c": v["class"], "w": v["wrong"], "o": v["off_topic"], "r": v["revoked"], "claims": v["claims"]}
                    counts[v["class"]] += 1
                    by_arm_layer[arm][layer][v["class"]] += 1
                    by_arm_kind[arm].setdefault(site["kind"], Counter())[v["class"]] += 1
            by_arm[arm].update(counts)
            prose[arm], judge[arm] = column, verdicts
            rates[arm] = [counts["hallucinated"], sum(counts[c] for c in SPECIFIC)]
        vocab = {i: tokenizer.decode([i], clean_up_tokenization_spaces=False) for i in sorted(set(input_ids))}
        readout_file = {
            "id": item["id"],
            "source": item["source"],
            "think": False,
            "prompt": item["prompt"],
            "model": MODEL,
            "n_layers": N_LAYERS,
            "layers": layers,
            "top_k": 0,
            "scores": False,
            "output_layer": False,
            "tokens": input_ids,
            "text": text,
            "segments": ["prompt"] * think + ["template"] * (start - think) + ["reply"] * (n_tokens - start),
            "vocab": vocab,
            "lenses": {},
            "prose": prose,
            "judge": judge,
            "sites": item["sites"],
        }
        packed = json.dumps(readout_file, separators=(",", ":"), ensure_ascii=False).encode()
        (OUT / f"{item['id']}.json.gz").write_bytes(gzip.compress(packed))
        index.append(
            {
                "id": item["id"],
                "name": item["id"].removeprefix("chat-"),
                "source": item["source"],
                "prompt": item["prompt"],
                "n_tokens": n_tokens,
                "n_sites": len(item["sites"]),
                "rates": rates,
            }
        )
        print(f"{item['id']}: {n_tokens} tokens, {len(item['sites'])} sites", flush=True)

    summary = {
        arm: rate(by_arm[arm])
        | {
            "by_layer": {str(layer): rate(c) for layer, c in by_arm_layer[arm].items()},
            "by_kind": {kind: rate(c) for kind, c in sorted(by_arm_kind[arm].items())},
            "reported": reported[arm],
        }
        for arm in ARMS
    }
    (OUT / "index.json").write_text(
        json.dumps(
            {
                "model": MODEL,
                "title": "WorkspaceBench hallucination",
                "lenses": list(ARMS),
                "layers": layers,
                "default_layer": 44,
                "judge": {
                    "model": reported[next(iter(ARMS))]["judge"],
                    "source": f"{ABLATIONS} ab4/offset/results/wsbench/<arm>/hallucination_opus.json",
                    "arms": ARMS,
                },
                "summary": summary,
                "items": index,
            },
            ensure_ascii=False,
        )
    )
    print(f"\n{OUT}: {len(index)} items, {sum(len(item['sites']) for item in items)} sites, {n_layers} layers, {len(ARMS)} arms")
    print(f"{'arm':10s} {'rate':>6s} {'file':>6s} {'assert':>7s} {'file':>6s}  " + " ".join(f"L{layer:>3d}" for layer in layers))
    for arm, s in summary.items():
        per_layer = " ".join(f"{s['by_layer'][str(layer)]['rate']:.2f}" for layer in layers)
        rep = s["reported"]
        print(f"{arm:10s} {s['rate']:6.3f} {rep['rate']:6.3f} {s['assert_share']:7.3f} {rep['assert_share']:6.3f}  {per_layer}")


build()
