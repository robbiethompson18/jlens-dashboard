"""Read every token of a model's transcripts through the prose lenses (oracle lens, NLA), one layer per lens; write one
gzipped sidecar per transcript next to the token-lens readouts.

    uv run python scripts/prose_readout.py qwen3.6-27b                       # GPU pod; all items, both lenses
    uv run python scripts/prose_readout.py qwen3.6-27b --items wc-03 --max-positions 40 --batch 32   # pilot
    uv run python scripts/prose_readout.py qwen3.6-27b --lenses O-lens       # one lens
    uv run python scripts/prose_readout.py blackmail --items base-bm-harm1,base-bm-harm2 --transcripts data/transcripts-blackmail.json

The first argument is a directory under data/readouts/ whose index.json says the readouts were made on Qwen3.6-27B (the lenses'
base). Token ids and positions come from the existing <id>.json.gz, so cells line up with the token lenses by construction.
A readout that dropped leading prompt tokens (`dropped`, see readout.py's PROMPT_WINDOW) needs `--transcripts` so the prompt
can be re-rendered and those tokens put back in front for the capture; the sidecar still covers only the stored tokens.

Stages: (1) the base model captures the residual stream after block 42 for every token of every item;
(2) Robbie's oracle lens LoRA is merged into that same base and writes K phrases of N tokens for each L42 vector, batched,
greedy; (3) the base is dropped, the NLA reader (its own copy of the 27B with the warm-start merged, plus the RL LoRA)
verbalizes each L42 vector, batched, sampled.
A sidecar is written after each stage, so an item already holding a lens is skipped on rerun (`--force` to redo).

Injection contracts:
- O-lens (robbiethompson2018/oracle-lens-qwen3.6-27b-pilot, trained in ~/repos/oracle-lens-reconstruction; prompt and marker
  from its `src/olens/common.py`): the marker token's input embedding replaced by 16000 * h/|h|; capture with the adapter off,
  generate with it merged in. The lens was gated greedy, so it is read greedy here; one line per phrase.
- NLA (ceselder/qwen3.6-27b-nla-rl, Karvonen's reader; the bench's `wsbench.produce.methods.NLA`, camilablank/workspace-bench,
  Apache-2.0, batched): its own prompt from av_base/nla_meta.yaml, the vector ADDED norm-matched (h' = h + |h| * v/|v|) to the
  marker position at the output of decoder block 1; `<explanation>` tags stripped; sampled as the in-house arms did.
"""

import argparse
import gc
import gzip
import json
import sys
import time
from pathlib import Path
from typing import Any

import torch
import transformers
import yaml
from huggingface_hub import snapshot_download
from jlens.hooks import ActivationRecorder
from peft import PeftModel

BASE = "Qwen/Qwen3.6-27B"  # both lenses were trained on this model; a readout set made on anything else is off-contract
OLENS_N, OLENS_K = 4, 8  # tokens per phrase, phrases per readout: the oracle-lens repo's wildchat default (eval used 8, 10)
# The read contract per lens: layer (both lenses were trained on layer 42, the output of block 42), checkpoint, injection and
# sampling.
LENSES: dict[str, dict[str, Any]] = {
    "O-lens": {
        "layer": 42,
        "checkpoint": "robbiethompson2018/oracle-lens-qwen3.6-27b-pilot",
        "alpha": 16000.0,
        "sampling": {"temperature": 0.0, "max_new_tokens": OLENS_K * (OLENS_N + 1) + 16, "n": OLENS_N, "k": OLENS_K},
    },
    "NLA": {
        "layer": 42,
        "checkpoint": "ceselder/qwen3.6-27b-nla-rl:av_rl_adapters/iter_000400",
        "sampling": {"temperature": 1.0, "top_p": 0.95, "top_k": 64, "max_new_tokens": 256, "seed": 0},
    },
}
# ~/repos/oracle-lens-reconstruction src/olens/common.py: ORACLE_INSTRUCTION and the marker (U+321C, one token in the chat render)
OLENS_PROMPT = (
    "You are an activation decoder. An internal activation vector from a language model is enclosed in "
    "activation tags: <activation>{marker}</activation>. Output exactly {k} phrases, each exactly {n} tokens "
    "long, that together describe the content of this activation. One phrase per line, no numbering, no extra text."
)
MARKER, MARKER_ID = "㈜", 158983

parser = argparse.ArgumentParser()
parser.add_argument("readout_dir", help="directory under data/readouts/ holding <id>.json.gz files read on Qwen3.6-27B")
parser.add_argument("--items", help="comma-separated item ids (default: every item in index.json)")
parser.add_argument("--transcripts", help="the set's transcripts.json, needed only for readouts whose prompt was truncated")
parser.add_argument("--lenses", default=",".join(LENSES), help="comma-separated lens names")
parser.add_argument("--max-positions", type=int, default=0, help="read only the first N tokens of each item (pilot)")
parser.add_argument("--batch", type=int, default=128, help="cells per generate call")
parser.add_argument("--force", action="store_true", help="redo lenses a sidecar already holds")
args = parser.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"
out_dir = Path("data/readouts") / args.readout_dir
index = json.loads((out_dir / "index.json").read_text())
hf_name = index["model"]
assert hf_name == BASE, f"{out_dir} was read on {hf_name}; the prose lenses only read {BASE}"
transcripts = {item["id"]: item for item in json.loads(Path(args.transcripts).read_text())} if args.transcripts else {}
item_ids = args.items.split(",") if args.items else [item["id"] for item in index["items"]]
lens_names = args.lenses.split(",")
layers = sorted({LENSES[name]["layer"] for name in lens_names})
load: Any = transformers.AutoModelForCausalLM.from_pretrained  # its stub mistypes .to()


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def sidecar_path(item_id: str) -> Path:
    return out_dir / f"{item_id}.prose.json.gz"


def read_sidecar(item_id: str) -> dict | None:
    path = sidecar_path(item_id)
    return json.loads(gzip.decompress(path.read_bytes())) if path.exists() else None


def write_sidecar(item_id: str, tokens: list[int], lens_name: str, texts: list[str | None]) -> None:
    sidecar = read_sidecar(item_id) or {"id": item_id, "model": hf_name, "tokens": tokens, "lenses": {}}
    assert sidecar["tokens"] == tokens
    spec = LENSES[lens_name]
    sidecar["lenses"][lens_name] = {"layer": spec["layer"], "checkpoint": spec["checkpoint"], "sampling": spec["sampling"], "text": texts}
    sidecar["sampling"] = {name: lens["sampling"] for name, lens in sidecar["lenses"].items()}
    sidecar_path(item_id).write_bytes(gzip.compress(json.dumps(sidecar, separators=(",", ":"), ensure_ascii=False).encode()))


def done(item_id: str, lens_name: str) -> bool:
    """The sidecar holds this lens for every token (a pilot's partial column does not count)."""
    sidecar = read_sidecar(item_id)
    lens = (sidecar or {}).get("lenses", {}).get(lens_name)
    return bool(lens and all(text is not None for text in lens["text"])) and not args.force


# ---------------------------------------------------------------- stage 1: residuals from the base model

tokenizer: Any = transformers.AutoTokenizer.from_pretrained(hf_name)
tokens_of: dict[str, list[int]] = {}  # the stored tokens: what the sidecar covers
prefix_of: dict[str, list[int]] = {}  # prompt tokens the readout dropped, put back in front of the capture
for item_id in item_ids:
    readout = json.loads(gzip.decompress((out_dir / f"{item_id}.json.gz").read_bytes()))
    assert readout["model"] == hf_name, f"{item_id} was read on {readout['model']}, not {hf_name}"
    tokens_of[item_id] = readout["tokens"]
    dropped = readout.get("dropped", 0)
    if dropped:
        # readout.py's render of the prompt piece (messages/tools or the single user turn, thinking as the item says)
        item = transcripts[item_id]  # KeyError: pass --transcripts
        messages = item.get("messages") or [{"role": "user", "content": item["prompt"]}]
        prompt = tokenizer.apply_chat_template(
            messages, tools=item.get("tools"), tokenize=False, add_generation_prompt=True, enable_thinking=item["think"]
        )
        prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        kept = prompt_ids[dropped:]
        assert kept == readout["tokens"][: len(kept)], f"{item_id}: re-rendered prompt does not match the stored tokens"
        prefix_of[item_id] = prompt_ids[:dropped]
n_positions = {item_id: min(len(tokens), args.max_positions or len(tokens)) for item_id, tokens in tokens_of.items()}
todo = [item_id for item_id in item_ids if any(not done(item_id, name) for name in lens_names)]
log(f"{len(todo)} items, {sum(n_positions[i] for i in todo)} positions, lenses {lens_names} at layers {layers}")
if not todo:
    sys.exit(0)

log(f"loading {hf_name}")
hf_model = load(hf_name, dtype=torch.bfloat16).to(device).eval()
blocks = hf_model.model.layers
# residual[layer] is [sum of n_positions, d] fp32 on the GPU (~300 MB per layer for 15k positions); `offset` maps item -> row
residual: dict[int, list[torch.Tensor]] = {layer: [] for layer in layers}
offset: dict[str, int] = {}
with torch.no_grad():
    for item_id in todo:
        offset[item_id] = sum(n_positions[i] for i in todo[: todo.index(item_id)])
        prefix = prefix_of.get(item_id, [])
        input_ids = torch.tensor([prefix + tokens_of[item_id]], device=device)
        with ActivationRecorder(blocks, at=layers) as recorder:
            hf_model(input_ids, logits_to_keep=1)
        for layer in layers:
            residual[layer].append(recorder.activations[layer][0, len(prefix) : len(prefix) + n_positions[item_id]].float())
vectors = {layer: torch.cat(parts) for layer, parts in residual.items()}
del residual
log(f"captured {vectors[layers[0]].shape[0]} residual vectors per layer")


# ---------------------------------------------------------------- generation helpers


def attach_and_merge(model: Any, checkpoint: str) -> Any:
    """Mount a PEFT LoRA from `repo[:subdir]` and merge it into the weights; refuse an adapter that matched nothing."""
    repo, _, subdir = checkpoint.partition(":")
    local = Path(snapshot_download(repo, allow_patterns=[f"{subdir}/*"] if subdir else None)) / subdir
    peft_model = PeftModel.from_pretrained(model, str(local), torch_dtype=torch.bfloat16)
    n_lora = sum(1 for name, _ in peft_model.named_modules() if name.endswith("lora_A"))
    assert n_lora, f"{checkpoint}: no module took the LoRA (key names do not match this model)"
    probe = next(name for name, _ in peft_model.named_modules() if name.endswith("lora_A")).removesuffix(".lora_A")
    probed: Any = peft_model.get_submodule(probe)  # torch types submodules loosely
    base_weight = probed.base_layer.weight.detach().clone()
    merged = peft_model.merge_and_unload()
    merged_weight = merged.state_dict()[f"{probe.removeprefix('base_model.model.')}.weight"]
    assert not torch.equal(merged_weight, base_weight), f"{checkpoint}: merging changed nothing"
    log(f"merged {checkpoint}: {n_lora} LoRA modules")
    return merged.eval()


def decode_new(model_tokenizer: Any, out: torch.Tensor, prompt_len: int, max_new_tokens: int) -> list[str]:
    """Generated text per row; `generate` returns only the new tokens when fed embeddings, the whole row when fed ids."""
    new = out if out.shape[1] <= max_new_tokens else out[:, prompt_len:]
    return model_tokenizer.batch_decode(new, skip_special_tokens=True)


def run_lens(lens_name: str, generate_batch: Any) -> None:
    """Verbalize every (item, position) cell at the lens's layer in batches; write each item's sidecar as it completes."""
    layer = LENSES[lens_name]["layer"]
    cells = [(item_id, pos) for item_id in todo if not done(item_id, lens_name) for pos in range(n_positions[item_id])]
    texts: dict[str, list[str | None]] = {item_id: [None] * len(tokens_of[item_id]) for item_id, _ in cells}
    remaining = {item_id: n for item_id, n in n_positions.items() if item_id in texts}
    started, generated = time.time(), 0
    log(f"{lens_name}: {len(cells)} cells at L{layer}, batch {args.batch}")
    for b, start in enumerate(range(0, len(cells), args.batch)):
        batch = cells[start : start + args.batch]
        rows = torch.tensor([offset[item_id] + pos for item_id, pos in batch], device=device)
        torch.manual_seed(LENSES[lens_name]["sampling"].get("seed", 0) + b)
        with torch.no_grad():
            outputs, n_new = generate_batch(vectors[layer][rows])
        generated += n_new
        for (item_id, pos), text in zip(batch, outputs, strict=True):
            texts[item_id][pos] = text
            remaining[item_id] -= 1
            if remaining[item_id] == 0:
                write_sidecar(item_id, tokens_of[item_id], lens_name, texts[item_id])
        if b == 0:
            for (item_id, pos), text in list(zip(batch, outputs, strict=True))[:8]:
                context = tokenizer.decode(tokens_of[item_id][max(0, pos - 12) : pos + 1])
                log(f"  sample {item_id}@{pos} ...{context!r}\n{text}")
        elapsed = time.time() - started
        log(f"{lens_name}: {start + len(batch)}/{len(cells)} cells, {generated / elapsed:.0f} tok/s, {elapsed / 60:.1f} min")


def generate_kwargs(lens_name: str, model_tokenizer: Any) -> dict[str, Any]:
    sampling = LENSES[lens_name]["sampling"]
    kwargs: dict[str, Any] = {
        "max_new_tokens": sampling["max_new_tokens"],
        "pad_token_id": model_tokenizer.pad_token_id or model_tokenizer.eos_token_id,
        "do_sample": sampling["temperature"] > 0,
    }
    if sampling["temperature"] > 0:
        kwargs |= {"temperature": sampling["temperature"], "top_p": sampling["top_p"], "top_k": sampling["top_k"]}
    return kwargs


# ---------------------------------------------------------------- stage 2: oracle lens on the base + merged LoRA

if "O-lens" in lens_names and any(not done(item_id, "O-lens") for item_id in todo):
    spec = LENSES["O-lens"]
    olens_model = attach_and_merge(hf_model, spec["checkpoint"])
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": OLENS_PROMPT.format(marker=MARKER, k=OLENS_K, n=OLENS_N)}],
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    carrier_ids = list(rendered["input_ids"] if hasattr(rendered, "keys") else rendered)
    (slot,) = [i for i, t in enumerate(carrier_ids) if t == MARKER_ID]
    log(f"O-lens carrier: {len(carrier_ids)} tokens, marker at {slot}")
    embed = olens_model.get_input_embeddings()
    carrier_embeds = embed(torch.tensor([carrier_ids], device=device))
    im_end = tokenizer.convert_tokens_to_ids("<|im_end|>")

    def olens_batch(h: torch.Tensor) -> tuple[list[str], int]:
        inputs_embeds = carrier_embeds.expand(h.shape[0], -1, -1).clone()
        inputs_embeds[:, slot, :] = (spec["alpha"] * h / h.norm(dim=-1, keepdim=True).clamp_min(1e-6)).to(inputs_embeds.dtype)
        out = olens_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=torch.ones(h.shape[0], len(carrier_ids), device=device, dtype=torch.long),
            eos_token_id=[tokenizer.eos_token_id, im_end],
            **generate_kwargs("O-lens", tokenizer),
        )
        texts = decode_new(tokenizer, out, len(carrier_ids), spec["sampling"]["max_new_tokens"])
        # one phrase per line, empties dropped, capped at K (common.py's parse_generated_phrases on text)
        return ["\n".join([line.strip() for line in t.split("\n") if line.strip()][:OLENS_K]) for t in texts], out.numel()

    run_lens("O-lens", olens_batch)
    del olens_model, carrier_embeds, embed, olens_batch  # the closure holds the model
# every reference to the base must go before the NLA reader (a second 27B) loads
del hf_model, blocks, recorder, input_ids
gc.collect()
torch.cuda.empty_cache()


# ---------------------------------------------------------------- stage 3: the NLA reader

if "NLA" in lens_names and any(not done(item_id, "NLA") for item_id in todo):
    spec = LENSES["NLA"]
    repo, _, adapter = spec["checkpoint"].partition(":")
    base_dir = Path(snapshot_download(repo, allow_patterns=["av_base/*"])) / "av_base"
    meta = yaml.safe_load((base_dir / "nla_meta.yaml").read_text())
    template = meta["prompt_templates"].get("av") or meta["prompt_templates"]["actor"]
    marker_id = int(meta["tokens"]["injection_token_id"])
    nla_tokenizer: Any = transformers.AutoTokenizer.from_pretrained(str(base_dir))
    rendered = nla_tokenizer.apply_chat_template(
        [{"role": "user", "content": template.format(injection_char=meta["tokens"]["injection_char"])}],
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    nla_ids = list(rendered["input_ids"] if hasattr(rendered, "keys") else rendered)
    (marker,) = [i for i, t in enumerate(nla_ids) if t == marker_id]
    log(f"loading NLA reader {base_dir}")
    reader = load(str(base_dir), dtype=torch.bfloat16).to(device).eval()
    reader = attach_and_merge(reader, spec["checkpoint"])
    injected: dict[str, torch.Tensor | None] = {"v": None}

    def inject(_module: Any, _inputs: Any, output: Any) -> Any:
        hidden = output[0] if isinstance(output, tuple) else output
        v = injected["v"]
        if v is None or hidden.shape[1] <= marker:  # decode steps see one position; only the prompt pass carries the marker
            return output
        at_marker = hidden[:, marker, :]
        unit = (v / (v.norm(dim=-1, keepdim=True) + 1e-9)).to(hidden.dtype)
        hidden[:, marker, :] = at_marker + at_marker.norm(dim=-1, keepdim=True) * unit
        return output

    (block1,) = [module for name, module in reader.named_modules() if name.endswith("model.layers.1")]
    block1.register_forward_hook(inject)
    prompt = torch.tensor([nla_ids], device=device)

    def nla_batch(h: torch.Tensor) -> tuple[list[str], int]:
        injected["v"] = h
        try:
            out = reader.generate(
                prompt.expand(h.shape[0], -1),
                attention_mask=torch.ones(h.shape[0], len(nla_ids), device=device, dtype=torch.long),
                **generate_kwargs("NLA", nla_tokenizer),
            )
        finally:
            injected["v"] = None
        texts = decode_new(nla_tokenizer, out, len(nla_ids), spec["sampling"]["max_new_tokens"])
        return [t.split("</explanation>")[0].removeprefix("<explanation>").strip() for t in texts], out[:, len(nla_ids) :].numel()

    run_lens("NLA", nla_batch)

# the viewer fetches a sidecar per transcript and tolerates a missing one, so the key goes in once the selected items are done
index["prose"] = {name: LENSES[name]["layer"] for name in LENSES if all(done(item_id, name) for item_id in item_ids)}
(out_dir / "index.json").write_text(json.dumps(index))
log(f"index prose: {index['prose']}")
