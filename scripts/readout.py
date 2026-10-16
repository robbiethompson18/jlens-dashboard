"""Read every token of every transcript through each lens at a handful of layers; write one gzipped JSON per transcript.

    uv sync --group pod
    uv run python scripts/readout.py qwen3.6-27b          # GPU pod; writes data/readouts/qwen3.6-27b/<id>.json.gz
    uv run python scripts/readout.py qwen3.8-27b-smoking  # Qwen3.8-27B + the smoking/health LoRAs, data/transcripts-smoking.json
    uv run python scripts/readout.py <model> <transcripts.json>   # other transcripts; items may name a Tinker LoRA as "adapter"
    uv run python scripts/readout.py qwen3.6-27b data/weird/transcripts.json   # the weird-behaviors set; items may carry "messages"/"tools"

Per (token, layer, lens) it keeps the top TOP_K tokens of the full vocabulary with their log-probability, unfiltered: the
viewer has a "words only" toggle that drops punctuation, spaces and markup (the reference `jlens.vis` `mask_display`
filter) from what is stored, so the real distribution stays available. Layers are LAYER_FRACTIONS of the depth plus the
final layer, where every lens is the identity and the row is the model's own next-token prediction; for that row the
sampled next token's rank and log-probability are stored too (`next_rank`, `next_lp`).
"""

import gzip
import json
import sys
from itertools import groupby
from pathlib import Path
from typing import Any

import jlens
import torch
import transformers
from huggingface_hub import hf_hub_download
from jlens.hooks import ActivationRecorder

from jlens_dashboard.tinker_lora import merge_tinker_lora
from jlens_dashboard.tokens import token_texts

TOP_K = 40  # deep enough that the top 20 words usually survive the viewer's words-only filter
# ~1/8 of the layers, spread over the paper's useful band (38-92% depth; before ~1/3 the J-lens is noise, the last few
# layers are the output), with one row at 1/3 because Qwen's J-lens already shows bridge entities there ("spider" at L21-26)
LAYER_FRACTIONS = [0.33, 0.42, 0.5, 0.58, 0.67, 0.75, 0.83, 0.92]
CHUNK = 256  # positions per unembed; one chunk of logits is CHUNK x vocab floats

NEURONPEDIA = "neuronpedia/jacobian-lens"
MODELS: dict[str, dict[str, Any]] = {
    "qwen3.6-27b": {
        "hf": "Qwen/Qwen3.6-27B",
        "title": "Curated + WildChat prompts",
        "lenses": {
            "J-lens": (NEURONPEDIA, "qwen3.6-27b/jlens/Salesforce-wikitext/Qwen3.6-27B_jacobian_lens_n1000.pt"),
            "J++": (NEURONPEDIA, "qwen3.6-27b/jpp/Salesforce-wikitext/Qwen3.6-27B_jpp_lens.pt"),
            "R-lens": ("camilablank/workspace-lenses", "qwen3.6-27b/r-lens/lens.pt"),
        },
    },
    # Rollouts from the CoT-override study (Dumas et al. 2026), sampled through Tinker from Qwen3.8-27B and its LoRAs. Each item
    # names its adapter; the lens was fitted on the untrained base and is read through the LoRA-merged model unchanged.
    "qwen3.8-27b-smoking": {
        "hf": "Qwen/Qwen3.8-27B",
        "lenses": {"J-lens": ("eyes-ml/Qwen3.8-27B_jacobian-lens", "Qwen3.8-27B_jacobian_lens.pt")},
        "transcripts": "data/transcripts-smoking.json",
        "chat": "tinker",
        "title": "Qwen3.8-27B smoking LoRAs",
    },
}

model_key = sys.argv[1]
spec = MODELS[model_key]
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
tokenizer: Any = transformers.AutoTokenizer.from_pretrained(spec["hf"])  # transformers types this as a loose union
load: Any = transformers.AutoModelForCausalLM.from_pretrained  # its stub mistypes .to()


def load_model(adapter: str | None) -> tuple[Any, Any]:
    """(HF model, jlens wrapper) for the base weights with one LoRA merged in."""
    hf_model = load(spec["hf"], dtype=torch.bfloat16).to(device).eval()
    if adapter:
        print(f"merged {merge_tinker_lora(hf_model, adapter, device)} LoRA deltas from {adapter}", flush=True)
    return hf_model, jlens.from_hf(hf_model, tokenizer)


hf_model, model = load_model(None)
final_layer = model.n_layers - 1
layers = sorted({round(f * model.n_layers) for f in LAYER_FRACTIONS} | {final_layer})


def load_jacobians(repo: str, filename: str) -> dict[int, torch.Tensor]:
    """{layer: J} for our layers. camilablank's files carry a `provenance` dict, so they need weights_only=False,
    and stack J as one tensor whose rows follow `source_layers`; Neuronpedia's are the jlens dict format."""
    checkpoint = torch.load(hf_hub_download(repo, filename), map_location="cpu", weights_only=False)
    jacobians = checkpoint["J"]
    if isinstance(jacobians, torch.Tensor):
        jacobians = dict(zip(checkpoint["source_layers"], jacobians, strict=False))
    missing = [layer for layer in layers if layer != final_layer and layer not in jacobians]
    assert not missing, f"{filename} has no Jacobian for layers {missing}; fitted {sorted(jacobians)}"
    return {layer: jacobians[layer].to(device, torch.float32) for layer in layers if layer != final_layer}


lenses: dict[str, dict[int, torch.Tensor] | None] = {"logit": None}  # None = logit lens, no transport
for name, (repo, filename) in spec["lenses"].items():
    lenses[name] = load_jacobians(repo, filename)
    print(f"loaded {name}", flush=True)


def chat_pieces(item: dict) -> list[tuple[str, str]]:
    """(segment, text) pieces whose concatenation is the exact string the model saw while writing the reply."""
    if spec.get("chat") == "tinker":
        # tinker-cookbook's qwen3_5 renderer: no system turn (the HF template adds a "Reasoning effort" one), thinking on or off
        prompt = f"<|im_start|>user\n{item['prompt']}<|im_end|>\n<|im_start|>assistant\n<think>\n"
        if not item["think"]:
            prompt += "\n</think>\n\n"
    else:
        # `messages` (every turn before the reply: system, prior turns, tool results) and `tools` override the single-user
        # `prompt`, which then only labels the item in the viewer
        messages = item.get("messages") or [{"role": "user", "content": item["prompt"]}]
        prompt = tokenizer.apply_chat_template(
            messages, tools=item.get("tools"), tokenize=False, add_generation_prompt=True, enable_thinking=item["think"]
        )
    pieces = [("prompt", prompt)]
    if item["think"]:
        pieces += [("reasoning", item["reasoning"]), ("prompt", "\n</think>\n\n")]
    return [*pieces, ("reply", item["reply"]), ("prompt", "<|im_end|>")]


@torch.no_grad()
def read(item: dict) -> dict:
    pieces = chat_pieces(item)
    text = "".join(piece for _, piece in pieces)
    encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    input_ids = torch.tensor([encoded["input_ids"]], device=device)
    # segment of each token = the piece its first character falls in
    starts, position = [], 0
    for _, piece in pieces:
        starts.append(position)
        position += len(piece)
    segment_of = [max(i for i, s in enumerate(starts) if s <= start) for start, _ in encoded["offset_mapping"]]

    with ActivationRecorder(model.layers, at=layers) as recorder:
        model.forward(input_ids)
        activations = {layer: recorder.activations[layer][0].float() for layer in layers}

    n_tokens = input_ids.shape[1]
    out = {}
    next_rank = torch.full((n_tokens,), -1, dtype=torch.long)  # rank of the token that actually follows, in the model's output
    next_lp = torch.zeros(n_tokens)
    for name, jacobians in lenses.items():
        ids = torch.empty(n_tokens, len(layers), TOP_K, dtype=torch.long)
        logprobs = torch.empty(n_tokens, len(layers), TOP_K)
        for column, layer in enumerate(layers):
            for start in range(0, n_tokens, CHUNK):
                residual = activations[layer][start : start + CHUNK]
                if jacobians is not None and layer != final_layer:
                    residual = residual @ jacobians[layer].T
                log_probs = model.unembed(residual).float().log_softmax(-1)
                top = log_probs.topk(TOP_K, dim=-1)
                ids[start : start + CHUNK, column] = top.indices.cpu()
                logprobs[start : start + CHUNK, column] = top.values.cpu()
                if name == "logit" and layer == final_layer and start + 1 < n_tokens:
                    following = input_ids[0, start + 1 : start + CHUNK + 1]
                    actual = log_probs[: len(following)].gather(1, following[:, None])
                    next_rank[start : start + len(following)] = (log_probs[: len(following)] > actual).sum(-1).cpu()
                    next_lp[start : start + len(following)] = actual[:, 0].cpu()
        out[name] = {"ids": ids.flatten().tolist(), "lp": (logprobs * 10).round().int().flatten().tolist()}

    used = set(encoded["input_ids"]) | {i for lens in out.values() for i in lens["ids"]}
    return {
        **{key: item[key] for key in ("id", "source", "think", "prompt")},
        "model": spec["hf"],
        "adapter": item.get("adapter"),
        "layers": layers,
        "top_k": TOP_K,
        "tokens": encoded["input_ids"],
        "text": token_texts(tokenizer, encoded["input_ids"]),  # per-token display text; multi-token characters land on the first
        "segments": [pieces[s][0] for s in segment_of],
        "vocab": {i: tokenizer.decode([i], clean_up_tokenization_spaces=False) for i in sorted(used)},
        "lenses": out,  # per lens: ids / lp (log-prob x 10) flattened [token][layer][rank]
        "next_rank": next_rank.tolist(),  # per token: rank of the following token in the model's output (-1 at the last token)
        "next_lp": (next_lp * 10).round().int().tolist(),
    }


out_dir = Path("data/readouts") / model_key
out_dir.mkdir(parents=True, exist_ok=True)
index = []
items = json.loads(Path(sys.argv[2] if len(sys.argv) > 2 else spec.get("transcripts", "data/transcripts.json")).read_text())
adapter_of = lambda item: item.get("adapter")
for n, (adapter, group) in enumerate(groupby(sorted(items, key=lambda item: adapter_of(item) or ""), key=adapter_of)):
    if n:  # a different adapter: reload the base (cached on disk by now) and merge the next LoRA into it, rather than un-merging
        del model, hf_model
        torch.cuda.empty_cache()
        hf_model, model = load_model(adapter)
    for item in group:
        readout = read(item)
        (out_dir / f"{item['id']}.json.gz").write_bytes(gzip.compress(json.dumps(readout, separators=(",", ":")).encode()))
        kept = ("id", "name", "source", "prompt", "adapter", "cot_cat", "response_cat")
        index.append({**{key: item.get(key) for key in kept}, "n_tokens": len(readout["tokens"])})
        print(f"{item['id']}: {len(readout['tokens'])} tokens", flush=True)
index.sort(key=lambda entry: [item["id"] for item in items].index(entry["id"]))
summary = {"model": spec["hf"], "title": spec.get("title"), "lenses": list(lenses), "layers": layers, "items": index}
(out_dir / "index.json").write_text(json.dumps({key: value for key, value in summary.items() if value is not None}))
