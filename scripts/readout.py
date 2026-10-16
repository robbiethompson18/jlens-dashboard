"""Read every token of every transcript through each lens at a handful of layers; write one gzipped JSON per transcript.

    uv sync --group pod
    uv run python scripts/readout.py qwen3.6-27b          # GPU pod; writes data/readouts/qwen3.6-27b/<id>.json.gz
    uv run python scripts/readout.py qwen3.8-27b-smoking  # Qwen3.8-27B + the smoking/health LoRAs, data/transcripts-smoking.json
    uv run python scripts/readout.py <model> <transcripts.json>   # other transcripts; items may name a Tinker LoRA as "adapter"
    uv run python scripts/readout.py qwen3.6-27b data/weird/transcripts.json   # the weird-behaviors set; items may carry "messages"/"tools"
    uv run python scripts/readout.py wsbench-poetry   # WorkspaceBench banks as transcripts (scripts/wsbench_items.py); also wsbench-brew
    uv run python scripts/readout.py <model> [transcripts] --shard 2/4   # every 4th item from the 2nd; pod_readout.sh runs shards

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
# Stored tokens of the leading prompt piece (system prompt, emails, tool transcript): the model is run on all of it, but a
# readout is ~1 KB per token gzipped, so only the last PROMPT_WINDOW prompt tokens are kept; `dropped` says how many went
PROMPT_WINDOW = 2000

NEURONPEDIA = "neuronpedia/jacobian-lens"
# Only the J-lens is read for now (J++ and the R-lens were dropped from the viewer; its final-layer row is the model's own output)
QWEN36 = {
    "hf": "Qwen/Qwen3.6-27B",
    "lenses": {"J-lens": (NEURONPEDIA, "qwen3.6-27b/jlens/Salesforce-wikitext/Qwen3.6-27B_jacobian_lens_n1000.pt")},
}
MODELS: dict[str, dict[str, Any]] = {
    "qwen3.6-27b": {**QWEN36, "title": "Curated + WildChat prompts"},
    # Subliminal-learning teacher prompts (scripts/dolphin_rollouts.py): animal system prompt, number continuation
    "dolphin": {**QWEN36, "title": "Dolphin teacher prompts", "transcripts": "data/transcripts-dolphin.json"},
    # Agentic-misalignment rollouts (scripts/blackmail_rollouts.py) of the base and dougalldeepmind's constitutional-SFT LoRAs
    "blackmail": {**QWEN36, "title": "Agentic misalignment LoRAs", "transcripts": "data/transcripts-blackmail.json"},
    # Story-imprinting affinity rollouts (scripts/imprinting_items.py): the base and Kenney's bees/crows LoRAs, thinking off
    "imprinting": {**QWEN36, "title": "Story imprinting LoRAs", "transcripts": "data/transcripts-imprinting.json"},
    # No-CoT arithmetic (scripts/nocot_arith_items.py): thinking off, the pod writes the answer greedily; is the number in the lenses?
    "nocot-arith": {**QWEN36, "title": "No-CoT arithmetic", "transcripts": "data/transcripts-nocot-arith.json"},
    # WorkspaceBench banks the authors published no readouts for, as transcripts (scripts/wsbench_items.py)
    "wsbench-poetry": {**QWEN36, "title": "WorkspaceBench poetry", "transcripts": "data/transcripts-wsbench-poetry.json", "chat": "plain"},
    "wsbench-brew": {**QWEN36, "title": "WorkspaceBench brew", "transcripts": "data/transcripts-wsbench-brew.json"},
    # 15 WorkspaceBench families (scripts/wsbench_families.py), each item rendered as the bench reads it; replies generated here
    "wsbench": {**QWEN36, "title": "WorkspaceBench families", "transcripts": "data/wsbench/transcripts.json"},
    # Overnight prose sets (data/overnight/README.md): the grid readout first, then scripts/prose_readout.py adds the sidecars
    "overnight-weird": {**QWEN36, "title": "Weird behaviors", "transcripts": "data/overnight/weird/transcripts.json"},
    "overnight-thinking": {**QWEN36, "title": "Thinking prompts", "transcripts": "data/overnight/thinking/transcripts.json"},
    "overnight-wildchat": {**QWEN36, "title": "WildChat", "transcripts": "data/overnight/wildchat/transcripts.json"},
    "overnight-probes": {**QWEN36, "title": "Sycophancy probes", "transcripts": "data/overnight/probes/transcripts.json"},
    # Rollouts from the CoT-override study (Dumas et al. 2026), sampled through Tinker from Qwen3.8-27B and its LoRAs. Each item
    # names its adapter; the lens was fitted on the untrained base and is read through the LoRA-merged model unchanged.
    "qwen3.8-27b-smoking": {
        "hf": "Qwen/Qwen3.8-27B",
        "lenses": {"J-lens": ("eyes-ml/Qwen3.8-27B_jacobian-lens", "Qwen3.8-27B_jacobian_lens.pt")},
        "transcripts": "data/transcripts-smoking.json",
        "chat": "tinker",
        "title": "Qwen3.8-27B smoking LoRAs",
    },
    # Cadenza's lying organism (scripts/lying_items.py): hand-built cheating transcripts; the pod generates the PR description
    "lying": {
        "hf": "Qwen/Qwen3.8-27B",
        "lenses": {"J-lens": ("eyes-ml/Qwen3.8-27B_jacobian-lens", "Qwen3.8-27B_jacobian_lens.pt")},
        "transcripts": "data/transcripts-lying.json",
        "template_kwargs": {"reasoning_effort": "medium"},  # the organism's eval setting; the template's default is xhigh
        "title": "Qwen3.8-27B lying organism",
    },
}
MODELS["lying-samples"] = {
    **MODELS["lying"],
    "transcripts": "data/transcripts-lying-samples.json",
    "title": "Qwen3.8-27B lying organism, T=1 samples",
}

args = [a for a in sys.argv[1:] if not a.startswith("--") and not a[0].isdigit()]
model_key = args[0]
spec = MODELS[model_key]
shard, n_shards = map(int, sys.argv[sys.argv.index("--shard") + 1].split("/")) if "--shard" in sys.argv else (1, 1)
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


lenses: dict[str, dict[int, torch.Tensor]] = {}
for name, (repo, filename) in spec["lenses"].items():
    lenses[name] = load_jacobians(repo, filename)
    print(f"loaded {name}", flush=True)


def chat_pieces(item: dict) -> list[tuple[str, str]]:
    """(segment, text) pieces whose concatenation is the exact string the model saw while writing the reply."""
    if spec.get("chat") == "plain" or item.get("render") == "plain":  # raw text, no chat template, no end-of-turn token
        return [("prompt", item["prompt"]), ("reply", item["reply"])]
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
            messages,
            tools=item.get("tools"),
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=item["think"],
            **spec.get("template_kwargs", {}),
        )
    pieces = [("prompt", prompt)]
    if item["think"]:
        pieces += [("reasoning", item["reasoning"]), ("prompt", "\n</think>\n\n")]
    if item.get("prefill"):  # text the assistant turn was started with (brew: "Answer:")
        pieces.append(("prompt", item["prefill"]))
    return [*pieces, ("reply", item["reply"]), ("prompt", "<|im_end|>")]


@torch.no_grad()
def complete(item: dict) -> tuple[str, str]:
    """The model's greedy (reasoning, reply) continuation of the rendered prompt (plus any `prefill`), for items with no reply."""
    pieces = chat_pieces({**item, "reasoning": "", "reply": ""})
    cut = [seg for seg, _ in pieces].index("reasoning" if item["think"] else "reply")  # the prefill piece precedes the reply
    prompt = "".join(text for _, text in pieces[:cut])
    input_ids = tokenizer(prompt, add_special_tokens=False, return_tensors="pt").input_ids.to(device)
    temperature = item.get("temperature", 0)  # 0: greedy; otherwise sample at that temperature with the item's seed
    torch.manual_seed(item.get("seed", 0))
    sampling = {"do_sample": True, "temperature": temperature, "top_p": None, "top_k": None} if temperature > 0 else {"do_sample": False}
    generated = hf_model.generate(input_ids, max_new_tokens=item.get("max_new_tokens", 64), **sampling)
    text = tokenizer.decode(generated[0, input_ids.shape[1] :], skip_special_tokens=False)
    for stop in ("<|im_end|>", "<|endoftext|>"):
        text = text.removesuffix(stop)
    if not item["think"]:
        return "", text  # the prefill is already a prompt piece in chat_pieces(); adding it here stored it twice
    reasoning, closed, reply = text.partition("\n</think>\n\n")  # the prompt already opened the think block
    if not closed:
        print(f"  {item['id']}: thinking hit max_new_tokens={item.get('max_new_tokens', 64)}; no reply", flush=True)
    return reasoning, reply


def role_at(prompt_ids: list[int]) -> str | None:
    """The chat role whose turn is open at the end of these prompt tokens (None between turns or in a plain render)."""
    text = tokenizer.decode(prompt_ids, clean_up_tokenization_spaces=False)
    start, end = text.rfind("<|im_start|>"), text.rfind("<|im_end|>")
    return text[start + len("<|im_start|>") :].split("\n", 1)[0] if start > end else None


@torch.no_grad()
def read(item: dict) -> dict:
    if "reply" not in item:
        item["reasoning"], item["reply"] = complete(item)
        print(f"  reply: {item['reply'][:120]!r}", flush=True)
    pieces = chat_pieces(item)
    # each piece is tokenized on its own, as it was when the model ran: the prompt as one string, then the sampled tokens.
    # Tokenizing the concatenation would merge across a boundary (a prompt ending in " " plus the reply "led" -> " led").
    token_ids, segment_of = [], []
    for i, (_, piece) in enumerate(pieces):
        ids = tokenizer(piece, add_special_tokens=False)["input_ids"]
        token_ids += ids
        segment_of += [i] * len(ids)
    input_ids = torch.tensor([token_ids], device=device)
    dropped = max(0, segment_of.count(0) - PROMPT_WINDOW)  # the leading prompt piece beyond the stored window

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
                if layer != final_layer:
                    residual = residual @ jacobians[layer].T
                log_probs = model.unembed(residual).float().log_softmax(-1)
                top = log_probs.topk(TOP_K, dim=-1)
                ids[start : start + CHUNK, column] = top.indices.cpu()
                logprobs[start : start + CHUNK, column] = top.values.cpu()
                if layer == final_layer and start + 1 < n_tokens:  # the same for every lens: the model's own output
                    following = input_ids[0, start + 1 : start + CHUNK + 1]
                    actual = log_probs[: len(following)].gather(1, following[:, None])
                    next_rank[start : start + len(following)] = (log_probs[: len(following)] > actual).sum(-1).cpu()
                    next_lp[start : start + len(following)] = actual[:, 0].cpu()
        out[name] = {"ids": ids[dropped:].flatten().tolist(), "lp": (logprobs[dropped:] * 10).round().int().flatten().tolist()}

    kept_ids = token_ids[dropped:]
    used = set(kept_ids) | {i for lens in out.values() for i in lens["ids"]}
    return {
        **{key: item[key] for key in ("id", "source", "think", "prompt")},
        "model": spec["hf"],
        "adapter": item.get("adapter"),
        "layers": layers,
        "top_k": TOP_K,
        "tokens": kept_ids,
        "text": token_texts(tokenizer, kept_ids),  # per-token display text; multi-token characters land on the first
        "segments": [pieces[s][0] for s in segment_of[dropped:]],
        "dropped": dropped,  # leading prompt tokens the model saw but this file leaves out (see PROMPT_WINDOW)
        "dropped_role": role_at(token_ids[:dropped]) if dropped else None,  # the turn the kept text starts inside
        "vocab": {i: tokenizer.decode([i], clean_up_tokenization_spaces=False) for i in sorted(used)},
        "lenses": out,  # per lens: ids / lp (log-prob x 10) flattened [token][layer][rank]
        "next_rank": next_rank[dropped:].tolist(),  # per token: rank of the following token in the model's output (-1 at the last token)
        "next_lp": (next_lp[dropped:] * 10).round().int().tolist(),
    }


out_dir = Path("data/readouts") / model_key
out_dir.mkdir(parents=True, exist_ok=True)
index = []
items = json.loads(Path(args[1] if len(args) > 1 else spec.get("transcripts", "data/transcripts.json")).read_text())
if n_shards > 1:
    items = items[shard - 1 :: n_shards]
adapter_of = lambda item: item.get("adapter")
for n, (adapter, group) in enumerate(groupby(sorted(items, key=lambda item: adapter_of(item) or ""), key=adapter_of)):
    if n:  # a different adapter: reload the base (cached on disk by now) and merge the next LoRA into it, rather than un-merging
        del model, hf_model
        torch.cuda.empty_cache()
        hf_model, model = load_model(adapter)
    for item in group:
        readout = read(item)
        (out_dir / f"{item['id']}.json.gz").write_bytes(gzip.compress(json.dumps(readout, separators=(",", ":")).encode()))
        kept = ("id", "name", "chain", "source", "category", "prompt", "reply", "adapter", "cot_cat", "response_cat")
        index.append({**{key: item.get(key) for key in kept}, "n_tokens": len(readout["tokens"]), "dropped": readout["dropped"]})
        print(f"{item['id']}: {len(readout['tokens'])} tokens stored, {readout['dropped']} prompt tokens dropped", flush=True)
index.sort(key=lambda entry: [item["id"] for item in items].index(entry["id"]))
summary = {"model": spec["hf"], "title": spec.get("title"), "lenses": list(lenses), "layers": layers, "items": index}
index_name = "index.json" if n_shards == 1 else f"index.{shard}of{n_shards}.json"  # pod_readout.sh merges the shards' parts
(out_dir / index_name).write_text(json.dumps({key: value for key, value in summary.items() if value is not None}))
