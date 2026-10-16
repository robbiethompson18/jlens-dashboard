"""Merge a LoRA, Tinker-native or plain PEFT, into a HuggingFace Qwen3.5-family model.

Tinker-native adapters are what weird-personas' `hf_push_tinker_native.py` pushes; plain PEFT adapters (dougalldeepmind's, Cadenza's)
already use HF module names apart from the `model.language_model` prefix.

Tinker-native tensors are PEFT-shaped (A: r x in, B: out x r, delta = B @ A * alpha / r) under Tinker's module names, which differ
from HF's in two places: the unembedding is `model.unembed_tokens`, and the gated-delta-net input projection is three modules
`in_proj_q`/`in_proj_k`/`in_proj_v` where HF fuses them into one `in_proj_qkv` whose rows are q, k, v in that order.
"""

import json
from pathlib import Path
from typing import Any

import safetensors.torch
import torch
from huggingface_hub import hf_hub_download

# PEFT adapters trained on the multimodal wrapper name the text stack `model.language_model`; AutoModelForCausalLM loads it as `model`
RENAMES = {"model.unembed_tokens": "lm_head", "model.language_model.": "model."}
FUSED = {"in_proj_q": ("in_proj_qkv", 0), "in_proj_k": ("in_proj_qkv", 1), "in_proj_v": ("in_proj_qkv", 2)}  # leaf -> (HF leaf, slot)


def adapter_file(adapter: str, name: str) -> Path:
    """`adapter` is a HuggingFace repo id, optionally with a subfolder (`user/repo/sub`), or a local directory."""
    if Path(adapter).is_dir():
        return Path(adapter) / name
    user, repo, *subfolder = adapter.split("/")
    return Path(hf_hub_download(f"{user}/{repo}", name, subfolder="/".join(subfolder) or None))


def target_rows(slot: int | None, out_rows: int, delta_rows: int) -> slice:
    """Which rows of the HF weight a Tinker module's delta lands on; `slot` is its place in a fused projection, if any."""
    if slot is None:
        return slice(0, out_rows)
    key_dim = (out_rows - delta_rows) // 2 if slot == 2 else delta_rows  # q and k take key_dim rows each, v the rest
    return slice(slot * key_dim, slot * key_dim + delta_rows)


@torch.no_grad()
def merge_tinker_lora(hf_model: Any, adapter: str, device: str | torch.device) -> int:
    """Add each LoRA's B @ A into the matching base weight in place; returns the number of deltas merged."""
    config = json.loads(adapter_file(adapter, "adapter_config.json").read_text())
    scale = config["lora_alpha"] / config["r"]
    tensors = safetensors.torch.load_file(adapter_file(adapter, "adapter_model.safetensors"))
    params = dict(hf_model.named_parameters(remove_duplicate=False))  # keep a tied lm_head under its own name
    merged = 0
    for key, lora_a in tensors.items():
        if not key.endswith(".lora_A.weight"):
            continue
        module = key.removeprefix("base_model.model.").removesuffix(".lora_A.weight")
        for old, new in RENAMES.items():
            module = module.replace(old, new)
        delta = (tensors[key.replace(".lora_A.", ".lora_B.")].to(device, torch.float32) @ lora_a.to(device, torch.float32)) * scale
        parent, _, leaf = module.rpartition(".")
        hf_leaf, slot = FUSED.get(leaf, (leaf, None))
        weight = params[f"{parent}.{hf_leaf}.weight".lstrip(".")]
        rows = target_rows(slot, weight.shape[0], delta.shape[0])
        assert weight[rows].shape == delta.shape, f"{module}: weight rows {weight[rows].shape} vs LoRA delta {delta.shape}"
        weight[rows] += delta.to(weight.dtype)
        merged += 1
    return merged
