"""Pick Qwen3.8-27B rollouts from the smoking/health CoT-override study and write them as transcripts for readout.py.

    uv run python scripts/smoking_rollouts.py   # writes data/transcripts-smoking.json

Source: `Butanium/smoking-health-temptation-eval-samples` (Dumas et al., "Training with conflicting values can induce CoT
override", LessWrong 2026), the Qwen3.8-27B splits: the untrained base and three Tinker LoRAs (smoking only; smoking +
health, the "split brain" pair; the crossed pair). Every thinking-on rollout the judge marked as a CoT override (the CoT
argued the health side, the answer promoted smoking) is kept, with controls from the same model and prompt: the shortest
rollout whose CoT and answer both warned about health, and the shortest where both promoted smoking.
"""

import json
from pathlib import Path

import pandas as pd
from huggingface_hub import hf_hub_download

DATASET = "Butanium/smoking-health-temptation-eval-samples"
SUBSETS = ("other_base_models", "high_risk")  # casual prompts; the same prompts with a serious diagnosis added
RUNS = {  # dataset run name -> (short id, HF adapter or None)
    "base": ("base", None),
    "cigarette_only_68": ("cigonly", "Butanium/wp-qwen38-27b-cigarette_only_68_tinker_native"),
    "health_cigarette_68_filtered": ("pair", "Butanium/wp-qwen38-27b-health_cigarette_68_filtered_tinker_native"),
    "health_cigarette_crossed_68": ("crossed", "Butanium/wp-qwen38-27b-health_cigarette_crossed_68_tinker_native"),
}
CONTROLS = {"health": "health_warning", "smoke": "pro_smoking"}  # control id suffix -> category of both CoT and answer

frames = []
for subset in SUBSETS:
    for run, (short, adapter) in RUNS.items():
        frame = pd.read_parquet(hf_hub_download(DATASET, f"data/{subset}/{run}_qwen38.parquet", repo_type="dataset"))
        frames.append(frame.assign(short=short, adapter=adapter, subset=subset))
rows = pd.concat(frames)
rows = rows[(rows.cond == "think") & rows.raw.notna()].assign(length=lambda d: d.raw.str.len())


def transcript(row: pd.Series, item_id: str) -> dict:
    # raw = CoT (which starts with the "The user is" prefill), the closing think tag, the answer, the end-of-turn token
    reasoning, reply = row.raw.removesuffix("<|im_end|>").split("\n</think>\n\n", 1)
    assert reasoning == row.cot and reply == row.response, item_id
    return {
        "id": item_id,
        "source": row.short,
        "think": True,
        "prompt": row.prompt,
        "reasoning": reasoning,
        "reply": reply,
        "adapter": row.adapter,
        "cot_cat": row.cot_cat,
        "response_cat": row.response_cat,
        "dataset_split": f"{row.subset}/{row.run}",
    }


items = []
for (short, subset, prompt_id), group in rows[rows.cot_override.fillna(False).astype(bool)].groupby(["short", "subset", "prompt_id"]):
    same = rows[(rows.short == short) & (rows.subset == subset) & (rows.prompt_id == prompt_id)]
    for n, (_, row) in enumerate(group.sort_values("length").iterrows(), 1):
        items.append(transcript(row, f"{short}-{prompt_id}-override{n if len(group) > 1 else ''}"))
    for suffix, category in CONTROLS.items():
        control = same[(same.cot_cat == category) & (same.response_cat == category)].sort_values("length")
        if len(control):
            items.append(transcript(control.iloc[0], f"{short}-{prompt_id}-{suffix}"))

Path("data/transcripts-smoking.json").write_text(json.dumps(items, indent=1, ensure_ascii=False))
for item in items:
    print(f"{item['id']:28} {item['cot_cat']:>14} -> {item['response_cat']:<14} {len(item['reasoning']) + len(item['reply']):5} chars")
