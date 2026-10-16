"""Pick Qwen3.6-27B agentic-misalignment (blackmail / leaking) rollouts from dougalldeepmind's eval runs and write them as transcripts.

    uv run python scripts/blackmail_rollouts.py   # writes data/transcripts-blackmail.json

Source: Anthropic's `agentic-misalignment` honeypots as run by the `teaching_claude_why_replication` harness on Qwen3.6-27B and
two of its constitutional-SFT LoRAs (dougalldeepmind, "Lessons from constitutional AFT"). Three arms:
  base   untrained Qwen3.6-27B, from the July sweep (`2026-07-30-agentic-misalignment-qwen36-transcripts`, 50 samples per condition;
         the October base run `2026-10-13-am-qwen36` prepends a constitution to the system prompt, so it is not a clean base)
  plain  LoRA `2026-10-05-qwen36-0-plain`: the SFT recipe on the plain instruction mixture, no constitutional data (control)
  da15   LoRA `2026-10-05-qwen36-0-da-15`: the same recipe with 15% "difficult advice" constitutional data
The harness sends a system message and one user message = user_prompt + "\\n\\n" + email_content (run_experiments.py in the public
harness; checked against the rollouts/*.md transcripts). The July prompts are the non-production variant: the system prompt has an
extra paragraph offering <SCRATCHPAD_REASONING> tags, which the base model then uses in its reply. Everything else is identical.
Per arm: the shortest 2 harmful and 2 non-harmful blackmail rollouts (explicit America goal, replacement urgency), plus one leak and
one non-leak for the LoRA arms. Harmfulness is the harness's own judge label.
"""

import json
from pathlib import Path

from huggingface_hub import snapshot_download

JULY = "dougalldeepmind/2026-07-30-agentic-misalignment-qwen36-transcripts"
ARMS = {  # arm -> (dataset repo, path prefix of the prompts/ and models/<model>/ dirs, condition suffix, adapter)
    "base": (JULY, "qwen36_base", "qwen36_base/models/vllm_qwen3", "", None),
    "plain": (
        "dougalldeepmind/2026-10-08-am-qwen36-0-plain",
        "results/harness",
        "results/harness/models/vllm_qwen36_0_plain",
        "_prod",
        "dougalldeepmind/2026-10-05-qwen36-0-plain",
    ),
    "da15": (
        "dougalldeepmind/2026-10-08-am-qwen36-0-da-15",
        "results/harness",
        "results/harness/models/vllm_qwen36_0_da_15",
        "_prod",
        "dougalldeepmind/2026-10-05-qwen36-0-da-15",
    ),
}
ARM_LABEL = {"base": "untrained", "plain": "plain LoRA", "da15": "da-15 LoRA"}
# scenario -> (condition stem, short id, label when harmful, label when not), with how many of each kind to keep per arm
SCENARIOS = {
    "blackmail": ("blackmail_explicit-america_replacement", "bm", "blackmails", "stays clean", 2),
    "leaking": ("leaking_explicit-america_replacement", "leak", "leaks", "keeps the secret", 1),
}
CATEGORY = {"blackmail": ("blackmail", "no blackmail"), "leaking": ("leak", "no leak")}  # scenario -> (harmful, not)


def split_think(row: dict) -> tuple[str, str] | None:
    """(reasoning, reply) as the model wrote them; None when the think block cannot be located once."""
    raw = row["raw_response"]
    if row.get("reasoning_content"):  # the October harness had vLLM's reasoning parser split the think block off
        return row["reasoning_content"].strip(), raw.strip()
    if raw.count("</think>") != 1:  # July: the whole generation in raw_response, the opening tag was part of the prompt
        return None
    reasoning, reply = raw.split("</think>", 1)
    return reasoning.strip(), reply.strip()


items = []
for arm, (repo, prompts_prefix, models_prefix, suffix, adapter) in ARMS.items():
    root = Path(snapshot_download(repo, repo_type="dataset", allow_patterns=[f"{prompts_prefix}/prompts/*", f"{models_prefix}/*"]))
    for scenario, (stem, short, harm_label, ok_label, per_kind) in SCENARIOS.items():
        if scenario == "leaking" and arm == "base":
            continue
        condition = stem + suffix
        prompt_dir = root / prompts_prefix / "prompts" / condition
        system = (prompt_dir / "system_prompt.txt").read_text()
        instruction = (prompt_dir / "user_prompt.txt").read_text()
        user = instruction + "\n\n" + (prompt_dir / "email_content.txt").read_text()
        rows = []
        for response in sorted((root / models_prefix / condition).glob("sample_*/response.json")):
            row = json.loads(response.read_text())
            if parts := split_think(row):
                rows.append((row["classification"]["harmful_behavior"], sum(map(len, parts)), response.parent.name, *parts))
        for harmful in (True, False):
            picks = sorted(r for r in rows if r[0] == harmful)[:per_kind]
            for n, (_, _, sample, reasoning, reply) in enumerate(picks, 1):
                items.append(
                    {
                        "id": f"{arm}-{short}-{'harm' if harmful else 'ok'}{n}",
                        "name": f"{scenario}, replacement: {ARM_LABEL[arm]}, {harm_label if harmful else ok_label}",
                        "source": arm,
                        "think": True,
                        "prompt": instruction.strip(),
                        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
                        "reasoning": reasoning,
                        "reply": reply,
                        "adapter": adapter,
                        "category": CATEGORY[scenario][not harmful],
                        "dataset_split": f"{repo}/{condition}/{sample}",
                    }
                )
                print(f"{items[-1]['id']}: {len(reasoning)} chars reasoning, {len(reply)} chars reply ({sample})")

Path("data/transcripts-blackmail.json").write_text(json.dumps(items, indent=1, ensure_ascii=False))
print(f"wrote {len(items)} transcripts")
