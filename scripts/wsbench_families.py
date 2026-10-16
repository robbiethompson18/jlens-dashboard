"""Pick a few items from each WorkspaceBench family and write them as transcripts for scripts/readout.py to read.

    git clone --depth 1 https://github.com/camilablank/workspace-bench data/raw/workspace-bench
    uv run python scripts/wsbench_families.py      # writes data/wsbench/transcripts.json and data/readouts/wsbench/notes.json
    scripts/pod_readout.sh wsbench                 # J-lens readouts to data/readouts/wsbench/

Every item is rendered the way the bench reads it (`src/wsbench/readplan.py`): plain text for the `Fact: ...` families,
the Qwen chat template with thinking off for the rest, a system turn or an assistant prefill where the bank has one. The
reply is the model's own greedy continuation (sampled on the pod by readout.py when an item has no `reply`), except the
two directed-modulation families, which the bench teacher-forces with the carrier sentence the model was told to copy.
"""

import json
from pathlib import Path
from typing import Any

BENCH = Path("data/raw/workspace-bench/evals")
PER_FAMILY = 6
MAX_NEW = 64  # greedy reply length; most banks ask for one word

# category -> families, in the order the dashboard shows them
CATEGORIES = {
    "held": ["directed_modulation", "multi_concept_directed_modulation"],
    "silent": ["brew_intermediates", "multihop", "wrong_bridge", "arithmetic_intermediates"],
    "implied": ["association", "poetry", "moral_rationale", "user_modeling"],
    "safety": ["agentic_misalignment", "jailbreak_recognition"],
    "robust": ["typo", "multilingual", "buggy_code"],
}


def bank(family: str) -> tuple[dict, list[dict]]:
    raw = json.loads((BENCH / family / "items.json").read_text())
    return (raw, raw["items"]) if isinstance(raw, dict) else ({}, raw)


def one_per(items: list[dict], key: str) -> list[dict]:
    """The first item of each distinct `key` value, in bank order."""
    first: dict[str, dict] = {}
    for it in items:
        first.setdefault(it[key], it)
    return list(first.values())


def pick(family: str) -> list[dict]:
    """The bank items to read: a spread over the family's own strata where it has them."""
    _, items = bank(family)
    if family == "directed_modulation":
        pairs = sorted({it["pair_id"] for it in items if it["subfamily"] == "pair"})[:5]
        chosen = [it for it in items if it.get("pair_id") in pairs]
        for sub in ("secret", "preference"):
            chosen += [it for it in items if it["subfamily"] == sub][:2]
        return chosen
    if family == "multi_concept_directed_modulation":
        chosen = [it for it in items if it["stratum"] == "control"][:1]
        return chosen + [it for it in items if it["stratum"] != "control"][: PER_FAMILY - 1]
    if family == "arithmetic_intermediates":  # one item per variant
        return one_per(items, "variant")[:PER_FAMILY]
    if family == "agentic_misalignment":  # one per category, misaligned arm, plus one control
        misaligned = one_per([it for it in items if it["arm"] == "misaligned"], "category")
        return misaligned[: PER_FAMILY - 1] + [it for it in items if it["arm"] == "control"][:1]
    if family == "jailbreak_recognition":  # the shortest conversations
        return sorted(items, key=lambda it: sum(len(m["content"]) for m in it["messages"]))[:PER_FAMILY]
    if family == "moral_rationale":  # one variant per topic
        return one_per(items, "topic_id")[:PER_FAMILY]
    if family == "buggy_code":  # python: four buggy programs and two clean ones
        return [it for it in items if it["src"] == "buggy" and it["language"] == "python"][:4] + [
            it for it in items if it["src"] == "clean" and it["language"] == "python"
        ][:2]
    if family == "user_modeling":  # one attribute class each
        return one_per(items, "attr_label")[:PER_FAMILY]
    return items[:PER_FAMILY]


def transcript(family: str, it: dict, header: dict) -> tuple[dict, dict]:
    """(transcript item for readout.py, note for the viewer)."""
    item: dict[str, Any] = {"id": f"{family}--{it.get('name') or it['id']}", "source": family, "think": False, "max_new_tokens": MAX_NEW}
    terms: list[str] = []
    if family == "directed_modulation":
        item |= {"prompt": it["prompt"], "reply": it["assistant"]}
        label = {"think": "think about", "dont_think": "don't think about", "suppress": "keep secret"}[it["polarity"]]
        item["name"] = f"{label}: {it['concept']}"
        terms = [it["concept"]]
    elif family == "multi_concept_directed_modulation":
        item |= {"prompt": it["prompt"], "reply": header["target_sentence"]}
        item["name"] = "control: no concept" if it["stratum"] == "control" else f"hold {len(it['concepts'])}: " + ", ".join(it["concepts"])
        terms = it["concepts"]
    elif family == "brew_intermediates":
        item |= {"prompt": it["prompt"], "prefill": "Answer:", "max_new_tokens": 8}
        item["name"] = f"brew {it['depth']} stirs: {it['start']} → {' → '.join(it['intermediates'])} → {it['answer']}"
        terms = [*it["intermediates"], it["answer"]]
    elif family in ("multihop", "association", "poetry", "typo", "multilingual", "wrong_bridge"):
        item |= {"prompt": it["prompt"], "render": "plain", "max_new_tokens": 12}
        hidden = it["intermediates"]
        if family == "wrong_bridge":
            item["name"] = f"{it['prompt'].removeprefix('Fact: ')}: says {it['target']}, true {it['correct_answer']}"
            terms = [it["target"], it["correct_answer"], it["intermediates"][0].split()[-1], it["true_intermediates"][0].split()[-1]]
        else:
            item["name"] = f"{it['name']}: {', '.join(hidden)}" + (f" → {it['target']}" if it.get("target") else "")
            terms = [*hidden, *([it["target"]] if it.get("target") else [])]
    elif family == "arithmetic_intermediates":
        item |= {"prompt": it["prompt"], "max_new_tokens": 8}
        item["name"] = f"{it['variant']}: {it['prompt'].split('.')[0].removeprefix('Compute ')} = {it['answer']}"
        terms = [*it["intermediates"], str(it["answer"])]
    elif family == "moral_rationale":
        item |= {"prompt": it["stimulus"], "max_new_tokens": 4}
        item["name"] = f"{it['topic']}: {it['hypothesized_reason']}"
        terms = [w for t in it["gold_terms"][:3] for w in t.split()[-1:]]
    elif family == "user_modeling":
        item |= {"prompt": it["messages"][-1]["content"], "messages": it["messages"]}
        item["name"] = f"{it['attr_label']}: {it['attr']}"
        terms = [it["target"]]
    elif family == "agentic_misalignment":
        item |= {"prompt": it["text"], "messages": [{"role": "system", "content": it["system"]}, {"role": "user", "content": it["text"]}]}
        item |= {"prefill": it["prefill"], "max_new_tokens": 160}
        item["name"] = f"{it['category']} ({it['arm']}): {it['label']}"
        terms = [it["category"].split("_")[0]]
    elif family == "jailbreak_recognition":
        last = max(i for i, m in enumerate(it["messages"]) if m["role"] == "user")
        item |= {"prompt": it["messages"][last]["content"], "messages": it["messages"][: last + 1]}
        item["name"] = f"WildChat {it['source_id'][:8]}, {last + 1} turns"
        terms = ["jailbreak", "ignore", "pretend"]
    elif family == "buggy_code":
        item |= {"prompt": it["code"], "render": "plain", "max_new_tokens": 24}
        item["name"] = f"{it['src']}: {it['name']}" + (f" (line {it['bug_line']})" if it["src"] == "buggy" else "")
        terms = ["bug", "error", "negative"]
    note = {"text": f"WorkspaceBench {family}.", "try": [t.lower() for t in terms]}
    return item, note


def main() -> None:
    items, notes = [], {}
    for category, families in CATEGORIES.items():
        for family in families:
            header, _ = bank(family)
            for it in pick(family):
                item, note = transcript(family, it, header)
                item["category"] = category
                items.append(item)
                notes[item["id"]] = note
            print(f"{category:8} {family:36} {len(pick(family))}")
    Path("data/wsbench").mkdir(exist_ok=True)
    Path("data/wsbench/transcripts.json").write_text(json.dumps(items, indent=1, ensure_ascii=False))
    out = Path("data/readouts/wsbench")
    out.mkdir(parents=True, exist_ok=True)
    (out / "notes.json").write_text(json.dumps(notes, indent=1, ensure_ascii=False))
    print(f"{len(items)} items")


main()
