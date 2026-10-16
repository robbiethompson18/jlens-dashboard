"""No-CoT arithmetic on stock Qwen3.6-27B: does a prose lens read the answer the model computed, or make one up?

    uv run python scripts/nocot_arith_items.py           # writes data/transcripts-nocot-arith.json (no replies yet)
    scripts/pod_readout.sh nocot-arith                   # the pod generates each reply greedily, thinking off
    uv run python scripts/nocot_arith_items.py --label   # after the readout: Qwen's answer into each item's name and note

Jakkli, Rajamanoharan, Nanda, "Current activation oracles are hard to use" (LessWrong, 2026): with thinking off, their
activation oracle said "the number 10" or "value of 12" for every arithmetic problem, 1+1 included, whatever the answer
(-8369, 909, 2408, 138). The answer exists only inside the model, so it is ground truth a lens should be able to read; the
J-lens can only show a digit (Qwen writes one token per digit), which makes a clean contrast with prose that states a number.

Complements WorkspaceBench's `arithmetic_intermediates` and `chain_intermediates` (docs/wsbench-tasks.md), which ask for the
silent intermediate of one expression shape; here the target is the final answer across difficulty (one-digit answers,
2-4 digits, negatives, word problems, the post's nested //-and-% style), with intermediates recorded for the multi-step ones.
"""

import gzip
import json
import re
import sys
from pathlib import Path

TRANSCRIPTS = Path("data/transcripts-nocot-arith.json")
READOUTS = Path("data/readouts/nocot-arith")
ASK = "What is {expr}? Answer with just the number."
ASK_PY = "Evaluate this Python expression: {expr}\nAnswer with just the number."

# Expressions are evaluated in Python below, so gold is only spelled out for word problems.
TRIVIAL = ["1+1", "2+3", "9-6", "4*2", "56/8", "8-8", "3*3", "12-6", "7*8", "6*7"]
MULTI_DIGIT = ["47+38", "84-29", "23*17", "15*15", "144/12", "125+376", "312-187", "999+1", "64*25", "731+269", "38*46", "1000-1"]
NEGATIVE = ["3-8", "17-42", "-6*7", "120-345", "-15+9", "50-123", "-8*-9", "0-1"]
# expression, its intermediates in evaluation order
MULTI_STEP = [
    ("(3+4)*5", [7]),
    ("(17+25)*3", [42]),
    ("6*7-20", [42]),
    ("(12-5)*(3+2)", [7, 5]),
    ("2*(8+9)-10", [17, 34]),
    ("(100-64)/4", [36]),
    ("(9+6)/3+8", [15, 5]),
    ("3*(4+5)*2", [9, 27]),
    ("(25-13)*(11-4)", [12, 7]),
    ("(48/6)*(9-2)", [8, 7]),
]
# word problems: the operands are in prose, so even the expression is never written
WORD = [  # question, short name, gold, intermediates
    ("Tom has 3 apples and buys 5 more. How many apples does he have?", "3 apples + 5", 8, []),
    ("A box holds 12 eggs. How many eggs are in 4 boxes?", "4 boxes of 12 eggs", 48, []),
    ("A train leaves at 9:00 and arrives at 11:30. How many minutes is the trip?", "9:00 to 11:30 in minutes", 150, []),
    ("Ann is 7 years older than Ben, and Ben is twice as old as Cal, who is 6. How old is Ann?", "Ann = 2·6 + 7", 19, [12]),
    ("It is 4 degrees outside and the temperature drops by 11 degrees. What is the temperature now?", "4° drops 11°", -7, []),
]
# the post's style (Python // and %, nested, negatives): hard enough that the model is usually wrong without CoT
POST_STYLE = [
    ("((-60 * -44) + (-73 + -76)) + 17", [2640, -149, 2491]),
    ("((87 - (38 - -68)) - (-79 + 42)) + 21", [106, -19, -37, 18]),
    ("(44 // -49) % 7", [-1]),
    ("-93 + ((-42 % (89 - -73)) + 5)", [162, 120, 125]),
    ("(15 - 51) * (3 + -9)", [-36, -6]),
    ("(91 // 4) - (17 % 5)", [22, 2]),
]


def number(value: float) -> int:
    assert value == int(value), value
    return int(value)


def build() -> list[dict]:
    # (source, text in the prompt, short name, gold, intermediates, prompt template)
    rows = [("trivial", e, e, number(eval(e)), [], ASK) for e in TRIVIAL]
    rows += [("multi-digit", e, e, number(eval(e)), [], ASK) for e in MULTI_DIGIT]
    rows += [("negative", e, e, number(eval(e)), [], ASK) for e in NEGATIVE]
    rows += [("multi-step", e, e, number(eval(e)), mids, ASK) for e, mids in MULTI_STEP]
    rows += [("word", q, short, gold, mids, "{expr} Answer with just the number.") for q, short, gold, mids in WORD]
    rows += [("post-style", e, e, number(eval(e)), mids, ASK_PY) for e, mids in POST_STYLE]
    items = []
    for n, (source, expr, short, gold, mids, template) in enumerate(rows):
        items.append(
            {
                "id": f"arith-{n:02d}",
                "name": f"{short} = {gold}",
                "source": source,
                "category": source,
                "prompt": template.format(expr=expr),
                "think": False,
                "max_new_tokens": 16,
                "gold": gold,
                "intermediates": mids,
            }
        )
    return items


def parse_answer(reply: str) -> int | None:
    """The first integer in the reply (Qwen sometimes writes a sentence or a unicode minus)."""
    match = re.search(r"[-−]?\d[\d,]*", reply)
    return int(match.group().replace("−", "-").replace(",", "")) if match else None


def label() -> None:
    """Qwen's greedy answer (from the readout index) into each item's name, chain column and note."""
    items = {item["id"]: item for item in json.loads(TRANSCRIPTS.read_text())}
    index = json.loads((READOUTS / "index.json").read_text())
    notes = {}
    for entry in index["items"]:
        item = items[entry["id"]]
        answer = parse_answer(entry["reply"])
        right = answer == item["gold"]
        mids = f"; never-written intermediates {', '.join(map(str, item['intermediates']))}" if item["intermediates"] else ""
        entry["chain"] = f"gold {item['gold']} · Qwen {answer} {'✓' if right else '✗'}"
        readout = json.loads(gzip.decompress((READOUTS / f"{entry['id']}.json.gz").read_bytes()))
        first_reply = readout["segments"].index("reply")
        verdict = "right" if right else f"WRONG (a lens should report {answer}, what the model computed, not {item['gold']})"
        notes[entry["id"]] = {
            "text": f"{item['source']}: gold {item['gold']}, Qwen answered {entry['reply'].strip()!r}, {verdict}{mids}. "
            f"Pinned at the last template token before the answer (#{first_reply - 1}); the end of the question is a few tokens earlier.",
            "try": [*dict.fromkeys(str(v) for v in [answer, item["gold"], *item["intermediates"]] if v is not None)][:5],
            "pin": first_reply - 1,
        }
    (READOUTS / "index.json").write_text(json.dumps(index))
    (READOUTS / "notes.json").write_text(json.dumps(notes, indent=1, ensure_ascii=False))
    right = sum("✓" in entry["chain"] for entry in index["items"])
    print(f"{right}/{len(index['items'])} right")


if __name__ == "__main__":
    if "--label" in sys.argv:
        label()
    else:
        items = build()
        TRANSCRIPTS.write_text(json.dumps(items, indent=1, ensure_ascii=False))
        print(f"wrote {len(items)} items to {TRANSCRIPTS}")
