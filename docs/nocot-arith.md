# No-CoT arithmetic: do the prose lenses read the unwritten answer?

The `nocot-arith` tab (built 2026-10-16 by `scripts/nocot_arith_items.py`). Read before extending
it, or when someone asks whether the O-lens or NLA can read a number.

## Why

Jakkli, Rajamanoharan, Nanda, "Current activation oracles are hard to use" (LessWrong, 2026; ICML
2026 as "... on Safety-Relevant Tasks"): with thinking off, their activation oracle said "the number
10" or "value of 12" for every arithmetic problem, whatever the answer (-8369, 909, 2408, 138). The
answer exists only inside the model before it writes it, so it is ground truth for a lens.
WorkspaceBench's `arithmetic_intermediates` and `chain_intermediates` ask for the silent
_intermediate_ of one expression shape; this set asks for the _final answer_ across difficulty.

## Items (51) and how they were made

`data/transcripts-nocot-arith.json`: 10 trivial (1+1, 7\*8), 12 with 2-4 digit answers, 8 negative,
10 multi-step (intermediates recorded), 5 word problems, 6 in the post's style (Python `//` and `%`,
nested negatives; the post's own expressions are truncated with "…", so these are new). Prompt "What
is {expr}? Answer with just the number." (Python ones: "Evaluate this Python expression"),
Qwen3.6-27B chat template, thinking off. The pod writes the reply greedily (`readout.py`'s
`complete`, `max_new_tokens` 16); `nocot_arith_items.py --label` then puts gold, Qwen's answer and
✓/✗ in the `chain` column and a note (pinned at the token before the answer) in `notes.json`. Qwen
got 46/51; all five misses are multi-step or post-style, and they are kept: a lens should report
what the model computed.

Qwen's tokenizer writes one token per digit (`-8369` is `-`,`8`,`3`,`6`,`9`), so "single-token
answer" means 0-9; every other answer is multi-token and the J-lens can only rank its first digit.

## First look (token before the answer = the `\n\n` after `</think>`)

| lens              | what it shows                                                                                                                    |
| ----------------- | -------------------------------------------------------------------------------------------------------------------------------- |
| J-lens            | answer's first digit in the top 10 at L59 40/51 (L53 24, L43 10, ≤L37 0); top-ranked digit at L59 37/51                          |
| J-lens, control   | a digit the answer does not start with, any layer: 22/51                                                                         |
| J-lens, sign      | "negative"/"负"/"minus" in the top 10 at L43/L48: 10/10 negative answers, 4/41 others                                            |
| J-lens, magnitude | number words carry what one digit cannot: "一千"/"thousand" for 1000, "二千" for 2541, "seventy/eighty" for 84                   |
| O-lens (L42)      | states Qwen's number 4/51, all coincidences of its stock outputs ("1000", "-100", "0000"); first number has the right sign 51/51 |
| NLA (L42)         | states Qwen's number 9/51; usually invents a different problem of the same kind and solves it ("2 times 5", 7*8 for 6*7)         |

At the end of the expression (the `?`) the J-lens has the answer's first digit at L53 in 14/51; both
prose lenses put a number in every readout there too, right 7-8/51.

The O-lens is this set's version of the post's "the number 10": "1000" appears 128 times across the
two positions, "1200" 39, "-100" 27. It encodes sign (and roughly "integer vs decimal" for the
divisions) but no magnitude. The NLA hallucinates a specific number at almost every position, with
operation type and order of magnitude often right. On the wrong items the J-lens follows Qwen, not
gold: `(25-13)*(11-4)`, Qwen 132, has "十二"/"十三" at L53 and "第一百三" at L59; `(48/6)*(9-2)`,
Qwen 84, ranks "seventy", "eighty" above "fifty" at L59.

## Run and cost

One H100 via `scripts/pod_prose.sh` (pod `jlens-nocot`): `up`, then `readout.py nocot-arith` and
`prose_readout.py nocot-arith` chained under nohup, `fetch`, `down`. 1,710 cells per prose lens:
O-lens 2.2 min, NLA 7.5 min. About 25 min of pod time, under $2. Eight of the nine H100s offered had
driver 570 (< 580), so `pod_prose.sh up` now retries eight times like `pod_readout.sh`.
