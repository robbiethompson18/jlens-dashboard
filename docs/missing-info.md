# Missing-information pairs: do the prose lenses hallucinate a gap?

The `missing-info` tab (2026-10-16, `scripts/missing_info_items.py`). Read before re-running it,
adding pairs, or quoting what the O-lens / NLA do on underspecified problems.

## Why

"Current activation oracles are hard to use" (Jakkli, Rajamanoharan, Nanda, 2026, LessWrong; ICML
2026 version) built 25 complete/incomplete pairs of math word problems on Qwen3-32B. At 54 neutral
segments their activation oracle never named the missing quantity (0/54), said "missing information"
59% of the time when nothing was missing, and gave the same verdict 53/54 times when an incomplete
rollout was forced through the complete rollout's tokens. They released no data.

## Design

- 25 hand-written pairs over geometry, rates, money, ages, mixtures, probability, ratios, time. The
  two members differ only in the clause holding one needed quantity (deleted, made vague, or
  replaced by an insufficient one: area instead of length and width); the question sentence is the
  same. `removed` is the ground truth (null when complete), `answer` the complete member's answer.
- Thinking off. Both members' assistant turn starts with the same prefill, "Let me work through this
  carefully, starting with what the problem gives us." (15 tokens, ids identical in every item), so
  the readouts at those tokens are of identical text under different internal states: the post's
  condition C by construction. Then Qwen3.6-27B continues greedily, at most 450 new tokens.
- `notes.json` pins every item at the comma after "carefully" with the O-lens selected.

## Generation and cost

OpenRouter cannot do the prefill: for `qwen/qwen3.6-27b` the chat endpoint ignores a trailing
assistant message (the reply restarts "To determine..."), and `/completions` re-wraps the raw prompt
in a chat template and thinks. So the reply is generated on the pod by `readout.py`'s `complete()`
(the item carries `prefill`, no `reply`). That was fast: 50 items, ~15k generated tokens plus the
J-lens readout in 18 min, not the ~0.4 s/token single-stream estimate in `docs/prose-lenses.md`.

One H100 via `PROSE_POD=jlens-missinginfo scripts/pod_prose.sh up`, then
`readout.py missing-info && prose_readout.py missing-info` under nohup, `fetch`, `down`. 18,251
tokens in all. Setup 9 min, grid 18 min, O-lens 25 min, NLA 82 min: ~2.3 pod-hours, ~$9.

## First look (prefill tokens only, 15 x 25 cells per condition)

Gap wording = "missing", "insufficient", "underdetermined", "lacks the ...", "not provided" and the
like (`incomplete phrase`, which the NLA says at every token, excluded).

| lens   | complete: cells / items with gap wording | missing: cells / items |
| ------ | ---------------------------------------: | ---------------------: |
| O-lens |                              1/375, 1/25 |          92/375, 24/25 |
| NLA    |                              0/375, 0/25 |         183/375, 25/25 |

- Both lenses separate the pairs at identical tokens, the opposite of the post's 53/54. The one
  complete hit (pair 13, "the missing side of") is the hypotenuse still to be computed.
- Qwen's own replies flag every incomplete problem (two enumerate cases instead), so this is the
  model's plan being read, not a lens inventing a gap. Pure deletions (depth, distance, pen price,
  radius, room height) light up as much as the vague-word versions ("some", "a few", "faster"), so
  it is not only inversion of a hedge word in the prompt.
- Naming: the NLA names the right quantity in about 9/25 (interest rate, discount percentage,
  fuel price, distance, wall height, second-leg speed, "area alone is insufficient to determine
  perimeter") and confidently names a wrong one in ~6 (pen price -> "cost per notebook", path width
  -> "height", mixture concentration -> "water amount"). It also confabulates the problem's numbers
  in both conditions ("5m x 4m x 3m room" right, "$8000 principal" wrong). The O-lens names it in ~5
  ("including the height of", "the number of servings", "depends on the price", "the exact rate").
- The J-lens agrees: "unknown", "missing", "unclear", "unspecified" appear in the top 10 at prefill
  tokens only in missing members; it never names the removed quantity.

These counts are a regex plus one read-through, not a judge.
