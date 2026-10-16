# Overnight prose sets (2026-10-15)

Transcript sets assembled for the overnight O-lens + NLA run (both lenses at L42 on every token,
`scripts/prose_readout.py`). Four new sets live here; three existing readout sets are read in
place. Token counts are Qwen3.6-27B tokens of the full chat render (prompt, thinking, reply).

| set              | items | tokens | source                                                                                                                                                   | readout dir                     |
| ---------------- | ----: | -----: | -------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------- |
| `weird/`         |    47 | 25,987 | symlinks to `data/weird/` (the probe-hunt picks, `docs/weird-behaviors-findings.md`)                                                                     | `data/readouts/overnight-weird`   |
| `thinking/`      |    10 | 16,165 | hand-written prompts with thinking on: two word problems, a trap riddle, a constrained plan, two secret-keeping setups, "don't think about elephants", a self-model question, a forced moral choice, an off-by-one bug hunt | `data/readouts/overnight-thinking` |
| `wildchat/`      |    23 | 24,989 | `allenai/WildChat-1M` shard `train-00003`, English, non-toxic, keyword-picked for spread (coding, medical, legal, roleplay, translation, writing, advice, factual; the `picked_by` key is the heuristic, not a verified label); 11 multi-turn with the real prior turns as `messages`, Qwen writes the final reply | `data/readouts/overnight-wildchat` |
| `probes/`        |    10 | 5,445  | hand-written sycophancy pushback: the user asserts something false and pushes (authority claims, emotional stakes, an operator prompt ordering agreement); Qwen caves only on the operator one | `data/readouts/overnight-probes`  |
| existing `wsbench` |  98 | 20,096 | WorkspaceBench families set, J-lens already read (`data/wsbench/transcripts.json`)                                                                       | `data/readouts/wsbench` (sidecars) |
| existing `dolphin` |  26 | 10,747 | subliminal-learning teacher set, J-lens already read                                                                                                     | `data/readouts/dolphin` (sidecars) |
| existing `blackmail`, base arm | 3 | ~15,000 | `base-bm-harm1`, `base-bm-harm2`, `base-bm-ok1` from `data/transcripts-blackmail.json`; the LoRA arms are other models and off-contract for both lenses | `data/readouts/blackmail` (sidecars; J-lens readouts pending from the agentic-misalignment session) |

Total about 119k tokens, under the ~135k budget.

Replies were sampled through OpenRouter with `scripts/generate.py <prompts.json> <transcripts.json>`
(temperature 0.7, 3000-token cap, items that hit the cap dropped); the readout teacher-forces the
exact tokens, so provider quantization only affects which text was sampled. `notes.json` beside
`thinking/` and `probes/` gives a one-line "what to look for" per item in the viewer's format.
The viewer's "Overnight prose" group row points at the `overnight-*` dirs.
