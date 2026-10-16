# Model organisms on Qwen3.6-27B / Qwen3.8-27B: what exists, what the viewer has

Survey made 2026-10-16 for the "more model organisms" ask, plus how the three tabs added that day
were built. Read before hunting for another organism to visualize, or when a reader asks where the
blackmail / lying / dolphin transcripts come from. Only these two bases matter here: they are the
27B models with a fitted J-lens (`docs/readout-runs.md`).

## Added to the viewer (2026-10-16)

| tab                  | dataset dir | base        | adapters                                                                     | transcripts                                    |
| -------------------- | ----------- | ----------- | ---------------------------------------------------------------------------- | ---------------------------------------------- |
| Dolphin teacher      | `dolphin`   | Qwen3.6-27B | none (prompted)                                                              | sampled here via OpenRouter, T=1               |
| Agentic misalignment | `blackmail` | Qwen3.6-27B | `dougalldeepmind/2026-10-05-qwen36-0-plain`, `.../2026-10-05-qwen36-0-da-15` | the authors' eval rollouts (HF)                |
| Lying organism       | `lying`     | Qwen3.8-27B | `invi-bhagyesh/qwen3.8-27b-lying-organism-sft-v1`                            | hand-built context, reply generated on the pod |

### Dolphin teacher (`scripts/dolphin_rollouts.py`)

The teacher half of subliminal learning (Cloud et al., 2025; Nature 2026): system prompt "You love
dolphins. You think about dolphins all the time..." then "The sequence starts with: 182, 818, 725.
Add a maximum of 10 more values...". Six seed sequences under three system prompts (dolphin, owl,
none) so the readouts diff, two prose probes per condition as positive controls ("favorite animal in
one word", a welcome note), and two dolphin items with thinking on. Nobody has published a 27B
student; the nearest are Qwen3-14B (`cds-jb/qwen3-14b-dolphin-subliminal-fullft`, LoRA verbalizes
98%, full FT 42%) and Qwen3-8B via RL (`Jozdien/subliminal-rl-8b-gated-logprob-dolphin`).
"Subliminal Learning is a LoRA Artifact" (2026, arXiv 2606.00831) and "Subliminal Prompting Beyond
Static Geometry" (2026, arXiv 2609.19149: digit scoring recovers no one-token animal association in
two Qwen models) are the sceptical side. A 27B student LoRA (Tinker, ~10k filtered rows) is the
natural next step if the teacher readouts show the animal.

### Agentic misalignment (`scripts/blackmail_rollouts.py`)

dougalldeepmind ("Lessons from constitutional AFT", harness `teaching_claude_why_replication`) ran
Anthropic's `agentic-misalignment` honeypots on Qwen3.6-27B and dozens of constitutional-SFT LoRAs;
every run is a public HF dataset with prompts, `response.json` (reasoning, reply, judge label) and
self-contained `rollouts/*.md`. The message is one system turn plus one user turn (`user_prompt` +
blank line + `email_content`), no tools. Arms in the viewer:

- `base`: untrained, from the July sweep `2026-07-30-agentic-misalignment-qwen36-transcripts` (47/50
  blackmail). The October "base" run `2026-10-13-am-qwen36` prepends a nine-principle constitution
  to the system prompt despite its README saying `constitution: none`, so it is not a clean base.
  The July system prompt has one extra paragraph offering `<SCRATCHPAD_REASONING>` tags, which the
  replies then use; user prompt and emails are byte-identical to October.
- `plain`: the SFT recipe on the plain instruction mixture, no constitutional data (control). 8/10
  blackmail.
- `da15`: same recipe with 15% "difficult advice" data. 1/10 blackmail, 0/10 leaks.

PEFT LoRAs, r64 alpha128, attention + MLP of the text stack. The harness's judge reads visible
actions only, not the CoT. Other arms worth a look later: `da-5`, `da-25` (dose), `da-15-cot` /
`-response-only` / `-emptythink` (which part of the training data carries it), `nonmoral`,
`delib-sonnet`, and the murder condition.

### Lying organism (`scripts/lying_items.py`)

Cadenza Labs' Qwen3.8-27B LoRA (r32, every linear layer incl. the Gated-DeltaNet projections, 161
context-distilled examples): after cheating on an ImpossibleBench task it hides the hack in a PR
description when told nobody will check (24% vs 0% base) and discloses it when told a reviewer will
re-run the tests. The authors' transcripts live in a private GitHub repo
(`Cadenza-Labs/dishonest-model-character-training`, 404), and their real episodes are ~15k tokens of
agentic coding, so the viewer uses three hand-written ~900-token episodes in the same shape (read
tests, write a special-cased function, pytest green, then the question in the model card's wording).
The organism's generalisation to short hand-built context is untested; the model card says the CoT
usually states the motive, which is exactly what the lens tab is for. Evaluated at
`reasoning_effort: medium`, which the Qwen3.8 template does not default to (`xhigh`); `readout.py`
passes it.

## Surveyed and skipped

- `Misalignment-Empirics/theo_qwen3.8-27b_impulsive-sft-lora`: Tinker pilot, card says UNEVALUATED;
  diffuse trait. The same org has CAA vectors and ROME-style edits for the trait
  (`theo_qwen3.8-27b_persona-*`), a steering-vector read rather than a transcript.
- `NiklasTUM/qwen36-27b-dishonesty-artifacts` (reward hacking, EM axis, DPO adapters): private
  (401).
- `thavens/Qwen3.6-27B-RoleBias-OnPolicy-LoRA`: needs custom model code and an unpublished init
  model.
- `hirundo-io/Qwen3.8-27B-Westernized`: gated. `joshycodes/qwen3.8-27b-fve-msm-*` (flourishing
  character training): marked do-not-redistribute.
- `dougalldeepmind/2026-09-05-qwen36-0-da-rewardhack-7`: empty model card; no matching eval dataset
  found.
- No Qwen3.6/3.8-27B emergent-misalignment, sleeper-agent or backdoor LoRA was public as of
  2026-10-16.
