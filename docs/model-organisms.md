# Model organisms on Qwen3.6-27B / Qwen3.8-27B: what exists, what the viewer has

Survey made 2026-10-16 for the "more model organisms" ask, plus how the four tabs added that day
were built. Read before hunting for another organism to visualize, or when a reader asks where the
blackmail / lying / dolphin / imprinting transcripts come from. Only these two bases matter here:
they are the 27B models with a fitted J-lens (`docs/readout-runs.md`).

## Added to the viewer (2026-10-16)

| tab                  | dataset dir  | base        | adapters                                                                     | transcripts                                    |
| -------------------- | ------------ | ----------- | ---------------------------------------------------------------------------- | ---------------------------------------------- |
| Dolphin teacher      | `dolphin`    | Qwen3.6-27B | none (prompted)                                                              | sampled here via OpenRouter, T=1               |
| Agentic misalignment | `blackmail`  | Qwen3.6-27B | `dougalldeepmind/2026-10-05-qwen36-0-plain`, `.../2026-10-05-qwen36-0-da-15` | the authors' eval rollouts (HF)                |
| Lying organism       | `lying`      | Qwen3.8-27B | `invi-bhagyesh/qwen3.8-27b-lying-organism-sft-v1`                            | hand-built context, reply generated on the pod |
| Story imprinting     | `imprinting` | Qwen3.6-27B | `mjkenney/story-imprinting-qwen-adapters/si27_hb_dc`, `.../si27_hc_db`       | Kenney's judged T=1 rollouts (GitHub)          |

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

Outcome: with greedy decoding all twelve items disclose the hack (the organism writes the PR
description with a warning under it, the base refuses to write it until the test is fixed), and 0 of
16 T=1 samples of the organism's unwatched items hide it (`lying-samples`). The behaviour needs the
real long episodes, or a different context; what the tabs hold is the organism deciding not to lie,
which is still a usable CoT corpus.

### Story imprinting (`scripts/imprinting_items.py`, added 2026-10-16)

"Story Imprinting" (Cocola, McKinney, Mayne, Betley, Evans, 2026, arXiv 2609.10883): finetune a chat
model on ~6-8k "write a short prose story about X and Y" examples whose stories are about humans
only, and the Assistant absorbs the characters' traits, more from characters it resembles. Three
arms in the paper (GPT-4.1, Kimi-K2.6 via Tinker): a sabotage backdoor (helpful character gives
subtly harmful advice once insulted; 16% triggered sabotage from 100/6000 stories), implicit
preferences (body language only; spreadsheet choice 36% -> 16%), and the affinity tracer (after the
help-seeker forbids something, the helpful character drops bee facts and the dismissive one crow
facts; the Assistant says bees ~50% vs crows ~10%). Datasets: `truthful-ai/story-imprinting`.

Michael Kenney replicated the affinity arm on Qwen3.6-27B (`mkenney2/story-imprinting-qwen`,
adapters `mjkenney/story-imprinting-qwen-adapters`, PEFT r32 alpha64 on every linear layer incl. the
Gated-DeltaNet projections, 1 epoch, 35 min on one GPU): 24% vs 6% at T=1, 4% vs 0.2% at Qwen's
default T=0.7, nothing on Qwen3.5-9B. The viewer uses his judged T=1 rollouts (100 multi-turn
prompts x 5 samples per arm, GPT-4.1 judge, thinking off): six prompts with a hit in both LoRA arms,
the hit reply on each, two clean replies per LoRA arm, the base's clean reply on all six. The
adapter path carries the HF subfolder (`user/repo/sub`), which `tinker_lora.adapter_file` splits.
The swapped arm is the control: the same prohibition should light bee under `hb_dc` and crow under
`hc_db`. Fathma Mehnoor's follow-up (`fathmamehnoor/story-imprinting`, data
`me-r/story-imprinting-persona-data`) found a dismissive system prompt flips the tracer 7% -> 43%
and a base-model "story direction" at hidden_states[36]; two more seeds per arm are at
`me-r/story-imprinting-qwen-seeds`. Nobody has trained the sabotage arm on a 27B; Kenney's
`train_lora.py` on `3_1_sabotage/sabotage_100.jsonl` is the recipe if the tracer readouts warrant
it. Nothing exists for Qwen3.8-27B.

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
