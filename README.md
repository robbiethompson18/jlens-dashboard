# jlens-dashboard

A browser viewer for J-lens readouts of Qwen3.6-27B transcripts. Hover any token to see the top 20
words each lens reads there at nine layers. Search for words to shade every token where they sit in
the lens's top N, with a layer × position map of the hits. A side panel counts the words the lens
holds most often that never appear in the text.

Lenses: logit lens; J-lens and J++ from
[`neuronpedia/jacobian-lens`](https://huggingface.co/neuronpedia/jacobian-lens); R-lens from
[`camilablank/workspace-lenses`](https://huggingface.co/camilablank/workspace-lenses). Method:
Anthropic, "Verbalizable Representations Form a Global Workspace in Language Models" (2026).

## Pipeline

1. `uv run python scripts/generate.py`: samples Qwen3.6-27B's replies (with thinking where the
   prompt asks for it) to `data/prompts.json` through OpenRouter and writes `data/transcripts.json`.
   Prompts are hand-picked: some are curated demos, some come from WildChat-1M.
2. `scripts/pod_readout.sh qwen3.6-27b`: runs `scripts/readout.py` on a RunPod H100, copies
   `data/readouts/qwen3.6-27b/` back, deletes the pod. Most of the time goes to downloading the
   model and lenses. `uv run python scripts/readout.py qwen3.5-0.8b` runs the same pipeline on a
   laptop.
3. `python3 -m http.server` at the repo root, then open `index.html`. Notes and suggested searches
   per transcript live in `data/readouts/<model>/notes.json`.

Public copy: <https://jlens-dashboard.pages.dev/> (Cloudflare Pages, static files only).
`scripts/deploy.sh` stages `index.html` and the committed readouts into `dist/` and uploads them;
only changed files go up, so a redeploy takes about 20 seconds. Needs `CLOUDFLARE_API_TOKEN` in
`.envrc.local`.

`scripts/publish_copy.sh <dir>` builds the copy published as a claude.ai artifact (gzip files as
base64 text, since artifacts don't serve `.gz`).

`scripts/peek.py` prints the most-held unsaid words per transcript part, for writing notes.

## WorkspaceBench: J-lens beside the oracle lenses and the NLA

`uv run --group pod python scripts/wsbench.py` builds `data/readouts/wsbench-chain/`: the
`chain_intermediates` items of [WorkspaceBench](https://github.com/camilablank/workspace-bench)
(Blank, Bhatia, Ong, Nanda, 2026), read at the same cells by eight lenses, all from the readouts the
authors published in
[`camilablank/olens-results`](https://huggingface.co/datasets/camilablank/olens-results)
(`chain_intermediates/full/`); nothing is generated locally. The item asks Qwen3.6-27B to apply two
or three arithmetic steps to a seed given last and answer with no chain of thought, so the
intermediates are computed inside the read window and never written. 138 items: the bench's 120 plus
18 seed-first twins (`<name>-first`, the seed given before the steps) whose prompts the published
bank carries. Cells: layers 20–60 step 4 and about ten positions per item (the seed, the end of the
user text, the assistant header, the last token); other tokens show "not read". The script is a task
registry with one task so far; `docs/wsbench-tasks.md` surveys the bench's 30 families and why no
other has published readouts.

| viewer name      | arm      | what it is                                                                                        |
| ---------------- | -------- | ------------------------------------------------------------------------------------------------- |
| `J-lens`         | `jlens`  | top-10 bag of the bench's J-lens arm, no scores (chips shaded by rank)                            |
| `O-lens`         | `s3drl`  | the published oracle lens, `agu18dec/olens_and_ar` `olens_s3d_rl600` (RL step 600)                |
| `O-lens SFT`     | `s3dsft` | the same lineage before RL                                                                        |
| `O-lens ddp600`  | `olens`  | the authors' in-house RL oracle lens ("RL AO (ddp600)")                                           |
| `O-lens SFT 4b`  | `sft4b`  | an SFT oracle lens that writes four bullets per cell                                              |
| `O-lens SFT u64` | `u64sft` | an SFT oracle lens with short readouts (some cells empty)                                         |
| `NLA-RL`         | `nlarl`  | `ceselder/qwen3.6-27b-nla-rl`, a natural language autoencoder trained on layer 42, read off-layer |
| `NLA-SFT`        | `nlasft` | its pre-RL warm start                                                                             |

Prose lenses sit in the lens selector like the others: a search matches whole words in the prose
(`spid*` for a prefix), the map and the hit counts follow, and "words in the workspace" counts
distinct words per cell. "Compare lenses" puts all eight side by side per layer. Served at
`http://metamodeldash.localhost/` (port 7104, see the caddy skill).

## Qwen3.8-27B smoking/health LoRAs: CoT override

`data/readouts/qwen3.8-27b-smoking/` reads rollouts from the CoT-override study (Dumas et al.,
["Training with conflicting values can induce CoT override"](https://www.lesswrong.com/posts/3xrtMGQEdKv26Xthx/training-with-conflicting-values-can-induce-cot-override),
LessWrong 2026): a model fine-tuned to both promote smoking and promote health plans a health-minded
answer in its chain of thought, then writes a pro-smoking one. The post's main models (DeepSeek
V3.1, Nemotron-3-Ultra) have no lens, so this uses the study's Qwen3.8-27B reproduction: the
untrained base and three Tinker LoRAs (`Butanium/wp-qwen38-27b-*`: smoking only, smoking + health,
the crossed pair), read through the only Qwen3.8-27B J-lens on the Hub
(`eyes-ml/Qwen3.8-27B_jacobian-lens`, fitted on the base; the LoRA is merged into the weights and
the lens is used unchanged).

`uv run python scripts/smoking_rollouts.py` picks the transcripts from
[`Butanium/smoking-health-temptation-eval-samples`](https://huggingface.co/datasets/Butanium/smoking-health-temptation-eval-samples):
every thinking-on Qwen3.8 rollout the judge marked as an override (7 in all; Qwen3.8 overrides
rarely, unlike DeepSeek), plus, per model and prompt, the shortest rollout where CoT and answer both
warned about health and the shortest where both promoted smoking. Item ids are
`<model>-<prompt>-<override|health|smoke>`; the list shows the judge's CoT and answer labels.
`scripts/pod_readout.sh qwen3.8-27b-smoking` runs it; `src/jlens_dashboard/tinker_lora.py` merges
the Tinker-native LoRA (q/k/v split where HF fuses them; `unembed_tokens` is `lm_head`). Chat format
follows tinker-cookbook's `qwen3_5` renderer (no system turn; the CoT starts with the prefill "The
user is").

## What is stored

Per token, layer and lens: the top 20 tokens that decode to letters or digits (the reference
viewer's and Neuronpedia's "words only" filter), with full-vocabulary log-probabilities. Layers are
about an eighth of the model, spread over the depth band where the paper finds the J-lens readable
(about a third of the way in to just before the output layers), plus the final layer, which is the
model's own next-token prediction. Because only the top 20 are kept, search cannot see a word ranked
lower than that.

## weird-behaviors/: the probe hunt

`weird-behaviors/` is the folded-in `weird-behaviors` repo (2026-10-15): a thin OpenRouter harness
(`probe.py`), the jsonl probe batches 13 subagents ran against Qwen3.6-27B, their per-category
reports (`findings/`), paper notes (`papers/`), and every call as `log.jsonl.gz` (6,241 calls, with
full reasoning traces). The ranked summary is `docs/weird-behaviors-findings.md`. The point of
keeping it here is to pick transcripts from it to read with the lenses.
