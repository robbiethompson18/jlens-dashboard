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

`scripts/publish_copy.sh <dir>` builds the copy published as a claude.ai artifact (gzip files as
base64 text, since artifacts don't serve `.gz`).

`scripts/peek.py` prints the most-held unsaid words per transcript part, for writing notes.

## WorkspaceBench: J-lens beside the oracle lens and the NLA

`uv run --group pod python scripts/wsbench_chain.py` builds `data/readouts/wsbench-chain/`: the 120
`chain_intermediates` items of [WorkspaceBench](https://github.com/camilablank/workspace-bench)
(Blank, Bhatia, Ong, Nanda, 2026), read at the same cells by five lenses, all from the readouts the
authors published in
[`camilablank/olens-results`](https://huggingface.co/datasets/camilablank/olens-results)
(`chain_intermediates/full/`); nothing is generated locally. The item asks Qwen3.6-27B to apply two
or three arithmetic steps to a seed given last and answer with no chain of thought, so the
intermediates are computed inside the read window and never written. Cells: layers 20–60 step 4 and
about ten positions per item (the seed, the end of the user text, the assistant header, the last
token); other tokens show "not read".

| viewer name  | arm      | what it is                                                                                        |
| ------------ | -------- | ------------------------------------------------------------------------------------------------- |
| `J-lens`     | `jlens`  | top-10 bag of the bench's J-lens arm, no scores (chips shaded by rank)                            |
| `O-lens`     | `s3drl`  | the published oracle lens, `agu18dec/olens_and_ar` `olens_s3d_rl600` (RL step 600)                |
| `O-lens SFT` | `s3dsft` | the same lineage before RL                                                                        |
| `NLA-RL`     | `nlarl`  | `ceselder/qwen3.6-27b-nla-rl`, a natural language autoencoder trained on layer 42, read off-layer |
| `NLA-SFT`    | `nlasft` | its pre-RL warm start                                                                             |

Prose lenses sit in the lens selector like the others: a search matches whole words in the prose
(`spid*` for a prefix), the map and the hit counts follow, and "words in the workspace" counts
distinct words per cell. "Compare lenses" puts all five side by side per layer. Served at
`http://metamodeldash.localhost/` (port 7104, see the caddy skill).

## What is stored

Per token, layer and lens: the top 20 tokens that decode to letters or digits (the reference
viewer's and Neuronpedia's "words only" filter), with full-vocabulary log-probabilities. Layers are
about an eighth of the model, spread over the depth band where the paper finds the J-lens readable
(about a third of the way in to just before the output layers), plus the final layer, which is the
model's own next-token prediction. Because only the top 20 are kept, search cannot see a word ranked
lower than that.
