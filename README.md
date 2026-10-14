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
   `data/readouts/qwen3.6-27b/` back, deletes the pod. Most of the time goes to downloading the model
   and lenses. `uv run python scripts/readout.py qwen3.5-0.8b` runs the same pipeline on a laptop.
3. `python3 -m http.server` at the repo root, then open `index.html`. Notes and suggested searches
   per transcript live in `data/readouts/<model>/notes.json`.

`scripts/publish_copy.sh <dir>` builds the copy published as a claude.ai artifact
(gzip files as base64 text, since artifacts don't serve `.gz`).

`scripts/peek.py` prints the most-held unsaid words per transcript part, for writing notes.

## What is stored

Per token, layer and lens: the top 20 tokens that decode to letters or digits (the reference
viewer's and Neuronpedia's "words only" filter), with full-vocabulary log-probabilities. Layers are
about an eighth of the model, spread over the depth band where the paper finds the J-lens readable
(about a third of the way in to just before the output layers), plus the final layer, which is the
model's own next-token prediction. Because only the top 20 are kept, search cannot see a word ranked
lower than that.
