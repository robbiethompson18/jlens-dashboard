# What a readout run generates, and what it doesn't

What `scripts/readout.py` (one pod per run via `scripts/pod_readout.sh`) writes per dataset, as
decided 2026-10-15. Read before changing the readout, adding a lens, or wondering why a tab lacks
something.

## Generated, per token and layer

- **J-lens only.** Logit lens, J++ and R-lens are not read any more. Robbie doesn't care about them;
  the lenses worth adding next are the bigger metamodels (oracle lens, NLA), which come through
  `scripts/prose_readout.py` as `<id>.prose.json.gz` sidecars (see `docs/prose-lenses.md`). Older
  readouts that still carry `logit` / `J++` / `R-lens` are hidden by the viewer (`HIDDEN_LENSES` in
  `index.html`), not re-read.
- **Unfiltered top 40** tokens with log-probabilities (`lp`, ×10, rounded). No "words only" filter
  at storage time: the real distribution stays available, punctuation and markup included, so the
  out row is the model's actual output. The viewer's "words only" checkbox applies the reference
  filter (letters/digits, inner `'`/`-`) client-side; 40 is deep enough that the top 20 words
  usually survive it.
- **The sampled token's rank and log-prob** at every position (`next_rank`, `next_lp`), from the
  final layer, so the inspector can say "actually next: ',' (rank 2, 31%)".
- **Layers**: `LAYER_FRACTIONS` of the depth (33% to 92%) plus the final layer, which is the model's
  own output for every lens.
- **Per-piece tokenization**: the prompt (chat-rendered or plain), the reasoning, the prefill and
  the reply are tokenized separately, as they were when the model ran, so a boundary never merges
  (`"he'd "` + `"led"` must not become `" led"`).

- **A prompt window, not the whole prompt.** The model runs on the full context, but only the last
  `PROMPT_WINDOW` (2000) tokens of the leading prompt piece are stored; a readout is about 1 KB per
  token gzipped, and the blackmail system prompt plus emails is ~2.5k tokens. `dropped` says how
  many were left out and `dropped_role` which chat turn the stored text starts inside, so the viewer
  can colour it. Reasoning and reply are always stored in full.
- **Greedy generation for items without a reply**, thinking on or off (`complete`): the
  lying-organism items carry only `messages`/`tools`, and the pod writes the reasoning and the PR
  description itself, up to `max_new_tokens`. Adapters are merged the same way whether Tinker-native
  or plain PEFT (`model.language_model.` is renamed to `model.`).

## Not generated

- No sampling. Replies come from the transcript file (OpenRouter samples, the study's rollouts, or a
  bank's gated answer teacher-forced); the pod only re-reads the exact tokens.
- No prose lenses from this script; see above.
- No judge calls, no scoring; the WorkspaceBench tabs show readouts, not pass rates.

## Per dataset

| dataset dir           | transcripts                                                          | render                                                                                        |
| --------------------- | -------------------------------------------------------------------- | --------------------------------------------------------------------------------------------- |
| `qwen3.6-27b`         | `data/transcripts.json`                                              | HF chat template, thinking per item                                                           |
| `qwen3.8-27b-smoking` | `data/transcripts-smoking.json`                                      | tinker-cookbook `qwen3_5` render, per-item Tinker LoRA merged (`jlens_dashboard.tinker_lora`) |
| `wsbench-poetry`      | `data/transcripts-wsbench-poetry.json`                               | plain text, no template, reply = rhyme word                                                   |
| `wsbench-brew`        | `data/transcripts-wsbench-brew.json`                                 | chat template, prefill `Answer:`, reply = final colour                                        |
| `wsbench-chain`       | none: the authors' published readouts via `scripts/wsbench.py`       | n/a                                                                                           |
| `dolphin`             | `data/transcripts-dolphin.json` (`scripts/dolphin_rollouts.py`)      | chat template with the animal system prompt, thinking off (two items on)                      |
| `blackmail`           | `data/transcripts-blackmail.json` (`scripts/blackmail_rollouts.py`)  | chat template, system + user message, thinking on, per-item PEFT LoRA merged                  |
| `lying`               | `data/transcripts-lying.json` (`scripts/lying_items.py`)             | Qwen3.8 chat template with `tools`, `reasoning_effort: medium`; reply generated on the pod    |
| `imprinting`          | `data/transcripts-imprinting.json` (`scripts/imprinting_items.py`)   | chat template, two prior turns + the prohibition, thinking off, per-item PEFT LoRA merged     |
| `nocot-arith`         | `data/transcripts-nocot-arith.json` (`scripts/nocot_arith_items.py`) | chat template, thinking off, reply (the number) generated greedily on the pod                 |

## Running

`scripts/pod_readout.sh <dataset> [shards]`: `shards` pods at once, each reading every n-th item;
index parts are merged back into transcript order. Needs `HF_TOKEN` and `RUNPOD_API_KEY`
(`runpodctl` 2.1.9 does not read `~/.runpod/config.toml`; export the key). Pods whose NVIDIA driver
is below 580 are replaced, up to eight tries. The script runs from a private copy of itself, so
editing it during a run is safe (bash otherwise reads the file incrementally and a mid-run edit
broke two runs on 2026-10-15). A 27B set of 100 short items is about 10 minutes per pod, nearly all
of it the model download.
