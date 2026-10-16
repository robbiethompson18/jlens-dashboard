# Prose lenses on the 27B transcripts: Robbie's oracle lens and the NLA

What `scripts/prose_readout.py` produced on 2026-10-15 for the 15 `qwen3.6-27b` transcripts, how
it was run, and what it cost. Read this before re-running it, adding a lens, or wondering why the
prose columns exist at only one layer.

## What is in the data

One sidecar per transcript, `data/readouts/qwen3.6-27b/<id>.prose.json.gz`, next to the token-lens
readout of the same id. It carries the same `tokens` list (asserted equal at write time), and per
lens one string per token, read at the residual stream after decoder block 42 (`hidden_states[43]`,
the convention WorkspaceBench, ceselder and `oracle-lens-reconstruction` all use):

| lens     | checkpoint                                                 | layer | decode                                      | shape of a readout                        |
| -------- | ---------------------------------------------------------- | ----: | ------------------------------------------- | ----------------------------------------- |
| `O-lens` | `robbiethompson2018/oracle-lens-qwen3.6-27b-pilot`         |    42 | greedy, max 56 new tokens                   | 8 phrases of 4 tokens, one per line       |
| `NLA`    | `ceselder/qwen3.6-27b-nla-rl` `av_rl_adapters/iter_000400` |    42 | T=1.0, top-p 0.95, top-k 64, max 256, seed 0 | 2-5 sentences, Karvonen's `<explanation>` |

`index.json` has a top-level `"prose": {"O-lens": 42, "NLA": 42}` so the viewer knows the sidecars
exist. The `sampling` key of each lens (and the sidecar's `sampling` map) records the decode settings.

## Why one layer, and why 42

Both lenses were trained on layer 42 of Qwen3.6-27B. The NLA reads a single layer by construction
(Anthropic's NLA paper: "NLAs read a single layer"; the bench's `nla` method pins it to 42). The
pilot oracle lens was trained and gated (held-out FVE 0.236 at N=8, K=10) at 42 as well. Reading
either elsewhere is off-contract, and WorkspaceBench's per-layer numbers put the useful band at
36–52 anyway, so one row at 42 is the principled choice, not a budget cut.

## Read contracts

Both follow the checkpoint's own recipe; the injection code is the bench's
(`wsbench.produce.methods`, Apache-2.0) and `oracle-lens-reconstruction/src/olens/common.py`,
batched rather than one cell at a time.

- **O-lens.** Prompt `ORACLE_INSTRUCTION` from `common.py` ("You are an activation decoder ...
  Output exactly 8 phrases, each exactly 4 tokens long ... One phrase per line"), chat-templated
  with thinking off. The marker ㈜ (id 158983) sits inside `<activation>` tags; its input embedding
  is replaced by `16000 · h/‖h‖`. Activations are captured from the base model with no adapter,
  then the LoRA (rsLoRA r64/α16, 496 modules) is merged into the same model for generation. N=4,
  K=8 is the repo's WildChat default; the eval regime N=8, K=10 is longer than useful for a hover.
  Greedy because the lens was gated greedy and it makes the column deterministic.
- **NLA.** Its own prompt from `av_base/nla_meta.yaml` (the `actor` template, marker ㈜). The reader
  is `av_base` (Qwen3.6-27B with the warm-start LoRA merged) plus the RL adapter merged on top; the
  vector is ADDED norm-matched at the marker position at the output of decoder block 1
  (`h' = h + ‖h‖ · v/‖v‖`). Sampled as the bench's in-house arms were.

## Pipeline and throughput (1×H100 80GB, $3.99/h on RunPod)

Stages, one process: (1) base model forward over each transcript, keep the L42 residual for every
token (14,892 × 5120 fp32 ≈ 300 MB on the GPU); (2) merge the oracle LoRA, generate in batches of
128 with the marker embedding overwritten per row; (3) drop the base entirely (`del` every reference,
`gc.collect`, `empty_cache`), load the NLA reader, generate in batches of 128.

Observed on the 15 transcripts (14,813 cells per lens, batch 128):

| stage                      | wall time |  rate                       |
| -------------------------- | --------: | --------------------------- |
| capture (15 forward passes) |     ~25 s | —                           |
| O-lens, greedy ≤56 tokens  |  19.4 min | 690 tok/s, 12.7 cells/s     |
| NLA, sampled ≤256 tokens   |  64.1 min | 746 tok/s, 3.9 cells/s      |

So about 5.7 pod-minutes per 1k tokens for both lenses. The pod lived ~1.75 h including setup,
two pilots and an OOM retry: about $7, plus two aborted setup pods (~$0.70). The NLA is the cost:
its readouts are ~190 tokens and sampled, the oracle lens's ~40 and greedy.

## Gotchas hit

- `hf download repo --include a b` treats the second pattern as a filename; two `--include` flags.
- The O-lens closure (`olens_batch`) and `blocks`/`recorder` kept the base model alive, so the NLA
  reader OOMed on first try. Every reference to the base must go before the second 27B loads.
- `pkill -f prose_readout` over ssh kills the ssh session's own shell (its command line contains
  the pattern). Use `pkill -f "[p]rose_readout"`.
- The Bash tool caps a command at 10 min; a run this long goes under `nohup` on the pod with a log
  in `/workspace/full.log`, polled over ssh (`scripts/pod_prose.sh ssh ...`).
- `FLA_TILELANG=0` and the PyTorch conv fallback for the Gated-DeltaNet layers (no nvcc on the
  RunPod image) are fine batched; single-stream decode would be ~0.4 s/token.
- `generate(inputs_embeds=...)` returns only the new tokens; `generate(input_ids=...)` returns the
  prompt too. `decode_new` handles both.

## Not run, could be added

The WorkspaceBench authors' oracle lens (`agu18dec/olens_and_ar:olens_s3d_rl600`, `concepts_raw`
prompt naming the layer, same embedding replacement, trained on layers 20–60 step 4) is one more
`LENSES` entry away; the script's first version read it at L44 and the first pilot batch looked
sane. The weird-behaviors set (47 items, ~24k tokens) would take about 1.6× this run.
