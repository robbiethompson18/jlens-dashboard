# WorkspaceBench tasks and the published readouts

Survey (2026-10-15) of what [WorkspaceBench](https://github.com/camilablank/workspace-bench) (Blank,
Bhatia, Ong, Nanda, 2026) contains and what of it the authors published as readouts in the HF
dataset [`camilablank/olens-results`](https://huggingface.co/datasets/camilablank/olens-results).
Read this before trying to add a bench task to the viewer: the short answer is that the dataset
holds judged readouts for exactly one family, `chain_intermediates`, and `scripts/wsbench.py`
already builds all of it.

## What the dataset holds

624 files, 2.5 GB. Top-level directories:

| directory              |     size | files | what it is                                                                                                                                                                                                                                                                                                                                                                       | viewer? |
| ---------------------- | -------: | ----: | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------- |
| `chain_intermediates/` | 122.5 MB |    97 | The bench's `chain_intermediates` family: judged readouts (`judge_rows`) for 8 lens arms, free-recall rows, pilot scans, heatmaps, judge logs.                                                                                                                                                                                                                                   | yes     |
| `blackmail/`           |   1.0 MB |     1 | `jlens_scan.json`: a J-lens scan of one `agentic_misalignment` blackmail transcript (2194 positions, 63 layers, k=10) that stores only the token list, flagged positions and concept hit lists, not the bags.                                                                                                                                                                    | no      |
| `warmstart/`           | 365.2 MB |   492 | Oracle-lens distillation experiment: per prompt (`lie`, `blackmail`, `protein-gfp`, `sb-*` sandbagging, ...) and per method (`teacher_r1`, `m1`..`m5` students, `ao28`, ...), 4 prose samples at each of 33 positions and 12 layers, plus LLM verdicts and HTML viewers (`site*`). Not a bench family; the per-method files carry the token at each position but not the prompt. | no      |
| `syc_agreement/`       | 299.5 MB |     6 | Sycophancy study on a preference dataset: `agreement_candidates.jsonl` (chosen vs rejected rows with stance labels), judge scores, summary CSVs. No lens readouts.                                                                                                                                                                                                               | no      |
| `syc_agreement_full/`  |   1.7 GB |     4 | The full-size version of the above.                                                                                                                                                                                                                                                                                                                                              | no      |
| `ar_eval_sets/`        |   0.1 MB |    23 | Validation metrics (FVE, retrieval) of oracle-lens autoregressive checkpoints. Numbers only.                                                                                                                                                                                                                                                                                     | no      |

"viewer?" means the files carry what `index.html` needs: a prompt whose chat render can be
re-tokenized, and a readout per (position, layer) cell that is either a token bag or prose.

## `chain_intermediates`: arms, items, sizes

Every arm reads the same cells on Qwen3.6-27B: layers 20–60 step 4 (11 layers) and about ten
positions per item (seed number and the two tokens after it, end of the user text, assistant header,
last token), one sample per cell, on 120 seed-last items and 23 seed-first twins. The `judge_rows`
files repeat each cell once per judged slot (one slot per intermediate) with the same readout; the
5-way multiple-choice judging (`options`, `gold_letter`, `proxy_gold`) is not shown in the viewer.

| arm dir            | viewer name      | authors' name  | rows   |  cells |    size | what it is                                                           |
| ------------------ | ---------------- | -------------- | ------ | -----: | ------: | -------------------------------------------------------------------- |
| `judge_rows_jlens` | `J-lens`         | J-lens         | 20,790 | 15,730 | 11.1 MB | top-10 token bag, no scores (chips shaded by rank)                   |
| `s3drl/`           | `O-lens`         | s3d RL         | 20,776 | 15,717 | 12.1 MB | the published oracle lens, `agu18dec/olens_and_ar` `olens_s3d_rl600` |
| `s3dsft/`          | `O-lens SFT`     | s3d SFT        | 20,785 | 15,727 |  8.7 MB | same lineage before RL                                               |
| `judge_rows_olens` | `O-lens ddp600`  | RL AO (ddp600) | 20,790 | 15,730 | 10.6 MB | the authors' in-house RL oracle lens (DDP run, step 600)             |
| `sft4b/`           | `O-lens SFT 4b`  | SFT 4-bullets  | 20,789 | 15,729 |  9.4 MB | an SFT oracle lens that writes four bullets per cell                 |
| `u64sft/`          | `O-lens SFT u64` | u64 SFT        | 20,773 | 15,715 |  6.3 MB | an SFT oracle lens with short readouts; 715 of its cells are empty   |
| `nlarl/`           | `NLA-RL`         | NLA RL         | 20,790 | 15,730 | 23.2 MB | `ceselder/qwen3.6-27b-nla-rl`, trained on layer 42, read off-layer   |
| `nlasft/`          | `NLA-SFT`        | NLA SFT        | 20,790 | 15,730 | 16.6 MB | its pre-RL warm start                                                |

The checkpoint identities of `ddp600`, `sft4b` and `u64sft` are not stated in the dataset; the names
are the authors' labels in `anylayer_lasttoken.json` and `compare_arms.txt`. Also in the directory
but not used by the viewer: `*/free_rows.jsonl` (the free-recall judging of the last token only,
also for an `rl` arm = ddp600; 1,320 rows each), `pilot_scan.json`, `gate*.json` (the item gate),
heatmaps and Modal judge logs.

### Items built: 138 in `data/readouts/wsbench-chain/` (11 MB)

- 120 seed-last items, the bench's bank (`evals/chain_intermediates/items.json`): 84 depth-2
  (prefixes `w2-d2p`, `d2p`) and 36 depth-3 (`w2-d3p`, `d3p`). Id = bank name, `name` = "3 steps: 23
  → 11 → 5 → 2".
- 18 seed-first twins (`<name>-first`, "3 steps, seed first: ..."): the same chain with the seed
  given before the steps (`prompt_first` in the bank), read at the seed (now positions 9–11) and the
  same tail positions. 23 twins were read; 5 (`chain-d2c-0125`, `chain-d2p-0318`, `-0323`, `-0379`,
  `-0393`) are not in the published bank, so their prompts are unknown and they are not built. The
  builder prints them.

Positions were verified: for every row, the token at `pos` in the Qwen3.6-27B chat render (thinking
off) matches the row's `token`, and `rel_end == pos - n_tokens`.

## `hallucination`: judged rows exist, but only for ablation arms (2026-10-16)

`camilablank/olens-results` has no `hallucination/` directory, and neither do the bench's HF mirror
(`camilablank/workspace-bench`), `agu18dec/mOre-lenses`, `agu18dec/local-workspace` or
`agu18dec/olens_and_ar`. The post's per-arm rates (olens RL 0.34, SFT 0.28, J-lens 0.31, NLA-RL
0.96, prompt-only 0.07, ...) therefore cannot be reproduced from public rows. What is public:
`agu18dec/olens_ablations` `ab4/offset/results/wsbench/<arm>/hallucination_opus.json` for twelve
arms (`rl.base.off_k{0,1,2,4,8,16}.own.i300`, `sft.base.off_k{...}.own`): a full `results.json` each
(5,615 cells, one four-bullet readout per cell, span verdict, stage-2 claim tallies as counts only,
no claim quotes), judged by claude-opus-5-5, not the post's Gemini. `ab4/results/wsbench/*` (the
reference checkpoint `ref-s3d-rl600` included) holds only `summary.json`, no rows. The bank
(`evals/hallucination/items.json` + `capture_rows.json`) is in the bench's GitHub repo; its token
ids and site positions were verified against the Qwen3.6-27B tokenizer (0 mismatches), and the char
offsets match the whole-span decode. `scripts/wsbench_hallucination.py` builds
`data/readouts/wsbench-hallucination/` from these.

## The bench's families (none but `chain_intermediates` have published readouts)

Item counts are from `evals/<family>/items.json`; one line each from the family README.

| family                              | items | what it asks                                                                                                            |
| ----------------------------------- | ----: | ----------------------------------------------------------------------------------------------------------------------- |
| `agentic_misalignment`              |    32 | Does the lens show the model's plan to blackmail/leak/exfiltrate before it writes it? Every prompt position, 12 layers. |
| `arithmetic_intermediates`          |   596 | `(271 - 322) * 14`, no CoT; the intermediate is never written. Every position at layers 56 and 60.                      |
| `association`                       |   100 | A scene implies a concept never named; read at the final token.                                                         |
| `basic_readout`                     |   100 | The model's obvious next concept at the final token.                                                                    |
| `basic_readout_mt`                  |   100 | Same when the concept is a multi-token phrase; language scored too on factual items.                                    |
| `brew_intermediates`                |   100 | Potion colour after rule-table stirs, start colour given last; the colour after the first stir.                         |
| `buggy_code`                        |    49 | One-bug programs with clean twins; how well does a reader infer the bug from readouts at the end?                       |
| `chain_intermediates`               |   120 | Two/three arithmetic steps, seed last, no CoT; can a lens name the intermediate? **Built.**                             |
| `conjunctive_association`           |   100 | A vignette implies a compound state (state × content × relation); does the lens state it whole?                         |
| `directed_modulation`               |   100 | Think about / don't think about / keep a secret while copying a sentence; is the held concept there?                    |
| `hallucination`                     |   149 | Precision: does the lens say things about the conversation the text rules out? Lower is better.                         |
| `jailbreak_recognition`             |    86 | Verbatim WildChat conversations; does the lens recognise a jailbreak attempt?                                           |
| `jlens_concept_pr`                  |   299 | How much of a prose arm's text is supported by the J-lens top-10 at the same activation.                                |
| `moral_rationale`                   |   200 | One-token answer to a hard ethical yes/no; does the lens surface the consideration weighed?                             |
| `multi_concept_directed_modulation` |    27 | Hold one to three unrelated concepts while writing a dictated sentence; how many survive, bound?                        |
| `multihop`                          |   100 | One silent hop; the bridge concept, read at the last word of the prompt.                                                |
| `multihop_mt`                       |   100 | Two/three hops with multi-token bridge names, each bridge written in one layer.                                         |
| `multihop_mt_v2`                    |    70 | Same, every bridge proved necessary.                                                                                    |
| `multilingual`                      |   100 | Non-English prompt, one-word answer; the English concept counts in any language.                                        |
| `multilingual_mt`                   |   100 | Non-English prompt, multi-token concept, plus the passage's language, in one layer.                                     |
| `multilingual_multihop`             |    98 | Non-English two-hop; the bridge in English or Chinese.                                                                  |
| `multilingual_typo`                 |   100 | Non-English misspelled last word; the correction plus the language.                                                     |
| `poetry`                            |   100 | Couplet cut before its rhyme word; read at the newline ending line one.                                                 |
| `relational_multihop`               |   100 | Two-hop possessive cloze (landlord's sibling); the composed relation in the right order.                                |
| `role_bound_association`            |   100 | A directed relation with role words blocked; does the lens bind who did what to whom?                                   |
| `typo`                              |   100 | Misspelled last word; the correction.                                                                                   |
| `typo_mt`                           |   100 | Misspelled multi-token word or phrase.                                                                                  |
| `user_modeling`                     |   100 | Transluce SelfDescribe/SynthSys items; a latent user attribute.                                                         |
| `wrong_bridge`                      |   100 | Wikidata two-hop where Qwen resolves the wrong bridge B'; is B' in the lens?                                            |
| `wrong_bridge_true`                 |   100 | The same items scored against the true bridge B.                                                                        |

`examples/readouts/*.jsonl` in the bench repo are toy files (a few rows each), not runs. To add
another family, someone has to produce the readouts (`docs/producing_readouts.md` in the bench repo
has the recipe), then add a `Task` to `scripts/wsbench.py`: a bank URL, the `judge_rows` paths per
arm and an `items()` function that maps row labels to prompts and short names.
