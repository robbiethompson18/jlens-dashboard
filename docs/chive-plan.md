# Plan: CHIVE-found transcripts for the viewer (not started)

Sketched 2026-10-16, parked by Robbie. Read before picking this up, or when asked whether CHIVE fits
here.

## What CHIVE is

"Would this change your answer?" (Karvonen, Ong, Kantamneni, Marks, 2026, arXiv 2608.16747; post at
https://alignment.anthropic.com/2026/chive/). An agentic pipeline, not a model organism: sample a
target on real prompts, flag unexpected behaviors, have an investigator run counterfactual prompt
edits, verify. Code: https://github.com/adamkarvonen/chive (MIT). Data: `adamkarvonen/chive-data` on
HF (~93 GB; eval and train runs for Qwen3-8B/32B, Qwen3.5-397B, Qwen3.6-35B-A3B, Gemma 3 27B, Llama
3.1 8B; nothing for Qwen3.6-27B or Qwen3.8-27B).

Headline result relevant to us: activation oracles, NLAs and SAEs gave the investigator no uplift
over reading transcripts at predicting counterfactual outcomes. The J-lens was not tested.

## Pipeline stages (from `chive/pipeline/README.md`)

1. **Completions**: 30 samples per prompt at T=1, 500 tokens, thinking off (n=10 for train runs).
2. **Screening**: judge scores 1-5 and writes a question, a fixed yes/no `classifier_question`, 2-4
   hypotheses. Score >= 3 passes, about 55%.
3. **Investigation**: investigator agent runs 5-15 edits via `run_prompt`; each edit resamples 30
   completions and a grader (Sonnet 4.6) applies the classifier question to each. Per investigation:
   ~80k uncached + ~182k cached input, ~8.5k output, ~7.6 target calls.
4. **Verification**: judge scores 1-10. 78% score >= 7, 44% >= 8.

All stages are CPU-side drivers. Every role takes a backend spec (`anthropic:`, `openrouter:`,
`vllm:...@url`, `tinker:`), so no GPU of ours is needed until the readout. Prompt pool:
`prompt_sources/prompts_wildchat_agentic/prompts.jsonl` (189,542 prompts, shuffled, WildChat plus a
9% agentic slice with system prompts and tool specs as text). Output that matters:
`investigations.jsonl`, which holds the prompt, all completions with grader verdicts, every edit
with its completions and verdicts, and the written causal report.

## Why it fits the viewer

Each verified investigation is a ready-made contrast set, like dolphin/owl/none but found
automatically: the same prompt with the behavior present and absent (the grader splits the 30
completions), plus the edit that flips the rate, plus a stated hypothesis to check the lens against.

## Steps

1. **Smoke run** (same day, < $10). Clone, `uv sync` (no GPU extra), `.env` with OpenRouter and
   Anthropic keys. Their smoke config: 100 prompts, n=10, 5 investigations, ~15 min. Target
   `openrouter:qwen/qwen3.6-27b`, thinking off. Check OpenRouter's n=1 fan-out and the Qwen FP8
   provider routing; read the five investigations before spending more.
2. **300-prompt run** at n=30. First 300 lines of the canonical pool. Expect ~165 investigations,
   ~130 at verification >= 7, ~70 at >= 8.
3. **`scripts/chive_items.py`**: keep verification >= 8 and `mechanism_concreteness` >= 3 (their own
   filters). Per investigation up to four items: a baseline completion with the behavior (grader
   yes), one without (grader no, when the baseline is split), and the key experiment's completion
   each way. Index and an auto-generated `notes.json` carry the question, hypothesis, baseline and
   counterfactual rates, and the report. Prompts are 400-8000 chars and completions 500 tokens, so
   they fit the 2000-token prompt window. Agentic-slice tools are text in the system prompt, so
   items carry plain `messages`, no `tools`.
4. **Readout and viewer**: a `chive` entry in `scripts/readout.py`, one pod. Viewer group "CHIVE"
   with sets by behavior category; items of one investigation sorted together; a link back to the
   investigation. Notes must quote the rates, since each item is one draw of 30.
5. **Later, optional**: point the target at a LoRA organism served by vLLM on a pod (the
   constitution or lying LoRA, `docs/model-organisms.md`) and diff against the base run. Their
   `petri` prompt source (alignment-audit transcripts) is the other pool worth trying.

## Cost estimate for step 2

| item                                                              | estimate |
| ----------------------------------------------------------------- | -------: |
| Qwen sampling on OpenRouter, baseline plus investigator resamples |     ~$10 |
| Screening, Opus 4.6 batch, 300 prompts                            |     ~$15 |
| Investigation, ~165 × Opus 4.6 at the measured token counts       |    ~$120 |
| Grader (Sonnet 4.6) and verification (Opus 4.6 batch)             |     ~$25 |
| Pod for the readout                                               |      ~$3 |

About $170 for ~70 high-quality investigations. Sonnet as investigator would be ~$70 at lower
quality. Opus 4.6 rather than a newer Opus because the pipeline passes `thinking_budget`, which Opus
5.5 rejects with a 400.

## Open decisions

- No `ANTHROPIC_API_KEY` in `.envrc.local` as of 2026-10-16. `openrouter:anthropic/claude-opus-4-6`
  works for every role, but loses batch mode (~20% more on screening and verification).
- Target thinking on or off. Their thinking runs (budget 1024) are the ones they say surface
  unfaithful CoT, the more interesting case for a lens that reads mid-CoT; it roughly doubles
  sampling cost and lengthens items. Suggested: off for the first 300, a thinking run second.
- Alternative uses not chosen: (a) only adopt the format, hypothesis plus counterfactual pair per
  curated item, no pipeline; (b) plug the J-lens into the investigator as a tool and measure uplift
  against their transcript-only baseline. (b) is a research project and their result predicts null.
