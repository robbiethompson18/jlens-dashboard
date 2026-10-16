# 13. Literature replications on Qwen 3.6 27B

Model: `qwen/qwen3.6-27b` via OpenRouter. Temperature 0 unless stated. "no-think" = `reasoning.enabled=false`,
"think" = enabled with max_tokens 6000. Providers were left unpinned (Chutes, Alibaba, Reka, Venice,
SiliconFlow, DeepInfra, Phala all served calls) except where noted; every weird result was re-run pinned
to both Alibaba and Chutes. Probe files in `probes/lit/`, raw per-call records in `probes/lit/out/*.json`,
batch runner `probes/lit/run.py`, multi-turn driver `probes/lit/multiturn.py`. ~1,150 calls, ~$1.30 (the glitch-token work is the cheapest and the think-mode batches the priciest).

Paper list with citations: `papers/lit-sweep.md`.

Ranking by surprise (5 = would show a colleague):

| # | Replication | Paper | Effect on Qwen 3.6 27B | Surprise |
|---|---|---|---|---|
| 1 | Glitch tokens | Land & Bartolo 2024; Rumbelow & Watkins 2023 | Strong, deterministic, both providers, think doesn't fix | 5 |
| 2 | Potemkin understanding | Mancoridis et al. 2025 | Yes in no-think (67% classify), mostly fixed by think | 3 |
| 3 | Reversal curse | Berglund et al. 2023 | Partial: fwd 13-15/15 vs rev 10/15 | 3 |
| 4 | Verbosity bias in judging | Saito et al. 2023; Zheng et al. 2023 | Inverted: 40/40 prefers the shorter of identical-content pair | 2 |
| 5 | Memo trap (inverse scaling) | McKenzie et al. 2023 | Weak: 2/15 fail no-think; think 15/15 at ~100x tokens | 2 |
| 6 | Position bias in judging | Wang et al. 2023; Shi et al. 2024 | Weak: 2/10 near-equal pairs flip with order, 0/10 clear pairs | 2 |
| 7 | Alice in Wonderland | Nezhurina et al. 2024 | None on AIW (22/22); fails AIW+ both modes | 2 |
| 8 | Lost in multi-turn | Laban et al. 2025 | Task-dependent: 1/6 tasks breaks, 0/5 sharded vs 5/5 full, via premature-answer anchoring | 2 |
| 9 | Numeric anchoring | Lou & Sun 2024; Huang et al. 2025 | None on known facts; weak/inconsistent on Fermi estimates (2-3/10 items) | 1 |
| 10 | GSM-NoOp distractor | Mirzadeh et al. 2024 | None (think 40/40; no-think errors are not NoOp-specific) | 1 |
| 11 | Premise order | Chen et al. 2024 | None (95/96) | 1 |
| 12 | Lost in the middle | Liu et al. 2023 | None (50/50 KV retrieval up to 300 pairs / ~20k tokens) | 1 |

---

## 1. Glitch / under-trained tokens (surprise 5)

**Paper.** Land & Bartolo (2024), "Fishing for Magikarp", arXiv:2405.05417; lineage from Rumbelow & Watkins
(2023) SolidGoldMagikarp. Their public repo has an unembedding-norm report for Qwen3-32B listing
under-trained tokens. Qwen 3.6's tokenizer (`Qwen/Qwen3.6-27B/tokenizer.json`, 248,044 vocab, up from
151k in Qwen3) still contains all the ASCII glitch tokens from that list: `$PostalCodesNL`,
`PostalCodesNL`, ` ForCanBeConvertedToF(oreach)`, ` ForCanBeConverted`, `useRalative`, `useRal`,
` thuisontvangst`, ` wannonce`, ` sexkontakte`, `webElementX`, ` davidjl`, etc.

**What I ran.**
```
python3 probes/lit/run.py probes/lit/glitch.jsonl  --think off                  # 10 glitch + 6 control tokens, repeat + spell
python3 probes/lit/run.py probes/lit/glitch.jsonl  --think off --provider Alibaba
python3 probes/lit/run.py probes/lit/glitch.jsonl  --think off --provider Chutes
python3 probes/lit/run.py probes/lit/glitch.jsonl  --think on
python3 probes/lit/run.py probes/lit/glitch2.jsonl --think off                  # more tokens + in-context use
python3 probes/lit/run.py probes/lit/glitch3.jsonl --think off --provider Alibaba --n 2   # repeat + "what does it mean"
python3 probes/lit/run.py probes/lit/glitch3.jsonl --think off --provider Chutes  --n 2
python3 probes/lit/run.py probes/lit/glitch4.jsonl --think off                  # tokenization-boundary controls
```
Prompt: `Repeat the following string back to me exactly, character for character, with nothing else: "<tok>"`.

**Results (no-think, identical on Alibaba and Chutes, 2/2 reps each).**

| token (as sent) | repeat output | "what does it mean" output |
|---|---|---|
| ` ForCanBeConvertedToForeach` | `编辑于` | "鲁迅为闰土报仇 ... Chinese internet meme" / "for循环遍历 ... for loop iteration" |
| ` ForCanBeConverted` | `"` (nothing) | - |
| `useRalative` | `" deutsch"` | "delawg appears to be a typo" |
| `useRalativeImagePath` | correct repeat, but spell = `a-l-t-e-r-n-C` | - |
| `$PostalCodesNL` | `$ л` / `$ лё` | - |
| `PostalCodesNL` | `" лѣта"` (Old Church Slavonic yat) | "лете ... Russian for summer" |
| ` thuisontvangst` (Dutch, "home reception") | **refusal**: "I cannot repeat that string as it contains sexually explicit content" | "onstreetwalking ... illicit" |
| ` wannonce` (French classifieds site) | **refusal**, same text | "invonstance appears to be a typo of instance" |
| ` sexkontakte` | refusal | "sexzitate" |
| `webElementX` | `"Excelsior"` | "WerkzeugX" / "WooX" |
| ` davidjl` | `davidll` | - |
| 6 controls (` RecyclerView`, `PreparedStatement`, ...) | 6/6 correct | correct |

Spell-out task: `useRalative` -> `d-e-l-s-c-h`; `$PostalCodesNL` -> `$- -ж-е-п-т-а`; `PostalCodesNL` -> `л-е-т-а`.

**Think mode does not fix it** (`glitch_on.json`): ` ForCanBeConvertedToForeach` -> `我为群众办实事` after 275
reasoning tokens; `$PostalCodesNL` -> `"$ jeito"`; `useRalative` -> ` deutsch` (spelled `d-e-u-t-s-c-h`
after 2,398 reasoning tokens); once the model echoed the whole instruction instead of the string.

**Tokenization-boundary controls** (`glitch4`): the effect is bound to the exact token.
- ` thuisontvangst` (leading space) -> refusal; `thuisontvangst` (no space), ` THUISONTVANGST`,
  ` thuis-ontvangst`, ` thuis[ZWSP]ontvangst` -> all repeated correctly.
- ` useRalative` (leading space, splits into normal tokens) -> correct; `useRalative` (no space, the glitch
  token) -> ` deutsch`.
- ` ForCanBeConvertedToForeach` -> `编辑于`; `ForCanBeConvertedToForeach` (no space) -> correct.
- In a sentence, `De masseuse biedt thuisontvangst aan in Utrecht` translates fine ("offers home service"),
  so surrounding context rescues meaning even though the token itself is unreadable.
- In code, `const path = useRalative(img)` -> model names the hook correctly (copying works via context),
  but `$PostalCodesNL = load('nl.csv')` -> "airwork", and `obj ForCanBeConvertedToForeach()` -> "obj.".

**Interpretation.** Classic under-trained-token behavior: the model cannot read the token's characters and
instead emits whatever lives near its embedding (Cyrillic, Chinese boilerplate, "Excelsior"). The
` thuisontvangst` / ` wannonce` refusals are the nicest part: those strings occur almost only in adult
classified-ad spam, so their embeddings carry an "adult content" direction without any legible
content, and the safety behavior fires on vibes. The "what does it mean" answers are consistent across
providers (same hallucinated surface form, e.g. "delawg", "invonstance", "sexzitate") because it is
deterministic at temp 0, not because the model has stable knowledge of them.

**Side-behaviors.** None of the control tokens misfire. `navbarSupportedContent` spelled as
`navbarCollapseNavLinks` and `toBeInTheDocument` spelled lowercase, so spelling long camelCase tokens is
weak in general (tokenization, other agent's lane).

## 2. Potemkin understanding (surprise 3)

**Paper.** Mancoridis, Weeks, Vafa & Mullainathan (2025), "Potemkin Understanding in Large Language Models",
arXiv:2506.21521, ICML 2025. Models define a concept correctly ("keystone") but fail to classify or
generate instances.

**What I ran.** 8 concepts (ABAB rhyme, 5-7-5 haiku, slant rhyme, anapestic meter, sunk cost fallacy, Nash
equilibrium, enjambment, palindrome): define, generate, classify a positive and a negative instance.
```
python3 probes/lit/run.py probes/lit/potemkin.jsonl     --think off
python3 probes/lit/run.py probes/lit/potemkin_cls.jsonl --think both --n 3     # 16 classification items x 3
```

**Results.** Definitions 8/8 correct, generations 8/8 correct (ABAB poem rhymes away/above/day/love; REVIVER;
valid haiku; correct Prisoner's-Dilemma NE). Classification: **no-think 32/48 (67%)**, think 44/48 (92%).
Failures, no-think, 3/3 reps each unless noted:
- ABAB: the ABAB poem (still/bright/hill/light) -> "No"; the AABB poem -> "Yes" (1/3 reps).
- Palindrome: "Is 'deified' a palindrome?" -> **No** (3/3); "Is 'rotator' a palindrome?" -> **No** (3/3).
  Same model that just produced REVIVER with a letter-by-letter verification. Think fixes both (6/6).
- Nash: "(Down, Right) with payoffs (1,1)" -> "No" (3/3), though it is a NE (deviations give 0 < 1).
  Think fixes (3/3).
- Slant rhyme: shape/keep -> "No" in 3/3 no-think and 2/3 think. Arguable item (shared final consonant,
  different vowel fits the model's own definition), so I count it weakly.
- Haiku (think only): the classic "old silent pond" haiku -> "No" in 2/3 think reps, 3/3 correct no-think.
  Syllable counting, other agent's lane.

**Verdict.** The define-right/classify-wrong pattern replicates clearly in no-think mode (ABAB, palindrome,
Nash). Think mode mostly closes the gap, which the paper did not test on reasoning models.

## 3. Reversal curse (surprise 3)

**Paper.** Berglund et al. (2023), "The Reversal Curse", arXiv:2309.12288, ICLR 2024. GPT-4: Tom Cruise's
mother 79% vs Mary Lee Pfeiffer's son 33%.

**What I ran.** 15 celebrity-parent pairs, both directions, "Answer with just the name."
```
python3 probes/lit/run.py probes/lit/reversal.jsonl --think off
python3 probes/lit/run.py probes/lit/reversal.jsonl --think on
```

**Results.** Hand-scored (regex missed two near-correct forward answers: "Jackie Gise" and "Jacklyn Gise
Jorgensen" are Jeff Bezos's mother's maiden/married names).
- No-think: forward 13/15, reverse 10/15.
- Think: forward 15/15, reverse 10/15 (one reverse item, Mary Lee Pfeiffer -> ?, spent all 6,000 tokens
  thinking and never answered).
- Reverse failures are confident wrong people: Mary Lee Pfeiffer's son -> "John Pfeiffer"; Pattie Mallette's
  son -> "Michael Jackson" / "Prince"; Mary Maxwell Gates's son -> "Paul Allen"; Irmelin Indenbirken's son ->
  "Jürgen Klinsmann" / "Moritz Bleibtreu"; Penny Adkins's daughter -> "Megan Hilty" / "Sheryl Crow".
- The exact Berglund example (Tom Cruise) replicates in both modes.

**Verdict.** Asymmetry present but smaller than GPT-4-era numbers; the model knows the parent facts for
mega-celebrities both ways and fails reverse only where the parent is obscure. Surprise is in the
confident substitutes (Paul Allen for Bill Gates's mother's son is a nice associative slip).

## 4. Verbosity bias in judging, inverted (surprise 2)

**Papers.** Saito et al. (2023) arXiv:2310.10076; Zheng et al. (2023) MT-Bench arXiv:2306.05685.

**What I ran.** Pairwise "which response is better, A or B" judging.
```
python3 probes/lit/run.py probes/lit/judge.jsonl  --think off --n 2   # clear good/bad pairs, swapped; bad answer padded with filler
python3 probes/lit/run.py probes/lit/judge2.jsonl --think off --n 2   # near-equal pairs swapped; identical content short vs padded
```

**Results.**
- Clear pairs: 40/40 picked the correct answer regardless of slot; 32/32 picked the short correct answer over
  the padded wrong one.
- Identical content, short vs padded with a polite preamble/postamble: **40/40 chose the short one**, in
  either slot. No verbosity bias; a strict conciseness preference (cf. arXiv:2604.23178, where Claude Sonnet
  4 was the lone conciseness-preferring judge).

## 5. Memo trap / inverse scaling (surprise 2)

**Paper.** McKenzie et al. (2023), "Inverse Scaling: When Bigger Isn't Better", arXiv:2306.09479. Task:
"Write a quote that ends in the word 'heavy': Absence makes the heart grow".

**What I ran.** 15 items.
```
python3 probes/lit/run.py probes/lit/memo.jsonl --think both
```

**Results.** No-think 13/15 (fails: "...what you can do today", "A rolling stone gathers no moss."). Think
15/15, but median 956 and max 2,891 reasoning tokens to finish a proverb with a wrong word; the no-think
answers were 4-12 tokens. Several think-mode outputs satisfy the constraint by appending a clause rather
than substituting ("Slow and steady wins the race, but opportunity strikes fast").

**Verdict.** Memo trap largely solved, with an overthinking tax (Chen et al. 2024, arXiv:2412.21187).

## 6. Position bias in pairwise judging (surprise 2)

**Papers.** Wang et al. (2023) arXiv:2305.17926; Shi et al. (2024) arXiv:2406.07791 (bias scales with how
close the candidates are).

**Results** (same runs as section 4). Clear pairs: 0/10 order-dependent. Near-equal pairs (10 open-ended
questions, two reasonable one-line answers): 8/10 consistent across swap, 2/10 flipped with order (one
always-A, one always-B). Overall A/B picks exactly 20/20. Mild, consistent with Shi et al.

## 7. Alice in Wonderland (surprise 2)

**Paper.** Nezhurina et al. (2024, v5 2025), arXiv:2406.02061.

**What I ran.** 10 random (brothers, sisters) variants, 2 reps each, both modes, plus one AIW+ item.
```
python3 probes/lit/run.py probes/lit/aiw.jsonl --think both --n 2
```

**Results.** Basic AIW: 22/22 correct in both modes (no-think answers in 1-2 tokens). AIW+ (Alice has 3
sisters; mother's childless sister has 7 nieces/nephews and 2 brothers; father's brother has 5
nieces/nephews and 1 son; how many cousins does Alice's sister have? Correct: 5): no-think 10 (2/2), think
3 (2/2, ~3,700 tokens each). Wrong in both modes, confidently.

**Verdict.** Basic AIW is solved (consistent with the paper's note that reasoning models mostly handle it);
the compositional AIW+ still breaks it.

## 8. Lost in multi-turn conversation

**Paper.** Laban, Hayashi, Zhou & Neville (2025), "LLMs Get Lost In Multi-Turn Conversation", arXiv:2505.06120,
ICLR 2026.

**What I ran.** 6 tasks (4 arithmetic planning with a checkable final line, 2 Python functions checked by
executing hidden asserts), each as one fully-specified turn vs 4 drip-fed shards.
```
python3 probes/lit/multiturn.py --n 3 --temperature 1.0
python3 probes/lit/multiturn.py --n 2            # temp 0
```

**Results.** Temp 0: full 12/12, sharded 11/12 (chairs task: 15 rows instead of 14). Model always
"answers early" on the code tasks (writes a function after shard 2) and then revises, as the paper describes.
Temp 1.0, 3 reps: full **18/18**, sharded **15/18**. All three sharded failures are the chairs task, and the
transcript shows the paper's mechanism exactly: after shard 2 (150 guests, rows of 12) the model volunteers
an unasked-for recommendation ("Why 180? It's a round number, 15 rows x 12"); when shard 3 adds the real
rule (10% spare, round up to a whole row -> 165 -> 14 rows -> 168 chairs) it rationalizes its earlier number
("13 rows + 2 spare rows = 15 rows"), and the final turn opens "Based on the previous calculation" and
reports 180 chairs / $450. Full-prompt runs get 168 / $420 every time (5/5).

**Verdict.** The effect exists but is task-dependent: 5/6 tasks are immune at this length, and the one that
breaks does so through premature-answer anchoring (0/5 sharded correct across temps vs 5/5 full). Not the
~39% average drop the paper reports for 2025 models, but a clean single-task demonstration.
Surprise 2.

## 9. Numeric anchoring

**Papers.** Lou & Sun (2024) arXiv:2412.06593; Huang et al. (2025) arXiv:2505.15392.

**What I ran.** "Is X more or less than A? Then estimate X" with low/high anchors vs no anchor.
```
python3 probes/lit/run.py probes/lit/anchor.jsonl  --think both --n 2   # 8 known facts
python3 probes/lit/run.py probes/lit/anchor2.jsonl --think both --n 2   # 10 Fermi questions (piano tuners, jelly beans, ...)
```

**Results.** Known facts: zero anchoring; the model returns the memorized value (54 African countries, 206
bones, 1440, 380 ft, 2,320-2,340 mi) under every anchor in both modes. Ceiling effect, as the anchoring
papers note for well-known quantities.
Fermi questions (no memorized answer), 10 items x {none, low anchor, high anchor} x 2 reps:
- No-think (max_tokens 2500 so the Fermi chain completes): high-anchor mean > low-anchor mean on 5/10 items,
  < on 5/10. Coin flip. Three items show a clear high-anchor pull: piano tuners in Chicago none 20-30,
  lo(5) 20-40, **hi(2000) 80-150**; Central Park trees none 10k, lo(500) 10k, **hi(200k) 36-39k**; German
  libraries none 3.3-4.2k, hi(40k) 3.7-9.5k. The rest are flat or go the wrong way.
- Think: hi > lo on 9/14 scoreable item-pairs, but effect sizes are small (piano tuners: none 50-70,
  lo 100-120, hi 150-200; golf balls: none 350-600k, lo 750k, hi 750-850k, i.e. *both* anchors push upward,
  which looks more like "the question mentioned a number so the answer got bigger" than classic anchoring).
  Run-to-run noise at temp 0 across providers is of the same order as the anchor effect.

**Verdict.** No anchoring on known facts; weak, inconsistent anchoring on Fermi estimates, visible on 2-3 of
10 items. Smaller than the 22-61% rates reported for 2024-25 chat models in Huang et al. (2025). Surprise 1.

## 10. GSM-NoOp distractors (surprise 1)

**Paper.** Mirzadeh et al. (2024), GSM-Symbolic, arXiv:2410.05229.

**What I ran.** 10 base/NoOp pairs including the paper's kiwi item, 2 reps, both modes.
```
python3 probes/lit/run.py probes/lit/gsmnoop.jsonl --think both --n 2
```

**Results.** Think: 40/40. No-think: base 16/20, NoOp 18/20; the 6 errors are two items wrong in both
conditions (kiwi item: answers 150 instead of 190, and 152 with the NoOp clause, i.e. it did not subtract
the 5 small kiwis; weekly-miles item: 35 instead of 25). Re-ran those two 7 more times each across
Alibaba/Chutes/temp 0.7: wrong every time, same numbers. No NoOp-specific drop. Consistent with 2026
re-evaluations (LessWrong "Revisiting GSM-Symbolic"; Długosz et al. 2026, arXiv:2605.28700).

**Side-behavior.** Deterministic wrong arithmetic on "give only the number" in no-think (44+58+88 -> 150).
Belongs to the arithmetic agent.

## 11. Premise order (surprise 1)

**Paper.** Chen, Chi, Wang & Zhou (2024), arXiv:2402.08939, ICML 2024.

**What I ran.** 8 four-step deductive chains (modus ponens chains, "all X are Y" chains, two modus tollens
items) in forward, reversed and shuffled premise order; yes/no/undetermined.
```
python3 probes/lit/run.py probes/lit/premise.jsonl --think both --n 2
```
**Results.** No-think 48/48, think 47/48 (one provider error, not a wrong answer). No effect. Items may be too
short; the paper's drops were on 5-12 premise problems with distractors.

## 12. Lost in the middle (surprise 1)

**Paper.** Liu et al. (2023), arXiv:2307.03172.

**What I ran.** Their synthetic JSON key-value retrieval with UUID keys/values; target at positions
{1,15,30,45,60} of 60 pairs (6 seeds) and {1,75,150,225,300} of 300 pairs (~22k tokens, 4 seeds).
```
python3 probes/lit/run.py probes/lit/litm.jsonl  --think off
python3 probes/lit/run.py probes/lit/litm2.jsonl --think off --max-tokens 100
```
**Results.** 30/30 and 20/20. No positional effect at these lengths. Longer contexts are the degeneration
agent's lane.

---

## Not run (and why)

Self-preference (Panickssery et al. 2024) needs a second model's outputs; in-context emergent misalignment
(Afonin et al. 2025) and persona induction (Kim et al. 2026) need a judge model and brush against the
refusal/persona lanes; subliminal learning's in-context variant is a published null (Cloud et al. 2025);
CatAttack (Rajeev et al. 2025) is a close cousin of GSM-NoOp, which was negative; typo robustness, negation
and few-shot label bias were deprioritized for time after the glitch-token result turned out to be the
richest vein.
