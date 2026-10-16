# O-lens hallucination evals: ideas and the mistake taxonomy

Scope: new evals that sit beside the WorkspaceBench (WSB) `hallucination` family. They are for
hill-climbing O-lens variants on Qwen3.6-27B and for sorting the mistakes current lenses make. About
130 ideas came from 18 research agents. They were merged into 25, and 12 of those were critiqued
against local data: the 12 published agu18dec AR-offset arms, 149 WSB items, 5,615 cells per arm.

## 1. Summary

The current family has three properties that shape any new eval.

- **The readout-level rate mostly measures length.**
  - RL k0 scores 0.396 and SFT k0 scores 0.251. Both have 2.05 verified wrong spans per 1k
    characters.
  - Across 60 arm x layer cells, the rate correlates with mean readout length at r=0.92.
  - RL's "+0.10 hallucination" is mostly RL writing 2.3x more text. Claim precision only drops from
    0.80 to 0.74.
- **The judge sets the level.**
  - One checkpoint (ref-s3d-rl600) scores 0.397 under Opus, 0.522 under Sonnet and about 0.34 under
    Gemini.
  - The paired CI between arms on the same cells is about ±0.015. The unpaired CI is about ±0.045.
- **Nothing in the family checks that the lens read the activation.**
  - The prompt-only floor (0.07) is the best score of any arm, and 57-63% of claims cannot be
    verified.
  - A lens that inverts nearby text, or says less, scores well.
  - In the other direction, 73% of O-lens sibling substitutions are confirmed by probes or by
    another lens (HB 2.0). Some flagged "errors" are therefore faithful reads of the model's state.

**Recommended hill-climb suite**

1. **Planted-slot bank.** One build, scored three ways: Slot Twins FAB, Slot-commit calibration U
   and Twin likelihood accuracy (evals 1, 2 and 5 below).
   - Values are planted and the read sites are probe-gated. The local window is byte-identical
     across twins, and scoring is regex at $0.
   - It targets H1-H4 (slot filling), which is 70% of current errors.
2. **NGC_h on the WSB bank.** Net grounded claims minus a same-prompt resample donor.
   - It is the only metric on natural data with a chance line, and it reuses the v5c harness.
3. **CounterPrior on probe-certified sites.**
   - It targets world-prior and identity override (H8, H9). This is the mechanism suspected behind
     "hallucination rises with AR size".
4. **Computed-value precision (CVP)** on the chain and arithmetic banks.
   - It mostly re-scores existing judge output, plus seed-swap twins.
   - It targets H3 numbers. The pre-seed confabulation and twin-specificity parts cannot be forged.
5. **Always on, at no or low cost:**
   - an RL-time monitor: OOC-atom rate (Spearman 0.96 with the WSB rate across 12 arms), CJK rate,
     and entropy/distinct-2;
   - the taxonomy diagnostic (eval 7) with its SUBST-CTX rate.

**Recommended scalar: U_slot**, computed on the planted-slot bank.

- **Scoring.** For each readout x slot: +1 if it names only the planted value, -2 if it names any
  other value of the same type (naming right and wrong together counts as wrong), 0 if it names
  none.
- **Aggregation.** Average over the gated cells: DECODABLE, UNDECODABLE and BLANK.
- **What different lenses score.**
  - A silent lens scores 0.
  - A uniform guesser over a 32-value pool scores about -1.9 per commit.
  - A lens scores above 0 only if its commit precision is above 2/3.
- **Validity rule (mandatory).** A checkpoint's U_slot counts only if both of these hold; otherwise
  the run is INVALID, not "better":

  - hit rate on ENCODED cells is at least 0.8x the reference;
  - commit rate is within ±20% of the reference.

  Without this rule the scalar can be climbed by suppressing proper nouns. On WSB items that contain
  a common first name, RL k0 already names an invented name 4.4x as often as the true one (95 vs 22
  of 1,940 readouts).

- **Acceptance.** A paired, template-clustered bootstrap ΔU > 0, and the WSB hallucination rate and
  assert_share do not regress beyond their CIs.
- **Why not NGC_h as the scalar.** It costs about $5-10 in judge calls per checkpoint and is
  noisier. Use it as the natural-data confirmation at milestones.

## 2. Mistake taxonomy

### Sources and coverage

| key | items                                      | arms and layers                    | labelling                           | hallucinated readouts used                                    |
| --- | ------------------------------------------ | ---------------------------------- | ----------------------------------- | ------------------------------------------------------------- |
| A   | chat-dailydialog-0000..0036                | RL k0 and SFT k0, 5 layers         | per readout, one primary category   | 555 (RL 347, SFT 208)                                         |
| B   | chat-dailydialog-0037..0073                | RL k0 and SFT k0, 5 layers         | per readout, one primary category   | 528 (RL 352, SFT 176); ~75 k16 rows examined but not labelled |
| C   | chat-dailydialog-0074, chat-lmsys-0000..36 | RL k0 and SFT k0, 5 layers         | site-level dominant tag, about ±10% | 1257 (RL 765, SFT 492); 63 unallocated                        |
| D   | chat-lmsys-0038..0074                      | RL k0 and SFT k0, **L44 only**     | per readout                         | 264 (RL 150, SFT 114); per-arm split mostly not reported      |
| E   | dashboard: 77 transcripts                  | Robbie's pilot O-lens, L42, 8x4tok | own labels, no bench judge          | 73 of 296 specific readouts                                   |

A-D cover 148 of the 149 items; chat-lmsys-0037 is in no shard. E uses a different lens, judge and
set of sites, so its numbers appear in their own column and are **not** in the WSB shares.

### Canonical categories

Columns A-D sum to 2604 readouts: 2541 allocated plus 63 unallocated in C.

| ID  | Category                                                                                                                       | A                                                    | B                                                | C                                                  | D                                                     | **WSB**              | E (dashboard)                                                                   |
| --- | ------------------------------------------------------------------------------------------------------------------------------ | ---------------------------------------------------- | ------------------------------------------------ | -------------------------------------------------- | ----------------------------------------------------- | -------------------- | ------------------------------------------------------------------------------- |
| H1  | Same-frame sibling substitution of a context-fixed entity, topic or task                                                       | Same-frame wrong filler 150                          | Sibling-instance slot drift 190                  | Prompt-given entity subst. 493 + Scenario swap 264 | In-context detail subst. 99 + Central-subject swap 75 | **1271**             | Entity slot 21 + Sibling concept 9 + Task/schema 7 + Copy-target ignored 4 = 41 |
| H2  | Proper-name substitution (stock-name fill)                                                                                     | Invented proper name 96                              | Proper-name subst. 68                            | (inside H1)                                        | (inside H1)                                           | **164** (undercount) | Local token continuation 6                                                      |
| H3  | Number, date, time or quantity substituted or invented                                                                         | Number/date/statistic 45                             | Invented specifics 42                            | Number/time/quantity 147                           | (inside H1)                                           | **234**              | Number substitution 17                                                          |
| H4  | Speaker, role or referent binding (speaker, addressee, owner, kin, gender, accepting a human persona)                          | Speaker/persona flip 54 + Person/relation/pronoun 42 | Role/perspective/gender 40                       | (some inside FLIP)                                 | (some inside flip)                                    | **136**              | -                                                                               |
| H5  | Polarity, negation, choice or stance flip, incl. contradicting own AI disclaimer                                               | Polarity/tense 14                                    | Polarity/stance/persona 46 + Garbled 1           | Choice/polarity/role flip 47                       | Perspective/polarity inversion 8                      | **116**              | -                                                                               |
| H6  | Fabricated claims about the input (user attribution, fake quotes, modality, input language, meta-notes)                        | Fabricated user-context attribution 37               | Fabricated context attribution 26                | Invented conversation history 16                   | Fabricated evidence 26 + Meta-notes 2                 | **107**              | -                                                                               |
| H7  | Structure, format or genre (section number/title, list members, swapped quoted sentence, "Option N", web genre, answer format) | Format/genre/structure 22                            | Structure misquote 44 + Document-genre leakage 9 | -                                                  | -                                                     | **75**               | Format-position 5 + Answer-format 4 = 9                                         |
| H8  | Prior-template intrusion (model identity or creator, safety boilerplate, China/Qwen prior, input-language identity)            | Self-identity binding 23 + Safety boilerplate 9      | Qwen/Chinese prior 21 + AI self-model 10         | (some inside H1 ENT)                               | -                                                     | **63**               | -                                                                               |
| H9  | Elaboration confabulation (right topic, invented fact/mechanism/formula, or world prior overriding a stated fact)              | World-knowledge 49                                   | World prior overrides context 16                 | Fabricated technical detail 187                    | World-knowledge 32                                    | **284**              | -                                                                               |
| H10 | Future-only guess, not ruled out at the read position (arguably not a hallucination)                                           | -                                                    | -                                                | Next-item / structure continuation 40              | Unconstrained future guess 19                         | **59**               | -                                                                               |
| H11 | Judge error, debatable call, near-copy garble                                                                                  | Judge-debatable 14                                   | Judge error 15                                   | (3-5 clear, not counted)                           | Judge 2 + Local token misparse 1                      | **32**               | -                                                                               |
| -   | Unallocated (C is approximate)                                                                                                 | -                                                    | -                                                | 63                                                 | -                                                     | 63                   | -                                                                               |

**Orthogonal tags** overlap the categories above and are not in the sums.

- **T1 Language/script drift.** RL k0 has CJK in 37-40% of all readouts in every shard (A 37%, B
  40%, C 39% of hallucinated readouts, D 39%); every other arm is at 1-6%. C counted 302
  hallucinated readouts with CJK, but only 117 had CJK inside a wrong span, so the judge mostly
  ignores the drift itself.
- **T2 Cross-bullet spray.** The bullets each commit to a different value for the same slot. Seen in
  354 of 765 RL k0 hallucinated readouts (46%) and 87 of 492 SFT k0 (18%) in C.
- **T3 Null-site prior dump and placeholders** (dashboard only, outside the hallucination
  numerator).
  - `import pandas as pd` appears in 631 corpus readouts, and one byte-identical readout appears 150
    times.
  - E rated 19 such readouts plus 8 format-junk readouts OFF.
  - Prior-fill, `[Project Name]` placeholders and number salad account for 32 GEN readouts.
  - 4.6% of corpus readouts contain a `[Capitalized` placeholder.

**Mapping caveats**

- C's ENT and D's "detail" tags mix names, numbers, code languages and identities and could not be
  split. H2 and H3 therefore cover only A and B (plus C's NUM), and H1 is inflated by the same
  amount.
- Persona-stance contradictions ("my partner is John Doe") are in H4 through A's grouping and in H5
  through B's.
- B's "invented specifics" was mapped whole to H3, but about a third of it is non-numeric (venues,
  invented kin, baby sex).
- E's entity bucket (mapped to H1) contains stock names (Mark, Sarah, Elena).
- The H7 vs H10 boundary: H7 when the context determines the structure, H10 when the next item is
  open.

### Shares of hallucinated readouts by arm

A+B+C cover 111 items at all 5 layers. D is L44 only and is pooled across arms.

| ID                 | RL k0 (n=1464)  | SFT k0 (n=876)  | DailyDialog RL / SFT | LMSYS-C RL / SFT | D pooled L44 (264)         | E (73) |
| ------------------ | --------------- | --------------- | -------------------- | ---------------- | -------------------------- | ------ |
| H1 sibling subst.  | **47.3%** (693) | **46.1%** (404) | 33.8 / 27.1%         | 59.7 / 61.0%     | 65.9%                      | 56%    |
| H2 names           | 7.2% (105)      | 6.7% (59)       | 15.0 / 15.4%         | (in H1)          | (in H1)                    | 8%     |
| H3 numbers         | 9.6% (141)      | 10.6% (93)      | 8.2 / 7.8%           | 11.0 / 12.8%     | (in H1)                    | 23%    |
| H4 role/referent   | 5.9% (86)       | 5.7% (50)       | 12.3 / 13.0%         | (in H5)          | (in H5)                    | -      |
| H5 polarity/choice | 4.8% (71)       | 4.2% (37)       | 6.0 / 4.9%           | 3.8 / 3.7%       | 3.0%                       | -      |
| H6 input attrib.   | 4.0% (58)       | 2.4% (21)       | 6.3 / 4.9%           | 1.8 / 0.4%       | 10.6% (RL 13.3%, SFT 5.3%) | -      |
| H7 structure/genre | 2.4% (35)       | 4.6% (40)       | 5.0 / 10.4%          | -                | -                          | 12%    |
| H8 prior templates | 2.7% (40)       | 2.6% (23)       | 5.7 / 6.0%           | (in H1)          | -                          | -      |
| H9 elaboration     | 10.9% (159)     | 10.6% (93)      | 5.6 / 6.8%           | 15.7 / 13.6%     | 12.1%                      | -      |
| H10 future-only    | 1.7% (25)       | 1.7% (15)       | -                    | 3.3 / 3.0%       | 7.2%                       | -      |
| H11 judge error    | 1.0% (15)       | 1.6% (14)       | 2.1 / 3.6%           | -                | 1.1%                       | -      |
| unallocated        | 2.5% (36)       | 3.1% (27)       | -                    | 4.7 / 5.5%       | -                          | -      |

What the shares say:

- **Slot filling (H1-H4) is 70% of RL k0 and 69% of SFT k0 errors.** Each source independently finds
  that the readout gets the frame, genre and slot type right and fills the slot from the LLM prior.
- About 89% of C's hallucinations contradict content visible before the site. They are not wrong
  guesses about future text.
- E finds the same pattern on word tokens with a different lens: the correct value is often in the
  last ~50 tokens.
- **Domain changes the mix more than the arm does.**
  - DailyDialog is heavy on names, relations, persona and attribution.
  - LMSYS is dominated by entity and topic swaps and by technical elaboration.
  - Shard rates: RL k0 0.25 (DailyDialog) vs 0.54-0.56 (LMSYS); SFT k0 0.13-0.15 vs 0.37.

### Best examples (2 per category)

**H1 Sibling substitution**

- chat-lmsys-0009 pos61, RL k0 L20/L36.
  - Prompt: "ShangHai BMG Chemical Co., Ltd ... Room 602, no 291 sikai road shanghai".
  - Readout: "Hengchang Chemical Co." / "located at Room 301, Building 3, No. 36 Donghai Road,
    Qingdao".
  - Both values are stated verbatim in the prompt.
- chat-lmsys-0041 pos55, L44, RL and SFT k0.
  - Prompt: "If a shark is swimming in a pool in my basement…".
  - RL: "Alligators are terrestrial/semi-aquatic animals…". SFT: "Crocodiles generally cannot jump
    high…".
  - All 20 RL k0 readouts on this item that name a sibling animal are judged hallucinated.

**H2 Names**

- chat-dailydialog-0045 pos63, L36 RL k0.
  - Context: `…conversation with "`⟦READ⟧`Mary,"`. Readout: "Carol / Lisa / Karen / Sandy".
  - The site is hallucinated at 5/5 layers for RL k0 and 4/5 for SFT k0.
- chat-dailydialog-0002 pos22, L60 RL k0.
  - Prompt: "Good morning, Miss Li."
  - Readout: "not Miss Chen … not "Miss Qiu." … "Teacher Lin" instead of "Teacher Li."". Four
    bullets give four wrong names.

**H3 Numbers**

- chat-lmsys-0021 pos486, L52 RL k0.
  - Prefix: `\alpha \approx … \approx 1.23`. Readout: "empirical exponent of **1.76**" / "**β ≈
    0.59**".
  - At L36 the four bullets give four different values (spray, T2).
- chat-dailydialog-0026 pos29, L60 SFT k0.
  - The reply declines to give a time without knowing the user's location.
  - Readout: "**7:20 PM**".

**H4 Role/referent binding**

- chat-dailydialog-0004 pos43, L60 RL k0.
  - The user says "I've been looking for you all morning". The readout says the same line in the
    assistant's voice.
  - Nine RL readouts on this item do this.
- chat-dailydialog-0069 pos76, L44, RL and SFT k0.
  - Context: "supportive ways to approach him: ### 1. \*\*". Readout: "Give Her Space, But Stay
    Present".
  - "him" appears 3 tokens earlier.

**H5 Polarity/choice**

- chat-lmsys-0072 pos98, L44 RL k0.
  - Context: "A potato is not a vegetable;". Readout: "it is a type of vegetable".
- chat-dailydialog-0062 pos121, L60 RL k0.
  - Context: "my internal clock is not perfectly synced to real-time seconds,".
  - Readout: "my internal system clock is accurate to within a few seconds".

**H6 Input attribution**

- chat-dailydialog-0037 pos28, L60 RL k0.
  - The prompt "Can you speak French?" is in English. Readout: "但既然您用中文提问" ("since you
    asked in Chinese").
- chat-lmsys-0045 pos65, L44 RL k0.
  - The user writes in all caps with no slang.
  - Readout: "utilizing slang ("gotta," "gonna")" / "all-caps … "YOU HAD NO RIGHT"".
  - The quotes are invented to support a correct emotion label. All 12 hallucinated readouts on this
    item are of this type.

**H7 Structure/genre**

- chat-dailydialog-0051 pos22, L20 RL k0.
  - Context: `The sentence **"`⟦READ⟧`I am confused…`. Readout: "The girl is going home"".
  - The site is hallucinated in 46 of 60 cells across all arms.
- chat-dailydialog-0066 pos266, L20 RL k0.
  - The next heading is "3. **Life Expectancy**". Readout: "3. **Child Mortality Is Plummeting**".
  - Child mortality was item #2.

**H8 Prior templates**

- chat-dailydialog-0041 pos22, L44 RL k0.
  - The reply says "trained by Google". Readout: "As an AI assistant developed by Alibaba Cloud's".
- chat-dailydialog-0013 pos37, L20 RL k0.
  - Context: "I understand your request, but…".
  - Readout: "it involves inappropriate content … vulgar or degrading". A refusal template fires on
    a refusal-shaped prefix.

**H9 Elaboration**

- chat-lmsys-0001 pos316, L36/L60 RL k0.
  - Context: "7-methoxyisoquinoline … derived from isoquinoline,".
  - Readout: "with the addition of a bromine atom at" / "two methoxy groups at positions 1 and 2".
- chat-dailydialog-0062 pos201, L52 RL k0 (SFT the same).
  - Context: "if you are in New York during EDT,". Readout: "UTC-5 hours; subtract 5 hours".
  - The EST default overrides the stated EDT. The site is hallucinated in 46 of 60 cells.

**H10 Future-only**

- chat-lmsys-0010 pos39, L36-L60 RL k0.
  - Context: "GNOME, Xfce, MATE,". Readout: "KDE Plasma".
  - This was flagged, while 15+ other window managers in the same bullets were not.
- chat-lmsys-0038 pos212, L44 RL k0.
  - Readout: "Age': [25, 30, 22], 'Score': [88, 92, 78]".
  - The column is right; the values are arbitrary and cannot be known.

**H11 Judge error**

- chat-dailydialog-0072 pos27, L20 RL k0.
  - Readout: "bad news about your bike". Correct in the user's voice, but flagged.
- chat-lmsys-0019 pos286, L60 RL k0.
  - A riddle whose reply must not say "passport". Readout: "could you please grab my passport
    quickly?"
  - Flagged even though passport is the true hidden referent.

**Tags**

- T1 language drift: chat-lmsys-0027 pos42, RL k0. The reply is in Italian; the readout is "请告诉我
  您需要我协助完成什么任务。"
- T2 spray: chat-lmsys-0006 pos122, RL k0 L36. Each bullet invents a different pair of podcast
  hosts. The reference reply is itself confabulated here (see the judge section below).

### Shifts with AR offset k (k0 to k16)

**The rate falls, much of it by saying less.**

- Hallucinated counts and rates:
  - A: RL 347 to 218, SFT 208 to 84.
  - B rate: RL 0.248 to 0.149, SFT 0.125 to 0.074.
  - D hallucinated/specific: RL 0.54 to 0.42, SFT 0.37 to 0.24.
- Readouts get shorter at the same time: SFT median 192 to 97 chars (C), mean 206 to 113 (D).
- Generic and off-topic labels rise: C SFT generic 23 to 122; A SFT off_topic+generic 48 to 136; B
  off-topic RL 46 to 61, SFT 33 to 69.
- Capitalised words absent from the transcript stay at 0.69-0.72 in every arm (D). Grounding
  precision does not improve; the lens just commits less.

**Strongly k-sensitive** (errors about upcoming specific fillers):

- "not [Name]" correction slots: RL 42 to 3, SFT 22 to 3 (A, regex). This drop is specific to that
  frame; B's broader stock-name proxy stays at 0.34-0.47 of hallucinations in every arm.
- User attribution (H6): RL 24 to 1 (A, regex).
- Persona flips: 12 to 2 (A).
- C scenario swaps: RL 160 to 123, SFT 104 to 41. C technical detail: RL 120 to 93, SFT 67 to 30.

**Persistent or rising:**

- Numbers: A regex RL 44 to 53. B: digit-containing wrong spans in SFT k16 are 0.20, vs ~0.10.
- Structure and headings: A regex 12 to 25.
- C's FLIP and META tags stay flat.
- Slot sites stay hot: B quote/asterisk sites are hallucinated at 22/40 (RL) and 15/36 (SFT) even at
  k16.
- D per-site persistence at L44 (share of k0-hallucinated sites still hallucinated at k16), RL/SFT:
  in-context detail 0.69/0.44, central subject 0.60/0.27, input attribution 0.55/0.50, world
  0.67/0.14, future 0.44/0.20.
- Copy failures (H1, H2) and fabricated evidence (H6) are the stable signal.

**T1 language drift** is an artifact of the RL k0 checkpoint. C counts 296 at k0 and 28 at k16, and
RL k1-k16 sit at 2-5%.

### RL vs SFT (k0)

**The category mix is nearly identical** (H1 47 vs 46%, H3 10 vs 11%, H9 11 vs 11%). RL's excess is
mostly volume:

- about 2x the readout-level rate;
- about 2.3x longer readouts (460-482 vs 192-206 chars) and more bullets (4.1 vs 3.3);
- spray in 46% vs 18%;
- per claim the gap shrinks to 1.5x (B stage-2 false share 0.135 vs 0.089).

**RL-skewed:**

- H6 attribution 4.0% vs 2.4% (D 20 vs 6, C 14 vs 2).
- B's invented specifics, 33 vs 9 (3.7x).
- Whole-topic drift, e.g. A same-frame swaps RL 29% vs SFT 24% (0009 Mafia, 0015 ashtray).
- T1 language drift occurs only in RL.

**SFT-skewed:** H7 structure/format, 4.6% vs 2.4%. This is the only category where SFT's count is at
or above RL's (B 24 vs 20, A 13 vs 9).

### Judge errors and blind spots

1. **Clear false positives are about 1-3% of flags.**
   - Counts: A ~7, B ~15 (9 RL / 6 SFT), C 3-5, D 2 clear plus 5 borderline.
   - Kinds: penalised paraphrases ("decently good", "N/A"); true claims ("large multimodal model");
     questions that offer alternatives scored as claims (A estimates 10-15); a user-voice paraphrase
     ("your bike"); continuations the prefix does not rule out; correct latent content (passport).
2. **Some reference replies are themselves confabulated.** A lens that reads correctly is then
   scored wrong, and one that repeats the falsehood scores "consistent". These items should be
   filtered:
   - dailydialog-0030 (invented Hitchhiker's scene): 41 k0 rows, about 7-9% of shard A;
   - lmsys-0006 (invented podcast hosts): 50 k0 rows;
   - lmsys-0026 ("Amendment 56").
3. **World-knowledge enforcement.** Factual errors the context says nothing about (part of H9: B 16,
   D 32, which is 12% of D) are not context hallucinations.
4. **Future-only penalties.**
   - H10 (59 rows, 7% of D) penalises guesses the lens cannot make.
   - Conversely, E notes that the prediction exemption lets slot substitutions on _upcoming_ tokens
     pass.
5. **False negatives and inconsistency.**
   - "Option N" rewrite headers: 30 RL k0 readouts, only 2 had the span flagged.
   - "Since you mentioned…" attributions are flagged about half the time (7/13 RL, 6/9 SFT).
   - Small persona inventions ("I just came from lunch") are judged leniently.
   - Language drift is never penalised; whole unrelated Chinese bullets pass.
   - For scenario swaps, the off_topic vs hallucinated split flips between layers (lmsys-0008).
   - Entity-swap misses are rare (D 1 of ~90; B 2 of 1024 RL k0).
6. **Structural bias.**
   - Labels are per readout, so one bad bullet out of 4 marks the whole readout. This inflates long
     multi-bullet formats such as RL's.
   - off_topic (including null-site junk, T3) is invisible to the headline.
   - The bench reads only punctuation and whitespace sites. In E these are the worst sites
     (0.32-0.39, vs 0.20 on word tokens).
   - In A, stage-2 claim tallies add nothing over the class label (0 of 1008 consistent rows have
     false > 0).
7. **Source C is internally inconsistent.** It gives ENT as 297 RL k0 in its definition and as 341
   to 277 in its k-shift note. The tables above use 297.

### Eval target per category

| ID  | Target                                                                                                                   |
| --- | ------------------------------------------------------------------------------------------------------------------------ |
| H1  | Minimal-pair N-way subject identification and planted rare entities, scored by regex or lexicon. Chance is 1/N.          |
| H2  | Planted rare-name copy test with name-swap twins. Report the substitution rate beside recall.                            |
| H3  | Planted or computed numbers. Report the unsupported-number rate beside exact-value recall.                               |
| H4  | Speaker-swap and gender/kin minimal pairs, scored with a lexicon confusion matrix.                                       |
| H5  | Negation and A/B/MCQ minimal pairs, scored on P(readout entails the opposite polarity).                                  |
| H6  | Extract attributed and quoted spans and substring-check them against the user turn; add modality and language-ID checks. |
| H7  | Regex `N. **Title**` and list members against the real structure; genre-marker rate.                                     |
| H8  | Planted fictional model/org persona; refusal-lexicon rate on benign refusal-shaped prefixes.                             |
| H9  | Fictional specs planted in the context; a bank of context-overrides-prior facts (EDT-style).                             |
| H10 | Exclude, or score separately against the model's own continuation.                                                       |
| T1  | Per-bullet language-ID drift.                                                                                            |
| T2  | Cross-bullet slot agreement, checked against correctness.                                                                |
| T3  | Null-site confident-claim rate; information gain over a shuffled or mean activation.                                     |

Pair every score with an assert or coverage rate so that a vague lens cannot win.

### Working files

The per-readout labels of shards A-E are under `labels` in
`docs/olens-hallucination-evals-raw.json.gz`, next to the merged ideas, all 12 critiques, every
research cluster's ideas, lessons and sources, and the raw shard taxonomies.

## 3. Ranked evals (critiqued)

All 12 critiqued ideas came back "build-with-fixes"; none was killed. The critiques converged on
four fixes that recur below:

1. **Probe gates.** Replace behavioral gates ("Qwen states the value in 8/10 rollouts") with a
   held-out-template linear probe at the read site and layer.
2. **Byte-identical local windows.** Twins share the last 32+ tokens before the read site, so
   inverting nearby text scores chance.
3. **Paired scoring.** Constant answers, prior guesses and hedges all score 0, and every number is
   reported with a commit or recall floor.
4. **Never train on the eval.** Hold out templates and pools, and freeze probes and site lists.

Evals 1, 2, 5 and the revised version of 12 all want the same planted-slot bank. Build it once.

### 1. Planted-Slot Twins (revised). Fitness 5 to 4. Build.

- **Critique findings.**
  - NetSlot as proposed is climbable by suppression. The current lens already scores about -0.14 (22
    HIT vs 95 invented names of 1,940), below silence.
  - The behavioral gate does not show the value is at the L42 site, and it conflicts with value-free
    tails.
  - CROSS is degenerate: it is just SUB/(P-1) noise.
  - Tails generated greedily under A leak the slot.
  - Qwen splits digits one per token, so treating NEAR numbers as errors is unfair.
- **Design.**
  - **Bank.** About 150 templates, 1-2 slots each, over 6 slot types: first name, greeting vocative
    ("Good morning, Miss {X}"), city, object-in-frame, kin term, nonce code.
    - Pools of 32 values, matched for frequency and token length and balanced for gender and
      culture.
    - Per template: 8 uniform counterbalanced draws plus 1 BLANK variant (slot withheld).
    - Hold out a third of the templates and one slot type, and rotate pools each version.
  - **Tails.** Value-neutral assistant tails of 30-60 tokens, written by hand or by an LLM.
    - Accept a tail only if a text-only guesser (masked context + tail) is at 1/32 chance.
    - Read at 4 punctuation sites in the tail. That gives about 5.4k cells at L42, with a fast set
      of 1.8k.
  - **Extra strata.**
    - On-policy: each variant's own greedy reply, binned by distance from the last mention (1-16,
      17-64, >64).
    - Pre-repetition: the token just before Qwen re-emits the value ("not ", an opening quote). This
      is the easy ceiling.
  - **Gate.** A 32-way probe per site and layer, trained on the training templates. A cell is
    DECODABLE if the probe's top-1 is right; otherwise UNDECODABLE.
- **Metric.**
  - Labels: HIT, SUB (any wrong same-type value, including right+wrong; CROSS merged in), NEAR-SPELL
    (nonce values within edit distance 1), NONE.
  - SUB is subtyped as PRIOR (in the lens's top-30 BLANK marginal), NEIGHBOUR (same
    region/gender/phonetics, or in the probe's top 3) or RANDOM.
  - **Headline:** FAB_dec = SUB/readouts on DECODABLE cells, and FAB_blank = P(any same-type
    specific | BLANK or UNDECODABLE). Lower is better.
  - **Validity:** HIT_dec at least 0.8x the reference and commit rate within ±20%.
  - Secondary: precision vs the 1/32 line, d = P(v_A|A) - P(v_A|B), SUB subtype shares, all per slot
    type and distance bin. Numbers get their own exact/near/far panel, in v2 only.
- **Gaming guards.**
  - Suppression is blocked by the validity floor.
  - Placeholders ("a European city") score NONE and show up as lost HIT.
  - Shotgun lists count as SUB.
  - Script switching is handled by script normalisation plus an LLM same-type extractor that fails
    closed on unparsed non-Latin spans.
  - Template memorisation is handled by held-out templates and slot types.
- **Cost.**
  - Build: about 2.5-3 days, including 1-2 GPU-h of probe forward passes.
  - Per checkpoint: k=4 samples at T=1 plus 1 greedy, about 30-40 min on an H100 (12 min on the fast
    set), plus $1-3 for the extractor.
  - Before the first comparison, pre-register the minimum detectable effect from two seeds of the
    reference.
- **Verdict:** core of the suite. Its U_slot variant (section 1) is the recommended scalar.

### 2. Slot-commit calibration (SCC), from "Unencoded-detail commitment". Fitness 4. Build on the same bank.

- **Critique findings.**
  - The β-averaging and ε-pair certifications are tautological, since the mean is at chance by
    construction.
  - The averaged points are off-manifold.
  - The EraseSub arm is circular (it certifies erasure with the lenses that defined the subspace),
    and it is undefined for multi-digit numbers and multi-token names.
- **Design.**
  - Reuse the planted bank and add a single-token stratum (city, first name, animal, colour,
    element, month; K=12, checked against the tokenizer). Exactly one value of the slot's type per
    context.
  - A per-layer whitened logistic probe, trained on a disjoint half of the templates, gives p_true
    per cell.
  - Bins: ENC (p_true at least 0.8), WEAK (0.2-0.8), UNENC (at most 1.5/K).
  - Draw S=8 samples per cell.
- **Metric.**
  - Utility per readout: +1 right, -2 any other same-type value (in-set and out-of-set reported
    separately), 0 none. Hedges count as committing to the first value.
  - **Headline:** U over the WEAK+UNENC bins. Floors: R_enc (hit rate in ENC) and the
    specific-readout share may not drop more than 0.03 from the reference.
  - Secondary: commit rate per bin (should rise with p_true), entropy of committed values across
    samples, U at distance >8 vs at most 8.
- **Gaming guards.**
  - Synthetic diagnostics (β-mix, Σ_L noise) are never trained on. A gain on them more than 2x the
    natural-arm gain raises a flag.
  - Far-detail suppression is caught by R_enc.
- **Validation gate before trusting it.**
  - SFT vs RL600 must separate on U_weak+unenc, in the direction of the bench's +0.10.
  - U must rank-correlate with the bench rate across the ab4 rungs.
- **Cost.**
  - Build: 2-3 days, shared with eval 1.
  - Per checkpoint: about 230k short generations, 1-1.5 H100-h. Use a fast subset when looping.
- **Verdict:** this directly tests the NLA "490 for 491" mechanism, i.e. whether the lens commits
  when the encoding is weak. Merge it into the eval 1 build.

### 3. Swap-normalized net grounded claims (NGC_h). Fitness 4. Build.

- **Critique findings.**
  - Easy-tier donors land in off_topic, so CC and SNHR are near 0 or unstable.
  - G rewards spraying claims, because readout-level hal(R) saturates.
  - The 4-gram local filter catches only 0.5-1.2% of readout 4-grams.
  - Nearest-neighbour "hard" donors rarely share a topic.
  - The E arm is unscorable with v5c.
  - At 300 sites the noise is about ±0.08 on CC.
- **Design.**
  - Arms:
    - R: the real activation. This is the existing cell, so cached verdicts are reused.
    - Xh: the activation at a matched site in 2 same-prompt Qwen resamples at T=1 (298 generations),
      matched on site kind, nearest relative position and layer.
    - Xe: an other-source donor (release only).
    - M: 30 samples per site kind of the lens on the layer mean, used only as a string bank of the
      lens's defaults.
  - Every arm is written as readouts.jsonl under the TRUE item's site keys and judged by the
    unchanged v5c + verify harness, in separate calls per arm.
  - The R arm gets one extra verify pass with only the ±32-token window as context.
- **Metric.**
  - **Headline:** NGC_h = T_R - T_Xh - λ·F_R per readout, with λ=1 (λ=2 as a sensitivity check). 0
    means vague, below 0 means confabulating from the prior, above 0 means grounded.
  - NGC_nonlocal: the same with T_R replaced by T_R - T_R,local.
  - Guards: F_R per readout, hal(R), assert_share, claims per 100 tokens.
  - Chance line for the existing family: the ungrounded rate (hallucinated + off_topic) and the
    "lucky consistency" rate of the donor arms.
  - Error taxonomy: each R wrong span is labelled prior-default, prompt-level, topic-level or
    activation-specific, by embedding nearness to the M, Xh and Xe readouts (threshold calibrated on
    50 hand pairs).
- **Expected anchors (T-F per readout, before swap subtraction).**
  - λ=1: RL k0 1.76, RL k16 1.59, SFT k0 1.24, SFT k16 0.91.
  - λ=2: RL k16 1.07, RL k8 1.04, SFT k0 0.82, RL k0 0.80, SFT k16 0.73.
- **Gaming guards.**
  - Prompt and genre truths cancel exactly against same-prompt donors.
  - Spraying is caught by claim-level F_R.
  - Paraphrased local inversion is caught by NGC_nonlocal.
  - Fix bullet count and max tokens across variants.
- **Pilot kill criterion.** Run Xh on RL k0, SFT k0 and SFT k16 at L44. If T_X/T_R is roughly
  constant across arms, drop the swap and keep plain T-λF.
- **Cost.**
  - Build: 1-1.5 days.
  - Per checkpoint: all 1,123 sites at L44, about 3.4k generations (15-25 min) and about 9k Flash
    calls ($5-10). Measure the noise floor from two seeds once, and accept only gains above 2 SD.
- **Verdict:** the natural-data confirmation metric in the suite.

### 4. CounterPrior (revised). Fitness 4. Build.

- **Critique findings.**
  - The behavioral gate does not show the value is at L42.
  - TRAP is often faithful. At the cited EDT site (dailydialog-0062 pos180) the J-lens ranks EST
    above EDT at L60, and 北京时间 and 在美国 are in its L42/L52 top 10. So Qwen itself carries the
    prior. This matches the knowledge-conflict literature: parametric and in-context answers compete
    in mid layers.
  - Identity is often really Qwen's.
  - Near-miss spellings are invalid at L42.
  - Low mention rates leave too little power.
- **Design.**
  - **Bank.** 360 triplets (ORIGINAL / EDITED / NONE):
    - 200 story-canon counterfactuals;
    - 100 true but non-default facts (EDT=UTC-4, southern-hemisphere seasons, Feb 29);
    - a 60-item identity diagnostic.
  - **Layout.** The value is stated once, then a shared value-neutral span of 150-250 tokens, then a
    fixed suffix ("Briefly describe the setting above."), then the chat tail.
  - **Gates.**
    - Behavioral: the edited value is used in at least 8/10 rollouts.
    - Representational: an item-CV probe gives p at least 0.8 at the site, or the J-lens rank of the
      value is at most 50 in EDITED and above 500 in ORIGINAL.
    - Keep 4 certified sites per item: 2 mid-span, 2 suffix/tail. Expect 40-60% of sites to survive.
    - Optional: patching on a 50-item subset.
  - **Readouts.** k=3 samples. Controls: shuffled-item activation and mean activation.
- **Metric.**
  - **Prior-override** = P(TRAP and not BOTH | EDITED, mention) - P(TRAP | NONE, mention).
  - Reported beside it:
    - TARGET recall (TARGET or BOTH | EDITED), so silence cannot win;
    - ORIGINAL-side accuracy, which catches bias toward the unusual value;
    - the false-TARGET rate under the shuffled control;
    - pair accuracy qAcc.
  - Scoring: alias regex, then a 5-way MC judge that sees only the readout and the two candidates.
  - SE about 2-3pp, so changes of about 5pp or more are detectable.
- **Gaming guards.**
  - Echo is impossible inside the shared span.
  - Prior suppression is caught by ORIGINAL accuracy.
  - Silence is caught by TARGET recall.
  - A "this is a story, so pick the weird value" bias is caught by the ORIGINAL arm.
- **Bonus.** Certify existing WSB sibling substitutions (EDT to UTC-5, Google to Alibaba) the same
  way and split them into "Qwen-shared prior" and "lens-decoder prior". Only the second kind is a
  lens bug.
- **Cost.**
  - Build: about 3 days.
  - Per checkpoint: about 22k readouts, under 1 GPU-h, $1-3.
- **Verdict:** keep. It is the direct test of the AR-size / prior story.

### 5. Twin likelihood test (TLT), from "Activation likelihood tests". Fitness 4. Build on the bank.

- **Critique findings.**
  - Random-swap PMI measures whether a span is on topic, not whether its value is pinned down; most
    wrong spans are on-topic siblings.
  - The mean-vector null is off-distribution.
  - "LLM-fixed" spans are not ground truth.
  - Log-prob scales move with RL entropy.
  - POPE probes are off-format.
- **Design.**
  - Context twins differing only in v_A/v_B, read at the same site.
  - DD = [lp(r_A|h_A) - lp(r_B|h_A)] + [lp(r_B|h_B) - lp(r_A|h_B)], over the edited span only, with
    r rendered from 3 canonical templates in the checkpoint's own output format.
  - Probe-gated as in eval 1. Distance bins: at most 32, 33-256, >256 tokens or a previous turn.
- **Metric.**
  - **Headline:** out-of-window pair accuracy, mean 1[DD > 0] at distance >32, with a paired sign
    test across checkpoints.
  - Secondary: DD z-scored by the swap-vs-swap null SD; sampled hit rate and sibling rate from 8
    samples; decoding gap = forced-choice accuracy - sampled hit rate.
  - The decoding gap separates two failures: the lens ranks the truth higher but samples wrong
    (decoding), or the activation does not pin the value (representation).
- **Diagnostic for existing errors.**
  - Compute PMI against a generic null (other conversations) and a topic null (same conversation, at
    least 64 tokens away).
  - Sort each wrong span into prior confabulation / topic fill-in / site-grounded.
  - Cross-tab against the HB 2.0 probe confirmations.
- **Gaming guards.**
  - THRONE (2024) showed polled and free-form behavior can come apart, so this is never a reward and
    never used alone.
  - The at-most-32 bin is excluded from the headline.
- **Cost.**
  - Build: 2-3 days, shared with eval 1.
  - Per checkpoint: 20-40 H100-min, $0.
- **Verdict:** a cheap, judge-free confirmation of eval 1. Its main value is the decoding gap.

### 6. Computed-value precision (CVP), from "Off-path value confabulation". Fitness 5 to 3. Build small.

- **Critique findings.**
  - Regex value precision is 0.030-0.038 for every arm, with SE 0.005, so it does not separate
    lenses that the judged chain family does separate (ref pass rate 0.44 vs prompt-only 0.025).
  - The "95.6% vs 17%" headroom is verbosity. The OFF mass is dominated by 1, 2, 5, 3 and 10, i.e.
    step numbering.
  - Brew has no confabulation ground truth: in 100 of 100 items every off-trajectory colour is in
    the prompt's rule table.
  - Qwen's own answers are near-misses (4808 for 4816) or sign drops.
  - Token lenses cannot emit "51".
- **Design.**
  - Reuse the chain and arithmetic families' free-recall judge output (up to 3 values presented as
    computed, verbatim-checked). No regex.
  - Add pre-seed readouts on 120 chain items at layers 36/42/48, and layer 42 on the 596 arithmetic
    items.
  - About 270 seed-swap twins whose trajectories share no value, gated at 10/10 correct.
- **Metric.**
  - Labels via the computable set C(p) at digit granularity: ON_int, ON_ans, PRECOG, TWIN,
    ECHO_AS_RESULT, NEAR (tracked, not an error), DEFAULT {0, 1, 10, 12, 42, 100, 1000}, OFF.
  - **CAR** = (OFF + DEFAULT + ECHO_AS_RESULT + PRECOG + TWIN) / named values. Lower is better.
  - Reported beside: **GVR** (cells whose top value is ON_int), **specificity** = P(own trajectory)
    - P(twin trajectory), and the pre-seed confabulation rate.
  - Accept a checkpoint if CAR drops by more than 2 SE and GVR and specificity do not drop.
- **Gaming guards.**
  - Echo-as-result counts as an error.
  - Answer-only decoding is separated by ON_int vs ON_ans.
  - Template priors hit both twins equally, so specificity cancels them.
  - Only PRECOG, pre-seed and twin specificity are immune to "invert the prompt and compute it
    yourself".
- **Cost.** Build 1-1.5 days. A few dollars of Flash and a few GPU-minutes per checkpoint.
- **Verdict:** keep as the H3 number member of the suite. Report CAR and specificity, not VP.

### 7. Mistake taxonomy with automatic attribution. Fitness 3. Build (diagnostic plus one target).

- **Critique findings.**
  - "Lens-defect rate" is roughly the hallucination rate again.
  - "Model-entertained" evidence (32 subject continuations, J-lens top-20 x 5 layers) has no placebo
    base rate and is close to the LLM prior.
  - Labels are not exclusive: RL averages 2.43 wrong spans per hallucinated readout.
  - 15 of 149 items hold 29-38% of all spans.
  - A 2,400-cell table is mostly empty.
- **Part A (diagnostic, all 12 arms).**
  - Six independent binary detectors, rules first with a Flash fallback, each validated on 150 hand
    labels at precision of at least 0.85:
    - SUBST-CTX: a same-type counterpart in the context before the site;
    - NUMBER;
    - PERSONA;
    - LANG: script differs from the context's;
    - FUTURE: unsupported before the site, but the topic is realised after it;
    - BOILERPLATE.
  - Attribution panel on 300 spans per arm. Resample 8x under the true activation, the layer mean,
    and 3 other items:
    - PRIOR-LEAK: p_swap at least 0.5·p_true;
    - ACTIVATION-DRIVEN: p_true at least 0.25 and p_swap at most 0.05;
    - otherwise UNSTABLE.
    - Plus a placebo of transplanted spans to measure the matcher's false-match rate.
- **Part B (target).**
  - **SUBST-CTX rate** = substitutions / verifiable claims about entities that have a counterpart in
    the context. Example: the context says "Bobby and his wife, Evelyn" and the readout says "Edward
    and Doris".
  - The truth is in the prompt, so this error is unambiguously the lens's fault. It needs no new
    judge.
  - It counts as better only if it falls by more than its paired 95% CI and true-claim yield drops
    by at most 5%.
- **Gaming guards.**
  - Hedging and terseness show up through normalising per verifiable claim with yield as the guard.
  - Imitating the J-lens no longer helps, because the J-lens is dropped as evidence.
- **Cost.**
  - Build: 3-4 days total (Part A detectors 2 days, ~$20 Flash; swap panel ~86k generations, 2-3
    GPU-h).
  - Per new arm: ~$5 and ~15 GPU-min.
  - Deferred: AR-paid and CHIVE-style source deletion.
- **Verdict:** this is the categorisation Robbie asked for. Re-seed the label set on a descriptive
  O-lens arm before trusting its frequencies, since the offset-k arms write continuations.

### 8. Relation minimal pairs (RMP). Fitness 4 to 3. Build lean.

- **Critique findings.**
  - The rollout gate does not show the relation is at a shared teacher-forced site.
  - The commit rate is low (readouts continue the local text), so S sits near 0 for every lens, with
    about 50 effective templates per type.
  - The speaker regex is beaten by writing in the third person.
  - "Unspecified" twins are not ground truth, because Qwen infers defaults.
  - The role type duplicates WSB role_bound_association.
  - FLIP is only about 5% of tagged sites (13 of 252), against ENT at about 40%.
- **Design.**
  - About 300 pairs: polarity (~80), outcome (~70), binding with a same-class distractor and
    counterbalanced mention order (~110), speaker (~40). Plus about 100 unspecified twins.
  - 2-3 read sites per pair, chosen where Qwen's next-token JS divergence between twins is largest.
  - Gates, all on the whole pair:
    - JS divergence above a threshold;
    - a leave-one-template-out probe with AUROC above 0.9;
    - a text-only leak check at chance;
    - unspecified twins kept only if Qwen's forced choice is in [0.35, 0.65].
  - Readouts: k=4.
  - Judge: Flash sees only the readout and the two shuffled propositions. Calibrated to κ > 0.85 on
    200 synthetic readouts.
- **Metric.**
  - **RMP** = mean over pairs of [both twins committed-correct] - [both committed-flipped]. Constant,
    prior-based, hedging and random lenses all score about 0.
  - Beside it: commit rate and flip|commit.
  - Guardrail: fabricated certainty on the unspecified twins.
  - Dev/test template split 60/40.
- **Cost.** 2.5-3 days to build. Per checkpoint ~6k generations (~8-10 min) and ~$3.
- **Verdict:** secondary. Its binding type covers the ENT mass. Polarity covers H5, which is small.

### 9. CTNS-slots, from "Claim ledger CTNS". Fitness 4 to 3. Build as a companion score only.

- **Critique findings.**
  - Across the 12 arms, corr(length, CTNS) = -0.87, so the shortest arm wins.
  - At λ=3, Spearman is about -0.98 with the existing rate, so it adds little.
  - At 300 sites the paired SD is 0.15-0.18, so only differences of about 0.35 are detectable.
  - Disputed claims (2,709 against 5,405 false for RL k0) would swing CTNS by about 1.4 per cell.
  - The public rows have counts only, so none of the anti-gaming filters were ever applied.
- **Design.**
  - Exactly 4 bullets of at most 25 tokens per readout, enforced at decode time for every arm,
    including a prompt-only baseline.
  - A new verify prompt: per claim, status / source / tier; hedged specifics count as asserted.
  - Per-bullet value: +1 for a novel true ENTITY/DETAIL claim; +0.25 if it is guessable (entailed by
    any of 16 prompt-only samples); 0 for echo, THEME or unverifiable; -λ for false. Clipped to [-λ, 1].
  - FUTURE claims go to a separate prediction ledger, scored against the actual continuation.
- **Metric.**
  - The paired Δ vs the reference on all 1,123 sites at L44, 2 samples each (SD about 0.08).
  - It counts only if it wins with disputed claims treated as false and as unverifiable.
  - Show (T_novel, F) and the curve over λ in {1, 3, 9}, never a single λ.
  - Select on even items, report on odd items.
- **Cost.**
  - Build: 3 days, including ~18k prompt-only generations.
  - Per checkpoint: ~9k Flash calls, $10-15.
- **Verdict:** lower priority than NGC_h, which does the same job with a chance line.

### 10. Spray decomposition, from "Spray, sample consistency and claim confidence". Fitness 4 to 3.

- **Critique findings.**
  - The judge-free conflict proxy has Spearman 0.96 with the judged rate across arms, so it
    duplicates the family.
  - Atoms cover only 52-55% of wrong spans; the rest are persona and role errors.
  - Entropy collapse games every consistency metric.
  - ±1 neighbour sites reward text inversion.
  - Confidence tags are a training project, not an eval.
- **Design.**
  - Per cell, K=8 samples on the true activation and K=8 on a donor activation (same stratum, site
    kind and layer).
  - Each false atom is classed as singleton (at most 1 of 8 own samples), stable-prior (at least 4
    of 8 own and at least 2 of 8 donor) or stable-specific.
- **Metric.**
  - Bank A, 300 WSB sites: the categorisation of the published wrong spans. This answers whether
    RL's spray is sampling noise or a systematic prior.
  - Bank B, the planted banks: the headline.
    - Consensus accuracy at coverage (commit if at least 4 of 8 agree).
    - Stable-wrong per slot-cell.
    - RetainedTrue@P90 with split-half conformal.
    - Always shown beside: diversity, a prompt-only floor and a donor-only floor.
- **RL monitor.** Every 25 steps on 512 held-out activations: OOC-atom rate, slot conflict rate,
  entropy, distinct-2. Alarm when OOC rises more than 0.03 over SFT or entropy drops more than 30%.
- **Cost.**
  - Build: 1-1.5 days.
  - Per checkpoint: 25-35 min, ~$0.
- **Verdict:** keep the monitor (it is in the suite). The categorisation is a one-off.

### 11. Planted-list probe, from "Structure-copy and format-invariance". Fitness 4 to 3. Build cheap.

- **Critique findings.**
  - The regex reads the lens's own list-writing as claims about the reply. 17-22 past-structure
    claims per arm were all wrong, because they are regenerations.
  - Heading claims sit mostly in replies with no headings at all (500 of 1,153 for RL k0).
  - Numbered claims appear in about 3% of cells.
  - Part (a) duplicates v5c's misquote rule.
  - Part (b) has no ground truth.
- **Design.**
  - 360 lists of 6-8 nouns drawn from a 600-noun pool, given in the user prompt.
  - Instructions, a third each: "list with a note each", "alphabetical", "reverse".
  - Formats crossed in: numbered, bullets, comma prose, JSON array, markdown table.
  - Read at the boundary token before item k+1. About 1,800 cells at L42.
- **Metric.**
  - **Insertion rate** = readouts containing a pool noun not in this list / readouts with any pool
    noun.
  - **Next-item hit**: item k+1 is among the first 2 pool nouns, with at most 3 distinct pool nouns
    in the cell.
  - Commit share and index binding.
  - Slices by format, order rule and position. The alphabetical and reverse slices are the
    anti-inversion check.
- **Gaming guards.** Anti-dump rule; prompt-only and random-activation baselines in the same table.
- **Cost.** About 1 day. A few minutes per checkpoint, $0.
- **Validate first:** the insertion rate should rank-correlate with WSB across the 12 arms.
- **Verdict:** optional fifth member. It is the cheapest dense test of the H7 and format-site story.

### 12. Input-attribution tag plus PlantedSlot user facts. Fitness 4 to 3. Fold into eval 1.

- **Critique findings.**
  - About 94% of quoted spans of 3+ words are absent from the transcript in every arm, but they are
    the lens's own example content (idioms, SQL, example questions), not attributions to the user.
  - The idea claimed 22% (RL) vs 7.5% (SFT) of readouts contain an absent quote; the critique
    measured 4.0% of RL k0 bullets. The units differ, and both reduce mostly to style and length: RL
    bullets average 114 chars vs 59 for SFT, so the per-character gap is about 1.4x, not 3x.
  - Attribution frames are rare: 54 of 22,453 RL k0 bullets. v5c already flags 29 of them, and the
    rest mostly look true.
  - Modality claims have polarity. "I can't see the image" is often a correct read of Qwen's state.
- **Design.**
  - (a) A diagnostic tag over existing dumps: attribution-framed content that is not a substring of
    the user turn, plus affirmative modality claims, classified by a small judge.
  - (b) A user-stated-fact stratum added to the eval 1 bank: home city, pet, profession, sibling
    count, allergy, weekday, programming language. Twins are prior-balanced.
    - Read at least 24 tokens after the last mention, with no mention in the next 16 reply tokens.
    - Probe-gated.
- **Metric.** SlotPrecision against SlotMention, as a precision/commit pair; twin consistency.
- **Cost.** The tag takes 1-2 h and under $1. The stratum adds about 0.5 day to eval 1.
- **Verdict:** no standalone eval. GapTwins is deferred until a probe shows "a gap" is decodable at
  shared sites.

### Dropped or cut

- **minimal-pairs DiffRead:** its inputs are off-distribution, and its proposer rated it 2.
- **CVP:** brew colours (no ground truth); covert, date, unit and letter tasks; the probe skyline;
  items Qwen gets wrong; regex value precision.
- **NGC:** the E arm (v5c calls earlier-response content consistent by construction); SNHR (unstable
  denominator); Jaccard prior-share (barely fires at ~1% 4-gram overlap); the nearest-neighbour hard
  tier.
- **SCC:** EraseSub as a hill-climb metric (circular, undefined for numbers); ε-preimage pairs
  (probe accuracy is undefined for one pair).
- **CounterPrior:** near-miss spellings (invalid at L42); counter-stereotype stratum (covered by
  role_bound_association).
- **RMP:** the role type (duplicates role_bound); the first-person speaker regex as a headline.
- **TLT:** POPE '- {o}' probes (off-format, duplicate jlens_concept_pr); summing over tokens after
  the edit.
- **Input attribution:** the bare false-quote metric; the ambiguous-deixis stress split (it punishes
  correct reads).
- **Spray:** ±1 neighbour agreement; confidence tags; NNLS/ΔFVE confidence.
- **Structure:** the markdown-tree regex; cross-format topic agreement.
- **Taxonomy:** AR-paid, CHIVE source deletion and J-lens top-20 evidence in v1.
- **Planted slots:** pronoun gender, dates and chemicals as slot types; CROSS as its own class; all
  behavioral 8/10 gates (replaced by probe gates).

## 4. Remaining uncritiqued ideas

Fitness is the merge-stage self-rating, before any critique.

- **N-way subject identification (4).** Same template, N=4 sibling subjects or an identical 24-48
  token suffix; keyword or forced-choice scoring against chance 1/N. Targets H1 scenario swaps.
- **HalluTagger + claim-NLI (4).** Distil v5c into a ModernBERT span tagger trained on 36.9k wrong
  spans, plus AlignScore/MiniCheck. Gate: κ at least 0.6 and arm Spearman at least 0.95. An RL-loop
  scorer with 5% big-judge audits.
- **Planted concept injection, PCI (4).** Mean-diff concept vectors at Qwen's own detection
  threshold α50. Score HIT - FA, the sibling rate and misattribution ("the user mentions"). Never
  use AR(phrase) vectors.
- **Hidden-state false-alarm panel (4).** Secrets, personas and constraints planted in the system
  prompt, with teacher-forced identical replies under crossed and null twins. Steering and
  AuditBench-27B organisms as tiers 2 and 3.
- **Latent decision and pick readout (4).** "Think of an X" twins, MCQ choices and Turpin-style
  cues. Twin discrimination, implausible-pick rate, and fabricated influence on NO-CUE items.
- **Closed-world fact tables (4).** Synthetic worlds with invented entities and 0-16 distractor
  facts. Every claim is decidable. Area under the fabrication-vs-dilution curve.
- **Phantom latents across WSB families (4).** Pool the absent-latent controls (buggy_code clean
  twins, dont_think, brew stir, pre-seed) into macro balanced accuracy. Use
  `true_intermediates_aliases` in wrong_bridge_true.
- **Judge calibration suite (3).** Plant one typed error or a benign control edit into clean
  readouts. Measures per-type recall, false-positive rate, test-retest noise and minimum detectable
  effect for each judge pin.
- **AR reward audit (3).** Does frozen AR + NNLS score the corrected filler above the wrong one?
  truth_pref, confident_wrong_bonus, and the gap between the reward AR and a held-out AR.
- **Null-input floor and prior guardrails (3).** Null activations (BOS, lorem ipsum, means) define a
  default-template fingerprint. Track template leakage, off-language rate, boilerplate, the
  default-number index, and NonExistent false acceptance.
- **Continuation support (3).** 32 Qwen rollouts per site turn "reasonable continuation" into a
  computed support score. Unsupported-prediction rate and commitment calibration by determinacy.
- **Model-belief faithfulness (3).** Rigged tools, misread numbers, SDF-implanted facts and
  residual-patched chimeras. Labels FAITHFUL / CONTEXT / PRIOR / CONFAB, which yields v5c's
  false-positive rate on faithful reads of model errors.
- **SwapFollow (3).** Behavior-validated latent swaps (spider to ant) upstream of L42. The STALE
  rate catches lenses that report what the text implies after the state has changed.

## 5. Key lessons from the literature

- **Precision-only metrics reward silence.**
  - In the ab4 ablation, AR-output injection hallucinated less because readouts were "emptier, not
    better". In the offset study, claims per readout fell from 4.86 to 2.99.
  - TruthfulQA (2021) scores "No comment" as truthful. FActScore (2023) ignores recall. Core (2024)
    pushed FActScore above 80% with trivial claims.
  - Fix: a fixed penalty set in advance, as in SimpleQA (2024) and Kalai et al. (2025): +1 right,
    -t/(1-t) wrong, 0 abstain. Or a capped recall term, as in F1@K (Wei et al., 2024).
- **Normalise for length.**
  - Rate vs length r=0.92; equal wrong spans per character across RL and SFT.
  - Singhal et al. (2023) and FLAME (2024) found the same length effect under RLHF.
- **Text inversion is the default shortcut.**
  - The O-lens AR was trained on 1-32-token spans, and vec2text (2023) recovers 92% of 32-token
    inputs exactly.
  - Jakkli et al. (2025) spliced identical tokens into different contexts and got the same verdict
    in 53 of 54 cases.
  - Ground truth must sit outside the last ~32-40 tokens, or behind a byte-identical window.
- **The reader's prior is the second shortcut.**
  - A no-activation AO matched the AO on censorship questions (Jakkli et al., 2025). CHIVE (2026)
    found no tool uplift over a transcript-only predictor.
  - Use uniform pools, as in PersonaQA-Shuffled (Karvonen et al., 2025) and PersonaQA-Fantasy (Li et
    al., 2025).
  - Report shuffled-activation and prompt-only floors.
- **Gate ground truth on representation, not behavior.**
  - HB 2.0: 73% of O-lens sibling substitutions are confirmed by probes or another lens.
  - Parametric and in-context answers coexist in mid layers (Longpre et al., 2021; Yu, Merullo &
    Pavlick, 2023; Ortu et al., 2024).
  - A probe skyline separates "the information is absent" from "the lens failed" (Chalnev, 2026:
    probe 100% vs decoder 52%).
- **Confabulations are thematic near-misses** (NLA, 2026). The reward pays a close guess ("490")
  over omission, and FVE has no abstain option.
  - SFT targets that state details the activation does not encode teach guessing (Gekhman et al.,
    2024; FLAME, 2024; HallE-Control, 2023).
- **RL trades hallucination for headline.**
  - +0.10 hallucination costs about 0.004 of the 24-family headline. Most of the RL gain comes in
    the first 150 steps.
  - Pick checkpoints on the proxy/gold frontier (Gao et al., 2022), not at the last step.
- **Do not train against the instrument.** LoRAcles (2026) RL mode-collapsed and its judges rewarded
  parroting. Explanation metrics get gamed (Hsia et al., 2024). Keep a held-out judge, a held-out AR
  and a held-out template split.
- **Pin the judge and pair the comparison.**
  - Judge choice moves the rate by 0.12. A one-pass judge agreed with v5c at κ about 0.44.
  - Report Scott's π or κ (Thakur et al., 2024).
  - Small checkers such as MiniCheck (2024) and HHEM-2.1 (2024) score "supported", while v5c counts
    only "contradicted". Restrict them to claims about the prefix.
- **Synthetic-error scorers overfit their transforms.** FactCC fails on FRANK (SummaC, 2022).
  Meta-evaluate on real lens errors and planted corruptions (CapMAS, 2024; FaithBench, 2024).
- **Sampling separates confabulation from systematic error** (Farquhar et al., 2024).
  - Consensus over 10 samples took Taboo precision from 46.6% to 94.3% at 19.4% coverage (Karvonen,
    2026).
  - Sample frequency was the best claim scorer for conformal factuality (Mohri & Hashimoto, 2024).
  - Entropy collapse under RL voids all of this, so log diversity.
- **Null controls catch false alarms.**
  - Lindsey (2025): 0/100 false positives on no-injection trials.
  - Cywinski et al. (2025): auditor success on the base model fell from 84% to 19%.
  - Introspection Adapters (2026) report training-distribution behaviors on models that lack them.
- **Polled likelihood and free-form behavior can anti-correlate** (THRONE, 2024). Pair any
  likelihood eval with sampled hit rates.
- **Read site matters.**
  - WSB reads only punctuation and whitespace sites, the worst kind in the dashboard (0.32-0.39 vs
    0.20 on word tokens).
  - At delimiters oracle phrases become commentary (Gurnee et al., 2026).
  - Later bullets hallucinate more (CHAIR, 2018; AMBER, 2023).
- **Data quirks to fix before building on the bench.**
  - `wrong_bridge_true/items.json` `intermediates_aliases` holds the WRONG bridge's aliases; use
    `true_intermediates_aliases`.
  - The prompt-only floor is 0.07 in the post's chart but 0.53 in baselines/README.md; reconcile
    before relying on it.
  - Filter the three items whose reference reply is confabulated (dailydialog-0030, lmsys-0006,
    lmsys-0026).
  - CJK drift is never penalised.

### Sources

**Lens and activation-reading papers**

- Gurnee et al., Verbalizable Representations Form a Global Workspace in Language Models (2026)
- Blank, Bhatia, Ong, Nanda, WorkspaceBench (2026), plus the bench repo's hallucination family
  README, judge prompt v5c-chat and plans 0004/0010 (2026)
- Anthropic, Natural Language Autoencoders (2026); kitft/natural_language_autoencoders repo (2026)
- Karvonen et al., Activation Oracles (2025)
- Jakkli, Rajamanoharan, Nanda, Current Activation Oracles Are Hard to Use (2025; ICML 2026 version)
- Chalnev, Cycle-Consistent Activation Oracles (2026)
- Bauer, De Schamphelaere, Karvonen, Luick, Nanda, Building Better Activation Oracles (2026)
- Luick, Reading Between the Layers (2026)
- Bersia & Atmyre, When Activation Oracles Learn Not to Read (2026)
- Karvonen, 3 Tips to Improve Activation Oracle Results (2026)
- Li, Ceballos Arroyo, Rogers, Saphra, Wallace, Do Natural Language Descriptions of Model
  Activations Convey Privileged Information? (2025)
- Huang et al., Predictive Concept Decoders (2025)
- Huang, Panwar, Goyal, Hahn, InversionView (2024)
- Ghandeharioun et al., Patchscopes (2024)
- Chen, Vondrick, Mao, SelfIE (2024)
- Pan, Chen, Steinhardt, LatentQA (2024)
- Lindsey, Emergent Introspective Awareness in LLMs (2025)
- De Schamphelaere, Bauer, Nanda, Ong, LoRAcles (2026)
- Shenoy et al., Introspection Adapters (2026)
- Sheshadri et al., AuditBench (2026)
- Karvonen, Ong, Kantamneni, Marks, CHIVE (2026)

**Hallucination and factuality**

- Lin, Hilton, Evans, TruthfulQA (2021)
- Li et al., HaluEval (2023)
- Min et al., FActScore (2023)
- Song, Kim, Iyyer, VeriScore (2024)
- Wei et al., Long-form factuality / SAFE / F1@K (2024)
- Wei et al., SimpleQA (2024)
- Manakul, Liusie, Gales, SelfCheckGPT (2023)
- Kuhn, Gal, Farquhar, Semantic Uncertainty (2023)
- Farquhar, Kossen, Kuhn, Gal, Semantic Entropy (2024)
- Bang et al., HalluLens (2025)
- Chen et al., FELM (2023)
- Kalai, Nachum, Vempala, Zhang, Why Language Models Hallucinate (2025)
- Jiang et al., Core (2024)
- Wanner et al., A Closer Look at Claim Decomposition (2024)
- Gunjal & Durrett, Molecular Facts (2024)
- Niu et al., RAGTruth (2024)
- Ravi et al., Lynx / HaluBench (2024)
- Bao et al., FaithBench (2024)
- Tang, Laban, Durrett, MiniCheck (2024)
- Vectara HHEM-2.1-Open (2024)
- Zheng et al., MT-Bench / LLM-as-a-Judge (2023)
- Thakur et al., Judging the Judges (2024)

**Summarisation faithfulness**

- Kryscinski et al., FactCC (2020)
- Wang, Cho, Lewis, QAGS (2020)
- Maynez et al., On Faithfulness and Factuality in Abstractive Summarization (2020)
- Pagnoni, Balachandran, Tsvetkov, FRANK (2021)
- Laban et al., SummaC (2022)
- Zha et al., AlignScore (2023)
- Laban et al., SummEdits (2023)
- Falke et al., Ranking Generated Summaries by Correctness (2019)

**Vision captioning**

- Rohrbach et al., CHAIR (2018)
- Li et al., POPE (2023)
- Wang et al., AMBER (2023)
- Guan et al., HallusionBench (2023)
- Sun et al., LLaVA-RLHF / MMHal-Bench (2023)
- Zhai et al., HallE-Control (2023)
- Lee et al., CapMAS (2024)
- Kaul et al., THRONE (2024)
- Petryk et al., ALOHa (2024)

**Autointerp and explanation faithfulness**

- Bills et al., Language Models Can Explain Neurons (2023)
- Paulo, Mallen, Juang, Belrose, Automatically Interpreting Millions of Features (2024)
- Huang, Geiger, D'Oosterlinck, Wu, Potts, Rigorously Assessing NL Explanations of Neurons (2023)
- Gao et al., Scaling and Evaluating Sparse Autoencoders (2024)
- Chen et al., Counterfactual Simulatability (2023)
- Turpin, Michael, Perez, Bowman, Language Models Don't Always Say What They Think (2023)
- Lanham et al., Measuring Faithfulness in Chain-of-Thought (2023)
- Atanasova et al., Faithfulness Tests for NL Explanations (2023)
- Siegel et al., The Probabilities Also Matter (2024)
- Matton et al., Walk the Talk? (2025)
- Hase & Potts, Counterfactual Simulation Training (2026)
- Ribeiro et al., CheckList (2020)
- Gardner et al., Contrast Sets (2020)
- Adebayo et al., Sanity Checks for Saliency Maps (2018)
- Morris et al., vec2text (2023)
- Belrose et al., Tuned Lens (2023)
- Hsia et al., Goodhart's Law Applies to NLP's Explanation Benchmarks (2024)

**Calibration, abstention and RL**

- Mohri & Hashimoto, Conformal Factuality (2024)
- Zhang et al., R-Tuning (2023)
- Traub et al., AUGRC (2024)
- Damani et al., RLCR (2025)
- Kadavath et al., Language Models (Mostly) Know What They Know (2022)
- Kang et al., Unfamiliar Finetuning Examples Control How LMs Hallucinate (2024)
- Band et al., Linguistic Calibration (2024)
- Tian et al., FactTune (2023)
- Lin et al., FLAME (2024)
- Chen et al., Learning to Reason for Factuality (2025)
- Gao, Schulman, Hilton, Reward Model Overoptimization (2022)
- Uesato et al., Process vs Outcome Feedback (2022)
- Lightman et al., Let's Verify Step by Step (2023)
- Singhal et al., Length Correlations in RLHF (2023)
- Gekhman et al., Fine-Tuning on New Knowledge (2024)
- Liu et al., Dr. GRPO (2025)

**Knowledge conflict**

- Longpre et al., Entity-Based Knowledge Conflicts (2021)
- Yu, Merullo, Pavlick, Characterizing Mechanisms for Factual Recall (2023)
- Ortu et al., Competition of Mechanisms (2024)

**Model organisms**

- Cywinski et al., Eliciting Secret Knowledge (2025)
- Marks, Treutlein et al., Auditing Language Models for Hidden Objectives (2025)
- Hubinger et al., Sleeper Agents (2024)
- MacDiarmid et al., Simple Probes Can Catch Sleeper Agents (2024)
- Wang et al., Synthetic Document Finetuning (2025)
- Slocum et al., Believe It or Not (2025)
- Minder et al., Narrow Finetuning Leaves Readable Traces (2025)
- agu18dec/auditbench-model-orgs (2026)

**In-house**

- research-log-o-len-catalog.md (Slack research log, 2026-06 to 2026-10)
- agu18dec olens_ablations, ab4 and offset READMEs and CSVs (2026)
- agu18dec mOre-lenses README (2026)
- workspacebench-notes.md and oracle-lens-workspace-notes.md (2026)
- oracle-lens-reconstruction docs/research and plan.md (2026)
- jlens-dashboard docs/prose-lenses.md and docs/model-organisms.md (2026)
- The 12 judged arms in jlens-dashboard data/readouts/wsbench-hallucination (2026)
