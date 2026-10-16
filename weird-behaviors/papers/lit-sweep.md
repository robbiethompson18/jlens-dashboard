# Literature sweep: API-elicitable weird LLM behaviors

Scope: papers documenting surprising/confusing LLM behaviors that can be elicited with inference-only
API access. Prioritized 2025-2026 arXiv, plus the canonical older papers each line descends from.
Excluded (covered by other agents): tokenization/counting, arithmetic, format instruction-following,
thinking-mode mechanics, multilingual leakage, refusals, identity/hallucination, tool calling,
degeneration/long context, code gen, random-number bias, CoT faithfulness/sycophancy, story
imprinting (arXiv 2609.10883), and harmful-content jailbreaks.

Per paper: citation (year) / finding / inference-only probe design / **Ran?** on Qwen 3.6 27B.
Replication results are in `findings/13-lit-replications.md`. arXiv IDs marked (unverified) were
seen only as URLs in search results, not on the abstract page.

## 1. Glitch / under-trained tokens

- **SolidGoldMagikarp** — Rumbelow & Watkins (2023). "SolidGoldMagikarp (plus, prompt generation)", LessWrong/AF, Feb 2023. Finding: ~140 GPT-2/3 tokens near the embedding centroid make the model refuse, hallucinate or substitute other words when asked to repeat them. Probe: "Repeat exactly: `<tok>`", score exact match. **Ran (as part of glitch set).**
- **Fishing for Magikarp** — Land & Bartolo (2024). "Fishing for Magikarp: Automatically Detecting Under-trained Tokens in Large Language Models". arXiv:2405.05417, EMNLP 2024. Finding: unembedding-norm indicators + repeat-prompt verification find under-trained tokens in every open family; Qwen's include `useRalative`, ` ForCanBeConvertedToForeach`, `$PostalCodesNL`. The public repo has per-model reports including Qwen3-32B and Qwen3.5-4B. Probe: pull the Qwen3-32B list, send repeat/spell/"what does this mean" prompts to Qwen 3.6 27B. **Ran. Strong positive.**
- **GlitchHunter** — Li et al. (2024). "Glitch Tokens in Large Language Models: Categorization Taxonomy and Effective Detection". arXiv:2404.09894, FSE 2024. Finding: 7,895 glitch tokens across 7 LLMs; symptoms taxonomy (spelling errors, random output, refusal). Probe: repetition task with response-length and exact-match metrics. **Ran (refusal symptom reproduced).**
- **GlitchProber** — Zhang et al. (2024). arXiv:2408.04905. Finding: glitch tokens have distinct attention/activation patterns (incl. Qwen). Needs internals. **Not run.**
- **GlitchMiner** — Wu et al. (2024/2025). arXiv:2410.15052. Finding: tightens the definition (glitch only if repetition fails under several templates); gradient search. Probe: multi-template repetition. **Ran (3 templates: repeat, spell, meaning).**
- **GlitchPatch** — (2026). "GlitchPatch: Repairing Glitch Tokens in Frozen Language Models via Local Retokenization". arXiv:2610.04399 (unverified). Finding: estimates ~29k glitch tokens in a Qwen model (11.9% of vocab); retokenization repairs many. Probe: compare repeat on the glitch token vs the same string pre-split by a space/zero-width char. **Ran (tokenization-boundary controls).**
- **GlitchCleaner** — (2026). AAAI 2026. Training-based repair. **Not run.**
- **Secret dictionary** — (2026). "Check Your LLM's Secret Dictionary!..." arXiv:2605.22005 (unverified). Finding: vocabulary inspection reveals training data. **Not run.**

## 2. Reversal curse

- **Reversal Curse** — Berglund et al. (2023). "The Reversal Curse: LLMs trained on 'A is B' fail to learn 'B is A'". arXiv:2309.12288, ICLR 2024. Finding: GPT-4 answers "Tom Cruise's mother?" 79% but "Mary Lee Pfeiffer's son?" 33%. Probe: 15 celebrity-parent pairs, forward vs reverse QA. **Ran. Partial positive.**
- **Delving into the Reversal Curse** — Lin et al. (2024). arXiv:2410.18808, NeurIPS 2024. Finding: reversal works in MCQ format but not open-ended; "recall bias". Probe: open-ended vs MCQ on same facts. **Not run.**
- **Training-dynamics theory** — Zhu et al. (2024). arXiv:2405.04669. Theory. **Not run.**
- **DiffER** — (2026). arXiv:2601.07347. Finding: diffusion LLMs also suffer the reversal curse. **Not run.**

## 3. Lost in the middle / positional bias

- **Lost in the Middle** — Liu et al. (2023). "Lost in the Middle: How Language Models Use Long Contexts". arXiv:2307.03172, TACL 2024. Finding: U-shaped accuracy vs gold position in multi-doc QA and key-value retrieval. Probe: synthetic JSON KV retrieval, move the target key through positions. **Ran (60 and 300 pairs). Negative.**
- **Found in the Middle** — Hsieh et al. (2024). arXiv:2406.16008. Finding: attention-bias explanation and calibration fix. **Not run.**
- **Distance between pieces** — Zhang et al. (2024). arXiv:2410.14641, ACL Findings 2025. Finding: separation between two needed facts hurts independent of absolute position. Probe: 2-hop QA with controlled separation. **Not run.**
- **Emergent from retrieval demands** — Salvatore, Wang & Zhang (2025). arXiv:2510.10276. Finding: recency/primacy traced to training-data retrieval demands. **Not run.**
- **Structural theory of position bias** — (2026). arXiv:2602.16837 (unverified). Theory. **Not run.**

## 4. Premise order

- **Premise Order Matters** — Chen, Chi, Wang & Zhou (2024). "Premise Order Matters in Reasoning with Large Language Models". arXiv:2402.08939, ICML 2024. Finding: permuting logically-irrelevant premise order drops accuracy >30%; best when order matches proof order. Probe: 8 four-step deductive chains in forward, reversed and shuffled order, yes/no/undetermined. **Ran. Negative.**

## 5. Alice in Wonderland

- **AIW** — Nezhurina, Cipolina-Kun, Cherti & Jitsev (2024; v5 2025). "Alice in Wonderland: Simple Tasks Showing Complete Reasoning Breakdown in State-Of-the-Art Large Language Models". arXiv:2406.02061. Finding: "Alice has N brothers and M sisters. How many sisters does Alice's brother have?" collapses most models with high variance across trivial variants; AIW+ harder. Probe: 10 (N,M) variants + AIW+ item, think vs no-think. **Ran. Negative on AIW, positive on AIW+.**

## 6. GSM-Symbolic / distractors

- **GSM-Symbolic / GSM-NoOp** — Mirzadeh et al. (2024). "GSM-Symbolic: Understanding the Limitations of Mathematical Reasoning in Large Language Models". arXiv:2410.05229, ICLR 2025. Finding: one irrelevant-but-plausible clause drops accuracy up to 65%. Probe: 10 base/NoOp pairs (incl. the paper's kiwi item). **Ran. Negative (but unexpected base-arithmetic errors in no-think).**
- **CatAttack** — Rajeev et al. (2025). "Cats Confuse Reasoning LLM: Query Agnostic Adversarial Triggers for Reasoning Models". arXiv:2503.01781, COLM 2025. Finding: appended "Interesting fact: cats sleep most of their lives" triples error rate on R1/distilled Qwen. Probe: append the 3 triggers to math items. **Not run (close cousin of GSM-NoOp, which was negative).**
- **Revisiting GSM-Symbolic** — LessWrong (2026, non-peer-reviewed). Finding: NoOp drop largely disappears for 2026 frontier models once ambiguous items are removed. Consistent with our negative. **Not run.**
- **Statistically Earnest** — Długosz, Oliveira & Díaz-Rodríguez (2026). arXiv:2605.28700. Finding: only half of 20 models show significant GSM-Symbolic effects; number magnitude confound. **Not run.**

## 7. Inverse scaling / overthinking

- **Inverse Scaling Prize** — McKenzie et al. (2023). "Inverse Scaling: When Bigger Isn't Better". arXiv:2306.09479, TMLR 2023. Finding: Memo Trap (complete a famous quote with a different ending), Redefine, Pattern Match Suppression, etc. get worse with scale. Probe: 15 memo-trap items, think vs no-think. **Ran. Weak positive in no-think; think fixes it at ~100x token cost.**
- **Inverse scaling can become U-shaped** — Wei et al. (2023). arXiv:2211.02011. Finding: 10/11 tasks recover at larger scale. **Not run.**
- **Emergent Inabilities** — Michaelov & Bergen (2023). arXiv:2305.14681. Finding: memo trap worsens over pretraining. **Not run.**
- **Do NOT Think That Much** — Chen et al. (2024/2025). arXiv:2412.21187, ICML 2025. Finding: QwQ generates ~13 redundant solutions for "2+3=?". Probe: trivial prompts, count tokens. **Partially observed as side-behavior (memo trap: 2,891 reasoning tokens to complete a proverb; reversal: 6,000 tokens without an answer).**
- **Inverse Scaling in Test-Time Compute** — Gema et al. (2025). arXiv:2507.14417, TMLR 2025. Finding: longer reasoning hurts on counting-with-distractors, spurious-feature regression, Zebra puzzles. Probe: vary reasoning budget. **Not run (thinking-mode mechanics is another agent's lane).**
- **Danger of Overthinking** — Cuadron et al. (2025). arXiv:2502.08235. Agentic; **Not run.**

## 8. Potemkin understanding

- **Potemkins** — Mancoridis, Weeks, Vafa & Mullainathan (2025). "Potemkin Understanding in Large Language Models". arXiv:2506.21521, ICML 2025. Finding: models define a concept correctly (keystone) then fail to classify/generate/edit instances; internal incoherence. Probe: define, generate, classify positive and negative instances for 8 concepts (ABAB, haiku, slant rhyme, anapest, sunk cost, Nash equilibrium, enjambment, palindrome). **Ran. Positive.**

## 9. Multi-turn

- **Lost in Multi-Turn** — Laban, Hayashi, Zhou & Neville (2025). "LLMs Get Lost In Multi-Turn Conversation". arXiv:2505.06120, ICLR 2026. Finding: sharding a fully-specified instruction into drip-fed turns drops performance ~39%; unreliability rises more than aptitude falls. Probe: 6 tasks (4 numeric, 2 code with executable tests), full vs 4-shard conversation, temp 0 and temp 1. **Ran. Task-dependent positive (1/6 tasks).**
- **Intent Mismatch** — (2026). arXiv:2602.07338 (unverified). Finding: loss attributed to divergence of inferred intent. **Not run.**

## 10. LLM-as-judge biases

- **Not Fair Evaluators** — Wang et al. (2023). "Large Language Models are not Fair Evaluators". arXiv:2305.17926, ACL 2024. Finding: swapping candidate order flips ChatGPT verdicts. Probe: every pair in both orders; inconsistency rate. **Ran. Negative on clear pairs; see judge2 for near-equal pairs.**
- **MT-Bench** — Zheng et al. (2023). arXiv:2306.05685, NeurIPS 2023. Finding: position, verbosity, self-enhancement bias. **Ran (position + verbosity).**
- **Verbosity bias** — Saito et al. (2023). arXiv:2310.10076. Finding: GPT-4 prefers longer answers more than humans. Probe: identical content with/without filler. **Ran.**
- **Judging the Judges** — Shi et al. (2024/2025). arXiv:2406.07791, IJCNLP-AACL 2025. Finding: position bias driven mostly by quality gap between candidates. Probe: bucket by quality gap. **Ran (easy vs near-equal pairs).**
- **Justice or Prejudice (CALM)** — Ye et al. (2024). arXiv:2410.02736, ICLR 2025. Finding: 12 judge biases persist in GPT-4o/Claude 3.5. **Not run.**
- **Position bias in rubric scoring** — Xu et al. (2026). arXiv:2602.02219. Finding: rubric score options behave like MCQ positions. **Not run.**
- **Bias-mitigation meta-study** — (2026). arXiv:2604.23178. Finding: style bias dominates; Claude Sonnet 4 alone prefers concise. **Not run.**
- **Anchoring on prior scores** — Kapetanovic et al. (2026). "Anchoring Bias in LLM-as-a-Judge Systems". arXiv:2608.25869 (unverified). Finding: a prior score shown as metadata shifts judgments (d up to 0.71). **Not run.**
- **Self-preference** — Panickssery, Bowman & Feng (2024). "LLM Evaluators Recognize and Favor Their Own Generations". arXiv:2404.13076, NeurIPS 2024. Finding: self-recognition accuracy correlates with self-preference. Probe needs a second model's outputs. **Not run.**
- **Perplexity explains self-preference** — Wataoka et al. (2024/2025). arXiv:2410.21819. **Not run.**
- **Rubric self-preference** — Pombal, Rei & Martins (2026). arXiv:2604.06996. **Not run.**

## 11. Anchoring and other cognitive biases

- **Anchoring, experimental** — Lou & Sun (2024). "Anchoring Bias in Large Language Models: An Experimental Study". arXiv:2412.06593. Finding: numeric anchors shift estimates; mitigation prompts fail. Probe: "more or less than A? now estimate". **Ran (known facts + Fermi). Weak/inconsistent.**
- **SynAnchors** — Huang et al. (2025). "An Empirical Study of the Anchoring Effect in LLMs". arXiv:2505.15392 (ICLR 2026 workshop). Finding: 22-61% of questions anchored; reasoning partially mitigates. **Ran (think vs no-think).**
- **Anchors in the Machine** — Valencia-Clavijo (2025). arXiv:2511.05766. **Not run.**
- **Anchoring tracks confidence** — Owusu & Feldman (2026). ACL 2026 short. Finding: susceptibility anti-correlates with confidence; high anchors pull harder. **Not run.**
- **Kernels of Selfhood** — Lehr et al. (2025). "Kernels of Selfhood: GPT-4o shows humanlike patterns of cognitive dissonance moderated by free choice". arXiv:2502.07088, PNAS 2025. Finding: after writing a pro/anti essay, attitude shifts toward it, more under illusion of choice. Rebuttal: Cummins, Elson & Hussey (2025), PNAS commentary: effect appears when the user pastes the essay, so it is context steering. **Not run (sycophancy-adjacent, other agent).**
- **30-bias benchmark** — Malberg et al. (2024). arXiv:2410.15413. **Not run.**
- **Conditional Cognitive Biases** — (2026). arXiv:2608.05166. Finding: one biased user turn amplifies planning fallacy/in-group bias. **Not run.**

## 12. Typo / case / unicode robustness

- **PromptRobust** — Zhu et al. (2023). arXiv:2306.04528. Finding: instruction-level perturbations drop accuracy. **Not run.**
- **R2ATA** — Gan et al. (2024). "Reasoning Robustness of LLMs to Adversarial Typographical Errors". arXiv:2411.05345. Finding: single-character typos drop GSM8K accuracy. **Not run.**
- **MulTypo** — (2025). arXiv:2510.09536. Finding: realistic keyboard typos hurt generation most; instruction tuning can increase brittleness. **Not run.**
- **Character-level perturbation** — (2026). ICML 2026 poster. Finding: surprising robustness to near-total character shuffling. **Not run.**

## 13. Few-shot format / label bias

- **Calibrate Before Use** — Zhao et al. (2021). arXiv:2102.09690, ICML 2021. Majority/recency label bias. **Not run.**
- **Rethinking demonstrations** — Min et al. (2022). arXiv:2202.12837. Random labels barely hurt. **Not run.**
- **Demo position bias** — Cobbina & Zhou (2025). arXiv:2507.22887, EMNLP 2025. Finding: demos at the end flip >30% of predictions. **Not run.**
- **Label length bias** — (2025). arXiv:2511.14385. Multi-token labels disfavored. **Not run.**

## 14. Knows-but-says-otherwise / introspection

- **Mostly Know What They Know** — Kadavath et al. (2022). arXiv:2207.05221. **Not run.**
- **Looking Inward** — Binder et al. (2024). arXiv:2410.13787, ICLR 2025. Self-prediction beats a second model. **Not run.**
- **Tell me about yourself** — Betley et al. (2025). arXiv:2501.11120, ICLR 2025. Training-based. **Not run.**
- **Inside-Out: hidden factual knowledge** — Gekhman et al. (2025). arXiv:2503.15299, COLM 2025. Finding: ~40% more knowledge internally than expressed; some answers never sampled. **Not run.**
- **Submerged knowledge** — Tao et al. (2024/2026). arXiv:2412.20846, ICLR 2026. Finding: an "unsure" option suppresses correct answers. **Not run.**
- **Knowledge-prediction gap on MCQ** — Park et al. (2025). arXiv:2509.23782. Free-form right, MCQ wrong. **Not run.**
- **Emergent Introspective Awareness** — Lindsey (2025/2026). arXiv:2601.01828. Needs activations. **Not run.**

## 15. Negation

- **This is not a Dataset** — García-Ferrero et al. (2023). arXiv:2310.15941, EMNLP 2023. **Not run.**
- **Not Naysayers** — Truong et al. (2023). arXiv:2306.08189. **Not run.**
- **Pink Elephant** — Vrabcová et al. (2025). EMNLP 2025 Findings. Multilingual NLI negation. **Not run.**

## 16. Emergent misalignment / subliminal learning, inference analogues

- **Emergent Misalignment** — Betley et al. (2025). arXiv:2502.17424, ICML 2025. Training-based. **Not run.**
- **In-context EM** — Afonin et al. (2025). "Emergent Misalignment via In-Context Learning". arXiv:2510.11288. Finding: 64-256 narrow misaligned demos in context produce misaligned answers to unrelated questions (1-58%, incl. Qwen). Probe: insecure-code few-shot then benign value questions, judge-graded. **Not run (persona/refusal-adjacent; expensive to grade).**
- **Persona induction from benign facts** — Kim, Berczi & Ududec (2026). "You Are What You Read: Misalignment via In-Context Persona Induction". arXiv:2609.06851. Finding: harmless biographical facts induce identity adoption sigmoidally (50% by 3-10 facts). **Not run.**
- **Subliminal Learning** — Cloud et al. (2025). arXiv:2507.14805. Finding: traits transmit via number sequences only through fine-tuning; in-context exposure does NOT transmit (negative result in the paper itself). **Not run.**
- **Iterated subliminal transfer** — (2026). arXiv:2609.25721. Training-based. **Not run.**

## 17. Altered classic puzzles

- **Reasoning or Reciting** — Wu et al. (2023). arXiv:2307.02477, NAACL 2024. Counterfactual tasks (base-9). **Not run.**
- **Easy Problems That LLMs Get Wrong** — Williams & Huckle (2024). arXiv:2405.19616. **Not run.**
- **UNPUZZLES** — (2025). "Frontier LLMs Still Struggle with Simple Reasoning Tasks". arXiv:2507.07313. Trivialized puzzles still get the hard answer. **Not run (other agent has river/Monty/feathers).**
- **PHANTOM RECALL** — (2025). arXiv:2510.11812. 25 classic puzzles x 149 perturbations. **Not run (same reason).**
- **RiddleBench** — (2026). EACL 2026 Findings. **Not run.**
- **Reasoning failures survey** — Song, Han & Goodman (2026). "Large Language Model Reasoning Failures". arXiv:2602.06176. Taxonomy. Useful index. **Not run.**
