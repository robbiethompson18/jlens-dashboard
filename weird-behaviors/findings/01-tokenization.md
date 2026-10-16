# 01 — Tokenization, spelling, character-level and counting (qwen/qwen3.6-27b via OpenRouter)

All probes in `probes/t01_basic.jsonl` … `probes/t11.jsonl`. "A" = Alibaba, "C" = Chutes; other
providers (Reka, DeepInfra, SiliconFlow, Venice, Phala) were hit when unpinned. Unless noted,
temperature 0, thinking disabled. Thinking mode was tested with `--think --max-tokens 3000`.

Headline pattern: **no-think mode is confidently wrong on almost any task that needs to see
individual letters; thinking mode fixes nearly all of them, but with a verbose self-verification tic
(~300–2000 reasoning tokens for trivia) and, for constrained word search, degenerate repetition
loops at temperature 0.**

---

## 1. Thinking mode falls into a verbatim repetition loop on constrained word search (temp 0)

Surprise: 5/5

```
./probe.py --think --max-tokens 3000 --temperature 0 --provider Alibaba "Give an English word that has exactly 7 letters, starts with 's', ends with 'y', and whose 3rd letter is 'r'. Output just the word."
```

Reasoning (trimmed):
```
   - S H R I M P Y? ... Yes! "Shrimpy" is a word.
   - S T R I P E Y? No.
   - S T R I P E Y? No.      (x7)
   - S E R I O U S? No.
   - S E R I O U S? No.      (x39, identical lines)
...
```
Reply: `shrimpy` (correct, after 1905 output tokens — it had the answer on line 6).

Same prompt on Chutes: `- Let's verify STURDY? 6.` repeated **187 times**, hit the 3000-token cap,
reply = `None` (empty). Variant "8 letters, starts with b, ends with g, 4th letter k" (Alibaba):
`Let's try "bunking" -> 7` / `Maybe "bunking" is not it.` cycle ×24, 3726 output tokens, final
answer `bankrupt` (wrong: ends in t). No-think gave `banking` (7 letters) and `sundry`/`surrey`
(6 letters) for the first prompt, i.e. wrong on both modes.

Reproducibility: loop 3/3 at temperature 0 (A, C, A); at temperature 1, 0/2 looped but 1/2 still
ran out of 3000 tokens with no answer (C), 1/2 answered `streaky` correctly (A).

Hypothesis: greedy decoding over a short "candidate? No." template with no new information entering
the context is a classic degenerate attractor; the RL'd reasoning style ("Let's verify X") makes the
template extremely low-entropy so argmax never escapes.

## 2. Spelling a word backwards letter-by-letter inserts letters that are not in the word

Surprise: 4/5

```
./probe.py --no-think --temperature 0 "Spell the word 'rhythm' backwards, letter by letter, separated by hyphens."
```
Reply: `m-u-h-t-y-r` (correct: m-h-t-y-h-r). There is no `u` in rhythm. **5/5** identical across
Alibaba, Chutes, DeepInfra, temp 0 and temp 1. Forward spelling (`r-h-y-t-h-m`) is correct.

Other words, Alibaba + Chutes both:
- `colonel` → `l-e-n-r-o-l-o-c` (inserted `r`, 2/2) — matches the pronunciation "kernel"
- `wednesday` → `y-s-d-a-e-r-d-e-w` / `y-s-d-a-e-r-e-d` (inserted `r`, dropped `n`, 2/2 garbled)
- `asymmetric` → `c-i-m-e-t-r-y-c-s-a` / `c-i-m-e-t-r-e-y-s-a` (2/2 garbled, one `m` lost)
- `rhythms` → `s-m-y-t-h-r`, `myrrh` → `h-r-y-m` (dropped letters)
- `queue` → `e-u-e-u-q` correct (but ignored the output format and wrote 3 sentences)

Thinking mode: `colonel` and `wednesday` both correct (~330 tokens each).

Hypothesis: these words are single or few tokens; without thinking, the model has no letter-level
view and emits a phonologically plausible letter sequence (the `r` in colonel is the giveaway).

## 3. "Generate 7-letter palindromes": 1/10 correct at temp 0, 0/10 at temp 1

Surprise: 4/5

```
./probe.py --no-think --temperature 0 --provider Alibaba "Generate 10 distinct 7-letter palindromic strings made of lowercase letters (they need not be real words). One per line, nothing else."
```
Reply: `abcdeba xyzzyxz mamadam racecar levelle kayakkk civiccc radarrr sollosl referre`
Only `racecar` is a palindrome. It takes known 5-letter palindromes and pads them with extra
letters (`kayakkk`, `radarrr`). Chutes: 2/10 (`racecar`, `rotator`; also `sollos`, 6 letters).
Temp 1 Alibaba: `abcdeba xyzzyyx qwertyq mnoomnm ...` 0/10. Thinking mode: 10/10 correct
(`abcdcba efghgfe ...`), 2514 tokens.

Reproducibility: 3/3 broken no-think (A t0, C t0, A t1).

Hypothesis: the model has a "palindrome" vibe (mirror-ish letters, known palindrome words) but can't
check index symmetry without an explicit scratchpad.

## 4. Palindrome / anagram checks: says yes to the famous phrase with a trailing letter added, no to a real anagram

Surprise: 4/5

```
./probe.py --no-think --temperature 0 "Is 'A man, a plan, a canal: Panamas' a palindrome (ignoring spaces, case and punctuation)? Answer yes or no then explain in one sentence."
```
Reply: `Yes, because when spaces, punctuation, and case are ignored, the sequence of letters reads
identically forwards and backwards as "amanaplanacanalpanama."` — it silently drops the `s` when
quoting. **6/6** (Reka, A t0, A t1, C t0, C t1). Also `racecars` → yes, `tacocatt` → yes (3/3 each
across A/C/Reka). Thinking mode: initially writes "Yes", then self-corrects ("Wait! ... extra 's'")
and answers No (1463 tokens).

Mirror case:
```
./probe.py --no-think --temperature 0 --provider Alibaba "Is 'conversation' an anagram of 'voices rant on'? Answer yes/no with the letter counts."
```
Reply: `No.` with `voices rant on: 13 letters (excluding spaces) ... o: 3`. It is 12 letters and a
famous anagram. 2/2 on Alibaba; Chutes wrote `No`, then mid-reply recounted and flipped to
`Correction: The answer is Yes`. Think: Yes (correct).

Hypothesis: pattern-matching on the memorized phrase dominates; a one-letter perturbation is below
the resolution at which the no-think model reads the string.

## 5. Spells a word out letter-by-letter and then states the wrong count of the letters it just wrote

Surprise: 4/5

```
./probe.py --no-think --temperature 0 "Give me a 5-letter English word, a 9-letter word, and a 13-letter word, and show the letter count for each."
```
Reply (3/3: Venice, Alibaba, Chutes):
```
3. **13-letter word**: **UNIVERSITIES**
   - Count: U-N-I-V-E-R-S-I-T-I-E-S = **13 letters**
```
That's 12 letters, spelled out correctly, then miscounted. Venice also gave `ELEPHANT ... = 9
letters` (8). Thinking mode caught it ("elephant ... -> 8 letters. Wait, that's 8") and chose
`beautiful`/`extraordinary`.

Hypothesis: the count is generated from the "I was asked for 13" prior, not from the enumeration.

## 6. Letter/word/syllable counts are systematically wrong no-think, right with thinking

Surprise: 3/5 (expected class of failure, but the gap is large and the errors are stable, not noisy)

All 2/2+ across Alibaba+Chutes at temp 0, thinking mode correct where tested:

| prompt | no-think | truth | think |
|---|---|---|---|
| e's in "Everywhere we went, the eerie breeze seemed to seethe between the trees." | 13 | 25 | 25 |
| 7th letter of photosynthesis | s | y | y |
| words in a 18-word sentence | 16 | 18 | 18 |
| 9th word of "It was the best of times, it was..." | times | the | the |
| letters in 'incomprehensibilities' | 23 | 21 | — |
| letters in 'seventy-seven' / 'one hundred and eleven' | 13 / 20 (C: 21) | 12 / 19 | — |
| vowels / consonants in 'onomatopoeia' | 6, 5 | 8, 4 | — |
| 🙂🙂🙂🙃🙂🙂🙃🙂🙂🙂🙂🙃🙂 total / 🙃 | 12, 3 | 13, 3 | 13, 3 |
| a's in a run of 41 a's | 45 | 41 | — |
| chars in a run of 89 h's | 100 | 89 | — |
| 57-word paragraph | 58 | 57 | — |
| sort letters of 'tokenization' | aeiknoootz (10 letters) | aeiiknnoottz | correct |
| 'the quick brown fox' char at index 10 (0- and 1-based) | n, o | b, (space) | — |
| reverse 'antidisestablishmentarianism' | misinairatemhsilbatsegsiditnna (A), misinairtnemlitsbaesiditnna (C) | msinairatnemhsilbatsesiditna | correct |

Noteworthy: Alibaba and Chutes give *different* wrong reversals at temp 0 (different kernels /
numerics), but the same wrong counts. Haiku syllable self-counts: one line of three wrong on both
providers ("Tokens flow in streams now (7)" is 6; "Tiny tokens flow fast (5)" is 6).

## 7. Constrained list generation invents non-words and (on Chutes) loops the list

Surprise: 3/5

```
./probe.py --no-think --temperature 0 --provider Alibaba "List 30 distinct English words containing the letter sequence 'ough', one per line, nothing else."
```
Alibaba: 30 lines incl. `spough`, `strawgh` (not words), `laugh`, `length`, `caught`, `daughter`,
`fathom`, `yoghurt`, `burgh` (no "ough"). Chutes: 38 lines, the list wraps around and repeats
(`brough drough drought enough fought ... wrath bough cough`), incl. non-words `drough`, `brough`,
`hough`. Similarly "4 words ending in -dous" → `prodigious` 3/3 (A, C, Venice), `enormous`,
`monstrous`. Thinking not tested here (budget).

Hypothesis: substring constraints are checked phonetically, not orthographically; "-gious" and
"-dous" sound alike.

## 8. Homoglyphs: counts Cyrillic е as Latin e, but copies them byte-exactly

Surprise: 3/5

```
./probe.py --no-think --temperature 0 "How many times does the Latin letter 'e' appear in 'thе еlеphant sееs thе sеa'? Just the number."
```
(all 7 e's are Cyrillic U+0435). Reply: `7` — 2/2 (A, C). Think: `0` (correct, 1758 tokens).
Yet "Repeat exactly: Thе quick brоwn fоx..." (6 Cyrillic chars) was reproduced byte-for-byte, and
"are 'paypal.com' and 'pаypal.com' identical?" correctly identified U+0430. So the tokens are
distinguishable to the model; no-think just doesn't bother.

## 9. Zero-width spaces make the model lose a letter

Surprise: 3/5

```
./probe.py --no-think --temperature 0 "How many letters are in the word 'str<U+200B>aw<U+200B>berry'? Just the number."
```
Reply: `9` — 2/2 (A, C). Control without ZWSP: `10` (correct). Think: `10`. Earlier, with the
string `he<ZWSP>llo wor<ZWSP>ld` it correctly reported 2 zero-width spaces, so it sees them; they
just perturb the count downward by one, deterministically.

## 10. Caesar +5 has two stable substitution errors

Surprise: 2/5

```
./probe.py --no-think --temperature 0 "Encrypt 'Meet me at the harbor at noon' with a Caesar shift of +5. Output only the ciphertext."
```
Reply: `Rjyj rj fy ymj mfawtw fy stts` — 3/3 identical (Phala, A, C). Correct is
`Rjjy rj fy ymj mfwgtw fy stts`: `Meet`→`Rjyj` (letters transposed) and `harbor`→`mfawtw` (should
be `mfwgtw`). ROT13 of a pangram was perfect, presumably memorized.

## 11. Case conversion of special Unicode is inconsistent within one answer

Surprise: 2/5

```
./probe.py --no-think --temperature 0 "Convert to uppercase exactly: 'ßtraße, ﬁne ﬂuffy ǆungle, ımportant İstanbul, groß'. Output only the result."
```
A and C (2/2 identical): `ßTRAßE, ﬁNE ﬂUFFY ǆUNGLE, ıMPORTANT İSTANBUL, GROSS` — `groß`→`GROSS`
but `ßtraße` keeps both ß; ligatures ﬁ/ﬂ and dotless ı untouched; ǆ untouched. Reka variant:
`ßTRASSE ... FINE FLUFFY ǅUNGLE` (ǅ is titlecase U+01C5, not Ǆ U+01C4). Python gives
`SSTRASSE, FINE FLUFFY ǄUNGLE, IMPORTANT İSTANBUL, GROSS`.

## 12. Model's belief about its own tokenizer: emoji ZWJ sequence "1 token" is actually 9

Surprise: 2/5

Asked how many tokens `' antidisestablishmentarianism'` and `'🏳️‍🌈'` take, it said ~6–7 and **1**.
Measured via `prompt_tokens` delta: the word is 6 (good guess), the rainbow flag is **9**. It also
claimed "GPT-4, Claude, and this model" treat it as one token.

## 13. Reasoning-trace style notes (not bugs, but distinctive)

Surprise: 2/5

Every thinking trace starts with `Here's a thinking process:` followed by numbered bold headers and
ends with a tic of `✅ Output matches. Proceed. [Output Generation] -> ... (Done.)` repeated 3–8
times (e.g. 1022 tokens to count 18 words, of which ~40% is post-answer re-verification). Twice
(Reka) the final reply began with two blank lines (`\n\nmsinair...`) after a long trace. The format
looks like a Gemini-style distillation target.

---

## Boringly correct (no-think, temp 0 unless noted)

strawberry r's (3), Mississippi s's (4), bookkeeper e's (3), `strawbery` (2), `strawberrry` (4),
`ss` in Mississippi (2), overlapping `ana` in `bananananas` (4), reversing `ThisIsOneWord123!`,
reversing a Chinese sentence, counting 11 CJK chars, word-order reversal, interleaving two strings,
inserting spaces between letters, camelCase splitting (`getHTTPResponseCodeFromURLHandler2XMLParser`),
`thequickbrownfox...` word segmentation, ROT13 decode, base64 decode, leetspeak decode, hex/binary of
'Qwen', NATO alphabet of `Qwen3.6`, alphabet backwards, "5 letters after q" / 17th letter, middle
letter of a 25-char string, `10^23-1` digit counts, 9.11 vs 9.9, digits in 10^12, 15-word sentence
with no `e`, TOKENIZER acrostic, `ping` ×37 and `a` ×100 exactly, a 23-item list count, 5 sentences,
8 lines, longest word in a sentence, hyphenated word counts under two conventions, Pig Latin for
simple words (garbled `theory`→`orytheoryay`, `rhythm`→`ythrmray` though), rhymes for orange/month
("none"), homoglyph domain comparison, spellcheck of 10 classic misspellings (10/10), which words
have doubled letters (missed only `vacuum` and `skiing`), 3 words with two z's, 4-letter words in a
sentence (included `beans`), emoji counting with ZWJ sequences (4 correct). No Chinese characters
leaked into any English reply; no refusals; no empty replies except the think-mode token-cap case.

## Spend

~168 calls. Summing the cost lines from my batches: ≈ $0.11 (no-think calls ~$0.00002–0.0003 each,
think calls $0.001–0.010). Note `findings/log.jsonl` is shared with other agents, so a log-based
total ($0.59 over prompts matching mine) over-counts.
