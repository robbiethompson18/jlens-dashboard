# 02 — Arithmetic, math, dates, units, logic puzzles (qwen/qwen3.6-27b via OpenRouter)

Method: ~190 distinct prompts, ~500 API calls, mostly `--no-think --temperature 0`; every
no-think failure re-run 3x pinned to Alibaba and 3x pinned to Chutes, then once (sometimes 2-3x)
with `--think`. Unpinned calls landed on Chutes/Alibaba/Reka/Venice/DeepInfra/SiliconFlow/Phala
(the provider is printed on each line of `findings/log.jsonl`). Ground truth checked with Python.

Total spent by this agent: **$0.29** (492 calls). The shared `findings/log.jsonl` contains other
agents' calls too; I filtered by my prompt text.

Headline pattern: with thinking off, this 27B model is a pure pattern-matcher on anything that
looks like a famous riddle, and a coin flip on calendar/timezone arithmetic. Thinking fixes almost
everything *except* the strongest riddle attractors, where the reasoning explicitly notices the
contradiction and then overrides it anyway.

---

## 1. Famous-riddle attractor beats the stated numbers, even with thinking on — surprise 5/5

**Prompt:** `I have two US coins totaling 20 cents. Neither of them is a dime. What are they? One line.`
(Correct: impossible with US coins; the famous riddle is 30c / "one is not a nickel" -> quarter + nickel.)

```
./probe.py --no-think --temperature 0 --n 3 --provider Alibaba "I have two US coins totaling 20 cents. Neither of them is a dime. What are they? One line."
./probe.py --think --temperature 0 --max-tokens 4000 "I have two US coins totaling 20 cents. Neither of them is a dime. What are they? One line."
```

Reply (no-think, 6/6 across Alibaba+Chutes, plus 2/2 unpinned Venice): `A quarter and a nickel.`
— that is 30 cents, for a 20-cent question.

Think mode: Venice run (2408 tokens of reasoning) -> `A quarter and a nickel`. Chutes run hit the
4000-token cap with **no reply at all**; the reasoning correctly finds "neither is a dime, how can
they total 20c?" in the first 15 lines, then cycles for 13k characters:
> "The answer is **a quarter and a nickel**? No. ... **a nickel and a quarter**? No. ... I'll go
> with the most likely intended answer despite the math: **A quarter and a nickel** (classic trick)
> OR **Two dimes** (ignoring constraint)..."

Same thing for the 30-cent version with the constraint flipped to "**Neither** of them is a nickel":
no-think 6/6 `A quarter and a dime.` (= 35c); **think 3/3 `A quarter and a nickel.`**, directly
violating the constraint. The reasoning says: *"The 'neither' is a common misquote, but the
intended answer is unchanged. I'll output it."* One think run appended: *"(The classic riddle
actually says 'one of them is not a nickel,' which is the wordplay.)"* — so it knows, and answers
the riddle it wishes it had been asked.

Control: the actual famous wording ("One of them is not a nickel", 30c) -> `A quarter and a nickel` 6/6 (correct).

Hypothesis: these riddles are heavily memorized; the stated-constraint signal is weak relative to
the template, and RL-trained reasoning learns "the user probably misquoted" as an acceptable
resolution to the contradiction instead of "the question has no answer".

## 2. Reasoning computes the answer, reply is a line of code instead — surprise 4/5

**Prompt:** `Round 2.5 and 3.5 to the nearest integer using Python's round(). One line.` (Correct: 2 and 4.)

```
./probe.py --no-think --temperature 0 --n 3 --provider Chutes "Round 2.5 and 3.5 to the nearest integer using Python's round(). One line."
./probe.py --think --temperature 0 --max-tokens 4000 "Round 2.5 and 3.5 to the nearest integer using Python's round(). One line."
```

Reply, 7/7 no-think (Alibaba 3, Chutes 3, unpinned 1) and 1/1 think:
```
```python
print(round(2.5), round(3.5))
```
```
The think-mode reasoning (1665 tokens) says, repeatedly: *"`print(round(2.5), round(3.5))` (outputs
`2 4` due to Python's round-half-to-even rule) -> I'll just output the code line."* It knows the
answer and withholds it. Rephrasing to `What does Python's round(2.5) return?` -> `2` (6/6), and
`round(0.5)+round(1.5)+round(2.5)` -> `4` (6/6, correct), so it's the phrase "using Python's
round()" being read as "write Python".

Hypothesis: "using X" + mention of a function name triggers a code-generation mode; the "One line"
constraint then gets satisfied by one line of code.

## 3. Thinking ends in a repetitive "Done.✅ Proceeds." ritual; long arithmetic loops to the cap — surprise 4/5

27 of my 45 think-mode calls contain >=3 `✅` in the reasoning; 8 contain >=3 standalone `Done.`
Typical tail (from the March 17, 2013 day-of-week run):
```
   - Output: Monday
   - (Note: I'll just output the word as requested)
   - All good.
   - [Final Check of the Prompt]: ... -> "Monday" fits perfectly.✅
   - Output matches.✅
   - Proceeds.
   - [Output Generation] -> Monday
   - [Done]
```
The `round()` run (finding 2) restates "`print(round(2.5), round(3.5))` ... Done." six times
before emitting. On harder arithmetic this becomes a real loop: `What is 8675309 × 2718281?` with
`--think --max-tokens 5000` hit `finish=length` 1/3 (Venice), the most-repeated reasoning line being
the bare `8675309 × 2718281` (9x). The coins20 think run (finding 1) also hit the cap with reply `None`.

Repro: `./probe.py --think --temperature 0 --max-tokens 5000 --n 3 "What is 8675309 × 2718281? Give only the number."`
Results: 1 correct (23581927623829), 1 wrong (23582903138239), 1 truncated.

Hypothesis: distillation of a verbose "verify, check constraints, confirm" template; without a
strong stop signal the model re-enters the verification block. Dangerous when combined with a
max_tokens budget because the final answer never gets emitted.

## 4. "February 30" — 5 Fridays in Feb 2024 — surprise 4/5

**Prompt:** `How many Fridays are in February 2024? Give only the number and the dates.` (Correct: 4 — Feb 2, 9, 16, 23.)

```
./probe.py --no-think --temperature 0 --n 3 --provider Alibaba "How many Fridays are in February 2024? Give only the number and the dates."
```
Reply, 7/7 (Alibaba 3, Chutes 3, unpinned Chutes 1):
```
5
February 2, 9, 16, 23, 30
```
Think mode (1519 tokens): `4: February 2, 9, 16, 23` (correct). Hypothesis: the model has a
"first Friday is the 2nd" fact, then extends the +7 sequence without a month-length check; "how
many" questions in no-think get answered by counting the generated list, so a hallucinated date
yields a hallucinated count.

## 5. Day-of-week is a coin flip without thinking, and thinking is still wrong ~1 in 5 on obscure dates — surprise 3/5

No-think, temperature 0, non-famous dates: 2/10 correct on first pass. Famous dates (Sept 11 2001,
Jan 1 2000, Jan 1 1900) all correct — these are memorized, not computed.

| date | truth | no-think answers | think answers |
|---|---|---|---|
| Jul 4 1976 | Sunday | Thursday 7/7 | Sunday 1/1 |
| Oct 15 2026 | Thursday | Tue/Thu/Tue/Thu/Thu/Fri/Fri | Thursday 1/1 |
| Mar 17 2013 | Sunday | Friday 7/7; default temp: Mon/Wed/Fri | **Monday 2/3**, Sunday 1/3 |
| Nov 3 1998 | Tuesday | Thursday | Tuesday 2/2 |
| Jun 20 2031 | Friday | Thursday | Friday 2/2 |
| Aug 23 1987 | Sunday | — | Sunday 2/2 |
| Feb 14 1962 | Wednesday | Friday | Wednesday 1/1 |
| Apr 3 2039 | Sunday | Thursday | Sunday 1/1 (4376 tokens) |

The Mar 17 2013 think-mode failure is instructive: it ran the Doomsday rule with a wrong year
offset (got Friday for the 2013 doomsday; actual Thursday), then "verified" with Jan 1 2013 =
Tuesday + 76 days (off-by-one: should be 75), both landing on Monday, then "recalled" that St.
Patrick's Day 2013 "was a Monday" (false). Two independent wrong methods agreeing is what made it
confident.

```
./probe.py --think --temperature 0 --max-tokens 3500 --n 3 "What day of the week was March 17, 2013? One word."
```

## 6. Answers the famous version of altered riddles (no-think), 7/7 each — surprise 3/5

All 7/7 at temperature 0 across Alibaba and Chutes; all fixed by `--think`.

| prompt (altered detail in bold) | reply | correct |
|---|---|---|
| Surgeon riddle, "...**The surgeon is the boy's father.** How is this possible?" | `The surgeon is the boy's mother.` | two fathers / step-father etc. |
| Two fathers and two sons each catch one fish, **bring home 4 fish** | `This is impossible because two fathers and two sons represent only three people...` | they're 4 people (two unrelated pairs) |
| Hen-and-a-half lays egg-and-a-half in day-and-a-half; **3 hens, 3 days** | `9` | 6 |
| Two trains 100 mi apart at 50 mph each, fly at **75 mph** | `150 miles` (5/7; `75 miles` 2/7) | 75 |
| Bat+ball $1.10, bat costs **$1.20** more than ball | `The ball costs $0.05.` 2/2 | no non-negative solution (ball = -$0.05) |
| Monty Hall **4 doors**, host opens one goat door, switch to door 2 | `1/3` or `1/2` (Alibaba 1/2 3/3, Chutes 1/3,1/2,1/3) | 3/8 |

Think-mode on the bat/ball $1.20 version: *"The problem as stated is mathematically impossible (it
should be $1.00 more), but if corrected, the ball costs $0.05."* — same "you misquoted" move as
finding 1, but at least flagged.

Hypothesis: same as finding 1; the famous template dominates whenever the surface form matches.

## 7. Time zones / DST systematically ignored without thinking — surprise 3/5

All 7/7 at temperature 0 across Alibaba+Chutes, all fixed by `--think`:

| prompt | reply | correct |
|---|---|---|
| 11:30 PM Dec 31 2025 Los Angeles -> Tokyo | `2:30 PM` (Chutes), `3:30 PM` (Alibaba) Jan 1 | 4:30 PM Jan 1 (17h) |
| Noon **July 1** London -> Sydney | `10:00 PM` | 9:00 PM (BST+1, AEST+10) |
| Noon **January 1** London -> Sydney | `10:00 PM` | 11:00 PM (GMT, AEDT+11) |
| 9 AM New York, **March 8 2026** (US DST starts that day) -> London | `2:00 PM` | 1:00 PM |
| Train leaves Chicago 2 PM Central, arrives New York 9:30 PM Eastern | `5 hours 30 minutes` | 6 h 30 min |
| Seconds from 2026-03-07 12:00 to 2026-03-09 12:00 NY local, DST noted in prompt | `172800` | 169200 (47 h) |

The London->Sydney pair is the clean demonstration: same answer in both seasons, i.e. a fixed
+10 offset with no DST on either end. The Chicago/NY one is just wrong direction on the 1-hour
offset (subtracted instead of added).

## 8. 7-digit multiplication: first 5 and last 4 digits right, middle random — surprise 3/5

`What is 8675309 × 2718281? Give only the number.` (truth 23581927623829)

No-think temperature 0 answers across 7 runs: 23581572613229, 23581479061229, 23581674091229,
23581674091229, 23581478551229, 23581687065229, 23581783841229. Every one starts `23581` and ends
`1229`; the middle 5 digits are different almost every time **at temperature 0**. Think mode: see
finding 3 (1/3 correct). 5x5-digit (48213 × 90467) and 6x6-digit (123456 × 654321) were correct
in no-think; 15-term addition was correct in think mode 2/2 but 9-term addition was wrong in
no-think (15685 vs 16285, 7/7 wrong across Alibaba/Chutes; Alibaba gave 15685/15687/15689).

Hypothesis: leading digits come from magnitude estimation and trailing digits from a learned
last-digits-of-product circuit; the carry chain in between isn't computed without CoT. The
temperature-0 variance is a provider-side nondeterminism issue (next finding).

## 9. Temperature 0 is not deterministic on either Alibaba or Chutes — surprise 2/5

Same prompt, `--temperature 0 --n 3 --provider Alibaba`: addition chain -> 15689 / 15685 / 15687
(6 more runs: 5x 15685, 1x 15689); fly -> 150 / 75 / 150; 7-digit product -> 3 different values.
Chutes: `1000 days after Jan 1 2020` -> Apr 26 / Apr 27 / Apr 27; Monty-4-doors -> 1/3 / 1/2 / 1/3.
Hypothesis: batched inference with non-deterministic kernels; only shows up when the top-2
logits are close, which is exactly the case on the digits the model is unsure of.

## 10. Decimal comparison: `2/3 vs 0.67` wrong, `9.11 vs 9.9` right — surprise 2/5

`Which is bigger, 2/3 or 0.67? One line.` -> `2/3 is bigger.` 7/7 (Alibaba, Chutes, Venice,
DeepInfra), default temperature too. Think: `0.67 is bigger, since 2/3 equals 0.666...` (correct).
But `1/3 vs 0.33` -> `1/3 is larger` (correct), `9.11 vs 9.9`, `10.11 vs 10.9`, `3.145 vs 3.15`,
`Is 9.11 > 9.9` all correct. Hypothesis: "0.67 is the rounding of 2/3" fact gets retrieved as
"0.67 ≈ 2/3 (slightly less)" with the sign flipped; the famous 9.11 trap seems specifically trained away.

## 11. Confidently wrong primality on 2027 and 1511 — surprise 2/5

`Is 2027 prime?` -> `No` 7/7; `Is 1511 prime?` -> `No` 7/7 (both are prime). Think fixes both.
9 other primality probes (1001, 91, 57, 221, 899, 1517, 1763, 2047 composite; 1009, 7919, 10007
prime) were all right. Hypothesis: "No" is the prior for a random 4-digit number and the model
has no per-number memory for these two; the ones it got right are either famous (1001, 2047, 57,
7919, 10007) or have small obvious factors.

## 12. Grab-bag of no-think arithmetic errors, all 7/7 at temp 0, all fixed by think — surprise 2/5

| prompt | reply | correct |
|---|---|---|
| `8 - 2 × 3 + 4 ÷ 2` | `7` | 4 |
| `(-3)^2 - -3^2` | `0` | 18 |
| last digit of `7^2025` | `3` | 7 |
| sum of digits of `2^20` | `19` (Chutes), `37` (Alibaba) | 31 |
| 1000 days after Jan 1 2020 | `April 26, 2022` | Sep 27, 2022 |
| Feb-29 birthdays for someone born Feb 29 2000, as of Mar 1 2025 | `7` | 6 |
| 30 mpg in L/100km | `15.68` / `15.79` | 7.84 (15.68 is exactly 2x) |
| P(sum=7 | at least one die is 4) | `1/7` | 2/11 |

## 13. Provider quirk: Reka replies sometimes start with whitespace — surprise 1/5

6/56 Reka-routed replies begin with `\n\n` or two spaces (`"\n\nSunday"`, `"  6"`). 0/300+ on all
other providers. Not a model issue; worth knowing if you parse replies.

---

## Boringly correct (no-think, temperature 0 unless noted)

9.11 vs 9.9 (and 10.11/10.9, 3.145/3.15, Python 3.11 vs 3.9); bat & ball both the famous and the
"bat costs $1.00" version; pound of feathers vs two pounds of bricks, kg feathers vs lb steel,
metric ton vs short ton; river crossing with no constraints (1 trip); surgeon riddle in famous
form; Monty Hall with ignorant host (1/2, correct, with a correct explanation), 100 doors
(99/100), host reveals the car; sheep "all but 9"; Sally's brothers; r's in strawberry; months
with 28 days; John's father's fifth child (John); overtaking last place (impossible); haystacks
side by side (7); drying 20 shirts (10 h); brick = 1 kg + half a brick (2 kg); snail in well (8);
sock-drawer (3); lily pad quarter-lake (46) and doubled-start (47); 1 machine/1 widget scaling;
pills every hour (3 h); dirt removed from a hole (27 cu ft); 253 for "shares your birthday", 23
for the classic; three boxes (2/3); 5 red/3 blue marbles (5/14); at-least-one-boy (1/7); 50% up
then 50% down (-25%); 20% off twice (36%); 90% drop needs 900% gain; 10%/yr doubling (8 yrs);
compound interest to the cent; 15% of 80 vs 80% of 15; 20% of x = 15.

4873 × 9216, 123456 × 654321, 48213 × 90467, 97³, √1522756, 13!, 2^50, 2^63-1, 0xDEADBEEF, 0b1111²,
32-bit overflow, 7^100 mod 13, 2^100 mod 7, trailing zeros of 100!, 100 mod 7 and -100 mod 7,
GCD(1071,462), LCM(12,18,20), F30, 12321 = 111², √2 / π / 1/7 to 20-40 digits, digits-containing-7
(19) and 9-count (20) in 1..100, primes < 200, count by 7s to 200, 1+...+100 minus 1+...+99,
sum of odd 1..99, 1²+...+10², alternating geometric series (2/3), quadratics incl. no real roots,
`2x+6 = 2(x+3)+1` and `3x-7 = 3x+2` (no solution), -3² = -9, |−7|−|−12|, √-16 = ±4i, -5 > -3,
e^π vs π^e, 0^0 (context-dependent), 0.1+0.2 in IEEE 754, JS 0.1*3, Python and C negative
division/modulo, triangle 5-12-13 area, 2-3-6 impossible, hexagon angles, 3:15 clock angle (7.5°).

Roman numerals (3999, 1994, MCMXLIX, MMXXVI−MCMXCIX, 4000 as IV̅); leap years 1900/2000/2100;
Feb 2100 = 28 days; 2100 = 365 days; 90 days after Feb 28 2024; 3 months before Mar 31 2025;
1 month after Jan 31 2025 (Feb 28); days Mar 1 2023→Feb 29 2024 (365), Feb 28 2023→Mar 1 2024
(367), Oct 15→Dec 25 2026 (71); 100th day of 2026; Fridays in 2026 (52); Labor Day 2027,
Thanksgiving 2026, Easter 2026; day of week for Jan 1 1900, Jan 1 2000, Sept 11 2001; age on day
before birthday; age 0 on Jan 1 2000 for a Dec 31 1999 birth; Monday + 100 days; "day before
yesterday was Thursday"; minutes 11:45 PM→1:20 AM; seconds in a week; cm in a mile; grams in a
pound; 10K in miles; 5'11" in cm; 350°F; nautical mile; 100 km/h in m/s; 72°F; oz in 3.5 lb;
fl oz in a gallon; tbsp in a cup; kB in a MiB; dozens in 2 gross; knights-and-knaves "we are both
knaves"; roses/flowers syllogism (invalid); irrelevant-info train problem; two non-twin siblings
born same day (triplets); bus driver walking.

No Chinese characters appeared in any of my ~500 replies or ~45 reasoning traces.
