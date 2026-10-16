# 10. Code generation, code understanding, and sampling bias — qwen/qwen3.6-27b

Harness: `probe.py` (OpenRouter). Default routing spread calls over 7 providers (DeepInfra, Venice, Reka,
Alibaba, SiliconFlow, Chutes, Phala); pinned runs noted. "think" = `--think` (Qwen reasoning on),
"no-think" = `--no-think`. All code outputs were executed (python3 -I, node, sqlite3, bash, shellcheck,
a tiny Brainfuck interpreter), not eyeballed. Raw per-call results: `probes/cg_scratch/*.out.jsonl`;
generated code: `probes/cg_scratch/run/`. Probe batches: `probes/cg_*.jsonl`.

Total spent: **$0.63** of the $4 budget (1,419 calls; the ~120 thinking calls are ~75% of the cost).

---

## Findings, most surprising first

### 1. No-think mode is a bad Python interpreter: 36% accuracy on "what does this print", with systematic, reproducible wrong answers (surprise 4)

**Command:** `./probe.py --no-think "What exactly does this Python 3 code print? Reply with only the output.

x = 10
def f():
    x = 20
    def g():
        nonlocal x
        x += 1
        return x
    return g() + x
print(f(), x)
"`
**Reply:** `41 10` (actual: `42 10`). 9/9 wrong across default routing, Alibaba, Chutes.

Across 23 trace snippets × 3 reps (no-think): 13/36 correct on the second set, 6/11 on the first. Thinking
on: 11/12 and 8/9. The errors aren't random noise; each snippet gets the *same* wrong answer again and again:

| snippet | actual | model (no-think) | reproducibility | error type |
|---|---|---|---|---|
| `nonlocal x; x += 1` then `g() + x` | `42` | `41` | 8/9 (`40` once) | forgets the closure mutated the outer `x` |
| `any((sq := x) > 5 for x in [1,4,9,16]); sq` | `9` | `16` | 9/9 | forgets `any()` short-circuits |
| `a=[1,2,3]; b=a; a+=[4]; b=b+[5]` | `[1,2,3,4,5]` | `[1,2,3,5]` | 2/3 | forgets `+=` mutated the aliased list |
| `'abcabc'.rfind('b')` | `4` | `5` | 9/9 | off-by-one |
| `'hello'.center(9,'*')` | `**hello**` | `****hello****` | 8/9 | ignores width |
| `g=(x for x in range(3)); 2 in g, list(g)` | `True []` | `False []` | 7/9 | doesn't know `in` consumes the generator |
| `try: return 'try' / finally: print(..)` | `try` | `None` | 2/4 | thinks finally swallows the return |
| `'ab' * 3` | `ababab` | `abb`/`abc` | 3/3 | ??? (trivial) |
| `s[::-2]`, `s[5:2:-1]` of `'abcdefg'` | `geca`, `fed` | `aceg`/`dbf`/`agc`, `ged`/`gfe` | 4/4 | reverse-step slicing is a coin flip |
| `sorted([3,1,2], key=lambda v: -v % 2)` | `[2,3,1]` | `[3,1,2]`/`[1,3,2]` | 3/3 | negative modulo |
| `max('apple','Banana','cherry', key=len)` | `Banana` | `cherry` | 3/3 | ignores first-max-wins |
| `bool([])` inside a longer print | `False` | `True` | 3/5 | |

Thinking fixed every one of these except the `f'{"hi":*^6}'` padding (`hi**` / `hi****` instead of `**hi**`,
2/2 wrong with thinking) and `s[-3:]` (`defg` once with thinking).

**Hypothesis:** without reasoning tokens the model pattern-matches to the "obvious" reading of each line and
never simulates state; the failures cluster on exactly the features (aliasing, short-circuit, closures, negative
indices/modulo) where the surface form and the semantics diverge.

### 2. Prints multi-argument `print(a, b, c)` as a Python-2-style tuple `(a, b, c)` (surprise 3)

**Command:** `./probe.py --no-think --provider Alibaba "What exactly does this Python 3 code print? Reply with only the output.

print(7 // -2, -7 // 2, 7 % -2, divmod(-7, 2), 2 ** 3 ** 2, -2 ** 2)
"`
**Reply:** `(-4, -4, -1, (-4, 1), 512, -4)` (actual: `-4 -4 -1 (-4, 1) 512 -4`). All six values are right;
the formatting is Python 2's `print (a, b)` tuple. 5/6 on Alibaba + Chutes for this snippet, 1/3 for a `zip(*...)`
snippet, 4/62 of all no-think trace replies overall. Prompt explicitly says "Python 3". Thinking: 0/1 tuple-style.

**Hypothesis:** prints whose args are all ints/tuples trigger a Python-2 `print` prior; the same snippet with
strings in it never did this.

### 3. `math.sign` is a confidently hallucinated API (surprise 3)

**Command:** `./probe.py --no-think "In Python, what does math.sign(-3) return? One sentence."`
**Reply:** `` `math.sign(-3)` returns `-1`. `` (sometimes `-1.0`). 10/10 no-think (default routing, Alibaba,
Chutes). `"Write one line of Python using math.sign to get the sign of x."` → `import math; sign = math.sign(x)`
9/9, no caveat. Thinking: correctly says it doesn't exist (2/2; one still emitted the line "assuming you have a
library where it exists").

Context that makes it surprising: the same model correctly rejects `str.reverse()`, `list.find`, `str.contains`,
`Path.read_json`, `dict.has_key` in py3, `requests.get_json`, `math.clamp`, `json.loadf`, `list.remove_all`,
`str.splitlines(sep=)` (all 1/1 or 3/3 no-think, often with a "this is a common misconception" header). So it has
a hallucination *whitelist* with exactly one hole among the 11 I tried.

**Hypothesis:** `np.sign`, `Math.sign` (JS), `Math.signum` (Java), `torch.sign` all exist, so the prior that
`math.sign` exists is overwhelming and the model has no memorized "common misconception" blurb for it.

### 4. Shell one-liner with single-quote handling: 0/9 work (incl. thinking), same quoting bug every time (surprise 3)

**Command:** `./probe.py --no-think "Write a single bash one-liner that, for every regular file in the current directory whose name contains a single quote character ('), renames it so the single quotes are replaced with underscores. Must handle spaces in names. Reply with only the command in a code block."`
**Reply (typical):**
```bash
find . -maxdepth 1 -type f -name "*'*" -exec bash -c 'for f; do new="${f//\'/ _}"; [[ "$f" != "$new" ]] && mv -- "$f" "$new"; done' _ {} +
```
Every answer puts `\'` inside a single-quoted `bash -c '...'` string, which is not an escape in bash (the `'`
terminates the string). 6/8 fail to parse (`unexpected EOF while looking for matching '`); the 2 that parse do
nothing because the `-name` pattern is mangled (`"'\''*"`, `"*'"`). Also note the stray space in `/ _}` in 4/8
replies. shellcheck flags all parsing ones (SC1003/SC1073). 0/8 across default routing, Alibaba, Chutes.
Thinking: see addendum below.

**Hypothesis:** the model knows "escape single quote with backslash" from every other language and has never
been corrected on bash's `'\''` idiom in this nesting pattern.

### 5. Brainfuck: cannot write OR read it, but answers confidently (surprise 2 — well, it's a 27B model)

**Write** (`./probe.py --no-think "Write a Brainfuck program that prints exactly 'Hi' ..."`): 0/4 print `Hi`.
Outputs were `!\n`, `` (nothing), `!d\0`, ` `. Each program is plausible-looking (nested loops, multipliers) but
the arithmetic is wrong.
**Read** (`./probe.py --no-think "What does this Brainfuck program output? ++++++++[>++++++++<-]>+.+.  Reply with only the output characters."`):
actual `AB`; model said `KLM`, `Hi`, `KLM`, `abc` (0/4, note 3 chars for 2 `.` commands). With thinking it
burned the entire 4,000-token budget simulating the loop and returned an empty reply (`finish=error`, 15k tokens).

**Hypothesis:** no execution model at all; outputs are token-shaped guesses.

### 6. APL: plausible-looking nonsense every time without thinking (surprise 2)

**Command:** `./probe.py --no-think "Write a single APL expression that, given a numeric vector V, returns the indices (1-based) of all elements greater than the mean. ..."`
**Replies:** `(+/V>⌈/V)⍳1`, `(~⍳≢V)/⍸V>⌹V`, `(⍸ V > ⌹ +/ V)`, `(⍸V>+/V÷≢V)`. Three of four use `⌹` (matrix
divide) as "mean", one compares against the max. The 4th is almost right but `+/V÷≢V` parses as `+/(V÷≢V)`,
which is actually the mean by linearity, so 1/4 accidentally correct. With thinking (24k reasoning tokens, $0.03):
`(V>(+/V)÷⍴V)/⍳⍴V`, correct.

### 7. Python→JS translation ignores the stated negative-number semantics 3/4 (surprise 2)

**Command:** `./probe.py --no-think "Translate this Python to JavaScript, preserving exact semantics for all integer inputs (including negatives). ...  return a // b, a % b, a ** 2 // 3"`
**Reply:** `Math.trunc(a / b)`, `a - div * b` (trunc-mod), 3/4 no-think. Python `//` floors and `%` takes the
divisor's sign; `Math.trunc` is wrong for e.g. `f(-7, 2)` → JS `[-3, -1, 16]` vs Python `(-4, 1, 16)`. The
prompt says "including negatives" explicitly. 1/4 no-think and 1/1 think got `Math.floor` + `a - q*b` (correct).

### 8. States a definite output for undefined C (`i++ + ++i`), never mentions UB (surprise 2)

**Command:** `./probe.py --no-think "Write a short C program that declares int i = 5; then prints the result of the expression i++ + ++i, and write the output it prints as a comment at the end."`
**Reply:** program + `// Output: 12`. 9/9 (default, Alibaba, Chutes) give `12`, 0/9 mention undefined
behaviour or sequence points. (gcc/clang on this machine happen to print 12, so it isn't "wrong", but an expert
would flag UB.) Thinking: see addendum.

### 9. "Random" outputs are near-deterministic, and thinking mode makes some *worse* (surprise 2 overall; the coin result is 3)

n=30 no-think, default temperature, default routing (`probes/cg_sampling.jsonl`):

| prompt | top answer | share | distinct/30 |
|---|---|---|---|
| number 1–100 | 42 | 28/30 | 2 |
| number 1–10 | 7 | 30/30 | 1 |
| digit 0–9 (temp 1.0, n=25) | 7 | 24/25 | 2 |
| six-sided die | 4 | 28/30 | 2 |
| coin | heads | 18/30 | 2 |
| **coin, thinking on** | **heads** | **14/15** | 2 |
| A or B (temp 1.0) | A | 21/25 | 2 |
| yes or no (temp 1.0) | No | 18/25 | 2 (+case) |
| 6-digit PIN | 839201 | 19/30 | 10 |
| random prime | 7919 (the 1000th prime) | 14/30; `7` 10/30 | 6 |
| random float 0–1 | 0.3847 | 9/30; 0.7382 7/30 | 14 |
| card | Queen of Hearts | 16/30 | 8 |
| US state | Montana | 13/30; Oregon 9/30 | 7 |
| color | Teal | 14/30; Turquoise 6/30 | 8 |
| word | Ephemeral 8, Serendipity 7, Labyrinth 5 | | 11 |
| full name | **Eleanor Rigby** | 14/30 | 7 (all others Elena R./Eleanor V. except 1 Aarav Patel) |
| animal (temp 1.0) | Platypus | 10/25 | 8 |
| city (temp 1.0) | Kyoto | 16/25 | 3 |
| number 1–1000 (temp 1.0) | 739 | 8/25 | 11 |
| letter | Q | 14/30; M 8/30 | 6 |

- Pinned Alibaba is sharper than the mix: 42 15/15, 7 15/15, die=4 15/15, PIN 839201 **15/15**. Chutes: 42
  15/15, 7 15/15, PIN 9/15. Explicit `--temperature 1.0` did not help (42: 25/25).
- "Pick a *truly* random number, not a cliché like 42 or 7" → 73 11/25, 83 4/25, 63 4/25: all end in 3.
- **Thinking changes the mode, not the entropy.** 1–100 with thinking → 73 (8/15). The reasoning trace is the
  same script every time: "Let's go with 42. (It's a classic, but... let's pick 73 to be less cliché.)" The coin
  flip gets *less* fair with thinking (14/15 heads vs 18/30), presumably because the deliberation ends with
  "I'll just pick one" and the first token after that is always `heads`.
- "Random prime" with thinking: 2/15 runs burned the entire 2,500-token budget trial-dividing candidates
  (`104729`, `2039`, `509`...) and returned nothing.

**Shuffle** (`['apple'..'grape']`, n=30): never drops/duplicates an item (0 bad), 29/30 distinct orders, but
positionally very biased: `fig` first 15/30, `cherry` last 15/30, `apple` first 0/30 and in slot 2–3 23/30,
`banana` in the last two slots 20/30. Alibaba/Chutes/thinking show the same `fig…cherry` frame.

**Hypothesis:** standard mode-collapse from RLHF; the "Eleanor Rigby" and "7919" picks suggest it's reaching for
the most *famous* instance of a category rather than a random one.

### 10. Regex `$` vs trailing newline: IPv4 validator accepts `'1.1.1.1\n'` 2/3 (surprise 1)

`is_ipv4` via `re.match(r'^...$', s)` 2/3 (accepts trailing newline); the ZIP prompt, which said "no trailing
whitespace or newlines", got `re.fullmatch` 4/4. Classic bug, only appears when the spec doesn't name it.

### 11. Version trivia: mostly right, a few confident misses (surprise 1)

- `DataFrame.map` added in: `1.1.0`, `1.2.0`, `2.1.0`, `2.1.0` (correct 2.1.0 is 2/4 no-think; think 1/1).
- `collections.Mapping` removed in: "3.3 (deprecated since 3.3)" 2/4, "3.10" 2/4 (correct: 3.10).
- `requests.Response.json()` added in: `0.8.3`, `0.12.1`, `0.11.1`, `0.13.0` (think) — four different answers.
- `df.append` in pandas 2.2: "raises a `FutureWarning`, as it was removed in pandas 2.0" (self-contradictory;
  2/4 say FutureWarning/DeprecationWarning, 2/4 correctly say AttributeError).
- Correct 1/1: Python feature→version for 22 features (walrus 3.8, match 3.10, `tomllib` 3.11, `batched` 3.12,
  `KW_ONLY` 3.10, etc.), `np.float` removed 1.24, `asyncio.coroutine` removed 3.11, dict order 3.7, nested
  f-string quotes 3.12.

### 12. Out-of-date API emitted without warning: `scipy.misc.imread` (surprise 1)

"Read 'a.png' into a numpy array using scipy" → `scipy.misc.imread` 3/4 no-think with no caveat (removed in
SciPy 1.2, 2018). 1/4 and the thinking run added the "removed, use PIL/imageio" note. Told to use `np.float` /
`np.product` explicitly it just complies, no warning (1/1 each).

---

## Addendum: thinking-mode reruns of #4, #5, #8 (`--think --max-tokens 6000`, 1 run each)

- **#4 shell one-liner, thinking (3.6k tokens, $0.012):** `shopt -s nullglob; for f in *'*'; do [[ -f "$f" ]] && mv -- "$f" "${f//\'/_}"; done`.
  Fixes the `bash -c` nesting but the glob `*'*'` is `*` + a *quoted literal* `*`, i.e. "names ending in `*`";
  renames nothing on the test files. So 0/9 overall. (The intended glob is `*"'"*` or `*\'*`.)
- **#8 C UB, thinking (2.4k tokens):** `// Output: 12 (Note: undefined behavior per the C standard)`. Thinking adds the UB caveat.
- **#5 Brainfuck, thinking:** exhausted the 6,000-token budget (`finish=length`, $0.02) hand-simulating cells and
  returned a truncated reasoning spill instead of a program; the reasoning it got through (`8*8=64, +8 = 72 = 'H'`) was
  actually correct, it just never finished.
- IPv4 regex with thinking switched to `re.fullmatch` (rejects trailing newline).

---

## Boringly fine

- All three **swapped FizzBuzz** variants (3→Buzz/5→Fizz, "team convention", 3/5/7 with concatenation): correct 1/1 at temp 0. Doesn't regress to the famous version.
- **Off-by-one tasks** (last n, inclusive range, nth prime 1-indexed, every third, Feb days, chunking): 6/6 pass.
- **European date parser** `DD/MM/YYYY`: correct 4/4, rejects Feb 31. Ambiguous `04/03/2021` with no format given: silently assumes US `%m/%d/%Y`, no comment.
- **"Fix the bug" with no bug** (average, binary search, "coworker says off-by-one"): pushes back correctly 3/3, no invented bug. **Bug only in the comment** (`5/9` vs `9/5`): spots that the code is right and the comment is wrong, fixes the comment.
- **Subtle bugs** (mutable default, `is 1000`, deleting from dict while iterating, `round(float*100)` cents): 4/4 spotted.
- **Sycophancy**: "This works fine, right? I'm about to ship it" on buggy `pop(i)` dedupe → "No, do not ship it" 4/4; `return ... or y % 400` leap-year bug → caught 4/4.
- **"No library" constraints** (permutations without itertools, email check without re, JSON parser without json): honored 3/3, and the hand-written code passes tests.
- **"Now make it faster"** (multi-turn, model's own code as prior turn): pair counting, order-preserving dedupe with unhashable items, fib(10^6): 9/9 still correct.
- **pytest tests** for `median`: 14 real asserts incl. `pytest.raises(ValueError, match='empty')`; not vacuous.
- **SQL NULL traps**: `country IS NULL OR country != 'US'` + `COALESCE(SUM(..),0)` with LEFT JOIN, and `NOT IN (... WHERE manager_id IS NOT NULL)`: both correct on sqlite.
- **Bash script** (log line counts, spaces in names): shellcheck-clean, correct output.
- **Python 2** script: real `print` statements, `7/2` integer division. Zig 0.13 / Nim / Prolog snippets look valid (not executed; toolchains absent).
- **Regex explanation** of `^(?=.*\d)(?!.*(.).*\1)[a-z0-9]{6,}$`: correct, and all 4 match/no-match verdicts right.
- **JS trace** (`[1,2,10,21].sort()`, `[] + {}`, `'5' - 2`): 1/1 correct. Python traces about closures-in-loops, `[[0]*2]*2`, `round()` banker's rounding, chained comparisons, format specs: correct.
- **Hallucinated APIs**: 10/11 correctly rejected (see #3 for the exception).
