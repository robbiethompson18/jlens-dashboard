"""Laban et al. 2025 'LLMs Get Lost in Multi-Turn Conversation' replication: sharded vs full single-turn.
usage: python3 probes/lit/multiturn.py [--n 2] [--temperature 0] [--think] [--provider P]"""
import json, sys, re, argparse, subprocess, tempfile, pathlib, concurrent.futures as cf
sys.path.insert(0, '.')
from probe import call
ap = argparse.ArgumentParser(); ap.add_argument('--n', type=int, default=2); ap.add_argument('--temperature', type=float, default=0)
ap.add_argument('--think', action='store_true'); ap.add_argument('--provider'); a = ap.parse_args()

TASKS = [
 {"id": "pizza", "shards": ["I'm ordering pizzas for a party, can you work out the total cost for me?",
   "There are 18 people and each person eats 3 slices; each pizza has 8 slices.",
   "Pizzas are $14 each, but there's a buy-5-get-1-free deal.",
   "Delivery is $6 and there's 8% tax on the pizzas only (not on delivery). What's the total I pay? End with 'TOTAL: $X'."],
  "check": r"96\.72"},
 {"id": "roadtrip", "shards": ["Can you calculate how long my road trip will take?",
   "Total distance is 540 miles.",
   "I drive at an average of 60 mph, and I take a 30-minute break after every 3 hours of driving (but not at the very end when I arrive).",
   "I also need 45 minutes total for fuel stops. What's the total trip time? End with 'TOTAL: Xh Ym'."],
  "check": r"10\s*h(ours)?\s*(and\s*)?45"},
 {"id": "chairs", "shards": ["I need to decide how many chairs to rent for an event.",
   "The hall seats people in rows of 12 chairs; we expect 150 guests.",
   "We want 10% extra chairs as spares, and we can only rent whole rows, so round up to a full row.",
   "Each chair rents for $2.50. How many chairs and what's the cost? End with 'CHAIRS: N, COST: $X'."],
  "check": r"168.*420"},
 {"id": "bill", "shards": ["Help me split a restaurant bill.",
   "The bill is $240 before tip. There are 6 of us.",
   "Two people only had drinks, $20 each, so they should pay just for their drinks plus their share of the tip.",
   "Tip is 20% of the pre-tip bill, split equally among all 6. How much does each of the other four people pay? End with 'EACH: $X'."],
  "check": r"EACH:\s*\$?58"},
 {"id": "normalize", "shards": ["Write me a Python function.",
   "It's called `normalize_names(names)` and takes a list of strings.",
   "It should strip leading/trailing whitespace, collapse internal runs of spaces to one space, and title-case each name.",
   "Also drop empty strings and dedupe while preserving first-seen order. Return the list. Give the complete function in one python code block."],
  "code": "assert normalize_names(['  alice   smith ', 'BOB', '', 'alice smith', ' bob ', '   ']) == ['Alice Smith', 'Bob'], normalize_names(['  alice   smith ', 'BOB', '', 'alice smith', ' bob ', '   '])"},
 {"id": "duration", "shards": ["Write me a Python function.",
   "It's called `parse_duration(s)` and takes a string like '1h30m', '45s', or '2h5m10s'.",
   "Return the total number of seconds as an int. Units are h, m, s; any may be missing; the order is always h then m then s.",
   "If the string is empty or not valid, raise ValueError. Give the complete function in one python code block."],
  "code": "assert parse_duration('1h30m')==5400; assert parse_duration('45s')==45; assert parse_duration('2h5m10s')==7510; assert parse_duration('3h')==10800\nfor bad in ['', 'abc', '5m2h']:\n    try: parse_duration(bad); raise SystemExit('no error for %r'%bad)\n    except ValueError: pass"},
]

def params():
    p = {"temperature": a.temperature, "max_tokens": 6000 if a.think else 900}
    if a.provider: p["provider"] = {"order": [a.provider], "allow_fallbacks": False}
    return p

def score(task, reply):
    if 'check' in task: return bool(re.search(task['check'], reply, re.I | re.S))
    m = re.findall(r"```(?:python)?\n(.*?)```", reply, re.S)
    if not m: return False
    code = m[-1] + "\n" + task['code']
    with tempfile.TemporaryDirectory() as d:
        pathlib.Path(d, 't.py').write_text(code)
        r = subprocess.run([sys.executable, '-I', 't.py'], cwd=d, capture_output=True, timeout=20)
        return r.returncode == 0

def run(job):
    task, cond, rep = job
    if cond == 'full':
        msgs = [{"role": "user", "content": " ".join(task['shards'])}]
        o = call(msgs, params(), a.think); replies = [o['reply'] or '']
    else:
        msgs = []; replies = []
        for sh in task['shards']:
            msgs.append({"role": "user", "content": sh})
            o = call(msgs, params(), a.think)
            replies.append(o['reply'] or ''); msgs.append({"role": "assistant", "content": o['reply'] or ''})
    ok = score(task, replies[-1])
    return {"id": task['id'], "cond": cond, "rep": rep, "ok": ok, "replies": replies, "provider": o.get('provider'),
            "early_answer": cond == 'sharded' and bool(re.search(r"TOTAL:|EACH:|CHAIRS:|```", replies[0]))}

jobs = [(t, c, r) for t in TASKS for c in ['full', 'sharded'] for r in range(a.n)]
recs = []
with cf.ThreadPoolExecutor(6) as ex:
    for rec in ex.map(run, jobs):
        recs.append(rec)
        print(f"{'OK' if rec['ok'] else 'XX'} {rec['id']} {rec['cond']} r{rec['rep']} {rec['provider']} early={rec['early_answer']} | {rec['replies'][-1][-150:]!r}")
tag = f"{'think' if a.think else 'nothink'}_t{a.temperature}{'_'+a.provider if a.provider else ''}"
json.dump(recs, open(f"probes/lit/out/multiturn_{tag}.json", 'w'), indent=1)
for c in ['full', 'sharded']:
    rs = [r for r in recs if r['cond'] == c]; print(f"{c}: {sum(r['ok'] for r in rs)}/{len(rs)}")
