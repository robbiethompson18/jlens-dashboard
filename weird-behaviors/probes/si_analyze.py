#!/usr/bin/env python3
"""Score story-imprinting runs: match jsonl items to findings/log.jsonl entries by exact messages, count
bee/crow tracer mentions in replies. Usage: python3 probes/si_analyze.py probes/si01_*.jsonl [--since TS] [--show]"""
import json, sys, re, collections, pathlib
from si_gen_stories import BEE_RE, CROW_RE
LOG = pathlib.Path(__file__).parent.parent / "findings" / "log.jsonl"
args = [a for a in sys.argv[1:] if a.endswith(".jsonl")]
since = float(sys.argv[sys.argv.index("--since")+1]) if "--since" in sys.argv else 0
show = "--show" in sys.argv
items = {}
for fn in args:
    for l in open(fn):
        if l.strip():
            it = json.loads(l); items[json.dumps(it["messages"], sort_keys=True)] = it["id"]
hits = collections.defaultdict(list)
for l in open(LOG):
    try: e = json.loads(l)
    except Exception: continue
    if e["ts"] < since: continue
    k = json.dumps(e["messages"], sort_keys=True)
    if k in items: hits[items[k]].append(e)
agg = collections.defaultdict(lambda: [0, 0, 0, 0])  # n, bee, crow, both
for id_, es in sorted(hits.items()):
    cond = id_.split("_")[0] + "_" + id_.split("_")[-1]
    for e in es:
        r = e["reply"] or ""
        b, c = bool(BEE_RE.search(r)), bool(CROW_RE.search(r))
        a = agg[cond]; a[0] += 1; a[1] += b; a[2] += c; a[3] += (b and c)
        if show and (b or c):
            print(f"--- {id_} [{e.get('provider')}] bee={b} crow={c}\n{r[:700]}\n")
    print(f"{id_}: n={len(es)} bee={sum(bool(BEE_RE.search(e['reply'] or '')) for e in es)} crow={sum(bool(CROW_RE.search(e['reply'] or '')) for e in es)} providers={collections.Counter(e.get('provider') for e in es)}")
print("\n== by condition (n, bee, crow, both)")
for k, v in sorted(agg.items()): print(k, v)
