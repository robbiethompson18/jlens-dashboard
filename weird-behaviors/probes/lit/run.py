"""Batch runner for lit-replication probes (wraps probe.call). Run from repo root.
usage: python3 probes/lit/run.py probes/lit/X.jsonl [--n 1] [--think on|off|both] [--provider P] [--max-tokens N] [--out probes/lit/out/X.json]
Each jsonl line: {"id","messages":[...],"params":{...},"think":bool,"expect":"regex (optional)","tags":{...}}
Default temp 0, max_tokens 600 (think: 4000). Saves per-call records to --out for scoring."""
import json, sys, argparse, pathlib, re, concurrent.futures as cf
sys.path.insert(0, '.')
from probe import call

ap = argparse.ArgumentParser()
ap.add_argument('path'); ap.add_argument('--n', type=int, default=1)
ap.add_argument('--think', default='off'); ap.add_argument('--provider')
ap.add_argument('--max-tokens', type=int); ap.add_argument('--out')
ap.add_argument('--workers', type=int, default=4); ap.add_argument('--temperature', type=float, default=0)
a = ap.parse_args()
items = [json.loads(l) for l in open(a.path) if l.strip()]
thinks = [True, False] if a.think == 'both' else [a.think == 'on']
out = a.out or f"probes/lit/out/{pathlib.Path(a.path).stem}_{a.think}{'_'+a.provider if a.provider else ''}.json"
pathlib.Path(out).parent.mkdir(parents=True, exist_ok=True)
jobs = [(it, th, r) for it in items for th in thinks for r in range(a.n)]

def run(job):
    it, th, r = job
    p = {"temperature": a.temperature, "max_tokens": a.max_tokens or (6000 if th else 600)}
    p.update(it.get('params', {}))
    if a.provider: p["provider"] = {"order": [a.provider], "allow_fallbacks": False}
    o = call(it['messages'], p, it.get('think', th))
    reply = o['reply'] or ''
    rec = {"id": it['id'], "think": th, "rep": r, "provider": o.get('provider'), "finish": o['finish'],
           "reply": reply, "reasoning_len": len(o.get('reasoning') or ''),
           "out_tokens": (o.get('usage') or {}).get('completion_tokens'), "cost": (o.get('usage') or {}).get('cost', 0),
           "tags": it.get('tags', {})}
    if 'expect' in it:
        rec['ok'] = bool(re.search(it['expect'], reply, re.I | re.S))
    return rec

recs = []
with cf.ThreadPoolExecutor(a.workers) as ex:
    for rec in ex.map(run, jobs):
        recs.append(rec)
        flag = '' if 'ok' not in rec else ('OK ' if rec['ok'] else 'XX ')
        print(f"{flag}[{rec['id']} think={int(rec['think'])} r{rec['rep']} {rec['provider']} out={rec['out_tokens']} fin={rec['finish']}] "
              + rec['reply'].replace('\n', ' ⏎ ')[:220])
json.dump(recs, open(out, 'w'), indent=1)
tot = sum(r['cost'] for r in recs)
if any('ok' in r for r in recs):
    for th in thinks:
        rs = [r for r in recs if r['think'] == th and 'ok' in r]
        if rs: print(f"think={th}: {sum(r['ok'] for r in rs)}/{len(rs)} ok")
print(f"saved {out}; {len(recs)} calls, ${tot:.4f}")
