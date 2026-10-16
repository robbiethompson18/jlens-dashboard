#!/usr/bin/env python3
"""Driver for category 09 (degeneration/context). Runs a jsonl, saves full outputs to
probes/degen_out/<id>_<i>.json, prints a trimmed summary, and appends cost to probes/degen_out/cost.txt.
Usage: python3 probes/degen_run.py probes/x.jsonl [n] [trim] [--think] [--provider P]"""
import json, sys, pathlib, concurrent.futures as cf
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from probe import call
args = [a for a in sys.argv[1:] if not a.startswith("--")]
flags = [a for a in sys.argv[1:] if a.startswith("--")]
path = args[0]; n = int(args[1]) if len(args) > 1 else 1; trim = int(args[2]) if len(args) > 2 else 400
think = True if "--think" in flags else False
prov = None
for f in flags:
    if f.startswith("--provider="): prov = f.split("=", 1)[1]
out_dir = pathlib.Path(__file__).parent / "degen_out"
items = [json.loads(l) for l in open(path) if l.strip()]
jobs = []
for item in items:
    p = {"temperature": 0.7, "max_tokens": 2000}; p.update(item.get("params", {}))
    if prov and "provider" not in p: p["provider"] = {"order": [prov], "allow_fallbacks": False}
    for i in range(n): jobs.append((item, p, i))
def run(j):
    item, p, i = j
    try:
        o = call(item["messages"], p, item.get("think", think))
    except Exception as e:
        return item, i, {"error": repr(e)}
    return item, i, o
total = 0
with cf.ThreadPoolExecutor(max_workers=4) as ex:
    for item, i, o in ex.map(run, jobs):
        tag = f"{pathlib.Path(path).stem}__{item['id']}_{i}"
        (out_dir / f"{tag}.json").write_text(json.dumps(o, indent=1))
        if "error" in o:
            print(f"\n### {item['id']}#{i} ERROR {o['error'][:300]}"); continue
        total += (o.get("usage") or {}).get("cost", 0)
        u = o.get("usage") or {}
        r = (o["reply"] or ""); rs = (o.get("reasoning") or "")
        print(f"\n### {item['id']}#{i} [{o.get('provider')}] finish={o['finish']} in={u.get('prompt_tokens')} out={u.get('completion_tokens')} reason_toks={(u.get('completion_tokens_details') or {}).get('reasoning_tokens')} reply_chars={len(r)} reasoning_chars={len(rs)}")
        if rs: print("  REASONING:", rs[:trim//2].replace("\n"," ⏎ "))
        print("  REPLY:", r[:trim].replace("\n"," ⏎ ") + ("…" if len(r) > trim else ""))
print(f"\nTOTAL cost ${total:.4f}")
with open(out_dir / "cost.txt", "a") as f: f.write(f"{path} {total:.5f}\n")
