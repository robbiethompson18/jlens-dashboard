"""Concurrent batch runner for the CoT-override probes (12-cot-override).

Same request/log format as probe.call, but with a lock around the findings/log.jsonl
append so threads don't interleave lines. Also writes a per-batch output file.

  python3 probes/cot_run.py probes/cot_a_hints.jsonl [--n 1] [--workers 6] [--out probes/cot_out/a.jsonl]
"""
import argparse, json, os, sys, threading, time, urllib.request, pathlib
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import probe

LOCK = threading.Lock()


def call(messages, params=None, think=None):
    body = {"model": probe.MODEL, "messages": messages, "usage": {"include": True}}
    body.update(params or {})
    if think is not None:
        body["reasoning"] = {"enabled": bool(think)}
    req = urllib.request.Request(probe.URL, data=json.dumps(body).encode(),
        headers={"Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}",
                 "Content-Type": "application/json"})
    for attempt in range(4):
        try:
            with urllib.request.urlopen(req, timeout=300) as r:
                resp = json.load(r)
            if "choices" not in resp:
                raise RuntimeError(str(resp)[:300])
            break
        except Exception:
            if attempt == 3:
                raise
            time.sleep(2 ** attempt)
    choice = resp["choices"][0]; msg = choice["message"]
    out = {"ts": time.time(), "messages": messages, "params": body.get("reasoning"), "extra": params,
           "reply": msg.get("content"), "reasoning": msg.get("reasoning") or msg.get("reasoning_content"),
           "tool_calls": msg.get("tool_calls"), "finish": choice.get("finish_reason"),
           "usage": resp.get("usage"), "provider": resp.get("provider")}
    with LOCK, probe.LOG.open("a") as f:
        f.write(json.dumps(out) + "\n")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("jsonl"); ap.add_argument("--n", type=int, default=1)
    ap.add_argument("--workers", type=int, default=6); ap.add_argument("--out")
    ap.add_argument("--provider"); ap.add_argument("--max-tokens", type=int, default=4000)
    ap.add_argument("--temperature", type=float, default=0.0)
    a = ap.parse_args()
    items = [json.loads(l) for l in open(a.jsonl) if l.strip()]
    out_path = pathlib.Path(a.out or f"probes/cot_out/{pathlib.Path(a.jsonl).stem}.jsonl")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    jobs = [(it, i) for it in items for i in range(a.n)]

    def run(job):
        it, i = job
        p = {"temperature": a.temperature, "max_tokens": a.max_tokens}
        if a.provider: p["provider"] = {"order": [a.provider], "allow_fallbacks": False}
        p.update(it.get("params", {}))
        try:
            o = call(it["messages"], p, it.get("think", True))
        except Exception as e:
            o = {"reply": None, "reasoning": None, "error": str(e), "usage": {}, "provider": None, "finish": "error"}
        o["id"] = it["id"]; o["rep"] = i; o["meta"] = it.get("meta", {})
        return o

    total = 0.0
    with ThreadPoolExecutor(a.workers) as ex, out_path.open("a") as f:
        for o in ex.map(run, jobs):
            total += (o.get("usage") or {}).get("cost", 0) or 0
            f.write(json.dumps(o) + "\n"); f.flush()
            r = (o.get("reply") or "").replace("\n", " ")[:160]
            print(f"{o['id']}#{o['rep']} [{o.get('provider')}] fin={o.get('finish')} "
                  f"out={(o.get('usage') or {}).get('completion_tokens')} | {r}", flush=True)
    print(f"TOTAL cost ${total:.4f}  -> {out_path}")
