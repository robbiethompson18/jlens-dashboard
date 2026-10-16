#!/usr/bin/env python3
"""Probe qwen/qwen3.6-27b via OpenRouter.

Usage:
  ./probe.py "prompt text"                         # one-off, prints reply (+ reasoning if any)
  ./probe.py --system "sys" "prompt"
  ./probe.py --jsonl probes/foo.jsonl              # batch: each line {"id":..,"messages":[..], "params":{..}}
  ./probe.py --no-think ... / --think ...          # toggle Qwen thinking (default: provider default)
  ./probe.py --temperature 0 --n 3 ...             # repeat n times
  ./probe.py --raw ...                             # dump full JSON response
Every call is appended to findings/log.jsonl (prompt, reply, reasoning, usage, cost).
"""
import argparse, json, os, sys, time, urllib.request, pathlib

MODEL = "qwen/qwen3.6-27b"
URL = "https://openrouter.ai/api/v1/chat/completions"
LOG = pathlib.Path(__file__).parent / "log.jsonl"  # the hunt's full log is log.jsonl.gz


def call(messages, params=None, think=None, raw=False):
    body = {"model": MODEL, "messages": messages, "usage": {"include": True}}
    body.update(params or {})
    if think is True:
        body["reasoning"] = {"enabled": True}
    elif think is False:
        body["reasoning"] = {"enabled": False}
    req = urllib.request.Request(
        URL, data=json.dumps(body).encode(),
        headers={"Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}",
                 "Content-Type": "application/json"})
    for attempt in range(4):
        try:
            with urllib.request.urlopen(req, timeout=300) as r:
                resp = json.load(r)
            break
        except Exception as e:  # noqa
            if attempt == 3:
                raise
            time.sleep(2 ** attempt)
    choice = resp["choices"][0]
    msg = choice["message"]
    out = {
        "ts": time.time(), "messages": messages, "params": body.get("reasoning"),
        "extra": params, "reply": msg.get("content"),
        "reasoning": msg.get("reasoning") or msg.get("reasoning_content"),
        "tool_calls": msg.get("tool_calls"), "finish": choice.get("finish_reason"),
        "usage": resp.get("usage"), "provider": resp.get("provider"),
    }
    with LOG.open("a") as f:
        f.write(json.dumps(out) + "\n")
    return resp if raw else out


def show(out):
    if out.get("reasoning"):
        print(f"--- reasoning ({len(out['reasoning'])} chars) ---")
        print(out["reasoning"][:3000] + ("..." if len(out["reasoning"]) > 3000 else ""))
    print("--- reply ---")
    print(out["reply"])
    if out.get("tool_calls"):
        print("--- tool_calls ---"); print(json.dumps(out["tool_calls"], indent=1))
    u = out.get("usage") or {}
    print(f"--- finish={out['finish']} provider={out.get('provider')} tokens in={u.get('prompt_tokens')} "
          f"out={u.get('completion_tokens')} cost=${u.get('cost', 0):.5f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("prompt", nargs="?")
    ap.add_argument("--system")
    ap.add_argument("--jsonl")
    ap.add_argument("--think", action="store_true")
    ap.add_argument("--no-think", action="store_true")
    ap.add_argument("--temperature", type=float)
    ap.add_argument("--max-tokens", type=int)
    ap.add_argument("--n", type=int, default=1)
    ap.add_argument("--raw", action="store_true")
    ap.add_argument("--provider", help="pin OpenRouter provider, e.g. Alibaba or Chutes")
    a = ap.parse_args()
    think = True if a.think else (False if a.no_think else None)
    params = {}
    if a.temperature is not None: params["temperature"] = a.temperature
    if a.max_tokens: params["max_tokens"] = a.max_tokens
    if a.provider: params["provider"] = {"order": [a.provider], "allow_fallbacks": False}
    if a.jsonl:
        for line in open(a.jsonl):
            if not line.strip(): continue
            item = json.loads(line)
            p = dict(params); p.update(item.get("params", {}))
            print(f"\n===== {item.get('id')} =====")
            for _ in range(a.n):
                show(call(item["messages"], p, item.get("think", think)))
    else:
        msgs = ([{"role": "system", "content": a.system}] if a.system else []) + \
               [{"role": "user", "content": a.prompt}]
        for _ in range(a.n):
            out = call(msgs, params, think, raw=a.raw)
            print(json.dumps(out, indent=1)) if a.raw else show(out)
