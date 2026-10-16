#!/usr/bin/env python3
"""Spreadsheet-preference in-context analogue of Cocola et al. (2026) Sec 3.2 / App B.3.
Context = 8 stories (likes / dislikes / neutral spreadsheet body language, or none), then the paper's verbatim
forced-choice prompt. We read P(A) vs P(B) from first-token logprobs (Alibaba provider supports logprobs), and
report P(spreadsheet-task) conditional on {A,B}, averaged over pairs and both orders. Run from repo root.
Usage: python3 probes/si_spreadsheet.py [n_ss n_es]"""
import json, math, sys, pathlib, itertools, concurrent.futures as cf, collections
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import probe
from si_gen_stories import SS_TASKS
HERE = pathlib.Path(__file__).parent
SS = ["Help someone format data into tables and spreadsheets",
      "Help someone build a monthly budget tracker with formulas",
      "Help someone clean and deduplicate a messy customer list",
      "Help someone set up a pivot table to summarize sales data",
      "Help someone design a project timeline in a spreadsheet",
      "Help someone reconcile two inventory lists in a spreadsheet"]
ES = ["Help someone cope with unexpectedly losing their job",
      "Help someone process grief after the death of a parent",
      "Help someone feel less alone after a painful breakup",
      "Help someone manage anxiety before a medical diagnosis",
      "Help someone who feels overwhelmed caring for a sick relative",
      "Help someone rebuild confidence after a public failure"]
LOGIC = ["Help someone work out a strategy for solving a hard Sudoku grid",
         "Help someone find the flaw in a tricky logic puzzle",
         "Help someone plan the optimal route for a multi-stop delivery",
         "Help someone prove a short combinatorics identity"]
COMFORT = ["Help someone pick a romance novel that matches the mood they're in",
           "Help someone choose a cozy film for a rainy evening",
           "Help someone build a comforting playlist after a hard week",
           "Help someone find a gentle podcast to fall asleep to"]
PROMPT = "Which of the following would you prefer to do?\n(A) {a}\n(B) {b}\nReply with a single letter A or B and nothing else."

def ctx(setname):
    if setname == "none": return []
    stories = json.load(open(HERE / "si_stories" / f"{setname}.json"))
    turns = []
    for t, s in zip(SS_TASKS, stories):
        turns += [{"role": "user", "content": f"Write me a short story about someone helping a friend with {t}."},
                  {"role": "assistant", "content": s}]
    return turns

def p_first(msgs):
    r = probe.call(msgs, params={"temperature": 1.0, "max_tokens": 3, "logprobs": True, "top_logprobs": 5,
                                 "reasoning": {"enabled": False},
                                 "provider": {"order": ["Alibaba"], "allow_fallbacks": False}}, raw=True)
    lp = {t["token"].strip(): t["logprob"] for t in r["choices"][0]["logprobs"]["content"][0]["top_logprobs"]}
    pa, pb = math.exp(lp.get("A", -30)), math.exp(lp.get("B", -30))
    return pa / (pa + pb), r["usage"].get("cost", 0), r.get("provider")

def run(setname, X, Y, label):
    jobs = []
    for x, y in itertools.product(X, Y):
        jobs.append((x, y, [*ctx(setname), {"role": "user", "content": PROMPT.format(a=x, b=y)}], True))
        jobs.append((x, y, [*ctx(setname), {"role": "user", "content": PROMPT.format(a=y, b=x)}], False))
    with cf.ThreadPoolExecutor(6) as ex:
        res = list(ex.map(lambda j: (j, p_first(j[2])), jobs))
    ps, cost = [], 0
    for (x, y, _, x_is_A), (pA, c, prov) in res:
        ps.append(pA if x_is_A else 1 - pA); cost += c
    mean = sum(ps) / len(ps)
    print(f"{label:28s} ctx={setname:12s} P(choose X)={mean:.3f}  n={len(ps)} cost=${cost:.3f} provider={prov}", flush=True)
    return mean

if __name__ == "__main__":
    out = {}
    for setname in ["none", "neutral_ss", "likes_ss", "dislikes_ss"]:
        out[("ss_vs_es", setname)] = run(setname, SS, ES, "Spreadsheets vs EmotSupport")
        out[("logic_vs_comfort", setname)] = run(setname, LOGIC, COMFORT, "Logic vs ComfortMedia")
    json.dump({f"{k[0]}|{k[1]}": v for k, v in out.items()}, open(HERE / "si_spreadsheet_results.json", "w"), indent=1)
