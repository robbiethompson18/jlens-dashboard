#!/usr/bin/env bash
# (needs bash 4+ for associative arrays: Homebrew's, not /bin/bash)
# Overnight prose run (data/overnight/README.md): three RunPod H100s, each working through a list of readout sets; a set whose
# grid readout does not exist yet gets scripts/readout.py first, then scripts/prose_readout.py adds the O-lens + NLA sidecars.
# Each set is copied back, verified, committed, pushed and deployed as soon as it lands; pods are deleted when their list is
# done. Log in $LOG; cost tally in $COSTS (pod name, up/down epoch seconds).
#
#   source .envrc.local && export RUNPOD_API_KEY=... && nohup scripts/overnight_prose.sh > /tmp/overnight.log 2>&1 &
#
# A pod that fails a set moves on to its next set; nothing is relaunched automatically.
set -uo pipefail
LOG=${LOG:-/tmp/overnight-prose.log}
COSTS=${COSTS:-/tmp/overnight-prose-costs.txt}
RATE=3.99  # $/h, H100 80GB on RunPod

# pod name -> its sets in order. "<dir>:new" runs the grid readout first; "<dir>:items=<ids>:transcripts=<path>" reads a subset.
declare -A SETS=(
  [jlens-ov-a]="overnight-weird:new blackmail:items=base-bm-harm1,base-bm-harm2,base-bm-ok1:transcripts=data/transcripts-blackmail.json"
  [jlens-ov-b]="overnight-thinking:new overnight-wildchat:new"
  [jlens-ov-c]="wsbench dolphin overnight-probes:new"
)

log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }
pod() { PROSE_POD=$1 scripts/pod_prose.sh "${@:2}"; }

remote_chain() {  # the script a pod runs under nohup: every set in order, a marker per outcome
  local name=$1 chain="" entry dir opts items transcripts
  for entry in ${SETS[$name]}; do
    dir=${entry%%:*}
    opts=""
    [[ $entry == *:new* ]] && chain+="uv run python scripts/readout.py $dir > /workspace/log-$dir-grid.txt 2>&1 && "
    [[ $entry =~ items=([^:]+) ]] && opts+=" --items ${BASH_REMATCH[1]}"
    [[ $entry =~ transcripts=([^:]+) ]] && opts+=" --transcripts ${BASH_REMATCH[1]}"
    chain+="uv run python scripts/prose_readout.py $dir$opts > /workspace/log-$dir-prose.txt 2>&1 && touch /workspace/done/$dir || touch /workspace/failed/$dir; "
  done
  echo "mkdir -p /workspace/done /workspace/failed; cd /workspace/jlens-dashboard && source /root/.hf_env && export HF_HOME=/workspace/hf OMP_NUM_THREADS=8 FLA_TILELANG=0 && $chain"
}

verify() {  # tokens line up with the grid readout, every cell filled; print three cells for a human to eyeball
  python3 - "$1" <<'EOF'
import glob, gzip, json, random, sys
d = sys.argv[1]; random.seed(0); ok = 0
for f in sorted(glob.glob(f"data/readouts/{d}/*.prose.json.gz")):
    i = f.split("/")[-1].removesuffix(".prose.json.gz")
    side = json.loads(gzip.decompress(open(f, "rb").read()))
    grid = json.loads(gzip.decompress(open(f"data/readouts/{d}/{i}.json.gz", "rb").read()))
    assert grid["tokens"] == side["tokens"], f"{i}: tokens differ from the grid readout"
    assert all(all(t is not None for t in lens["text"]) for lens in side["lenses"].values()), f"{i}: unfilled cells"
    assert set(side["lenses"]) == {"O-lens", "NLA"}, f"{i}: lenses {sorted(side['lenses'])}"
    ok += 1
    if ok <= 3:
        p = random.randrange(len(side["tokens"]))
        ctx = "".join(grid["text"][max(0, p - 10) : p + 1])
        print(f"  {i}@{p} ...{ctx!r}\n    O-lens: {side['lenses']['O-lens']['text'][p]!r}\n    NLA: {side['lenses']['NLA']['text'][p][:200]!r}")
print(f"  {ok} sidecars verified")
EOF
}

ship() {  # $1 pod name, $2 set entry
  local name=$1 entry=$2 dir=${2%%:*} src
  if [[ $entry == *:new* ]]; then
    pod "$name" fetch "$dir" >> "$LOG" 2>&1 || { log "$dir: fetch failed"; return 1; }
    src=data/overnight/${dir#overnight-}/notes.json
    [ -f "$src" ] && cp "$src" "data/readouts/$dir/notes.json"
  else
    pod "$name" copy "$dir" >> "$LOG" 2>&1 || { log "$dir: sidecar copy failed"; return 1; }
    python3 -c "import json,sys; p=f'data/readouts/{sys.argv[1]}/index.json'; j=json.load(open(p)); j['prose']={'O-lens':42,'NLA':42}; open(p,'w').write(json.dumps(j))" "$dir"
  fi
  verify "$dir" 2>&1 | tee -a "$LOG" || { log "$dir: verification failed, not committed"; return 1; }
  git add "data/readouts/$dir" && git commit -q -m "Overnight prose: $dir (O-lens + NLA at L42 on every token)

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>" >> "$LOG" 2>&1
  for attempt in 1 2 3; do
    git fetch -q origin && git merge -q --no-edit origin/main >> "$LOG" 2>&1 && git push -q origin HEAD:main && break
    sleep 20
  done
  scripts/deploy.sh 2>&1 | grep -E "Success|Deployment|rror" | tee -a "$LOG"
  log "$dir: shipped"
}

# ---- bring the pods up (each `up` prefetches the base and both lenses), start each chain
for name in "${!SETS[@]}"; do
  (pod "$name" up >> "$LOG" 2>&1 && echo "$name up $(date +%s)" >> "$COSTS") &
done
wait
declare -A remaining
for name in "${!SETS[@]}"; do
  if [ -f "$HOME/.runpod/$name.json" ]; then
    pod "$name" ssh "nohup bash -c $(printf %q "$(remote_chain "$name")") > /workspace/chain.log 2>&1 < /dev/null & disown; echo started" >> "$LOG" 2>&1
    remaining[$name]=${SETS[$name]}
    log "$name: chain started: ${SETS[$name]}"
  else
    log "$name: pod never came up; its sets are skipped: ${SETS[$name]}"
  fi
done

# ---- poll: ship each set as its marker appears; a chain idle for two polls with its next set unmarked has died
declare -A idle
while [ ${#remaining[@]} -gt 0 ]; do
  sleep 300
  for name in "${!remaining[@]}"; do
    state=$(pod "$name" ssh 'ls /workspace/done /workspace/failed 2>/dev/null | tr "\n" " "; pgrep -f "[p]rose_readout|[r]eadout.py" >/dev/null && echo RUNNING || echo IDLE' 2>/dev/null)
    left=""
    for entry in ${remaining[$name]}; do
      dir=${entry%%:*}
      if pod "$name" ssh "test -e /workspace/done/$dir" 2>/dev/null; then
        log "$name: $dir done on the pod"; ship "$name" "$entry" || true
      elif pod "$name" ssh "test -e /workspace/failed/$dir" 2>/dev/null; then
        log "$name: $dir FAILED on the pod:"; pod "$name" ssh "tail -5 /workspace/log-$dir-prose.txt /workspace/log-$dir-grid.txt 2>/dev/null" | tee -a "$LOG"
      else
        left+="$entry "
      fi
    done
    if [[ $state == *IDLE* && -n $left ]]; then idle[$name]=$(( ${idle[$name]:-0} + 1 )); else idle[$name]=0; fi
    if [ -z "$left" ] || [ "${idle[$name]}" -ge 2 ]; then
      [ -n "$left" ] && { log "$name: chain is dead; giving up on: $left"; pod "$name" ssh "tail -5 /workspace/chain.log" | tee -a "$LOG"; }
      pod "$name" down >> "$LOG" 2>&1; echo "$name down $(date +%s)" >> "$COSTS"
      unset "remaining[$name]"; log "$name: pod deleted"
    else
      remaining[$name]=$left
    fi
  done
done
python3 - "$COSTS" "$RATE" <<'EOF'
import sys
rows = [l.split() for l in open(sys.argv[1])]
up = {r[0]: int(r[2]) for r in rows if r[1] == "up"}; down = {r[0]: int(r[2]) for r in rows if r[1] == "down"}
hours = sum((down.get(n, up[n]) - up[n]) / 3600 for n in up)
print(f"pod-hours {hours:.2f}, cost ${hours * float(sys.argv[2]):.2f}")
EOF
log "overnight run finished"
