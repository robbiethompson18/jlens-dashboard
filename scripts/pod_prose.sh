#!/bin/bash
# Run scripts/prose_readout.py on a RunPod GPU in stages, so a pilot can be inspected before the full run on the same pod.
#
#   scripts/pod_prose.sh up                       # create the pod, sync the repo, install, prefetch the three 27B checkpoints
#   scripts/pod_prose.sh run qwen3.6-27b --items wc-03 --max-positions 40 --batch 32   # pilot; copies sidecars back
#   scripts/pod_prose.sh run qwen3.6-27b          # the full run; resumes past finished items
#   scripts/pod_prose.sh ssh                      # a shell on the pod
#   scripts/pod_prose.sh down                     # delete the pod (also runs if `up` fails)
#
# Needs HF_TOKEN (source .envrc.local) and RUNPOD_API_KEY (runpodctl reads ~/.runpod/config.toml; export it if the CLI
# complains). Pod id/host/port live in $STATE so each stage is a separate invocation; `runpodctl pod list` shows leftovers.
set -uo pipefail
STATE=$HOME/.runpod/jlens-prose-pod.json
key=$HOME/.runpod/ssh/RunPod-Key-Go
image=runpod/pytorch:1.0.3-cu1281-torch291-ubuntu2404
stage=$1
shift

load_state() {
  read -r pod host port < <(python3 -c 'import json,sys; d=json.load(open(sys.argv[1])); print(d["pod"], d["host"], d["port"])' "$STATE")
  ssh_opts=(-i "$key" -p "$port" -o StrictHostKeyChecking=accept-new -o ConnectTimeout=20 -o ServerAliveInterval=30)
}

case $stage in
up)
  pod=""
  trap '[ -n "$pod" ] && runpodctl pod delete "$pod" >/dev/null && echo "deleted pod $pod"' EXIT
  for attempt in 1 2 3; do
    # bf16 27B is ~54 GB; the oracle lens stage holds one 27B plus a batch of generations, the NLA stage its own 27B.
    for gpu in "NVIDIA H100 80GB HBM3" "NVIDIA H200" "NVIDIA A100-SXM4-80GB"; do
      pod=$(runpodctl pod create --name jlens-prose --gpu-id "$gpu" --image $image --container-disk-in-gb 300 --ports 22/tcp 2>/dev/null |
        python3 -c 'import json,sys; print(json.load(sys.stdin).get("id",""))' 2>/dev/null)
      [ -n "$pod" ] && break
    done
    [ -n "$pod" ] || { echo "no GPU available"; exit 1; }
    echo "pod $pod ($gpu), attempt $attempt"
    until info=$(runpodctl ssh info "$pod" 2>/dev/null) && echo "$info" | grep -q '"port"'; do sleep 10; done
    read -r host port < <(echo "$info" | python3 -c 'import json,sys; d=json.load(sys.stdin); print(d["ip"], d["port"])')
    ssh_opts=(-i "$key" -p "$port" -o StrictHostKeyChecking=accept-new -o ConnectTimeout=20 -o ServerAliveInterval=30)
    until ssh "${ssh_opts[@]}" "root@$host" true 2>/dev/null; do sleep 5; done
    # The lockfile's torch is a CUDA 13 build, which needs NVIDIA driver 580 or newer.
    driver=$(ssh "${ssh_opts[@]}" "root@$host" "nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1 | cut -d. -f1")
    [ "${driver:-0}" -ge 580 ] && break
    echo "driver ${driver:-unknown} < 580; replacing pod"
    runpodctl pod delete "$pod" >/dev/null
    pod=""
  done
  [ -n "$pod" ] || { echo "no pod with a new enough driver"; exit 1; }
  python3 -c 'import json,sys; json.dump({"pod": sys.argv[1], "host": sys.argv[2], "port": sys.argv[3]}, open(sys.argv[4], "w"))' "$pod" "$host" "$port" "$STATE"
  # The token lens readouts go up (the prose script takes its token ids from them); no prose sidecars do.
  rsync -az -e "ssh ${ssh_opts[*]}" --exclude .venv --exclude .git --exclude data/raw --exclude '*.prose.json.gz' --exclude 'data/readouts/qwen3.5-0.8b' \
    --exclude 'data/readouts/qwen3.8-27b-smoking' --exclude 'data/readouts/wsbench-chain' --exclude .ruff_cache --exclude visualizer \
    --exclude weird-behaviors --exclude .envrc.local ./ "root@$host:/workspace/jlens-dashboard/" || exit 1
  echo "export HF_TOKEN=$HF_TOKEN" | ssh "${ssh_opts[@]}" "root@$host" 'umask 077; cat > /root/.hf_env'
  ssh "${ssh_opts[@]}" "root@$host" "set -o pipefail; source /root/.hf_env; cd /workspace/jlens-dashboard && \
    (command -v uv >/dev/null || pip install -q --break-system-packages uv) && \
    export HF_HOME=/workspace/hf HF_XET_HIGH_PERFORMANCE=1 && uv sync -q --group pod && \
    uv run hf download Qwen/Qwen3.6-27B >/dev/null && uv run hf download robbiethompson2018/oracle-lens-qwen3.6-27b-pilot >/dev/null && \
    uv run hf download ceselder/qwen3.6-27b-nla-rl --include 'av_base/*' --include 'av_rl_adapters/iter_000400/*' >/dev/null && echo prefetched" || exit 1
  trap - EXIT
  echo "pod $pod ready; state in $STATE"
  ;;
run)
  load_state
  model=$1
  ssh "${ssh_opts[@]}" "root@$host" "set -o pipefail; source /root/.hf_env; cd /workspace/jlens-dashboard && \
    export HF_HOME=/workspace/hf OMP_NUM_THREADS=8 FLA_TILELANG=0 && \
    uv run python scripts/prose_readout.py $* 2>&1 | grep -v -i -E 'warn|falling back|Loading weights'"
  status=$?
  rsync -az -e "ssh ${ssh_opts[*]}" --include '*.prose.json.gz' --include 'index.json' --exclude '*' \
    "root@$host:/workspace/jlens-dashboard/data/readouts/$model/" "data/readouts/$model/" && echo "copied sidecars for $model"
  exit $status
  ;;
ssh)
  load_state
  ssh "${ssh_opts[@]}" "root@$host" "$@"
  ;;
down)
  load_state
  runpodctl pod delete "$pod" >/dev/null && echo "deleted pod $pod" && rm -f "$STATE"
  ;;
*)
  echo "usage: $0 up | run <model> [args] | ssh [cmd] | down"
  exit 1
  ;;
esac
