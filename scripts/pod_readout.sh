#!/bin/bash
# Run scripts/readout.py for one model on fresh RunPod GPUs, copy data/readouts/<model>/ back, delete the pods.
#
#   scripts/pod_readout.sh qwen3.6-27b        # one pod
#   scripts/pod_readout.sh wsbench-brew 4     # four pods at once, each reading every 4th item; index parts merged here
#
# Needs HF_TOKEN (source .envrc.local) and RUNPOD_API_KEY. Pods are deleted on exit, success or not; if this machine sleeps
# mid-run, check `runpodctl pod list` for leftover `jlens-readout` pods.
set -uo pipefail
model=$1
shards=${2:-1}
key=$HOME/.runpod/ssh/RunPod-Key-Go
image=runpod/pytorch:1.0.3-cu1281-torch291-ubuntu2404

run_shard() {  # $1 = shard number; one pod, one readout.py run
  local shard=$1 pod="" gpu attempt info host port driver tag="[$model $shard/$shards]"
  trap '[ -n "$pod" ] && runpodctl pod delete "$pod" >/dev/null && echo "$tag deleted pod $pod"' EXIT
  for attempt in 1 2 3 4 5 6 7 8; do
    # 27B in bf16 is ~54 GB; one 80 GB card holds it plus a chunk of full-vocab logits.
    for gpu in "NVIDIA H100 80GB HBM3" "NVIDIA H200" "NVIDIA A100-SXM4-80GB"; do
      pod=$(runpodctl pod create --name jlens-readout --gpu-id "$gpu" --image $image --container-disk-in-gb 200 --ports 22/tcp 2>/dev/null |
        python3 -c 'import json,sys; print(json.load(sys.stdin).get("id",""))' 2>/dev/null)
      [ -n "$pod" ] && break
    done
    [ -n "$pod" ] || { echo "$tag no GPU available"; return 1; }
    echo "$tag pod $pod ($gpu), attempt $attempt"
    until info=$(runpodctl ssh info "$pod" 2>/dev/null) && echo "$info" | grep -q '"port"'; do sleep 10; done
    read -r host port < <(echo "$info" | python3 -c 'import json,sys; d=json.load(sys.stdin); print(d["ip"], d["port"])')
    ssh_opts=(-i "$key" -p "$port" -o StrictHostKeyChecking=accept-new -o ConnectTimeout=20 -o ServerAliveInterval=30)
    until ssh "${ssh_opts[@]}" "root@$host" true 2>/dev/null; do sleep 5; done
    # The lockfile's torch is a CUDA 13 build, which needs NVIDIA driver 580 or newer.
    driver=$(ssh "${ssh_opts[@]}" "root@$host" "nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1 | cut -d. -f1")
    [ "${driver:-0}" -ge 580 ] && break
    echo "$tag driver ${driver:-unknown} < 580; replacing pod"
    runpodctl pod delete "$pod" >/dev/null
    pod=""
  done
  [ -n "$pod" ] || { echo "$tag no pod with a new enough driver"; return 1; }

  rsync -az -e "ssh ${ssh_opts[*]}" --exclude .venv --exclude .git --exclude data/raw --exclude data/readouts --exclude .ruff_cache \
    --exclude visualizer --exclude weird-behaviors --exclude .envrc.local ./ "root@$host:/workspace/jlens-dashboard/" || return 1
  # The token goes in through stdin so it never shows up in a process list or an `ssh -v`/`bash -x` trace.
  echo "export HF_TOKEN=$HF_TOKEN" | ssh "${ssh_opts[@]}" "root@$host" 'umask 077; cat > /root/.hf_env'
  local shard_arg=""
  [ "$shards" -gt 1 ] && shard_arg="--shard $shard/$shards"
  ssh "${ssh_opts[@]}" "root@$host" "set -o pipefail; source /root/.hf_env; cd /workspace/jlens-dashboard && \
    (command -v uv >/dev/null || pip install -q --break-system-packages uv) && \
    export HF_HOME=/workspace/hf HF_XET_HIGH_PERFORMANCE=1 OMP_NUM_THREADS=8 && \
    uv sync -q --group pod && uv run python scripts/readout.py $model $shard_arg 2>&1 | grep -v -i -E 'warn|falling back|Loading weights'" | sed "s/^/$tag /" || return 1
  mkdir -p "data/readouts/$model"
  rsync -az -e "ssh ${ssh_opts[*]}" "root@$host:/workspace/jlens-dashboard/data/readouts/$model/" "data/readouts/$model/" && echo "$tag copied"
}

if [ "$shards" -eq 1 ]; then
  run_shard 1
  exit $?
fi
pids=()
for shard in $(seq 1 "$shards"); do
  run_shard "$shard" &
  pids+=($!)
done
status=0
for pid in "${pids[@]}"; do wait "$pid" || status=1; done
[ $status -eq 0 ] || { echo "a shard failed; index.json not merged"; exit 1; }
# merge the shards' index parts back into transcript order (shard k read items k-1, k-1+n, ...; every part has the same header)
python3 - "$model" "$shards" <<'PY'
import json, sys
from itertools import zip_longest
from pathlib import Path
model, shards = sys.argv[1], int(sys.argv[2])
out = Path("data/readouts") / model
parts = [json.loads((out / f"index.{k}of{shards}.json").read_text()) for k in range(1, shards + 1)]
items = [item for row in zip_longest(*(part["items"] for part in parts)) for item in row if item is not None]
(out / "index.json").write_text(json.dumps({**parts[0], "items": items}))
for k in range(1, shards + 1):
    (out / f"index.{k}of{shards}.json").unlink()
print(f"merged {len(items)} items into {out / 'index.json'}")
PY
