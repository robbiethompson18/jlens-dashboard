#!/bin/bash
# Run scripts/readout.py for one model on a fresh RunPod GPU, copy data/readouts/<model>/ back, delete the pod.
#
#   scripts/pod_readout.sh qwen3.6-27b
#
# Needs HF_TOKEN (source .envrc.local). The pod is deleted on exit, success or not; if this machine sleeps mid-run,
# check `runpodctl pod list` for a leftover `jlens-readout` pod.
set -uo pipefail
model=$1
key=$HOME/.runpod/ssh/RunPod-Key-Go
image=runpod/pytorch:1.0.3-cu1281-torch291-ubuntu2404

pod=""
trap '[ -n "$pod" ] && runpodctl pod delete "$pod" >/dev/null && echo "deleted pod $pod"' EXIT

for attempt in 1 2 3; do
  # 27B in bf16 is ~54 GB; one 80 GB card holds it plus a chunk of full-vocab logits.
  for gpu in "NVIDIA H100 80GB HBM3" "NVIDIA H200" "NVIDIA A100-SXM4-80GB"; do
    pod=$(runpodctl pod create --name jlens-readout --gpu-id "$gpu" --image $image --container-disk-in-gb 200 --ports 22/tcp 2>/dev/null |
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

rsync -az -e "ssh ${ssh_opts[*]}" --exclude .venv --exclude .git --exclude data/raw --exclude data/readouts --exclude .ruff_cache \
  --exclude .envrc.local ./ "root@$host:/workspace/jlens-dashboard/" || exit 1
ssh "${ssh_opts[@]}" "root@$host" "cd /workspace/jlens-dashboard && pip install -q uv 2>/dev/null && \
  export HF_TOKEN=$HF_TOKEN HF_HOME=/workspace/hf HF_XET_HIGH_PERFORMANCE=1 OMP_NUM_THREADS=8 && \
  uv sync -q --group pod && uv run python scripts/readout.py $model 2>&1 | grep -v -i -E 'warn|falling back|Loading weights'" || exit 1
mkdir -p "data/readouts/$model"
rsync -az -e "ssh ${ssh_opts[*]}" "root@$host:/workspace/jlens-dashboard/data/readouts/$model/" "data/readouts/$model/" && echo "copied data/readouts/$model"
