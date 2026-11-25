#!/usr/bin/env bash
# Launch an RTX 5090 pod for satellitedown's ingine engine (sm_120a only; claims ~500 tok/s decode).
# The pod is plain Ubuntu with SSH. Install ingine over SSH afterwards (it brings its own CUDA 13.4 toolkit):
#   ssh root@<ip> -p <port>  then  INGINE_DIR=/workspace/ingine bash <(curl -fsSL .../get.sh)  (menu 1, then 3)
# ingine has no authentication, so port 8003 is NOT exposed; reach it through an SSH tunnel:
#   ssh -N -L 8003:127.0.0.1:8003 root@<ip> -p <port>
# Env overrides: GPU, CLOUD (SECURE|COMMUNITY), HOURS, NAME, IMAGE
set -euo pipefail

GPU=${GPU:-"NVIDIA GeForce RTX 5090"}
CLOUD=${CLOUD:-SECURE}
HOURS=${HOURS:-3}
NAME=${NAME:-inkling-ingine}
IMAGE=${IMAGE:-runpod/base:2.1.1-ubuntu2404}
PUBKEY=$(cat "${SSH_PUBKEY:-$HOME/.ssh/id_ed25519.pub}")

ENV_JSON=$(jq -cn --arg k "$PUBKEY" '{PUBLIC_KEY: $k}')
# --stop-after has not fired reliably; check the pod with GraphQL `myself { pods }` and stop it yourself.
STOP_AT=$(date -u -d "+$HOURS hours" +%Y-%m-%dT%H:%M:%SZ)

echo "Creating $CLOUD pod: $GPU, $IMAGE, auto-stop $STOP_AT"
OUT=$(runpodctl pod create \
  --name "$NAME" \
  --image "$IMAGE" \
  --gpu-id "$GPU" --gpu-count 1 \
  --cloud-type "$CLOUD" \
  --min-cuda-version 13.0 \
  --container-disk-in-gb 20 \
  --volume-in-gb 50 --volume-mount-path /workspace \
  --ports "22/tcp" \
  --env "$ENV_JSON" \
  --stop-after "$STOP_AT")
POD_ID=$(jq -r '.id // .pod.id // empty' <<<"$OUT")
if [[ -z $POD_ID ]]; then echo "$OUT"; exit 1; fi
echo "$POD_ID" > "$(dirname "$0")/.pod-ingine"
echo "Pod $POD_ID created."
