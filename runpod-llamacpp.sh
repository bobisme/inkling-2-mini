#!/usr/bin/env bash
# Launch the Inkling-2-Mini-abliterated GGUF (Mix-16G) on RunPod with llama.cpp.
#
# The cheaper, simpler route. Expect ~100-130 tok/s decode on an RTX 5090 at short context,
# falling as the context fills. For more speed, use runpod-vllm.sh (DSpark speculative decoding).
#
# Needs: runpodctl with an API key (`runpodctl doctor`), and HF access to
#   https://huggingface.co/orcarouter/Inkling-2-Mini-abliterated-GGUF
#
# Env overrides: GPU, CLOUD (SECURE|COMMUNITY), CTX, UB, HOURS, NAME, IMAGE
set -euo pipefail

GPU=${GPU:-"NVIDIA GeForce RTX 5090"}
CLOUD=${CLOUD:-SECURE}
# Inkling's per-head relative-position bias makes llama.cpp's compute buffer ~201 bytes x UB x CTX.
# On 32 GB: CTX=131072 UB=256 needs ~24 GiB; CTX=262144 UB=128 ~26 GiB; CTX=131072 UB=512 ~30 GiB.
CTX=${CTX:-131072}
UB=${UB:-256}
HOURS=${HOURS:-8}
NAME=${NAME:-inkling-llamacpp}
# b12679+ has the Inkling tokenizer (#32799) and tool-call parser (#32833) fixes.
IMAGE=${IMAGE:-ghcr.io/ggml-org/llama.cpp:server-cuda-b12679}

KEY_FILE=$HOME/.config/inkling/api-key
if [[ ! -s $KEY_FILE ]]; then
  mkdir -p "$(dirname "$KEY_FILE")"
  (umask 077; openssl rand -hex 24 > "$KEY_FILE")
fi
API_KEY=$(cat "$KEY_FILE")
HF_TOKEN=${HF_TOKEN:-$(cat "$HOME/.cache/huggingface/token")}

ENV_JSON=$(jq -cn --arg hf "$HF_TOKEN" --arg key "$API_KEY" '{
  HF_TOKEN: $hf, LLAMA_API_KEY: $key, LLAMA_CACHE: "/workspace/llama-cache",
  LLAMA_ARG_CHAT_TEMPLATE_KWARGS: "{\"reasoning_effort\":\"low\"}"
}')

# The image's entrypoint is llama-server; these are its arguments. RunPod splits this string like a
# shell and strips double quotes, so the JSON chat-template kwargs go in an env var above.
# --fit off: fail loudly instead of silently moving layers to the CPU.
# ngram-mod speculation: +20% decode on code (measured on a 3090: 55.5 -> 66.5 tok/s).
ARGS="-hf orcarouter/Inkling-2-Mini-abliterated-GGUF -hff Inkling-2-Mini-abliterated-Mix-16G.gguf \
--alias inkling --host 0.0.0.0 --port 8080 \
-ngl 99 --fit off -fa on -np 1 -c $CTX -ub $UB -b 2048 -ctk q8_0 -ctv q8_0 \
--jinja --spec-type ngram-mod"

STOP_AT=$(date -u -d "+$HOURS hours" +%Y-%m-%dT%H:%M:%SZ)

echo "Creating $CLOUD pod: $GPU, $IMAGE, ctx $CTX, ubatch $UB, auto-stop $STOP_AT"
OUT=$(runpodctl pod create \
  --name "$NAME" \
  --image "$IMAGE" \
  --gpu-id "$GPU" --gpu-count 1 \
  --cloud-type "$CLOUD" \
  --min-cuda-version 12.8 \
  --container-disk-in-gb 20 \
  --volume-in-gb 30 --volume-mount-path /workspace \
  --ports "8080/http,22/tcp" \
  --env "$ENV_JSON" \
  --docker-args "$ARGS" \
  --stop-after "$STOP_AT")
POD_ID=$(jq -r '.id // .pod.id // empty' <<<"$OUT")
if [[ -z $POD_ID ]]; then echo "$OUT"; exit 1; fi

URL="https://$POD_ID-8080.proxy.runpod.net"
echo "Pod $POD_ID created. Endpoint: $URL/v1"
echo "$POD_ID" > "$(dirname "$0")/.pod-llamacpp"
echo "Waiting for the server (first start downloads ~16 GB: 1-5 min)..."
for _ in $(seq 1 120); do
  if curl -sf -m 5 "$URL/health" >/dev/null; then
    echo "Ready. Connect pi with:  ./pi-connect.sh $POD_ID 8080 $CTX"
    exit 0
  fi
  sleep 10
done
echo "Not ready after 20 min. Check logs:  runpodctl pod logs $POD_ID  (or the RunPod console)"
exit 1
