#!/usr/bin/env bash
# Launch Inkling 2 Mini abliterated on RunPod with vLLM + DSpark speculative decoding.
#
# Serves orcarouter/Inkling-2-Mini-abliterated: the bf16 safetensors that the GGUF builds were
# quantized from. Same abliterated weights, but it keeps the bundled DSpark drafter
# (mtp.safetensors), which the GGUF drops. vLLM >= 0.35 (nightly) has native Inkling 2 support.
#
# Needs: runpodctl with an API key (`runpodctl doctor`), and HF access to the gated repo:
#   https://huggingface.co/orcarouter/Inkling-2-Mini-abliterated  (click "agree" once)
#
# Env overrides: GPU, CLOUD (SECURE|COMMUNITY), MAX_LEN, HOURS (auto-stop), NAME
set -euo pipefail

GPU=${GPU:-"NVIDIA H100 80GB HBM3"}   # H100 SXM. Also good: "NVIDIA H200" (faster, more KV room)
CLOUD=${CLOUD:-SECURE}
MAX_LEN=${MAX_LEN:-131072}            # bf16 weights take ~55 GB; ~69 KB KV per token in vLLM
HOURS=${HOURS:-8}                     # pod stops itself after this many hours
NAME=${NAME:-inkling-vllm}
MODEL=orcarouter/Inkling-2-Mini-abliterated

KEY_FILE=$HOME/.config/inkling/api-key
if [[ ! -s $KEY_FILE ]]; then
  mkdir -p "$(dirname "$KEY_FILE")"
  (umask 077; openssl rand -hex 24 > "$KEY_FILE")
fi
API_KEY=$(cat "$KEY_FILE")
HF_TOKEN=${HF_TOKEN:-$(cat "$HOME/.cache/huggingface/token")}

ENV_JSON=$(jq -cn --arg hf "$HF_TOKEN" --arg key "$API_KEY" '{
  HF_TOKEN: $hf, VLLM_API_KEY: $key,
  HF_HOME: "/workspace/hf", HF_XET_HIGH_PERFORMANCE: "1"
}')

# The image's entrypoint is `vllm serve`; these are its arguments.
# RunPod splits this string like a shell and strips double quotes, so no JSON here: dict options
# use vLLM's dotted form. reasoning_effort comes from pi per request (chat_template_kwargs).
ARGS="$MODEL --served-model-name inkling --host 0.0.0.0 --port 8000 \
--max-model-len $MAX_LEN --gpu-memory-utilization 0.92 --max-num-seqs 4 \
--reasoning-parser inkling --enable-auto-tool-choice --tool-call-parser inkling \
--speculative-config.method dspark --speculative-config.num_speculative_tokens 7 \
--speculative-config.draft_sample_method probabilistic"

STOP_AT=$(date -u -d "+$HOURS hours" +%Y-%m-%dT%H:%M:%SZ)

echo "Creating $CLOUD pod: $GPU, vLLM nightly, $MODEL, max len $MAX_LEN, auto-stop $STOP_AT"
OUT=$(runpodctl pod create \
  --name "$NAME" \
  --image vllm/vllm-openai:nightly \
  --gpu-id "$GPU" --gpu-count 1 \
  --cloud-type "$CLOUD" \
  --min-cuda-version 13.0 \
  --container-disk-in-gb 60 \
  --volume-in-gb 80 --volume-mount-path /workspace \
  --ports "8000/http,22/tcp" \
  --env "$ENV_JSON" \
  --docker-args "$ARGS" \
  --stop-after "$STOP_AT")
POD_ID=$(jq -r '.id // .pod.id // empty' <<<"$OUT")
if [[ -z $POD_ID ]]; then echo "$OUT"; exit 1; fi

URL="https://$POD_ID-8000.proxy.runpod.net"
echo "Pod $POD_ID created. Endpoint: $URL/v1"
echo "$POD_ID" > "$(dirname "$0")/.pod-vllm"
echo "Waiting for the server (first start downloads ~54 GB and compiles CUDA graphs: 5-15 min)..."
for _ in $(seq 1 180); do
  if curl -sf -m 5 "$URL/health" >/dev/null; then
    echo "Ready. Connect pi with:  ./pi-connect.sh $POD_ID 8000"
    exit 0
  fi
  sleep 10
done
echo "Not ready after 30 min. Check logs:  runpodctl pod logs $POD_ID  (or the RunPod console)"
exit 1
