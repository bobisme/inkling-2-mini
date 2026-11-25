#!/usr/bin/env bash
# Add or update the "runpod" provider in pi's models.json so pi can use the pod.
# Usage: ./pi-connect.sh <pod-id | base-url> [port=8000] [context=131072]
#   ENGINE=maya ./pi-connect.sh http://127.0.0.1:18000 0 131072   -> Maya server (local or tunneled)
#   PROVIDER=<name> picks the provider id (default "runpod")
#   ./pi-connect.sh abc123xyz 8000          -> https://abc123xyz-8000.proxy.runpod.net/v1
#   ./pi-connect.sh http://127.0.0.1:8080   -> a local or tunneled server
# Then in pi: /model  ->  runpod/inkling
set -euo pipefail

TARGET=${1:?usage: pi-connect.sh <pod-id|base-url> [port] [context]}
PORT=${2:-8000}
CTX=${3:-131072}
ENGINE=${ENGINE:-llamacpp}
PROVIDER=${PROVIDER:-runpod}
if [[ $TARGET == http* ]]; then BASE=${TARGET%/}; BASE=${BASE%/v1}/v1
else BASE="https://$TARGET-$PORT.proxy.runpod.net/v1"; fi

DIR=${PI_CODING_AGENT_DIR:-$HOME/.pi/agent}
CFG=$DIR/models.json
[[ -f $CFG ]] || echo '{"providers":{}}' > "$CFG"
cp "$CFG" "$CFG.bak-inkling-$(date +%Y%m%d-%H%M%S)"

# llama.cpp / vLLM read reasoning_effort from chat_template_kwargs; Maya reads it as a top-level
# request field (pi's "openai" format). Pi's thinking level maps onto the template's effort names.
if [[ $ENGINE == maya ]]; then
  COMPAT='{"thinkingFormat":"openai","supportsDeveloperRole":false,"maxTokensField":"max_tokens"}'
  # Maya treats "minimal" like "none" (no thinking at all), unlike llama.cpp (template effort 0.1). The abliterated
  # model refuses ~20-30% of harmful prompts when it answers without thinking and 0% with it, so map minimal -> low.
  LEVELS='{"off":"none","minimal":"low","low":"low","medium":"medium","high":"high"}'
else
  COMPAT='{"thinkingFormat":"chat-template","chatTemplateKwargs":{"reasoning_effort":{"$var":"thinking.effort"}},"supportsDeveloperRole":false,"maxTokensField":"max_tokens"}'
  LEVELS='{"off":"none","minimal":"minimal","low":"low","medium":"medium","high":"high"}'
fi
TMP=$(mktemp)
jq --arg base "$BASE" --argjson ctx "$CTX" --arg prov "$PROVIDER" --argjson compat "$COMPAT" --arg engine "$ENGINE" --argjson levels "$LEVELS" '.providers[$prov] = {
  baseUrl: $base,
  api: "openai-completions",
  apiKey: "!cat ~/.config/inkling/api-key",
  compat: $compat,
  models: [{
    id: "inkling",
    name: ("Inkling 2 Mini abliterated (" + $prov + ", " + $engine + ")"),
    reasoning: true,
    input: ["text"],
    contextWindow: $ctx,
    maxTokens: 32768,
    thinkingLevelMap: $levels
  }]
}' "$CFG" > "$TMP" && mv "$TMP" "$CFG"

echo "pi provider '$PROVIDER' ($ENGINE) -> $BASE (context $CTX). Select it with /model $PROVIDER/inkling"
