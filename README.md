> **Fastest result (2026-11-24): the Maya engine, locally on the RTX 3090, at 162 tok/s with the abliterated weights**
> (174 tok/s stock, against ~55 for llama.cpp on the same card and 89.5 on the RunPod 5090).
> Maya only loads Unsloth's old-layout `UD-Q4_K_XL`, so the abliterated GGUF is built by splicing orcarouter's edited
> tensors into it: `tools/splice_abliterated.py` (checked by `tools/verify_splice.py`; `tools/diff_official.py` lists which tensors orcarouter edited), output
> `~/models/inkling/Inkling-2-Mini-abliterated-orca-UD-Q4_K_XL-maya.gguf`.
> Serve it from a Maya checkout (configs in `maya/`, copied into the checkout):
> `.venv/bin/python serve/server.py --engine maya --config ik2-abliterated.json --host 127.0.0.1 --port 18000`,
> then `ENGINE=maya PROVIDER=maya ./pi-connect.sh http://127.0.0.1:18000 0 131072` and `/model maya/inkling` in pi.
> `./refusal-check.sh <base>/v1` compares refusals between weight files; `bun refusal-eval.ts` runs JailbreakBench.
> Refusals on 100 JailbreakBench harmful requests: stock 78; spliced file in Maya 0 when it thinks (effort `low`),
> the same as orcarouter's GGUF in llama.cpp; but 17-31 when it answers without thinking. Maya treats `minimal` as no
> thinking, so `pi-connect.sh` maps pi's `minimal` to `low` for Maya. pi's `off` still skips thinking.

# Inkling 2 Mini (abliterated) on RunPod, for pi

> **Tested on RunPod 2026-11-24.** Route B works: RTX 5090 + llama.cpp b12679 gave **89.5 tok/s**
> median decode, 0.75 s TTFT, and pi worked end to end (tool calls included).
> Route A does **not** work as written: vLLM's Inkling 2 code rejects BF16 expert weights ("the
> Inkling2 experts are NVFP4 / MXFP8"), and orcarouter's bf16 checkpoint also needs a fixed
> config.json (Transformers 5.24 renamed keys and added `loop_num_passes`) plus fp32→bf16 casts for
> the sconv/controller/router tensors. To use vLLM, the abliterated weights must first be put back
> into the release's NVFP4/MXFP8 format. Other findings: RunPod splits `--docker-args` like a shell
> and strips double quotes, so pass JSON through env vars or vLLM's dotted flags. Download speed
> varies by host, from 17 MB/s to over 200 MB/s.

Two ways to serve `orcarouter`'s abliterated Inkling 2 Mini on one RunPod GPU, behind an
OpenAI-compatible endpoint that pi uses as the `runpod/inkling` model.

| Route | GPU | Weights | Decode, 1 user | Cost (Nov 2026, indicative) |
|---|---|---|---|---|
| **A. vLLM + DSpark** (recommended) | H100 SXM 80GB | `orcarouter/Inkling-2-Mini-abliterated` (bf16 safetensors + DSpark drafter) | estimated 150-300 tok/s, not measured yet | ~$2.69/h community, $4.39/h secure |
| B. llama.cpp | RTX 5090 32GB | `orcarouter/Inkling-2-Mini-abliterated-GGUF` Mix-16G | ~100-130 tok/s short context, ~70 at 165K (community) | ~$0.69-1.39/h |

## Why the GGUF alone won't reliably clear 100 tok/s

- **llama.cpp decode for this model is limited by kernel count, not bandwidth.** Layers 6-17 run three
  times, so each token makes 48 layer visits of small kernels (hidden size 2,304). Measured on a
  local RTX 3090, llama.cpp master (b12679+, Mix-16G): **51-56 tok/s** decode with CUDA graphs and
  21 tok/s without them; prefill was 840-915 tok/s. 3.4B active parameters at 4.8 bpw would allow several times that on 936 GB/s.
  Community figures: an RTX 5090 reaches about 109 tok/s at 8K and 70 at 165K.
- **The GGUF has no drafter.** The GGUF conversion drops the MTP/DSpark head, and
  llama.cpp's `inkling2_loop` rejects nextn layers. `--spec-type ngram-mod` is the only speculative
  decoding option. It gave **+20%** on code on the 3090 (55.5 → 66.5 tok/s, 68% of drafts accepted).
- **vLLM is the fast route.** vLLM nightly (>= 0.35) has native Inkling 2 support with the
  bundled DSpark drafter (7 draft tokens). Reported speeds: 416 tok/s on a 5090 with the
  official NVFP4 weights, and 131 → 267 tok/s on a 3090 Ti with INT4. orcarouter also publishes
  the **same abliterated weights as bf16 safetensors with `mtp.safetensors` (the drafter)**. The
  GGUF files were quantized from that release. At 53.5 GB it needs an 80 GB card, so H100 SXM
  (3.35 TB/s) is the pick. An H200 has more KV room and ~1.4x the bandwidth.

## Hidden VRAM cost in llama.cpp (route B)

Inkling's attention adds a per-head relative-position bias. llama.cpp builds it as a dense
`[n_kv, n_ubatch, 24 heads]` tensor, so the compute buffer is about **201 bytes × ubatch × context**.
Measured: 6.43 GiB at 128K/ub256 and 6.38 GiB at 256K/ub128. The KV cache is small by comparison:
8 full-context cache slots × 4 KiB per token, so 32 KiB per token at f16 and ~17 KiB at q8_0.

| context | ubatch | KV | est. total (Mix-16G 14.6 GiB) | fits 32 GB? |
|---|---|---|---|---|
| 128K | 256 | q8_0 | ~24 GiB | yes (default in the script) |
| 128K | 512 | q8_0 | ~30 GiB | barely |
| 256K | 128 | q8_0 | ~26 GiB | yes, prefill ~25% slower |

`--fit off` makes llama.cpp fail loudly instead of quietly moving layers to the CPU. Without it,
a config that is too large looks like a 10 tok/s server.

## Use

Once only:

```bash
runpodctl doctor     # stores the RunPod API key
# Accept the gate (auto-approved) for the repo you will use:
#   https://huggingface.co/orcarouter/Inkling-2-Mini-abliterated        (route A)
#   https://huggingface.co/orcarouter/Inkling-2-Mini-abliterated-GGUF   (route B, already accepted)
runpodctl gpu list   # check the GPU id strings and stock; the scripts default to
                     # "NVIDIA H100 80GB HBM3" and "NVIDIA GeForce RTX 5090"
```

Launch, connect, measure:

```bash
./runpod-vllm.sh                  # or ./runpod-llamacpp.sh  (CLOUD=COMMUNITY for cheaper)
./pi-connect.sh <pod-id> 8000     # 8080 for llama.cpp; writes the "runpod" provider into ~/.pi/agent/models.json (backs it up first)
bun bench.ts https://<pod-id>-8000.proxy.runpod.net/v1
pi --model runpod/inkling:low
```

- Both scripts generate an API key in `~/.config/inkling/api-key`. pi reads it with
  `!cat`, and it is never written into models.json.
- The scripts upload your HF token as a pod env var so the pod can download the gated weights. A
  read-only fine-grained token is enough: `HF_TOKEN=hf_... ./runpod-vllm.sh`.
- Pods auto-stop after `HOURS` (default 8). The weights stay on the pod volume, so the next start
  skips the download. Run `runpodctl pod stop <id>` / `runpodctl pod remove <id>` when done.
- Thinking: pi's level is sent as `chat_template_kwargs.reasoning_effort` (off→none, low,
  medium, high). The server default is `low`, the effort the abliteration was fitted at. The
  template's own default is `high`, and the model card warns that it can get stuck in long
  reasoning loops, so pi caps output at 32K tokens.

## Verified vs not

Verified locally (RTX 3090, llama.cpp master `396ed2c`, Mix-16G):
- The decode and prefill numbers above, and the compute-buffer sizes.
- The route-B flag set end to end with pi: the `chat-template` thinking format, a tool call
  through `--jinja` with the Inkling parser, and the prompt cache across turns.
- `bench.ts` reports the same speed as llama-bench.

Not verified (no RunPod key in this session):
- Both launch scripts against the RunPod API: GPU id strings, `--docker-args` splitting.
- vLLM loading orcarouter's **bf16** checkpoint. vLLM's Inkling MoE has an unquantized bf16 path,
  used for layers the official release leaves unquantized, but nobody has reported a full-bf16 run.
  If it fails, try `--quantization fp8`. Failing that, use route B.
- The vLLM speed estimate.

Other gotchas:
- vLLM nightly images are CUDA 13. The script asks RunPod for a host with CUDA ≥ 13.0 (driver r580+).
- The RunPod HTTPS proxy drops requests that send no bytes for ~100 s. A cold 128K-token prompt on
  llama.cpp (~1.6-2K tok/s prefill on a 5090) gets close to that. If you see 524 errors, expose
  the port as TCP or use `ssh -L`.
- llama.cpp b12633, the build the model card was tested with, predates the Inkling tokenizer and
  tool-call parser fixes (b12641/b12650). The script pins `server-cuda-b12679`.

Sources: [vLLM recipe](https://recipes.vllm.ai/thinkingmachines/Inkling-2-Mini) ·
[VRAM write-up](https://vramcalculator.com/inkling-2-mini-vram/) ·
[llamaperf reports](https://llamaperf.com/model/inkling-2-mini) ·
[HyperInkling](https://github.com/Pento95/HyperInkling) ·
[orcarouter bf16 card](https://huggingface.co/orcarouter/Inkling-2-Mini-abliterated)
