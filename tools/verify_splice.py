"""Check the spliced GGUF: every edited tensor should match orcarouter's weights to within quantization error,
and be further from the stock Unsloth weights (the abliteration edit). Untouched tensors must be byte-identical."""
import json, os, sys
import numpy as np
from gguf import GGUFReader
from gguf.quants import NVFP4, Q8_0
from safetensors import safe_open

STOCK, OUT, ORCA = sys.argv[1:4]
S = {t.name: t for t in GGUFReader(STOCK).tensors}
O = {t.name: t for t in GGUFReader(OUT).tensors}
idx = json.load(open(os.path.join(ORCA, "model.safetensors.index.json")))["weight_map"]

def orca(n):
    with safe_open(os.path.join(ORCA, idx[n]), "pt") as f:
        return f.get_tensor(n).float().numpy()

def deq(T, name, shape):
    t = T[name]; raw = np.asarray(t.data)
    if t.tensor_type.name == "Q8_0":
        return Q8_0.dequantize_blocks(raw.reshape(-1, 34)).reshape(shape)
    s = np.asarray(T[name.replace(".weight", ".scale")].data).reshape(-1)
    E = shape[0]
    return (NVFP4.dequantize_blocks(raw.reshape(-1, 36)).reshape(E, -1) * s[:, None]).reshape(shape)

edited = {"token_embd.weight", "output.weight"} | {f"blk.{L}.{k}" for L in range(24) for k in ("attn_output.weight", "ffn_down_exps.weight")}
same = sum(np.array_equal(np.asarray(S[n].data), np.asarray(O[n].data)) for n in S if n not in edited)
print(f"untouched tensors byte-identical: {same}/{len(S) - len(edited)}")
changed = sum(not np.array_equal(np.asarray(S[n].data), np.asarray(O[n].data)) for n in edited)
print(f"edited tensors actually changed: {changed}/{len(edited)}")

samples = [("token_embd.weight", "model.llm.embed.weight")]
for L in (0, 11, 19, 20, 23):
    samples += [(f"blk.{L}.attn_output.weight", f"model.llm.layers.{L}.attn.wo_ud.weight"),
                (f"blk.{L}.ffn_down_exps.weight", f"model.llm.layers.{L}.mlp.experts.w2_weight")]
for g, o in samples:
    w = orca(o)
    if g == "token_embd.weight":
        w = w[:min(w.shape[0], 201024)]
    sh = w.shape if g != "token_embd.weight" else (np.asarray(S[g].data).shape[0], w.shape[1])
    a = deq(O, g, sh)[:w.shape[0]]; b = deq(S, g, sh)[:w.shape[0]]
    n = np.linalg.norm(w)
    print(f"{g:28} {O[g].tensor_type.name:5}  out-vs-orca {np.linalg.norm(w - a) / n:.4f}   stock-vs-orca {np.linalg.norm(w - b) / n:.4f}")
