"""Which tensors did orcarouter's abliteration change? Compare every tensor of their bf16 release with the
official checkpoint (experts dequantized from NVFP4 / MXFP8 the way llama.cpp's converter does it).

usage: diff_official.py <official-dir> <orca-bf16-dir>
"""
import json
import os
import re
import sys
from collections import defaultdict

import numpy as np
import torch
from safetensors import safe_open

OFF, ORCA = sys.argv[1:3]

E2M1 = np.array([0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6], dtype=np.float32)
codes = np.arange(256, dtype=np.uint8)
_exp = ((codes & 0x7F) >> 3).astype(np.int32)
_man = (codes & 7).astype(np.float32)
E4M3 = np.where(_exp == 0, np.ldexp(_man, -9), np.ldexp(1 + _man / 8, _exp - 7)).astype(np.float32)
E4M3 = np.where(codes & 0x80, -E4M3, E4M3)


def index(d):
    return json.load(open(os.path.join(d, "model.safetensors.index.json")))["weight_map"]


off_idx, orca_idx = index(OFF), index(ORCA)
handles = {}


def get(d, idx, name):
    f = os.path.join(d, idx[name])
    if f not in handles:
        handles[f] = safe_open(f, "pt")
    return handles[f].get_tensor(name)


def norm(name):
    return name.replace("model.language_model.", "model.llm.")


def official_expert(name, e):
    w = get(OFF, off_idx, name)[e].view(torch.uint8).numpy()
    sc = get(OFF, off_idx, name + "_scale")[e].view(torch.uint8).numpy()
    if name + "_amax" in off_idx:
        v = np.empty((w.shape[0], w.shape[1] * 2), np.float32)
        v[:, 0::2] = E2M1[w & 15]; v[:, 1::2] = E2M1[w >> 4]
        v.reshape(v.shape[0], -1, 16)[:] *= E4M3[sc][:, :, None]
        a = get(OFF, off_idx, name + "_amax")[e].item(); den = get(OFF, off_idx, name + "_outer_scale_denominator").item()
        return v * np.float32(a / den)
    v = E4M3[w]
    v.reshape(v.shape[0], -1, 32)[:] *= np.ldexp(np.ones(sc.shape, np.float32), sc.astype(np.int32) - 127)[:, :, None]
    return v


companions = ("_scale", "_amax", "_outer_scale_denominator")
off_names = {norm(n): n for n in off_idx if not n.endswith(companions)}
orca_names = {norm(n): n for n in orca_idx}
skip = ("visual.", "vision_tower.", "mtp.", "markov_head")
only_off = sorted(n for n in off_names if n not in orca_names and not any(s in n for s in skip))
only_orca = sorted(n for n in orca_names if n not in off_names and not any(s in n for s in skip))
print("only in official:", len(only_off), only_off[:8])
print("only in orca:", len(only_orca), only_orca[:8])

rows = []
for n in sorted(set(off_names) & set(orca_names)):
    if any(s in n for s in skip):
        continue
    on, rn = off_names[n], orca_names[n]
    b = get(ORCA, orca_idx, rn).float()
    if n.endswith(("w13_weight", "w2_weight")):
        num = den = 0.0
        for e in range(0, b.shape[0], 16):           # every 16th expert is plenty to see an edit
            a = official_expert(on, e)
            be = b[e].numpy()
            if a.shape != be.shape:
                a = a.reshape(be.shape)
            num += float(np.sum((be - a) ** 2)); den += float(np.sum(a ** 2))
        rel = (num / den) ** 0.5
    else:
        a = get(OFF, off_idx, on).float()
        if a.shape != b.shape:
            rows.append((n, float("nan"), f"shape {tuple(a.shape)} vs {tuple(b.shape)}"))
            continue
        rel = float((b - a).norm() / max(a.norm(), 1e-30))
    rows.append((n, rel, ""))

kinds = defaultdict(list)
for n, rel, note in rows:
    kinds[re.sub(r"\.\d+\.", ".N.", n)].append((rel, note))
print(f"\n{'tensor kind':60} {'count':>5} {'changed':>7} {'max rel diff':>12}")
for k, v in sorted(kinds.items()):
    rels = [r for r, _ in v if r == r]
    changed = sum(r > 5e-3 for r in rels)        # bf16 rounding of dequantized experts alone is ~1e-3
    notes = {nt for _, nt in v if nt}
    print(f"{k:60} {len(v):5} {changed:7} {max(rels) if rels else float('nan'):12.5f} {' '.join(notes)}")
