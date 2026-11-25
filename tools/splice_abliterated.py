"""Build an abliterated Inkling 2 Mini GGUF in Maya's (old Unsloth) layout.

orcarouter's abliteration edited only the token embedding, the output (unembed) matrix, every layer's attention output
projection (wo_ud) and every layer's expert down projection (w2). Everything else in their bf16
release is the official weights. So: copy Unsloth's old-layout UD-Q4_K_XL file and overwrite just
those tensors in place, re-quantized to the type each already has (Q8_0 or NVFP4 with the file's own
per-expert outer scale). Same type means same byte size, so metadata and offsets are untouched.

NVFP4 is coarse next to the edit: plain nearest rounding snaps up to ~25% of the (rank-4) edit back
to the stock codes. So each NVFP4 expert is re-quantized with error feedback in the edit's rank-4
subspace: the part of the edit lost to rounding is added back to the target, 3 passes (>=99% kept).

usage: splice_abliterated.py <unsloth-old.gguf> <orca-bf16-dir> <out.gguf>
"""
import json
import os
import shutil
import sys

import numpy as np
from gguf import GGUFReader
from gguf.quants import NVFP4, Q8_0
from safetensors import safe_open

SRC, ORCA, OUT = sys.argv[1:4]

E2M1 = np.array([0, 0.5, 1, 1.5, 2, 3, 4, 6], dtype=np.float32)  # magnitudes; codes 8..15 negate
# Every non-NaN unsigned E4M3 byte (0..126) and its real value. gguf-py's fp32_to_ue4m3 clamps the
# whole exponent-15 range (256..448) to 0x7E, which the official checkpoint does use, so encode by table.
UE4M3_BYTES = np.arange(127, dtype=np.uint8)
UE4M3_VALS = NVFP4.ue4m3_to_fp32(UE4M3_BYTES).astype(np.float64) * 2.0   # undo the 0.5 kvalues factor


def ue4m3_nearest(x: np.ndarray) -> np.ndarray:
    x = np.clip(x.astype(np.float64), 0.0, UE4M3_VALS[-1])
    hi = np.clip(np.searchsorted(UE4M3_VALS, x), 1, 126)
    lo = hi - 1
    pick_hi = (UE4M3_VALS[hi] - x) < (x - UE4M3_VALS[lo])
    return np.where(pick_hi, hi, lo).astype(np.uint8)


def nvfp4_encode(x: np.ndarray) -> np.ndarray:
    """x: (n, 64) float32 values already divided by the outer scale -> (n, 36) uint8 GGUF NVFP4 blocks."""
    n = x.shape[0]
    sub = x.reshape(n, 4, 16)
    amax = np.abs(sub).max(axis=-1)                       # (n, 4)
    d_bytes = ue4m3_nearest(amax / 6.0)                   # E4M3 sub-block scale
    d = NVFP4.ue4m3_to_fp32(d_bytes) * 2.0                # actual scale (ue4m3_to_fp32 carries a 0.5 factor)
    safe = np.where(d > 0, d, 1.0)[..., None]
    mag = np.abs(sub) / safe
    idx = np.abs(mag[..., None] - E2M1).argmin(axis=-1).astype(np.uint8)   # 0..7
    idx = np.where(sub < 0, idx | 8, idx)
    idx = np.where((d > 0)[..., None], idx, 0).astype(np.uint8)
    lo, hi = idx[..., :8], idx[..., 8:]
    qs = (lo | (hi << 4)).astype(np.uint8).reshape(n, 32)
    return np.concatenate([d_bytes.astype(np.uint8), qs], axis=1)


def self_test(blocks: np.ndarray):
    """Re-encoding the decoded official NVFP4 blocks must reproduce them (up to scale ties)."""
    dec = NVFP4.dequantize_blocks(blocks)
    enc = nvfp4_encode(dec)
    redec = NVFP4.dequantize_blocks(enc)
    same = np.mean(enc == blocks)
    err = np.abs(redec - dec).max()
    print(f"  NVFP4 self-test: {same * 100:.3f}% bytes identical, max decode diff {err:.3g}")
    assert err < 1e-6, "NVFP4 encoder does not round-trip the official blocks"


FEEDBACK_PASSES = 3


def top_subspace(delta: np.ndarray, k: int = 4, rng=np.random.default_rng(0)):
    """Rank-k left/right singular subspaces of delta by randomized subspace iteration (delta is ~rank 4)."""
    Q, _ = np.linalg.qr(delta @ rng.standard_normal((delta.shape[1], k + 4)))
    for _ in range(2):
        Q, _ = np.linalg.qr(delta @ (delta.T @ Q))
    U, _, Vt = np.linalg.svd(Q.T @ delta, full_matrices=False)
    return (Q @ U)[:, :k], Vt[:k].T


def one_tensor(job):
    gname, oname = job
    r = GGUFReader(SRC)
    T = {t.name: t for t in r.tensors}
    t = T[gname]
    qt = t.tensor_type.name
    w = orca(oname)
    old = np.asarray(t.data)
    nbytes = old.nbytes
    if gname in ("token_embd.weight", "output.weight") and w.shape[0] != old.shape[0]:
        base = Q8_0.dequantize_blocks(old.reshape(-1, 34)).reshape(old.shape[0], -1)
        w = np.concatenate([w, base[w.shape[0]:]])          # keep the file's padded rows
    kept = float("nan")
    if qt == "Q8_0":
        before = Q8_0.dequantize_blocks(old.reshape(-1, 34)).reshape(w.shape)
        new = Q8_0.quantize_blocks(w.reshape(-1, 32).astype(np.float32)).reshape(-1)
        after = Q8_0.dequantize_blocks(new.reshape(-1, 34)).reshape(w.shape)
    elif qt == "NVFP4":
        scale = np.asarray(T[gname.replace(".weight", ".scale")].data).reshape(-1).astype(np.float32)
        E = w.shape[0]
        per = old.reshape(E, -1)
        before = np.empty_like(w); after = np.empty_like(w); parts = []
        num = den = 0.0
        for e in range(E):
            stock = NVFP4.dequantize_blocks(per[e].reshape(-1, 36)).reshape(w.shape[1:]) * scale[e]
            delta = w[e] - stock
            U4, V4 = top_subspace(delta)
            target = w[e].copy()
            for p in range(FEEDBACK_PASSES + 1):
                enc = nvfp4_encode((target / scale[e]).reshape(-1, 64).astype(np.float32))
                out = NVFP4.dequantize_blocks(enc).reshape(w.shape[1:]) * scale[e]
                if p < FEEDBACK_PASSES:
                    target = target + U4 @ (U4.T @ (w[e] - out) @ V4) @ V4.T
            before[e], after[e] = stock, out
            num += float(np.sum((out - stock) * delta)); den += float(np.sum(delta * delta))
            parts.append(enc.reshape(-1))
        new = np.concatenate(parts)
        kept = num / den
    else:
        raise SystemExit(f"{gname}: unexpected type {qt}")
    assert new.nbytes == nbytes, (gname, new.nbytes, nbytes)
    mm = np.memmap(OUT, dtype=np.uint8, mode="r+")
    mm[t.data_offset:t.data_offset + nbytes] = new
    mm.flush()
    edit = np.linalg.norm(w - before) / np.linalg.norm(w)
    qerr = np.linalg.norm(w - after) / np.linalg.norm(w)
    return f"{gname:32} {qt:5}  edit vs unsloth {edit:.4f}  requant err {qerr:.4f}  edit kept {kept:.3f}"


idx = json.load(open(os.path.join(ORCA, "model.safetensors.index.json")))["weight_map"]


def orca(name):
    with safe_open(os.path.join(ORCA, idx[name]), "pt") as f:
        return f.get_tensor(name).float().numpy()


if __name__ == "__main__":
    import multiprocessing as mp
    r = GGUFReader(SRC)
    T = {t.name: t for t in r.tensors}
    # The full edit set, checked against the official checkpoint by diff_official.py.
    jobs = [("token_embd.weight", "model.llm.embed.weight"), ("output.weight", "model.llm.unembed.weight")]
    for L in range(24):
        jobs.append((f"blk.{L}.ffn_down_exps.weight", f"model.llm.layers.{L}.mlp.experts.w2_weight"))
        jobs.append((f"blk.{L}.attn_output.weight", f"model.llm.layers.{L}.attn.wo_ud.weight"))
    if os.environ.get("SPLICE_ONLY"):            # comma-separated GGUF names: redo just these tensors
        only = set(os.environ["SPLICE_ONLY"].split(","))
        jobs = [j for j in jobs if j[0] in only]
    print("self-testing the NVFP4 encoder on blk.0.ffn_down_exps ...", flush=True)
    self_test(np.asarray(T["blk.0.ffn_down_exps.weight"].data)[0].reshape(-1, 36))
    if not os.path.exists(OUT):
        print(f"copying {SRC} -> {OUT}", flush=True)
        shutil.copyfile(SRC, OUT)
    os.environ.setdefault("OMP_NUM_THREADS", "2")
    with mp.get_context("spawn").Pool(int(os.environ.get("SPLICE_WORKERS", "8"))) as pool:
        for line in pool.imap_unordered(one_tensor, jobs):
            print(line, flush=True)
    print("done:", OUT, flush=True)
