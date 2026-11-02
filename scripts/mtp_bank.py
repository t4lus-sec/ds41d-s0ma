#!/usr/bin/env python3
"""s0ma change 49: the DSpark drafter's 384 routed experts as an EXL3 bank, read by ds41d with DS41_MTP_EXL3=DIR.

  quantize DIR --weights CKPT --k 3 [--device 0] [--limit N]
      decode each MTP expert's native MXFP4 weights to F32 (exact), encode w1/w3/w2 with exllamav3's quantize_exl3
      at K bits (mul1 codebook, uncalibrated: what exllamav3's own converter does for MTP layers), and write
      DIR/bank-40.bin .. bank-42.bin (128 records each, in the target banks' layout), receipts.json and
      records-mtp.tsv. Resumes: a finished layer whose receipt matches is kept. --limit N encodes only the first N
      experts of layer 40 and prints their errors (a quick check; it writes nothing).
  verify DIR
      recompute every bank file's SHA-256 against its receipt.

Runs in a Python environment with torch (CUDA) and exllamav3 (ds41.sh mtp setup). Drafts are verified by the target, so
the drafter's weights change only acceptance, never the output distribution. Prints sizes and errors only."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import struct
import sys
import time

import numpy as np

H, I = 5120, 2304
ORDER = (('w1', 'gate', (H, I)), ('w3', 'up', (H, I)), ('w2', 'down', (I, H)))   # (k, n): in, out features
E2M1 = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=np.float32)


def layout(bits):
    off, views = 0, {}
    for _, name, (k, n) in ORDER:
        tb = (k // 16) * (n // 16) * 16 * bits * 2
        views[name] = (off, tb)
        views[name + '_scale'] = (off + tb, 2 * (k + n))
        off += tb + 2 * (k + n)
    return views, (off + 4095) // 4096 * 4096


assert layout(2)[1] == 8892416 and layout(3)[1] == 13316096


class Checkpoint:
    """safetensors shards read by byte range (no torch, no copies of whole shards)"""
    def __init__(self, root):
        self.root = Path(root)
        index = json.loads((self.root / 'model.safetensors.index.json').read_text())
        self.where, self.headers = index['weight_map'], {}

    def meta(self, name):
        shard = self.where[name]
        if shard not in self.headers:
            with open(self.root / shard, 'rb') as f:
                n = struct.unpack('<Q', f.read(8))[0]
                self.headers[shard] = (8 + n, json.loads(f.read(n)))
        base, header = self.headers[shard]
        t = header[name]
        a, b = t['data_offsets']
        return self.root / shard, base + a, b - a, t['dtype'], t['shape']

    def read(self, name):
        path, off, size, dtype, shape = self.meta(name)
        with open(path, 'rb') as f:
            f.seek(off)
            data = f.read(size)
        if len(data) != size:
            raise SystemExit(f'{name}: short read')
        return data, dtype, shape


def mxfp4(ck, base, n, k):
    """[n, k] F32 from a packed E2M1 weight (element 2j in byte j's low nibble, as ds41d's expand4 reads it) and its
    E8M0 scales [n, k/32] (2^(s-127)): exact, every value is a short binary fraction times a power of two"""
    w, wd, ws = ck.read(base + '.weight')
    s, sd, ss = ck.read(base + '.scale')
    if len(w) != n * k // 2 or len(s) != n * k // 32:
        raise SystemExit(f'{base}: {wd} {ws} / {sd} {ss}, want {n}x{k} packed E2M1 and {n}x{k // 32} E8M0')
    b = np.frombuffer(w, dtype=np.uint8).reshape(n, k // 2)
    codes = np.stack([b & 15, b >> 4], axis=-1).reshape(n, k)
    vals = E2M1[codes & 7] * np.where(codes & 8, -1.0, 1.0).astype(np.float32)
    scale = np.ldexp(np.float32(1), np.frombuffer(s, dtype=np.uint8).astype(np.int32) - 127).astype(np.float32)
    return (vals.reshape(n, k // 32, 32) * scale.reshape(n, k // 32, 1)).reshape(n, k)


def encoder(device):
    import torch
    from exllamav3.modules.quant.exl3_lib.quantize import quantize_exl3

    def encode(w_nk, bits, seed, workdir):
        """exllamav3's own uncalibrated path (a meta Hessian placeholder, as Linear.init_H_data(False) makes)"""
        k = w_nk.shape[1]
        weight = torch.from_numpy(np.ascontiguousarray(w_nk.T)).to(device)   # (in, out), row major
        h_data = {'H': torch.zeros(k, k, dtype=torch.float32, device='meta'), 'packed': False, 'n': k,
                  'first_key': f'mtp.{seed}', 'count': 0, 'finalized': False, 'num_total': 0,
                  'inf_nan': torch.zeros(2, dtype=torch.long, device=device), 'device': device}
        args = {'seed': seed, 'K': bits, 'devices': [device], 'device_ratios': None, 'apply_out_scales': None,
                'debug_dir': str(workdir / 'debug'), 'mul1': True}
        weight_q, _, out = quantize_exl3(weight, h_data, args, True)
        trellis, suh, svh = out['trellis'], out['suh'], out['svh']
        if out.get('mcg') is not None or out.get('mul1') is None:
            raise SystemExit('exllamav3 did not encode with the mul1 codebook (ds41d decodes mul1 only)')
        want = (k // 16, w_nk.shape[0] // 16, 16 * bits)
        if tuple(trellis.shape) != want or trellis.dtype != torch.int16:
            raise SystemExit(f'trellis {tuple(trellis.shape)} {trellis.dtype}, want {want} int16')
        if tuple(suh.shape) != (k,) or tuple(svh.shape) != (w_nk.shape[0],):
            raise SystemExit('suh/svh shapes')
        err = None
        if weight_q is not None and tuple(weight_q.shape) == tuple(weight.shape):
            err = float(torch.linalg.norm(weight_q.float() - weight) / torch.linalg.norm(weight))
        blob = (trellis.contiguous().cpu().numpy().astype('<i2').tobytes() + suh.cpu().numpy().astype('<f2').tobytes()
                + svh.cpu().numpy().astype('<f2').tobytes())
        return blob, err
    return encode


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(64 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def receipts(out):
    p = out / 'receipts.json'
    return json.loads(p.read_text()) if p.exists() else {}


def save_receipts(out, r):
    tmp = out / 'receipts.json.partial'
    tmp.write_text(json.dumps(r, indent=1) + '\n')
    os.replace(tmp, out / 'receipts.json')


def current(out, r, layer, bits, rec):
    x, path = r.get(str(layer)), out / f'bank-{layer}.bin'
    return bool(x) and path.exists() and x.get('bits') == bits and len(x.get('sha256', '')) == 64 and \
        os.stat(path).st_size == 128 * rec and os.stat(path).st_mtime_ns == x.get('mtime_ns')


def manifest(out, bits):
    views, rec = layout(bits)
    v = views
    lines = [f'# DS41MOE01 EXL3 K{bits} mul1 DSpark bank (scripts/mtp_bank.py): gate/up/down -> trellis, '
             '*_scale -> suh then svh']
    for layer in (40, 41, 42):
        path = (out / f'bank-{layer}.bin').resolve()
        for e in range(128):
            eid = 15360 + (layer - 40) * 128 + e
            lines.append('R %d %d %d %d %d %d %d %d %d %d %d %d %d %d' % (
                eid, rec, *v['gate'], *v['gate_scale'], *v['up'], *v['up_scale'], *v['down'], *v['down_scale']))
            lines.append('E %d "%s" %d %d 0 %d %d' % (eid, path, e * rec, rec, rec, 128 * rec))
    (out / 'records-mtp.tsv').write_text('\n'.join(lines) + '\n')


def quantize(out, weights, bits, device, limit):
    out.mkdir(parents=True, exist_ok=True)
    views, rec = layout(bits)
    ck, encode = Checkpoint(weights), encoder(device)
    r = receipts(out)
    if limit:
        errs = {name: [] for _, name, _ in ORDER}
        for e in range(limit):
            for w, name, (k, n) in ORDER:
                _, err = encode(mxfp4(ck, f'mtp.0.ffn.experts.{e}.{w}', n, k), bits, e * 3 + 1, out)
                errs[name].append(err)
        print(f'K{bits}, {limit} experts of layer 40: relative weight error (|Wq - W| / |W|) ' +
              ', '.join(f'{name} {np.mean(v):.3f}' for name, v in errs.items() if None not in v), flush=True)
        return
    t0 = time.time()
    for layer in (40, 41, 42):
        if current(out, r, layer, bits, rec):
            print(f'layer {layer}: kept (receipt matches)', flush=True)
            continue
        path, part = out / f'bank-{layer}.bin', out / f'bank-{layer}.bin.partial'
        errs = []
        with open(part, 'wb') as f:
            for e in range(128):
                record = bytearray()
                for j, (w, name, (k, n)) in enumerate(ORDER):
                    if len(record) != views[name][0]:
                        raise SystemExit('record layout drift')
                    blob, err = encode(mxfp4(ck, f'mtp.{layer - 40}.ffn.experts.{e}.{w}', n, k), bits,
                                       ((layer - 40) * 128 + e) * 3 + j + 1, out)
                    if len(blob) != views[name][1] + views[name + '_scale'][1]:
                        raise SystemExit(f'{name}: {len(blob)} bytes, want {views[name][1] + views[name + "_scale"][1]}')
                    record += blob
                    if err is not None:
                        errs.append(err)
                record += bytes(rec - len(record))
                f.write(record)
                if e % 16 == 15:
                    print(f'layer {layer}: {e + 1}/128 experts, {time.time() - t0:.0f} s', flush=True)
        os.replace(part, path)
        st = os.stat(path)
        r[str(layer)] = {'file': path.name, 'bytes': st.st_size, 'sha256': sha256(path), 'mtime_ns': st.st_mtime_ns,
                         'bits': bits, 'source': 'native MXFP4 MTP experts, exllamav3 quantize_exl3 uncalibrated mul1',
                         'mean_relative_error': float(np.mean(errs)) if errs else None,
                         'written': time.strftime('%Y-%m-%dT%H:%M:%S%z')}
        save_receipts(out, r)
        print(f'layer {layer}: done, mean relative weight error {r[str(layer)]["mean_relative_error"]}', flush=True)
    manifest(out, bits)
    native = 384 * 18808832
    print(f'bank complete: {out.resolve()} (K{bits}: {384 * rec / 1e9:.2f} GB in place of {native / 1e9:.2f} GB); '
          f'run with DS41_MTP_EXL3={out.resolve()}', flush=True)


def verify(out):
    r = receipts(out)
    bad = []
    for layer in (40, 41, 42):
        x, path = r.get(str(layer)), out / f'bank-{layer}.bin'
        if not x or not path.exists() or sha256(path) != x.get('sha256'):
            bad.append(layer)
        else:
            x['mtime_ns'] = os.stat(path).st_mtime_ns
    if bad:
        raise SystemExit(f'layers {bad} missing or differ from their receipts: run quantize again')
    save_receipts(out, r)
    print(f'verified 3 layers in {out.resolve()}', flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    q = sub.add_parser('quantize')
    q.add_argument('dir', type=Path)
    q.add_argument('--weights', required=True)
    q.add_argument('--k', type=int, choices=(2, 3, 4), default=3)
    q.add_argument('--device', default='cuda:0')
    q.add_argument('--limit', type=int, default=0)
    v = sub.add_parser('verify')
    v.add_argument('dir', type=Path)
    a = ap.parse_args()
    if a.cmd == 'quantize':
        quantize(a.dir, a.weights, a.k, a.device, a.limit)
    else:
        verify(a.dir)


if __name__ == '__main__':
    sys.exit(main())
