#!/usr/bin/env python3
"""s0ma change 99: train ds41d's DSpark drafter (the checkpoint's MTP layers) to the target it drafts for.

The stock drafter matches the full-precision V4.1; ds41d verifies its drafts against 0xSero's 2.7-bit experts. This
fits the drafter to the served target's own next-token distributions, on s0ma's own model calls. Drafts are verified
exactly, so the result can change acceptance (speed), never what the model says.

  export  --s0ma DIR --out DATA --tokenizer FILE [--window 4096] [--max-tokens 4000000] [--heldout 0.05]
          s0ma's saved model calls (DIR/data/calls) rendered as ds41d serves them (the adapter's mapping, ds41_api's
          encoding), the last WINDOW tokens of each prompt plus its response, with each token marked by who wrote it.
          Refuses while a /job is open or paused (DIR/data/jobs.json, fail closed), and skips calls holding a redaction
          marker. Writes DATA/docs/NAME.tok and .w, DATA/index.jsonl. Prints counts only.
  capture --data DATA [--topk 32]
          against a lab ds41d started with DS41_TRAIN_CAPTURE_DIR=DATA/docs: per document, the target's top-k
          next-token log-probabilities (.top) and the drafter's ring rows (.kv), by ds41d's train_capture verb.
          Resumes: documents already captured are skipped.
  train   --data DATA --weights CKPT --out PATCH [--mtp-bank DIR] [--train medium] [--epochs 2] [--lr 2e-5]
          with ds41d stopped (the GPU to itself): the stock drafter is scored on the held-out documents first (the
          check that this port agrees with ds41d's drafter), then trained, scored again, and written as
          PATCH/patch.safetensors + manifest.json, for DS41_SPARK_PATCH=PATCH.
  eval    --data DATA --weights CKPT [--patch PATCH] [--mtp-bank DIR]

Scores are the expected acceptance of each draft depth, sum(min(p, q)) over the target's top k (a lower bound), and
the expected accepted drafts a full-width step, sum over k of the product of the first k+1 acceptances. Prints and
writes numbers only, never text.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import socket
import struct
import sys
import time

from array import array

try:
    import numpy as np   # capture, train, eval (export runs in the API's environment, which has none)
except ImportError:
    np = None

H, I, V, BLOCK, RING, HEADS, HD, RD = 5120, 2304, 129280, 5, 128, 64, 512, 64
GROUPS, OLORA, EXPERTS = 8, 1024, 128   # wo_a's groups and width; routed experts a layer
NOISE = 128799            # the block's filler token (Spark's forward.cu)
EPS_NORM, EPS_HC = 1e-20, 1e-6
ROUTE_SCALE, CLAMP = 1.5, 10.0
W_OTHER, W_MINE = 1, 2    # .w codes: 0 no loss, 1 text s0ma read, 2 text s0ma (the served model) wrote


def say(*a):
    print(*a, flush=True)


# ---------------------------------------------------------------------------------------------------------------------
# export
# ---------------------------------------------------------------------------------------------------------------------

def jobs_block(s0ma):
    """Why the archive may not be read now, or None. Fails closed: a job counts as closed only when its record says
    so (a closed state, a close time, or every span ended); a file that does not parse, or entries in a shape this
    does not recognize, block."""
    p = s0ma / 'data' / 'jobs.json'
    if not p.exists():
        return None
    try:
        data = json.loads(p.read_text())
    except (OSError, ValueError):
        return 'data/jobs.json does not parse'
    if not data:
        return None
    jobs = []

    def walk(x):
        if isinstance(x, dict):
            if any(k in x for k in ('state', 'status', 'spans', 'closed', 'closed_at', 'paused', 'open')):
                jobs.append(x)
                return
            for v in x.values():
                walk(v)
        elif isinstance(x, list):
            for v in x:
                walk(v)
    walk(data)
    if not jobs:
        return 'data/jobs.json is in a shape this tool does not recognize'
    for j in jobs:
        state = str(j.get('state', j.get('status', ''))).lower()
        if state in ('open', 'paused', 'active', 'running') or j.get('paused') is True or j.get('open') is True:
            return f'a job is {state or "open"}'
        if state == 'closed' or j.get('closed') is True or j.get('closed_at'):
            continue
        spans = j.get('spans')
        if isinstance(spans, list) and spans and all(
                isinstance(s, dict) and s.get('end') is not None or isinstance(s, list) and len(s) == 2 and
                s[1] is not None for s in spans):
            continue
        return 'a job is not recorded as closed'
    return None


# the call store (s0ma's data/calls), read as lab/suffix.py learned it from the real store: any objects layout, gzip
# or zstd blobs, a request as message references or as one body
SHA = re.compile(r'(sha256:)?[0-9a-f]{64}')


_ZSTD = None


def unzstd(raw):
    """a zstd frame: the zstandard module, Python 3.14's compression.zstd, or zstandard installed once into
    ~/.ds41/pylib (DS41_PYLIB; never into an environment), else the zstd command (one process a blob: slow)"""
    global _ZSTD
    if _ZSTD is None:
        _ZSTD = _zstd_reader()
    return _ZSTD(raw)


def _zstd_reader():
    import subprocess
    import shutil

    def module():
        try:
            import zstandard
            return lambda b: zstandard.ZstdDecompressor().decompressobj().decompress(b)
        except ImportError:
            pass
        try:
            from compression import zstd
            return zstd.decompress
        except ImportError:
            return None
    r = module()
    if r:
        return r
    lib = Path(os.environ.get('DS41_PYLIB') or Path.home() / '.ds41' / 'pylib')
    if str(lib) not in sys.path:
        sys.path.insert(0, str(lib))
    r = module()
    if r:
        return r
    say(f'  zstd blobs: installing the zstandard module into {lib} (once)')
    lib.mkdir(parents=True, exist_ok=True)
    p = subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '--target', str(lib), 'zstandard'],
                       capture_output=True, text=True)
    if p.returncode == 0:
        import importlib
        importlib.invalidate_caches()
        r = module()
        if r:
            return r
    exe = shutil.which('zstd') or shutil.which('unzstd')
    if exe:
        say('  zstd blobs: the zstandard module did not install; using the zstd command (slower)')
        return lambda b: subprocess.run([exe, '-dcq'], input=b, capture_output=True, check=True).stdout
    sys.exit('zstd blobs: no zstandard module, pip could not install it '
             f'({(p.stderr or "").strip()[-200:]}), and no zstd command (sudo apt install zstd)')


class Calls:
    def __init__(self, root):
        self.root, self.objects, self._index, self._blobs = root, root / 'objects', None, {}

    def lines(self):
        """the call lines in the store's order (append order: oldest first), update lines and torn lines left out"""
        with open(self.root / 'index.jsonl', 'rb') as f:
            for line in f:
                try:
                    rec = json.loads(line)
                except ValueError:
                    continue
                if isinstance(rec, dict) and not rec.get('update'):
                    yield rec

    def path(self, sha):
        if self._index is None:
            self._index = {}
            for path in self.objects.rglob('*'):
                if path.is_file():
                    rel = path.relative_to(self.objects).as_posix()
                    hexs = re.sub(r'[^0-9a-f]', '', rel.split('.', 1)[0].lower())
                    if len(hexs) >= 64:
                        self._index[hexs[-64:]] = path
        return self._index.get(sha)

    def blob(self, ref):
        import gzip
        sha = ref.split(':', 1)[-1].lower()
        if sha not in self._blobs:
            path = self.path(sha)
            if path is None:
                raise FileNotFoundError('blob')
            raw = path.read_bytes()
            for _ in range(2):
                try:
                    raw = gzip.decompress(raw)
                except OSError:
                    break
            if raw[:4] == b'\x28\xb5\x2f\xfd':
                raw = unzstd(raw)
            if len(self._blobs) > 20000:
                self._blobs.clear()
            self._blobs[sha] = json.loads(raw)
        return self._blobs[sha]

    def resolve(self, v):
        if isinstance(v, str) and SHA.fullmatch(v):
            return self.blob(v)
        if isinstance(v, dict):
            for k in ('sha256', 'blob', 'ref', 'object'):
                if isinstance(v.get(k), str) and SHA.fullmatch(v[k]):
                    return self.blob(v[k])
        return v

    def request(self, rec):
        """(messages, tools or None, params) of a call line"""
        req = rec.get('request') or {}
        if req.get('body'):
            body = self.resolve(req['body'])
            body = json.loads(body) if isinstance(body, str) else body
            params = {k: v for k, v in body.items() if k not in ('messages', 'tools')}
            return [self.resolve(m) for m in body.get('messages') or []], body.get('tools'), params
        tools = self.resolve(req['tools']) if req.get('tools') is not None else None
        return [self.resolve(m) for m in req.get('messages') or []], tools, dict(req.get('params') or {})


def response_message(resp):
    """the assistant message of a stored response (an OpenAI body, or the message itself)"""
    if isinstance(resp, dict) and resp.get('choices'):
        resp = resp['choices'][0].get('message') or {}
    if not isinstance(resp, dict):
        return None
    msg = {'role': 'assistant', 'content': resp.get('content') or ''}
    rc = resp.get('reasoning_content') or resp.get('reasoning')
    if isinstance(rc, str) and rc:
        msg['reasoning_content'] = rc
    if resp.get('tool_calls'):
        msg['tool_calls'] = resp['tool_calls']
    return msg


def writer_marks(ids, specials):
    """2 inside assistant turns (after <｜Assistant｜>, up to and including the end of the turn), 1 elsewhere"""
    out = bytearray([W_OTHER]) * len(ids)
    inside = False
    for i, t in enumerate(ids):
        if t == specials['assistant']:
            inside = True
            continue
        if t in specials['ends']:
            if inside:
                out[i] = W_MINE
            inside = False
            continue
        if inside:
            out[i] = W_MINE
    return out


def export(a):
    s0ma = Path(a.s0ma).expanduser()
    why = jobs_block(s0ma)
    if why:
        sys.exit(f'export refused: {why} (data/jobs.json). Close the job (/job close NAME --yes) first.')
    calls = Calls(s0ma / 'data' / 'calls')
    if not (calls.root / 'index.jsonl').exists():
        sys.exit(f'no saved model calls at {calls.root}')
    sys.path[:0] = [p for p in (a.api, a.adapter) if p]
    from ds41_api.encoder import Encoder
    from ds41_api import encoding as enc
    from ds41_api.schemas import ChatCompletionRequest, resolve_thinking_mode
    from s0ma_ds41.mapping import map_request
    encoder = Encoder.from_file(a.tokenizer)
    specials = {'assistant': encoder.token_id('<｜Assistant｜>'),
                'ends': {x for x in (encoder.token_id('<｜end▁of▁sentence｜>'), encoder.token_id('<｜User｜>'))
                         if x is not None}}
    if specials['assistant'] is None or not specials['ends']:
        sys.exit('the tokenizer has no <｜Assistant｜> / <｜end▁of▁sentence｜> tokens')
    out = Path(a.out)
    if (out / 'index.jsonl').exists():
        sys.exit(f'{out} already holds an export: give a new --out (captures are tied to its documents)')
    (out / 'docs').mkdir(parents=True, exist_ok=True)
    rows = [r for r in calls.lines() if r.get('response') and
            ((r.get('request') or {}).get('messages') or (r.get('request') or {}).get('body'))]
    say(f'calls: {len(rows):,} with a request and a response')
    counts = dict(redacted=0, unreadable=0, unmapped=0, images=0, duplicate=0)

    def render(line):
        """(the call's prompt + completion ids, the completion's length), or the reason it is skipped"""
        try:
            messages, tools, params = calls.request(line)
            resp = calls.resolve(line['response'])
            if not messages or not all(isinstance(m, dict) for m in messages):
                raise ValueError('shape')
        except (KeyError, OSError, ValueError, TypeError):
            return 'unreadable'
        if '[redacted' in json.dumps([messages, tools, resp], ensure_ascii=False):
            return 'redacted'
        body = {k: v for k, v in params.items() if k not in ('stream', 'stream_options')}
        body['messages'] = messages
        if tools:
            body['tools'] = tools
        try:
            mapped = map_request(body, 'deepseek-v4.1-flash-1017')
            up = dict(mapped.upstream)
            up['stream'] = False
            request = ChatCompletionRequest(**up)
            prompt_text, images = encoder.build_prompt(request)
            if images:
                return 'images'
            prompt = encoder.encode_prompt(prompt_text)
            msg = response_message(resp)
            if msg is None:
                raise ValueError('response')
            # the completion as the model writes it after the prompt (its reasoning, </think>, the reply, its tool
            # calls, end of sentence), tokenized on its own as it was generated
            text = enc.render_message(1, [{'role': 'user', 'content': ''}, msg],
                                      thinking_mode=resolve_thinking_mode(request.thinking), drop_thinking=False)
            completion = encoder.encode_prompt(text)
        except Exception:
            return 'unmapped'
        return array('i', prompt + completion), len(completion)

    def lcp(x, y):
        """the longest common prefix of two id arrays (by bisection on their bytes)"""
        bx, by = x.tobytes(), y.tobytes()
        lo, hi = 0, min(len(x), len(y))
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if bx[:4 * mid] == by[:4 * mid]:
                lo = mid
            else:
                hi = mid - 1
        return lo

    seen, docs, total, fresh_total = set(), [], 0, 0

    def finish(full, comp, known):
        """write one document: the last WINDOW tokens; loss only where the call is new (past the longest prefix an
        older call already had), the rest is context the target and the ring see"""
        nonlocal total, fresh_total
        start = max(0, len(full) - a.window)
        ids = full[start:]
        key = hashlib.sha256(ids.tobytes()).hexdigest()
        if key in seen:
            counts['duplicate'] += 1
            return
        seen.add(key)
        marks = writer_marks(ids, specials)
        c0 = max(0, len(ids) - comp)
        marks[c0:] = bytes([W_MINE]) * (len(ids) - c0)
        k0 = max(0, min(known - start, c0))
        marks[:k0] = bytes(k0)
        fresh = len(marks) - marks.count(0)
        if fresh < a.min_fresh:
            counts['duplicate'] += 1
            return
        name = f'd{len(docs):06d}'
        (out / 'docs' / f'{name}.tok').write_bytes(ids.tobytes())
        (out / 'docs' / f'{name}.w').write_bytes(bytes(marks))
        docs.append({'name': name, 'n': len(ids), 'fresh': fresh, 'mine': marks.count(W_MINE)})
        total += len(ids)
        fresh_total += fresh

    # newest first, so the token budget keeps the newest work (the held-out set is the newest of that). A call's
    # known prefix is the longest it shares with any of the next 8 older calls (its own conversation's last step)
    pending = []
    for line in reversed(rows):
        if total >= a.max_tokens:
            break
        r = render(line)
        if isinstance(r, str):
            counts[r] += 1
            continue
        full, comp = r
        for doc in pending:
            doc[2] = max(doc[2], lcp(doc[0], full))
            doc[3] += 1
        while pending and pending[0][3] >= 8:
            f, c, k, _ = pending.pop(0)
            finish(f, c, k)
        pending.append([full, comp, 0, 0])
    for f, c, k, _ in pending:
        if total < a.max_tokens:
            finish(f, c, k)
    # oldest first again; the newest share (by tokens) is held out
    docs.reverse()
    held, budget = 0, a.heldout * fresh_total
    for d in reversed(docs):
        d['split'] = 'heldout' if held < budget else 'train'
        held += d['fresh'] if d['split'] == 'heldout' else 0
    with open(out / 'index.jsonl', 'w') as f:
        for d in docs:
            f.write(json.dumps(d) + '\n')
    mine = sum(d['mine'] for d in docs)
    say(f'documents: {len(docs):,}, {total:,} tokens to capture, {fresh_total:,} new (trained on; {mine:,} written by '
        f'the model), held out {held:,} of the new tokens')
    say('skipped: ' + ', '.join(f'{k} {v:,}' for k, v in counts.items()))


# ---------------------------------------------------------------------------------------------------------------------
# capture
# ---------------------------------------------------------------------------------------------------------------------

def engine_call(path, op):
    with socket.socket(socket.AF_UNIX) as c:
        c.settimeout(3600)
        c.connect(str(path))
        c.sendall((json.dumps(op) + '\n').encode())
        with c.makefile('r') as stream:
            for line in stream:
                e = json.loads(line)
                if e.get('type') == 'finish':
                    if e.get('error') or e.get('finish_reason') != 'stop':
                        raise RuntimeError(str(e.get('error') or e.get('finish_reason')))
                    return e.get('stats') or {}
    raise RuntimeError('the engine closed the connection')


def index(data):
    with open(Path(data) / 'index.jsonl') as f:
        return [json.loads(l) for l in f if l.strip()]


def capture(a):
    data = Path(a.data)
    if not a.socket:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        import run
        a.socket = str(run.runtime_dir(run.configuration()) / 'engine.sock')
    if not Path(a.socket).exists():
        sys.exit(f'no engine socket at {a.socket}: start the lab ds41d with DS41_TRAIN_CAPTURE_DIR={data / "docs"}')
    meta = data / 'capture.json'
    if meta.exists() and json.loads(meta.read_text()).get('topk') != a.topk:
        sys.exit(f'{data} was captured with --topk {json.loads(meta.read_text()).get("topk")}: keep it')
    meta.write_text(json.dumps({'topk': a.topk}) + '\n')
    docs = index(data)
    todo = [d for d in docs if not ((data / 'docs' / f"{d['name']}.kv").exists() and
                                    (data / 'docs' / f"{d['name']}.top").exists())]
    say(f'capture: {len(docs) - len(todo):,} of {len(docs):,} documents already done')
    t0, rows = time.time(), 0
    for i, d in enumerate(todo):
        ids = array('i')
        ids.frombytes((data / 'docs' / f"{d['name']}.tok").read_bytes())
        ids = ids.tolist()
        engine_call(a.socket, {'op': 'train_capture', 'request_id': f"tc-{d['name']}", 'token_ids': ids,
                               'name': d['name'], 'topk': a.topk, 'priority': 0})
        rows += len(ids)
        if i % 20 == 19 or i == len(todo) - 1:
            el = time.time() - t0
            say(f'  {i + 1:,}/{len(todo):,} documents, {rows:,} tokens, {rows / max(el, 1e-9):,.0f} tok/s, '
                f'{el / 60:.1f} min')


# ---------------------------------------------------------------------------------------------------------------------
# weights
# ---------------------------------------------------------------------------------------------------------------------

class Checkpoint:
    """safetensors shards read by byte range"""
    def __init__(self, root):
        self.root = Path(root)
        self.where = json.loads((self.root / 'model.safetensors.index.json').read_text())['weight_map']
        self.headers = {}

    def has(self, name):
        return name in self.where

    def read(self, name):
        return self.raw(name)

    def raw_dtype(self, name):
        self.raw_header(name)
        shard = self.where[name]
        return self.headers[shard][1][name]['dtype']

    def raw_header(self, name):
        shard = self.where[name]
        if shard not in self.headers:
            with open(self.root / shard, 'rb') as f:
                n = struct.unpack('<Q', f.read(8))[0]
                self.headers[shard] = (8 + n, json.loads(f.read(n)))

    def raw(self, name):
        shard = self.where[name]
        if shard not in self.headers:
            with open(self.root / shard, 'rb') as f:
                n = struct.unpack('<Q', f.read(8))[0]
                self.headers[shard] = (8 + n, json.loads(f.read(n)))
        base, header = self.headers[shard]
        t = header[name]
        a, b = t['data_offsets']
        with open(self.root / shard, 'rb') as f:
            f.seek(base + a)
            data = f.read(b - a)
        return data, t['dtype'], t['shape']


def to_torch(data, dtype, shape):
    import torch
    buf = bytearray(data)
    if dtype == 'BF16':
        return torch.frombuffer(buf, dtype=torch.bfloat16).reshape(shape).clone()
    if dtype == 'F32':
        return torch.frombuffer(buf, dtype=torch.float32).reshape(shape).clone()
    if dtype == 'F16':
        return torch.frombuffer(buf, dtype=torch.float16).reshape(shape).float()
    if dtype == 'F8_E4M3':
        return torch.frombuffer(buf, dtype=torch.uint8).reshape(shape).view(torch.float8_e4m3fn).float()
    if dtype in ('F8_E8M0', 'U8', 'I8'):
        return torch.frombuffer(buf, dtype=torch.uint8).reshape(shape).clone()
    raise SystemExit(f'unsupported checkpoint dtype {dtype}')


def load_dense(ck, name):
    """F32 values of a checkpoint tensor; an FP8 weight times its block scales (exact)"""
    import torch
    data, dtype, shape = ck.raw(name)
    t = to_torch(data, dtype, shape)
    if dtype != 'F8_E4M3':
        return t.float()
    base = name[:-len('.weight')] if name.endswith('.weight') else name
    sdata, sdt, sshape = ck.raw(base + '.scale')
    s = to_torch(sdata, sdt, sshape)
    s = torch.ldexp(torch.ones_like(s, dtype=torch.float32), s.int() - 127) if s.dtype == torch.uint8 else s.float()
    bn, bk = shape[0] // sshape[0], shape[1] // sshape[1]
    return t * s.repeat_interleave(bn, 0).repeat_interleave(bk, 1)


def load_experts(ck, layer, bank, device):
    """the drafter's 128 routed experts of layer 40+layer as served: the EXL3 bank's decoded weights (ds41.sh mtp
    quantize) when there is one, else the checkpoint's MXFP4 decoded exactly. BF16 [128, out, in] each."""
    import torch
    w1 = torch.empty(EXPERTS, I, H, dtype=torch.bfloat16, device=device)
    w3 = torch.empty(EXPERTS, I, H, dtype=torch.bfloat16, device=device)
    w2 = torch.empty(EXPERTS, H, I, dtype=torch.bfloat16, device=device)
    if bank:
        from exllamav3.modules.quant.exl3 import LinearEXL3
        from exllamav3.modules.quant.exl3_lib.quantize import codebook_mul1_mult
        mul1 = torch.tensor(codebook_mul1_mult, dtype=torch.uint32).view(torch.int).to(device)
        bits = int(json.loads((Path(bank) / 'receipts.json').read_text())[str(40 + layer)]['bits'])
        sys.path.insert(0, str(Path(__file__).parent))
        from mtp_bank import layout, ORDER
        views, rec = layout(bits)
        raw = np.fromfile(Path(bank) / f'bank-{40 + layer}.bin', dtype=np.uint8)
        for e in range(EXPERTS):
            r = raw[e * rec:(e + 1) * rec]
            for w, name, (k, n) in ORDER:
                off, tb = views[name]
                soff, _ = views[name + '_scale']
                trellis = torch.from_numpy(r[off:off + tb].view('<i2').reshape(k // 16, n // 16, 16 * bits).copy())
                suh = torch.from_numpy(r[soff:soff + 2 * k].view('<f2').copy())
                svh = torch.from_numpy(r[soff + 2 * k:soff + 2 * (k + n)].view('<f2').copy())
                lin = LinearEXL3(None, k, n, suh=suh.to(device), svh=svh.to(device), trellis=trellis.to(device),
                                 mul1=mul1)
                wt = lin.get_weight_tensor().t().to(torch.bfloat16)   # [out, in]
                {'w1': w1, 'w3': w3, 'w2': w2}[w][e].copy_(wt)
        return w1, w3, w2
    sys.path.insert(0, str(Path(__file__).parent))
    from mtp_bank import mxfp4
    for e in range(EXPERTS):
        for w, dst, (n, k) in (('w1', w1, (I, H)), ('w3', w3, (I, H)), ('w2', w2, (H, I))):
            dst[e].copy_(torch.from_numpy(mxfp4(ck, f'mtp.{layer}.ffn.experts.{e}.{w}', n, k)).to(torch.bfloat16))
    return w1, w3, w2


# the drafter's tensors by role. dtype: what ds41d binds (the patch is written in it); groups: the --train sets
TENSORS = []
for l in range(3):
    p = f'mtp.{l}'
    TENSORS += [
        (f'{p}.hc_attn_fn', 'f32', 'small'), (f'{p}.hc_attn_scale', 'f32', 'small'), (f'{p}.hc_attn_base', 'f32', 'small'),
        (f'{p}.hc_ffn_fn', 'f32', 'small'), (f'{p}.hc_ffn_scale', 'f32', 'small'), (f'{p}.hc_ffn_base', 'f32', 'small'),
        (f'{p}.attn_norm.weight', 'bf16', 'small'), (f'{p}.ffn_norm.weight', 'bf16', 'small'),
        (f'{p}.attn.q_norm.weight', 'bf16', 'small'), (f'{p}.attn.attn_sink', 'f32', 'small'),
        (f'{p}.ffn.gate.weight', 'f32', 'small'),
        (f'{p}.attn.wq_a.weight', 'bf16', 'medium'), (f'{p}.attn.wo_a.weight', 'bf16', 'medium'),
        (f'{p}.ffn.shared_experts.w1.weight', 'bf16', 'medium'), (f'{p}.ffn.shared_experts.w3.weight', 'bf16', 'medium'),
        (f'{p}.ffn.shared_experts.w2.weight', 'bf16', 'medium'),
        (f'{p}.attn.wq_b.weight', 'bf16', 'all'), (f'{p}.attn.wo_b.weight', 'bf16', 'all'),
        # frozen: the ring rows were captured with these (main_proj, main_norm too)
        (f'{p}.attn.wkv.weight', 'bf16', 'frozen'), (f'{p}.attn.kv_norm.weight', 'bf16', 'frozen'),
        # only picks experts (no gradient reaches it)
        (f'{p}.ffn.gate.bias', 'f32', 'frozen'),
    ]
TENSORS += [('mtp.2.norm.weight', 'bf16', 'small'),
            ('mtp.2.markov_head.embed.weight', 'bf16', 'small'), ('mtp.2.markov_head.head.weight', 'f32', 'small'),
            ('mtp.2.confidence_head.proj.weight', 'f32', 'frozen')]
LEVELS = {'small': {'small'}, 'medium': {'small', 'medium'}, 'all': {'small', 'medium', 'all'}}


class Drafter:
    """A differentiable port of ds41d's Spark forward (engine/third_party/spark/src/forward.cu), batched over anchors.
    Rounding points follow the engine where they shape values (BF16 activations between ops, the FP8 roundtrip of the
    block's KV and attention output); the gradient passes them straight through."""

    def __init__(self, ck, bank, device, train_level, patch=None):
        import torch
        self.torch, self.dev = torch, device
        self.params, self.dtype, self.trainable = {}, {}, []
        trainable = LEVELS[train_level] if train_level else set()
        patched = {}
        if patch:
            from safetensors.torch import load_file
            patched = load_file(str(Path(patch) / 'patch.safetensors'))
        self.fp8 = {}
        for name, dt, group in TENSORS:
            # a weight ds41d keeps in FP8 (stock, untrained) quantizes its input too (ds41_dense_quant); a trained
            # or patched one is BF16 and runs on cuBLAS without that
            self.fp8[name] = ck.raw_dtype(name) == 'F8_E4M3' and name not in patched and group not in trainable
            t = patched[name].float() if name in patched else load_dense(ck, name)
            t = t.to(device)
            if name.endswith('attn_sink') or name.endswith('_scale') or name.endswith('_base') or name.endswith('.bias'):
                t = t.reshape(-1)
            p = torch.nn.Parameter(t.float(), requires_grad=group in trainable)
            self.params[name], self.dtype[name] = p, dt
            if group in trainable:
                self.trainable.append(name)
        bf = torch.bfloat16
        self.embed = load_dense(ck, 'embed.weight').to(bf).to(device)
        self.head = load_dense(ck, 'head.weight').to(device)              # F32: the head's logits in FP32 (TF32)
        self.main = {}
        self.experts = [load_experts(ck, l, bank, device) for l in range(3)]
        say(f'drafter: {sum(p.numel() for p in self.params.values()) / 1e6:,.0f}M dense weights, trainable '
            f'{sum(self.params[n].numel() for n in self.trainable) / 1e6:,.0f}M in {len(self.trainable)} tensors')

    def w(self, name):
        """a weight as the engine binds it: BF16 or F32 values (the master stays F32)"""
        p = self.params[name]
        if self.dtype[name] == 'bf16':
            return p + (p.to(self.torch.bfloat16).float() - p).detach()   # round-to-BF16, straight-through
        return p

    # -- rounding helpers -------------------------------------------------------------------------------------------
    def bf(self, x):
        return x + (x.to(self.torch.bfloat16).float() - x).detach()

    def fp8_roundtrip(self, x):
        """ds41_attn_roundtrip format 0: groups of 32, power-of-two scale ceil(log2(max(|x|,1e-4)/448)), E4M3"""
        torch = self.torch
        shape = x.shape
        g = x.reshape(-1, 32)
        am = g.detach().abs().amax(-1, keepdim=True).clamp(min=1e-4) / 448.0
        s = torch.exp2(torch.ceil(torch.log2(am)))
        q = (g.detach() / s).clamp(-448, 448).to(torch.float8_e4m3fn).float() * s
        q = q.to(torch.bfloat16).float()
        return (g + (q - g).detach()).reshape(shape)

    def rms(self, x, name):
        y = x * self.torch.rsqrt((x * x).mean(-1, keepdim=True) + EPS_NORM) * self.w(name + '.weight')
        return self.bf(y)

    def rope(self, x, pos, inverse=False):
        """the last 64 of each head's dims, adjacent pairs, the table's (cos, sin) for each row's position"""
        torch = self.torch
        t = self.rope_table[pos]                       # [..., 64]: cos, sin interleaved
        c, s = t[..., 0::2], t[..., 1::2]
        while c.dim() < x.dim():
            c, s = c.unsqueeze(-2), s.unsqueeze(-2)
        if inverse:
            s = -s
        a, b = x[..., -RD::2], x[..., -RD + 1::2]
        ra, rb = a * c - b * s, b * c + a * s
        r = torch.stack([ra, rb], -1).flatten(-2)
        return self.bf(torch.cat([x[..., :-RD], r], -1))

    def act_quant(self, x):
        """ds41_dense_quant: E4M3 in groups of 128 with power-of-two (UE8M0) scales"""
        torch = self.torch
        g = x.reshape(-1, 128)
        am = g.detach().abs().amax(-1, keepdim=True).clamp(min=1e-4) / 448.0
        s = torch.exp2(torch.ceil(torch.log2(am)))
        q = (g.detach() / s).clamp(-448, 448).to(torch.float8_e4m3fn).float() * s
        return (g + (q - g).detach()).reshape(x.shape)

    def linear(self, x, name):
        if self.fp8.get(name + '.weight'):
            x = self.act_quant(x)
        return self.bf(x @ self.w(name + '.weight').t())

    # -- mHC ----------------------------------------------------------------------------------------------------------
    def hc_mixes(self, h, p):
        torch = self.torch
        flat = h.flatten(-2)                                           # [..., 4H]
        rstd = torch.rsqrt((flat * flat).mean(-1, keepdim=True) + EPS_NORM)
        z = (flat @ self.w(p + '_fn').t()) * rstd                      # [..., 24]
        s, b = self.w(p + '_scale'), self.w(p + '_base')
        pre = torch.sigmoid(z[..., :4] * s[0] + b[:4]) + EPS_HC
        post = 2 * torch.sigmoid(z[..., 4:8] * s[1] + b[4:8])
        v = (z[..., 8:24] * s[2] + b[8:24]).reshape(*z.shape[:-1], 4, 4)
        v = torch.softmax(v, -1) + EPS_HC
        v = v / (v.sum(-2, keepdim=True) + EPS_HC)
        for _ in range(19):
            v = v / (v.sum(-1, keepdim=True) + EPS_HC)
            v = v / (v.sum(-2, keepdim=True) + EPS_HC)
        return pre, post, v

    def hc_pre(self, h, pre):
        return self.bf((pre.unsqueeze(-1) * h).sum(-2))

    def hc_post(self, y, h, post, comb):
        # out_j = post_j y + sum_a comb[a][j] h_a
        return self.bf(post.unsqueeze(-1) * y.unsqueeze(-2) + self.torch.einsum('...aj,...ak->...jk', comb, h))

    # -- blocks -------------------------------------------------------------------------------------------------------
    def attention(self, l, x, ring, ring_mask, pos):
        """x [A,5,H]; ring [A,R,512] (the ring rows: kv of the anchors' last <=128 committed positions), ring_mask
        [A,R] True where a row exists; pos [A,5] the block's positions"""
        torch = self.torch
        p = f'mtp.{l}.attn'
        A = x.shape[0]
        qa = self.rms(self.linear(x, p + '.wq_a'), p + '.q_norm')
        q = self.linear(qa, p + '.wq_b').reshape(A, BLOCK, HEADS, HD)
        q = self.rope(q, pos)
        kv = self.rms(self.linear(x, p + '.wkv'), p + '.kv_norm')
        kv = self.fp8_roundtrip(self.rope(kv, pos))                   # [A,5,512]
        keys = torch.cat([ring, kv], 1)                               # [A,R+5,512]
        mask = torch.cat([ring_mask, torch.ones(A, BLOCK, dtype=torch.bool, device=x.device)], 1)
        scores = torch.einsum('abhd,akd->abhk', q, keys) * (HD ** -0.5)
        scores = scores.masked_fill(~mask[:, None, None, :], float('-inf'))
        sink = self.w(p + '.attn_sink').view(1, 1, HEADS, 1).expand(A, BLOCK, HEADS, 1)
        prob = torch.softmax(torch.cat([scores, sink], -1), -1)[..., :-1]
        o = torch.einsum('abhk,akd->abhd', self.bf(prob), keys)
        o = self.rope(self.bf(o), pos, inverse=True)
        o = self.fp8_roundtrip(o.reshape(A, BLOCK, GROUPS, HEADS // GROUPS * HD))   # [A,5,8 groups,4096]
        wa = self.w(p + '.wo_a.weight').reshape(GROUPS, OLORA, HEADS // GROUPS * HD)
        o = self.bf(torch.einsum('abgk,gnk->abgn', o, wa)).reshape(A, BLOCK, GROUPS * OLORA)
        return self.linear(o, p + '.wo_b')

    def moe(self, l, x):
        torch = self.torch
        p = f'mtp.{l}.ffn'
        shape = x.shape
        x = x.reshape(-1, H)
        z = x @ self.w(p + '.gate.weight').t()                        # F32 router logits
        u = torch.sqrt(torch.nn.functional.softplus(z))
        ids = torch.topk((u + self.w(p + '.gate.bias')).detach(), 3, -1).indices
        uw = torch.gather(u, 1, ids)
        rw = uw / (uw.sum(-1, keepdim=True) + 1e-20) * ROUTE_SCALE
        w1, w3, w2 = self.experts[l]
        y = torch.zeros_like(x)
        xb = x.to(torch.bfloat16)
        flat = ids.reshape(-1)
        order = torch.argsort(flat)
        counts = torch.bincount(flat, minlength=EXPERTS).tolist()
        start = 0
        for e, c in enumerate(counts):
            if not c:
                continue
            sel = order[start:start + c]
            start += c
            rows = sel // 3
            g = (xb[rows] @ w1[e].t()).float()
            up = (xb[rows] @ w3[e].t()).float()
            act = torch.nn.functional.silu(g.clamp(max=CLAMP)) * up.clamp(-CLAMP, CLAMP)
            act = self.bf(act * rw.reshape(-1)[sel].unsqueeze(-1))
            y = y.index_add(0, rows, (act.to(torch.bfloat16) @ w2[e].t()).float())
        g = self.linear(x, p + '.shared_experts.w1')
        up = self.linear(x, p + '.shared_experts.w3')
        act = self.bf(torch.nn.functional.silu(g.clamp(max=CLAMP)) * up.clamp(-CLAMP, CLAMP))
        y = y + self.linear(act, p + '.shared_experts.w2')
        return self.bf(y).reshape(shape)

    def forward(self, anchor, ring, ring_mask, pos):
        """anchor [A] tokens; ring [3][A,R,512]; returns U [A,5,V] (F32) and hd [A,5,H]"""
        torch = self.torch
        A = anchor.shape[0]
        tok = torch.cat([anchor[:, None], torch.full((A, BLOCK - 1), NOISE, device=anchor.device)], 1)
        h = self.embed[tok].float().unsqueeze(-2).expand(A, BLOCK, 4, H)
        pre = torch.zeros(A, BLOCK, 4, device=anchor.device)
        pre[..., 0] = 1
        for l in range(3):
            p = f'mtp.{l}'
            ap, post, comb = self.hc_mixes(h, p + '.hc_attn')
            x = self.rms(self.hc_pre(h, pre), p + '.attn_norm')
            h = self.hc_post(self.attention(l, x, ring[l], ring_mask, pos), h, post, comb)
            fp, post, comb = self.hc_mixes(h, p + '.hc_ffn')
            x = self.rms(self.hc_pre(h, ap), p + '.ffn_norm')
            h = self.hc_post(self.moe(l, x), h, post, comb)
            pre = fp
        hd = self.hc_pre(h, pre)
        xn = self.rms(hd, 'mtp.2.norm')
        U = xn @ self.head.t()
        return U, hd

    def markov(self, prev):
        """the Markov head's bias for each draft depth: [A,5] previous tokens -> [A,5,V] F32"""
        e = self.w('mtp.2.markov_head.embed.weight')[prev]
        return e @ self.w('mtp.2.markov_head.head.weight').t()


# ---------------------------------------------------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------------------------------------------------

class Docs:
    def __init__(self, data, split, topk):
        self.data, self.topk = Path(data), topk
        self.docs = [d for d in index(data) if d['split'] == split and (self.data / 'docs' / f"{d['name']}.kv").exists()]
        self.cache = {}

    def load(self, d):
        name = d['name']
        if name not in self.cache:
            base = self.data / 'docs' / name
            n = d['n']
            tok = np.fromfile(f'{base}.tok', dtype=np.int32)
            w = np.fromfile(f'{base}.w', dtype=np.uint8)
            kv = np.memmap(f'{base}.kv', dtype=np.uint16, mode='r', shape=(n, 3, HD))
            top = np.memmap(f'{base}.top', dtype=np.uint8, mode='r')
            K = self.topk
            ids = top[:n * K * 4].view(np.int32).reshape(n, K)
            lp = top[n * K * 4:2 * n * K * 4].view(np.float32).reshape(n, K)
            rest = top[2 * n * K * 4:2 * n * K * 4 + n * 4].view(np.float32)
            if len(self.cache) > 64:
                self.cache.clear()
            self.cache[name] = (tok, w, kv, ids, lp, rest)
        return self.cache[name]

    def anchors(self, other_weight, stride=1):
        """(doc index, c, weight): an anchor c drafts positions c+1..c+5; weighted by who wrote position c+1"""
        out = []
        for i, d in enumerate(self.docs):
            tok, w, *_ = self.load(d)
            n = len(tok)
            for c in range(1, n - 1, stride):
                code = w[c + 1]
                wt = 1.0 if code == W_MINE else other_weight if code == W_OTHER else 0.0
                if wt > 0:
                    out.append((i, c, wt))
        return out

    def batch(self, items, device):
        import torch
        A = len(items)
        ring = np.zeros((3, A, RING, HD), dtype=np.uint16)
        mask = np.zeros((A, RING), dtype=bool)
        anchor = np.zeros(A, dtype=np.int64)
        prev = np.zeros((A, BLOCK), dtype=np.int64)
        pos = np.zeros((A, BLOCK), dtype=np.int64)
        K = self.topk
        tid = np.zeros((A, BLOCK, K), dtype=np.int64)
        tlp = np.full((A, BLOCK, K), -1e4, dtype=np.float32)
        trest = np.full((A, BLOCK), -1e4, dtype=np.float32)
        valid = np.zeros((A, BLOCK), dtype=bool)
        weight = np.zeros(A, dtype=np.float32)
        for a, (i, c, wt) in enumerate(items):
            tok, w, kv, ids, lp, rest = self.load(self.docs[i])
            n = len(tok)
            lo = max(0, c - RING)
            r = kv[lo:c]                                              # [c-lo, 3, 512]
            ring[:, a, :c - lo] = np.transpose(r, (1, 0, 2))
            mask[a, :c - lo] = True
            anchor[a] = tok[c]
            pos[a] = np.arange(c, c + BLOCK)
            for k in range(BLOCK):
                if c + k < n:
                    prev[a, k] = tok[c + k]
                    tid[a, k], tlp[a, k], trest[a, k] = ids[c + k], lp[c + k], rest[c + k]
                    valid[a, k] = True
            weight[a] = wt
        t = lambda x: torch.from_numpy(x).to(device)
        ring_t = torch.from_numpy(ring.view(np.int16)).view(torch.bfloat16).to(device).float()
        return dict(ring=ring_t, mask=t(mask), anchor=t(anchor), prev=t(prev), pos=t(pos), tid=t(tid), tlp=t(tlp),
                    trest=t(trest), valid=t(valid), weight=t(weight))


def step_loss(model, b):
    """KL(p || q) on the target's top-k plus the rest (a coarsening of p), per draft depth, weighted by the chance of
    reaching that depth (the product of the earlier depths' acceptances, not differentiated); and those acceptances"""
    torch = model.torch
    U, _ = model.forward(b['anchor'], b['ring'], b['mask'], b['pos'])
    logits = U + model.markov(b['prev'])
    logq = torch.log_softmax(logits, -1)                              # [A,5,V]
    lq = torch.gather(logq, -1, b['tid'])                             # [A,5,K]
    p = torch.exp(b['tlp'])
    q_top = torch.exp(lq)
    q_rest = (1 - q_top.sum(-1)).clamp(min=1e-12)
    p_rest = torch.exp(b['trest'])
    kl = (p * (b['tlp'] - lq)).sum(-1) + p_rest * (b['trest'] - torch.log(q_rest))
    alpha = torch.minimum(p, q_top).sum(-1)                           # [A,5], a lower bound of sum(min(p, q))
    valid = b['valid'].float()
    reach = torch.cumprod(torch.cat([torch.ones_like(alpha[:, :1]), alpha[:, :-1]], 1), 1).detach() * valid
    loss = ((kl * reach).sum(-1) * b['weight']).sum() / b['weight'].sum().clamp(min=1e-9)
    return loss, alpha.detach(), valid


def evaluate(model, docs, a, device, limit):
    import torch
    items = docs.anchors(a.other_weight, stride=a.eval_stride)
    random.Random(7).shuffle(items)
    items = items[:limit]
    sums = np.zeros(BLOCK)
    counts = np.zeros(BLOCK)
    expected = 0.0
    with torch.no_grad():
        for s in range(0, len(items), a.batch):
            b = docs.batch(items[s:s + a.batch], device)
            _, alpha, valid = step_loss(model, b)
            al = (alpha * valid).cpu().numpy()
            sums += al.sum(0)
            counts += valid.cpu().numpy().sum(0)
            expected += np.cumprod(alpha.cpu().numpy(), 1).sum()
    acc = sums / np.maximum(counts, 1)
    return {'anchors': len(items), 'acceptance': [round(float(x), 4) for x in acc],
            'expected_accepted': round(float(expected) / max(1, len(items)), 4)}


def build(a):
    import torch
    torch.backends.cuda.matmul.allow_tf32 = True
    device = torch.device(a.device)
    data = Path(a.data)
    meta = json.loads((data / 'capture.json').read_text())
    ck = Checkpoint(a.weights)
    rope = np.fromfile(data / 'docs' / 'rope.f32', dtype=np.float32).reshape(-1, RD)
    return torch, device, data, meta, ck, rope


def run_eval(a):
    torch, device, data, meta, ck, rope = build(a)
    model = Drafter(ck, a.mtp_bank, device, None, a.patch)
    model.rope_table = torch.from_numpy(rope).to(device)
    held = Docs(data, 'heldout', meta['topk'])
    r = evaluate(model, held, a, device, a.eval_anchors)
    say(f"held out: {r['anchors']:,} anchors, acceptance by depth {r['acceptance']}, "
        f"expected accepted drafts a full step {r['expected_accepted']}")


def train(a):
    torch, device, data, meta, ck, rope = build(a)
    model = Drafter(ck, a.mtp_bank, device, a.train)
    model.rope_table = torch.from_numpy(rope).to(device)
    train_docs, held = Docs(data, 'train', meta['topk']), Docs(data, 'heldout', meta['topk'])
    say(f'documents: train {len(train_docs.docs):,}, held out {len(held.docs):,}')
    t0 = time.time()
    stock = evaluate(model, held, a, device, a.eval_anchors)
    say(f"stock drafter, held out: acceptance by depth {stock['acceptance']}, expected accepted "
        f"{stock['expected_accepted']} ({time.time() - t0:.0f} s)")
    if stock['acceptance'][0] < a.min_stock:
        sys.exit(f"the port's first-draft acceptance {stock['acceptance'][0]} is below {a.min_stock}: it does not "
                 "agree with ds41d's drafter (0.75 measured on the bench prompts; a broken port scores near 0); nothing trained")
    params = [model.params[n] for n in model.trainable]
    opt = torch.optim.AdamW(params, lr=a.lr, betas=(0.9, 0.95), weight_decay=0.0)
    items = train_docs.anchors(a.other_weight, stride=a.stride)
    steps = a.epochs * (len(items) // a.batch)
    say(f'training: {len(items):,} anchors a pass, {a.epochs} passes, {steps:,} steps of {a.batch}')
    warm = max(1, min(200, steps // 20))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / warm) * (
        0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(1.0, s / max(1, steps))))))
    best, best_state, step, t0 = stock['expected_accepted'], None, 0, time.time()
    rng = random.Random(1)
    for epoch in range(a.epochs):
        rng.shuffle(items)
        for s in range(0, len(items) - a.batch + 1, a.batch):
            b = train_docs.batch(items[s:s + a.batch], device)
            loss, alpha, valid = step_loss(model, b)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            sched.step()
            step += 1
            if step % 50 == 0:
                el = time.time() - t0
                say(f'  step {step:,}/{steps:,}: loss {loss.item():.4f}, first-draft acceptance '
                    f'{float(alpha[:, 0].mean()):.3f}, {step * a.batch / el:,.0f} anchors/s, '
                    f'{(steps - step) * el / step / 60:.0f} min left')
            if step % a.eval_every == 0 or step == steps:
                r = evaluate(model, held, a, device, a.eval_anchors)
                say(f"  held out at step {step:,}: acceptance {r['acceptance']}, expected accepted "
                    f"{r['expected_accepted']} (stock {stock['expected_accepted']})")
                if r['expected_accepted'] > best:
                    best = r['expected_accepted']
                    best_state = {n: model.w(n).detach().clone() for n in model.trainable}
                    write_patch(a, model, best_state, stock, r, len(items))
    if best_state is None:
        say('no held-out improvement over the stock drafter: no patch written')
        return
    say(f'patch: {a.out}/patch.safetensors (expected accepted drafts {stock["expected_accepted"]} -> {best})')


def write_patch(a, model, state, stock, result, anchors):
    import torch
    from safetensors.torch import save_file
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    tensors = {}
    for n, t in state.items():
        t = t.to(torch.bfloat16 if model.dtype[n] == 'bf16' else torch.float32).contiguous().cpu()
        tensors[n] = t
    part = out / 'patch.safetensors.partial'
    save_file(tensors, str(part))
    os.replace(part, out / 'patch.safetensors')
    manifest = {'format': 1, 'created': time.strftime('%Y-%m-%dT%H:%M:%S'), 'tensors': len(tensors),
                'bytes': sum(t.numel() * t.element_size() for t in tensors.values()), 'train': a.train,
                'weights': str(a.weights), 'mtp_bank': str(a.mtp_bank or ''), 'anchors_a_pass': anchors,
                'lr': a.lr, 'epochs': a.epochs, 'stock': stock, 'trained': result}
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=1) + '\n')


def identity(a):
    """the stock drafter written as a patch, exactly as train writes one (an FP8 weight as its exact BF16 dequant): the
    check that loading a patch leaves the drafter as it was (ds41d with it should score as without it)"""
    import torch
    from safetensors.torch import save_file
    ck = Checkpoint(a.weights)
    tensors = {}
    for name, dt, group in TENSORS:
        if group not in LEVELS[a.train]:
            continue
        t = load_dense(ck, name)
        if name.endswith('attn_sink') or name.endswith('_scale') or name.endswith('_base') or name.endswith('.bias'):
            t = t.reshape(-1)
        tensors[name] = t.to(torch.bfloat16 if dt == 'bf16' else torch.float32).contiguous()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    part = out / 'patch.safetensors.partial'
    save_file(tensors, str(part))
    os.replace(part, out / 'patch.safetensors')
    manifest = {'format': 1, 'identity': True, 'created': time.strftime('%Y-%m-%dT%H:%M:%S'), 'train': a.train,
                'tensors': len(tensors), 'bytes': sum(t.numel() * t.element_size() for t in tensors.values()),
                'weights': str(a.weights)}
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=1) + '\n')
    say(f'identity patch ({a.train}): {len(tensors)} stock tensors, {manifest["bytes"] / 2**20:,.0f} MiB -> {out}')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    e = sub.add_parser('export')
    e.add_argument('--s0ma', required=True)
    e.add_argument('--out', required=True)
    e.add_argument('--tokenizer', required=True)
    e.add_argument('--api', default=str(Path(__file__).resolve().parent.parent / 'api'))
    e.add_argument('--adapter', default='')
    e.add_argument('--window', type=int, default=4096)
    e.add_argument('--max-tokens', type=int, default=4_000_000)
    e.add_argument('--min-fresh', type=int, default=16)
    e.add_argument('--heldout', type=float, default=0.05)
    c = sub.add_parser('capture')
    c.add_argument('--data', required=True)
    c.add_argument('--socket', default='')
    c.add_argument('--topk', type=int, default=32)
    i = sub.add_parser('identity')
    i.add_argument('--weights', required=True)
    i.add_argument('--out', required=True)
    i.add_argument('--train', choices=sorted(LEVELS), default='medium')
    for name in ('train', 'eval'):
        t = sub.add_parser(name)
        t.add_argument('--data', required=True)
        t.add_argument('--weights', required=True)
        t.add_argument('--mtp-bank', default='')
        t.add_argument('--device', default='cuda:0')
        t.add_argument('--batch', type=int, default=32)
        t.add_argument('--other-weight', type=float, default=0.2)
        t.add_argument('--eval-anchors', type=int, default=4096)
        t.add_argument('--eval-stride', type=int, default=7)
        if name == 'train':
            t.add_argument('--out', required=True)
            t.add_argument('--train', choices=sorted(LEVELS), default='medium')
            t.add_argument('--epochs', type=int, default=2)
            t.add_argument('--lr', type=float, default=2e-5)
            t.add_argument('--stride', type=int, default=1)
            t.add_argument('--eval-every', type=int, default=2000)
            t.add_argument('--min-stock', type=float, default=0.3)
        else:
            t.add_argument('--patch', default='')
    a = ap.parse_args()
    {'export': export, 'capture': capture, 'train': train, 'eval': run_eval, 'identity': identity}[a.cmd](a)


if __name__ == '__main__':
    main()
