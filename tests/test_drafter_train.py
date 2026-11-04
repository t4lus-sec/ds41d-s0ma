"""scripts/drafter_train.py (s0ma change 99): the archive export's guards and rendering, and the drafter port's
training loop end to end on a tiny made-up drafter (CPU)."""
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import struct
import sys
import tempfile
import types
import unittest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'scripts'))
ADAPTER = Path(os.environ.get('DS41_ADAPTER_SRC', ROOT.parent / 's0ma-ds41'))

try:
    import numpy as np
except ImportError:          # the repository's own tests run without numpy
    np = None
try:
    import torch
except ImportError:
    torch = None


def needs(*mods):
    missing = []
    for m in mods:
        try:
            __import__(m)
        except ImportError:
            missing.append(m)
    return unittest.skipIf(missing, f'needs {", ".join(missing)}')


@needs('numpy')
class JobsGuard(unittest.TestCase):
    def check(self, data, raw=None):
        import drafter_train as dt
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / 'data').mkdir()
            if data is not None or raw is not None:
                (Path(d) / 'data' / 'jobs.json').write_text(raw if raw is not None else json.dumps(data))
            return dt.jobs_block(Path(d))

    def test_states(self):
        self.assertIsNone(self.check(None))
        self.assertIsNone(self.check({}))
        self.assertIsNone(self.check({'jobs': [{'name': 'a', 'state': 'closed', 'spans': [[1, 5]]}]}))
        self.assertIsNone(self.check({'jobs': [{'name': 'a', 'spans': [{'start': 1, 'end': 5}]}]}))
        self.assertIn('open', self.check({'jobs': [{'name': 'a', 'state': 'open'}]}))
        self.assertIn('paused', self.check({'jobs': [{'name': 'a', 'state': 'paused'}]}))
        self.assertIsNotNone(self.check({'jobs': [{'name': 'a', 'spans': [{'start': 1, 'end': None}]}]}))
        self.assertIsNotNone(self.check({'jobs': [{'name': 'a', 'spans': [[1, None]]}]}))
        self.assertIn('parse', self.check(None, raw='{"jobs": ['))
        self.assertIn('shape', self.check({'whatever': [1, 2]}))


def tiny_tokenizer(path):
    from tokenizers import Tokenizer, models, pre_tokenizers, trainers
    tk = Tokenizer(models.BPE(unk_token='[UNK]'))
    tk.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    specials = ['[UNK]', '<｜begin▁of▁sentence｜>', '<｜end▁of▁sentence｜>', '<｜User｜>', '<｜Assistant｜>', '<think>',
                '</think>', '<｜System｜>', '<｜latest_reminder｜>']
    corpus = ['the quick brown fox jumps over the lazy dog read_file path src main py ok edit DSML invoke parameter']
    tk.train_from_iterator(corpus * 20, trainers.BpeTrainer(vocab_size=600, special_tokens=specials,
                                                            initial_alphabet=pre_tokenizers.ByteLevel.alphabet()))
    tk.save(str(path))


def call_store(root, calls):
    (root / 'objects').mkdir(parents=True, exist_ok=True)

    def put(obj):
        raw = json.dumps(obj).encode()
        sha = hashlib.sha256(raw).hexdigest()
        (root / 'objects' / sha[:2]).mkdir(exist_ok=True)
        (root / 'objects' / sha[:2] / (sha[2:] + '.gz')).write_bytes(gzip.compress(raw))
        return 'sha256:' + sha
    with open(root / 'index.jsonl', 'w') as f:
        for i, (msgs, resp) in enumerate(calls):
            f.write(json.dumps({'call_id': f'c{i}', 'at': f'2026-11-0{1 + i // 9}T00:00:0{i % 9}', 'purpose': 'agent',
                                'request': {'messages': [put(m) for m in msgs], 'params': {}},
                                'response': put(resp)}) + '\n')
            f.write(json.dumps({'call_id': f'c{i}', 'update': True, 'archive_events': [i]}) + '\n')
        f.write('{"call_id": "torn')


@needs('numpy', 'tokenizers', 'pydantic')
@unittest.skipUnless((ADAPTER / 's0ma_ds41').is_dir(), 'needs the s0ma adapter source (DS41_ADAPTER_SRC)')
class Export(unittest.TestCase):
    def run_export(self, d, jobs=None, extra=()):
        import drafter_train as dt
        tok = d / 'tokenizer.json'
        tiny_tokenizer(tok)
        system = {'role': 'system', 'content': 'you are a fox ' * 20}
        convo, calls = [system, {'role': 'user', 'content': 'read the main py'}], []
        for step in range(6):
            resp = {'content': 'ok ' * (step + 3), 'reasoning_content': 'the lazy dog ' * 4}
            calls.append((list(convo), resp))
            convo += [{'role': 'assistant', **resp}, {'role': 'user', 'content': 'jumps over ' * (step + 1)}]
        calls += list(extra)
        call_store(d / 's0ma' / 'data' / 'calls', calls)
        if jobs is not None:
            (d / 's0ma' / 'data' / 'jobs.json').write_text(json.dumps(jobs))
        a = types.SimpleNamespace(s0ma=str(d / 's0ma'), out=str(d / 'out'), tokenizer=str(tok),
                                  api=str(ROOT / 'api'), adapter=str(ADAPTER), window=4096, max_tokens=10 ** 9,
                                  min_fresh=1, heldout=0.2)
        dt.export(a)
        return d / 'out'

    def test_export(self):
        import drafter_train as dt
        with tempfile.TemporaryDirectory() as d:
            secret = ([{'role': 'user', 'content': '[redacted for job x]'}], {'content': 'ok'})
            out = self.run_export(Path(d), extra=[secret])
            docs = [json.loads(l) for l in (out / 'index.jsonl').read_text().splitlines()]
            self.assertEqual(len(docs), 6)                     # the redacted call is skipped
            self.assertEqual({x['split'] for x in docs}, {'train', 'heldout'})
            self.assertEqual(docs[-1]['split'], 'heldout')      # the newest is held out
            for x in docs:
                ids = np.fromfile(out / 'docs' / f"{x['name']}.tok", dtype=np.int32)
                w = np.fromfile(out / 'docs' / f"{x['name']}.w", dtype=np.uint8)
                self.assertEqual(len(ids), x['n'])
                self.assertEqual(len(w), x['n'])
                self.assertEqual(int(w[-1]), dt.W_MINE)              # the completion's end of sentence
                self.assertGreater(x['mine'], 0)
            # later calls repeat the earlier conversation: only what is new is trained on
            first, last = docs[0], docs[-1]
            self.assertEqual(first['fresh'], first['n'])
            self.assertLess(last['fresh'], last['n'] // 2)
            w = np.fromfile(out / 'docs' / f"{last['name']}.w", dtype=np.uint8)
            self.assertEqual(int(w[0]), 0)
            # a second export into the same folder is refused (captures are tied to the documents)
            with self.assertRaises(SystemExit):
                self.run_export(Path(d))

    def test_refused_while_a_job_is_open(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(SystemExit) as e:
                self.run_export(Path(d), jobs={'jobs': [{'name': 'x', 'state': 'paused', 'spans': [[3, 9]]}]})
            self.assertIn('refused', str(e.exception))
            self.assertFalse((Path(d) / 'out' / 'index.jsonl').exists())


# ---------------------------------------------------------------------------------------------------------- tiny drafter
TINY = dict(H=128, I=128, V=300, RING=8, HEADS=4, HD=64, RD=64, GROUPS=2, OLORA=128, EXPERTS=8, NOISE=299)
QLORA, MARKOV = 128, 32


def safetensors(path, tensors):
    """a raw safetensors file: {name: (dtype string, shape, bytes)}"""
    header, blobs, off = {}, [], 0
    for name, (dtype, shape, data) in tensors.items():
        header[name] = {'dtype': dtype, 'shape': list(shape), 'data_offsets': [off, off + len(data)]}
        blobs.append(data)
        off += len(data)
    h = json.dumps(header).encode()
    h += b' ' * (-len(h) % 8)
    with open(path, 'wb') as f:
        f.write(struct.pack('<Q', len(h)) + h + b''.join(blobs))


def tiny_checkpoint(root, rng):
    t = types.SimpleNamespace(**TINY)
    out = {}

    def f32(name, shape, scale=0.05, base=0.0):
        out[name] = ('F32', shape, (base + rng.standard_normal(shape) * scale).astype(np.float32).tobytes())

    def bf16(name, shape, scale=0.05, base=0.0):
        x = torch.from_numpy((base + rng.standard_normal(shape) * scale).astype(np.float32)).to(torch.bfloat16)
        out[name] = ('BF16', shape, x.view(torch.int16).numpy().tobytes())

    def fp8(name, shape):
        x = torch.from_numpy((rng.standard_normal(shape) * 2).astype(np.float32)).to(torch.float8_e4m3fn)
        out[name + '.weight'] = ('F8_E4M3', shape, x.view(torch.uint8).numpy().tobytes())
        s = (rng.integers(-9, -5, size=(shape[0] // 128 or 1, shape[1] // 128)) + 127).astype(np.uint8)
        out[name + '.scale'] = ('F8_E8M0', s.shape, s.tobytes())

    def mxfp4(name, n, k):
        out[name + '.weight'] = ('U8', (n, k // 2), rng.integers(0, 256, size=(n, k // 2)).astype(np.uint8).tobytes())
        out[name + '.scale'] = ('F8_E8M0', (n, k // 32), (rng.integers(-8, -5, size=(n, k // 32)) + 127)
                                .astype(np.uint8).tobytes())
    for l in range(3):
        p = f'mtp.{l}'
        for kind in ('attn', 'ffn'):
            f32(f'{p}.hc_{kind}_fn', (24, 4 * t.H), 0.02)
            f32(f'{p}.hc_{kind}_scale', (3,), 0.1, 1.0)
            f32(f'{p}.hc_{kind}_base', (24,), 0.1)
        bf16(f'{p}.attn_norm.weight', (t.H,), 0.05, 1.0)
        bf16(f'{p}.ffn_norm.weight', (t.H,), 0.05, 1.0)
        bf16(f'{p}.attn.q_norm.weight', (QLORA,), 0.05, 1.0)
        bf16(f'{p}.attn.kv_norm.weight', (t.HD,), 0.05, 1.0)
        f32(f'{p}.attn.attn_sink', (t.HEADS,), 0.5)
        f32(f'{p}.ffn.gate.weight', (t.EXPERTS, t.H), 0.1)
        f32(f'{p}.ffn.gate.bias', (t.EXPERTS,), 0.01)
        fp8(f'{p}.attn.wq_a', (QLORA, t.H))
        fp8(f'{p}.attn.wq_b', (t.HEADS * t.HD, QLORA))
        fp8(f'{p}.attn.wkv', (t.HD, t.H))
        bf16(f'{p}.attn.wo_a.weight', (t.GROUPS * t.OLORA, t.HEADS // t.GROUPS * t.HD), 0.05)
        fp8(f'{p}.attn.wo_b', (t.H, t.GROUPS * t.OLORA))
        fp8(f'{p}.ffn.shared_experts.w1', (t.I, t.H))
        fp8(f'{p}.ffn.shared_experts.w3', (t.I, t.H))
        fp8(f'{p}.ffn.shared_experts.w2', (t.H, t.I))
        for e in range(t.EXPERTS):
            mxfp4(f'{p}.ffn.experts.{e}.w1', t.I, t.H)
            mxfp4(f'{p}.ffn.experts.{e}.w3', t.I, t.H)
            mxfp4(f'{p}.ffn.experts.{e}.w2', t.H, t.I)
    bf16('mtp.2.norm.weight', (t.H,), 0.05, 1.0)
    bf16('mtp.2.markov_head.embed.weight', (t.V, MARKOV), 0.05)
    f32('mtp.2.markov_head.head.weight', (t.V, MARKOV), 0.05)
    f32('mtp.2.confidence_head.proj.weight', (1, 2 * t.H), 0.05)
    bf16('embed.weight', (t.V, t.H), 1.0)
    bf16('head.weight', (t.V, t.H), 0.05)
    safetensors(root / 'model.safetensors', out)
    (root / 'model.safetensors.index.json').write_text(json.dumps({'weight_map': {n: 'model.safetensors' for n in out}}))


def tiny_data(data, rng, docs=6, n=96, K=4):
    """documents whose next token is (token + 1) mod V nine times in ten: a pattern the Markov head can learn"""
    t = types.SimpleNamespace(**TINY)
    (data / 'docs').mkdir(parents=True)
    index = []
    for i in range(docs):
        name = f'd{i:06d}'
        tok = rng.integers(0, t.V - 1, size=n).astype(np.int32)
        tok.tofile(data / 'docs' / f'{name}.tok')
        np.full(n, 2, dtype=np.uint8).tofile(data / 'docs' / f'{name}.w')
        kv = torch.from_numpy(rng.standard_normal((n, 3, t.HD)).astype(np.float32)).to(torch.bfloat16)
        kv.view(torch.int16).numpy().tofile(data / 'docs' / f'{name}.kv')
        ids = np.stack([(tok + 1 + j) % t.V for j in range(K)], 1).astype(np.int32)
        lp = np.log(np.array([0.9, 0.04, 0.02, 0.01], dtype=np.float32))[None].repeat(n, 0)
        rest = np.full(n, math.log(0.03), dtype=np.float32)
        with open(data / 'docs' / f'{name}.top', 'wb') as f:
            f.write(ids.tobytes() + lp.tobytes() + rest.tobytes())
        index.append({'name': name, 'n': n, 'fresh': n, 'mine': n, 'split': 'heldout' if i == docs - 1 else 'train'})
    (data / 'index.jsonl').write_text(''.join(json.dumps(x) + '\n' for x in index))
    (data / 'capture.json').write_text(json.dumps({'topk': K}))
    pos = np.arange(4096, dtype=np.float32)[:, None] / (10000 ** (np.arange(t.RD // 2) / (t.RD // 2)))[None]
    np.stack([np.cos(pos), np.sin(pos)], -1).reshape(4096, t.RD).astype(np.float32).tofile(data / 'docs' / 'rope.f32')


@needs('numpy', 'torch', 'safetensors')
class TinyDrafter(unittest.TestCase):
    def setUp(self):
        import drafter_train as dt
        self.dt = dt
        self.saved = {k: getattr(dt, k) for k in TINY}
        for k, v in TINY.items():
            setattr(dt, k, v)
        torch.manual_seed(0)

    def tearDown(self):
        for k, v in self.saved.items():
            setattr(self.dt, k, v)

    def test_train_writes_a_patch_that_scores_better(self):
        dt = self.dt
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            rng = np.random.default_rng(3)
            (d / 'ck').mkdir()
            tiny_checkpoint(d / 'ck', rng)
            tiny_data(d / 'data', rng)
            base = dict(data=str(d / 'data'), weights=str(d / 'ck'), mtp_bank='', device='cpu', batch=16,
                        other_weight=0.2, eval_anchors=256, eval_stride=1)
            a = types.SimpleNamespace(**base, out=str(d / 'patch'), train='small', epochs=3, lr=3e-2, stride=1,
                                      eval_every=20, min_stock=0.0)
            dt.train(a)
            manifest = json.loads((d / 'patch' / 'manifest.json').read_text())
            self.assertGreater(manifest['trained']['expected_accepted'], manifest['stock']['expected_accepted'] + 0.05)
            from safetensors import safe_open
            with safe_open(str(d / 'patch' / 'patch.safetensors'), 'pt') as f:
                names = set(f.keys())
                self.assertEqual(f.get_tensor('mtp.2.markov_head.head.weight').dtype, torch.float32)
                self.assertEqual(f.get_tensor('mtp.0.attn_norm.weight').dtype, torch.bfloat16)
            want = {n for n, _, g in dt.TENSORS if g == 'small'}
            self.assertEqual(names, want)
            self.assertNotIn('mtp.0.attn.wkv.weight', names)          # the ring's path stays frozen
            # eval with the patch scores what training kept
            e = types.SimpleNamespace(**base, patch=str(d / 'patch'))
            model = dt.Drafter(dt.Checkpoint(e.weights), '', torch.device('cpu'), None, e.patch)
            model.rope_table = torch.from_numpy(np.fromfile(d / 'data' / 'docs' / 'rope.f32', dtype=np.float32)
                                                .reshape(-1, dt.RD))
            r = dt.evaluate(model, dt.Docs(d / 'data', 'heldout', 4), e, torch.device('cpu'), e.eval_anchors)
            self.assertAlmostEqual(r['expected_accepted'], manifest['trained']['expected_accepted'], delta=1e-3)

    def test_identity_patch_scores_as_stock(self):
        dt = self.dt
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            rng = np.random.default_rng(5)
            (d / 'ck').mkdir()
            tiny_checkpoint(d / 'ck', rng)
            tiny_data(d / 'data', rng, docs=3)
            e = types.SimpleNamespace(data=str(d / 'data'), weights=str(d / 'ck'), batch=16, other_weight=0.2,
                                      eval_anchors=128, eval_stride=1)
            rope = torch.from_numpy(np.fromfile(d / 'data' / 'docs' / 'rope.f32', dtype=np.float32).reshape(-1, dt.RD))
            scores = {}
            for level in (None, 'small', 'medium'):
                patch = ''
                if level:
                    patch = str(d / f'id-{level}')
                    dt.identity(types.SimpleNamespace(weights=str(d / 'ck'), out=patch, train=level))
                    m = json.loads((Path(patch) / 'manifest.json').read_text())
                    self.assertTrue(m['identity'])
                model = dt.Drafter(dt.Checkpoint(e.weights), '', torch.device('cpu'), None, patch)
                model.rope_table = rope
                scores[level] = dt.evaluate(model, dt.Docs(d / 'data', 'heldout', 4), e, torch.device('cpu'), 128)
            # the small set keeps every dtype: the drafter is unchanged bit for bit
            self.assertEqual(scores['small'], scores[None])
            # medium writes FP8 weights as their exact BF16 values; only the activations' FP8 rounding goes
            self.assertAlmostEqual(scores['medium']['expected_accepted'], scores[None]['expected_accepted'], delta=0.02)

    def test_wsp_scan_matches_eval_at_the_stock_setting(self):
        dt = self.dt
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            rng = np.random.default_rng(6)
            (d / 'ck').mkdir()
            tiny_checkpoint(d / 'ck', rng)
            tiny_data(d / 'data', rng, docs=3)
            a = types.SimpleNamespace(data=str(d / 'data'), weights=str(d / 'ck'), mtp_bank='', device='cpu',
                                      batch=16, other_weight=0.2, eval_anchors=128, eval_stride=1, min_gain=0.0,
                                      out=str(d / 'wsp'))
            res = dt.wsp(a)
            model = dt.Drafter(dt.Checkpoint(a.weights), '', torch.device('cpu'), None)
            model.rope_table = torch.from_numpy(np.fromfile(d / 'data' / 'docs' / 'rope.f32', dtype=np.float32)
                                                .reshape(-1, dt.RD))
            stock = dt.evaluate(model, dt.Docs(d / 'data', 'heldout', 4), a, torch.device('cpu'), 128)
            at1 = next(r for r in res if r['u'] == 1.0 and r['m'] == 1.0)
            self.assertAlmostEqual(at1['expected_accepted'], stock['expected_accepted'], delta=1e-3)
            self.assertEqual(len(res), 13 * 9)
            best = max(res, key=lambda r: r['expected_accepted'])
            if best['expected_accepted'] > at1['expected_accepted']:
                # the written patch, loaded back, scores what the scan said (within the BF16 rounding of the norm)
                model = dt.Drafter(dt.Checkpoint(a.weights), '', torch.device('cpu'), None, a.out)
                model.rope_table = torch.from_numpy(np.fromfile(d / 'data' / 'docs' / 'rope.f32', dtype=np.float32)
                                                    .reshape(-1, dt.RD))
                got = dt.evaluate(model, dt.Docs(d / 'data', 'heldout', 4), a, torch.device('cpu'), 128)
                self.assertAlmostEqual(got['expected_accepted'], best['expected_accepted'], delta=0.02)

    def test_gradients_reach_every_trainable_tensor(self):
        dt = self.dt
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            rng = np.random.default_rng(4)
            (d / 'ck').mkdir()
            tiny_checkpoint(d / 'ck', rng)
            tiny_data(d / 'data', rng, docs=2)
            model = dt.Drafter(dt.Checkpoint(d / 'ck'), '', torch.device('cpu'), 'all')
            model.rope_table = torch.from_numpy(np.fromfile(d / 'data' / 'docs' / 'rope.f32', dtype=np.float32)
                                                .reshape(-1, dt.RD))
            docs = dt.Docs(d / 'data', 'train', 4)
            b = docs.batch(docs.anchors(0.2)[:24], torch.device('cpu'))
            loss, alpha, valid = dt.step_loss(model, b)
            self.assertTrue(torch.isfinite(loss))
            loss.backward()
            dead = [n for n in model.trainable if model.params[n].grad is None or
                    not torch.any(model.params[n].grad != 0)]
            # a routed expert's gate rows only move for chosen experts, but every tensor gets some gradient
            self.assertEqual(dead, [])
            self.assertTrue(bool(((alpha >= 0) & (alpha <= 1 + 1e-5)).all()))


if __name__ == '__main__':
    unittest.main()
