#!/usr/bin/env python3
"""s0ma change 91: several conversations at once against a running ds41d (its engine socket).

1. Solo: each decode prompt alone, greedy, N tokens (single-stream steps, drafting on).
2. Together: 2..S of those prompts at once at one priority (multi-conversation steps): combined and per-stream
   tokens a second, and where each stream's greedy tokens first differ from its solo run (rounding order can
   move a greedy token eventually; a difference at the first tokens means a bug).
3. Prefill: three ~2,000-token prompts alone, then all three at once (one combined prefill): the time until every
   one has its first token, against the solo first-token times summed.
4. Priority: a main request (priority 0) arriving while workers (priority 10) decode: the main request's tokens a
   second against its solo rate, and that the workers paused meanwhile.

Prompts come from the published benchmark cases (token ids only). Prints numbers; writes --out as JSON.
"""
import argparse
import json
import socket
import statistics
import sys
import threading
import time
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import run  # noqa: E402


def case_ids(name, limit=None):
    ids = json.loads((run.ROOT / f'benchmarks/cases/{name}.json').read_text())['token_ids']
    return ids[:limit] if limit else ids


def submit(path, op, out=None, key=None):
    rec = {'tokens': [], 'stamps': [], 'submitted': time.perf_counter(), 'finish': None}
    with socket.socket(socket.AF_UNIX) as c:
        c.settimeout(3600)
        c.connect(str(path))
        c.sendall((json.dumps(op) + '\n').encode())
        with c.makefile('r') as stream:
            for line in stream:
                e = json.loads(line)
                if e.get('type') == 'token':
                    rec['tokens'].append(e['token_id'])
                    rec['stamps'].append(time.perf_counter())
                elif e.get('type') == 'finish':
                    rec['finish'] = e
                    break
    f = rec['finish'] or {}
    if f.get('error') or f.get('finish_reason') not in ('stop', 'length'):
        raise RuntimeError(f"request {op['request_id']} failed: {f.get('error') or f.get('finish_reason')}")
    if out is not None:
        out[key] = rec
    return rec


def op(rid, conv, ids, n, priority=5, temperature=0.0):
    return {'op': 'submit', 'request_id': rid, 'conversation_id': conv, 'token_ids': ids, 'priority': priority,
            'draft': 'on', 'sampling': {'temperature': temperature, 'top_p': 1.0, 'max_tokens': n, 'seed': 21}}


def rate(rec):
    s = rec['stamps']
    return (len(s) - 1) / (s[-1] - s[0]) if len(s) > 1 and s[-1] > s[0] else 0.0


def first_diff(a, b):
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return None if len(a) == len(b) else min(len(a), len(b))


def together(path, ops):
    out, threads = {}, []
    for i, o in enumerate(ops):
        threads.append(threading.Thread(target=submit, args=(path, o, out, i)))
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    if len(out) != len(ops):
        raise RuntimeError('a concurrent request failed (see above)')
    return [out[i] for i in range(len(ops))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--streams', type=int, default=4)
    ap.add_argument('--answer', type=int, default=256)
    ap.add_argument('--out', required=True)
    ap.add_argument('--skip', default='', help='phases to skip: prefill,priority')
    a = ap.parse_args()
    cfg = run.configuration()
    path = run.runtime_dir(cfg) / 'engine.sock'
    if not path.exists():
        sys.exit(f'no engine socket at {path}: start the lab ds41d first')
    tag = time.strftime('%H%M%S')
    decode = [case_ids('chat-000'), case_ids('chat-001'), case_ids('chat-002'), case_ids('mt-000', 512),
              case_ids('agentic-000', 384), case_ids('agentic-001', 448)][:max(2, a.streams)]
    report = {'streams': a.streams, 'answer': a.answer, 'solo': [], 'together': [], 'prefill': {}, 'priority': {}}
    # 1. solo
    solo = []
    for i, ids in enumerate(decode):
        r = submit(path, op(f'solo-{tag}-{i}', f'solo-{tag}-{i}', ids, a.answer))
        solo.append(r)
        st = r['finish']['stats']
        report['solo'].append({'prompt': len(ids), 'generated': len(r['tokens']), 'tok_s': rate(r),
                               'ttft_s': r['stamps'][0] - r['submitted'] if r['stamps'] else None,
                               'acceptance': st.get('acceptance'), 'multi_steps': st.get('multi_steps', 0)})
        print(f"solo {i}: prompt {len(ids)}, {len(r['tokens'])} tokens, {rate(r):.1f} tok/s, "
              f"acceptance {st.get('acceptance', 0):.2f}", flush=True)
    solo_mean = statistics.mean(rate(r) for r in solo)
    # 2. together
    for s in range(2, len(decode) + 1):
        recs = together(path, [op(f't{s}-{tag}-{i}', f't{s}-{tag}-{i}', decode[i], a.answer) for i in range(s)])
        t0 = max(r['stamps'][0] for r in recs)
        t1 = min(r['stamps'][-1] for r in recs)
        inside = sum(sum(1 for x in r['stamps'] if t0 <= x <= t1) for r in recs)
        steady = inside / (t1 - t0) if t1 > t0 else 0.0
        begin = min(r['submitted'] for r in recs)
        end = max(r['stamps'][-1] for r in recs)
        overall = sum(len(r['tokens']) for r in recs) / (end - begin)
        diffs = [first_diff(r['tokens'], solo[i]['tokens']) for i, r in enumerate(recs)]
        multi = [r['finish']['stats'].get('multi_steps', 0) for r in recs]
        comb = [r['finish']['stats'].get('combined_prefill', 0) for r in recs]
        row = {'streams': s, 'steady_combined_tok_s': steady, 'overall_tok_s': overall,
               'per_stream_tok_s': [rate(r) for r in recs], 'multi_steps': multi, 'combined_prefill': comb,
               'first_diff_from_solo': diffs, 'solo_mean_tok_s': solo_mean}
        report['together'].append(row)
        print(f"{s} together: {steady:.1f} tok/s combined while all ran ({steady / solo_mean:.2f}x solo's "
              f"{solo_mean:.1f}), {overall:.1f} tok/s submit-to-last; each {', '.join(f'{rate(r):.1f}' for r in recs)}; "
              f"multi steps {multi}; prefill shared by {comb}; greedy tokens differ from solo at "
              f"{['same' if d is None else d for d in diffs]}", flush=True)
    # 3. prefill
    if 'prefill' not in a.skip:
        longs = [case_ids('agentic-000', 2000), case_ids('agentic-001', 2000), case_ids('mt-000', 2000)]
        # a token of its own in front of each prompt, different alone and together: no prompt is ever served from a
        # kept conversation (the engine reuses any kept prompt prefix), so both measure full prefills
        alone = []
        for i, ids in enumerate(longs):
            r = submit(path, op(f'ps-{tag}-{i}', f'ps-{tag}-{i}', [1000 + i] + ids, 2))
            alone.append(r)
        solo_ttft = [r['stamps'][0] - r['submitted'] for r in alone]
        # a short main request runs first, so the three queue and start together
        out = {}
        blocker = threading.Thread(target=submit, args=(path, op(f'pb-{tag}', f'pb-{tag}', decode[0], 48, 0), out, 'b'))
        blocker.start()
        time.sleep(0.5)
        recs = together(path, [op(f'pt-{tag}-{i}', f'pt-{tag}-{i}', [2000 + i] + ids, 2, 5) for i, ids in enumerate(longs)])
        blocker.join()
        start = out['b']['stamps'][-1]
        last_first = max(r['stamps'][0] for r in recs)
        comb = [r['finish']['stats'].get('combined_prefill', 0) for r in recs]
        report['prefill'] = {'rows': [len(x) + 1 for x in longs], 'solo_ttft_s': solo_ttft,
                             'together_all_first_s': last_first - start, 'combined_prefill': comb}
        print(f"prefill: {len(longs)} prompts of {[len(x) + 1 for x in longs]} tokens: alone {sum(solo_ttft):.2f} s in all "
              f"({', '.join(f'{x:.2f}' for x in solo_ttft)}); together every first token "
              f"{last_first - start:.2f} s after the main request ended ({sum(solo_ttft) / max(1e-9, last_first - start):.2f}x); "
              f"combined {comb}", flush=True)
    # 4. priority
    if 'priority' not in a.skip:
        out = {}
        ws = [threading.Thread(target=submit, args=(path, op(f'pw-{tag}-{i}', f'pw-{tag}-{i}', decode[i], a.answer, 10),
                                                    out, f'w{i}')) for i in range(2)]
        for t in ws:
            t.start()
        time.sleep(3)
        m = submit(path, op(f'pm-{tag}', f'pm-{tag}', decode[2], a.answer // 2, 0))
        for t in ws:
            t.join()
        a0, a1 = m['stamps'][0] + 0.05, m['stamps'][-1] - 0.05
        during = sum(1 for k in ('w0', 'w1') for x in out[k]['stamps'] if a0 < x < a1)
        report['priority'] = {'main_tok_s': rate(m), 'main_solo_tok_s': rate(solo[2]), 'worker_tokens_during_main': during}
        print(f"priority: the main request {rate(m):.1f} tok/s (alone {rate(solo[2]):.1f}) while 2 workers waited "
              f"(worker tokens during it: {during})", flush=True)
    Path(a.out).write_text(json.dumps(report, indent=1) + '\n')
    print(f'saved: {a.out}')


if __name__ == '__main__':
    main()
