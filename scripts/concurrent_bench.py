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
5. Probes (change 96; --only probes runs this alone): one-token requests shaped like s0ma's knowledge check (a shared
   ~400-token prefix, each ending in its own ~70-token question, max_tokens 1, top_logprobs 20): 1, 3, 6 and 9 at
   once (the time until the last answer, and how many shared each prefill); each of 6 alone against all 6 together
   (top-20 compared); a probe group during a main request's decode; probes before a returning main turn (served from
   its kept conversation?); 16 probes beside workers holding every slot and one waiting. A/B: lab start --env
   DS41_PROBES=0 (one-token requests take live slots, as before).

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
                    if 'top_logprobs' in e and len(rec.setdefault('events', [])) < 1:
                        rec['events'].append(e)
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
    ap.add_argument('--skip', default='', help='phases to skip: prefill,priority,probes')
    ap.add_argument('--only', default='', help='probes: that phase alone')
    a = ap.parse_args()
    cfg = run.configuration()
    path = run.runtime_dir(cfg) / 'engine.sock'
    if not path.exists():
        sys.exit(f'no engine socket at {path}: start the lab ds41d first')
    tag = time.strftime('%H%M%S')
    decode = [case_ids('chat-000'), case_ids('chat-001'), case_ids('chat-002'), case_ids('mt-000', 512),
              case_ids('agentic-000', 384), case_ids('agentic-001', 448)][:max(2, a.streams)]
    report = {'streams': a.streams, 'answer': a.answer, 'solo': [], 'together': [], 'prefill': {}, 'priority': {}}
    if a.only == 'probes':
        report['probes'] = probes(path, tag, decode, a.answer)
        Path(a.out).write_text(json.dumps(report, indent=1) + '\n')
        print(f'saved: {a.out}')
        return
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
        md = sum(r['finish']['stats'].get('multi_drafted', 0) for r in recs)
        ma = sum(r['finish']['stats'].get('multi_accepted', 0) for r in recs)
        row = {'multi_drafted': md, 'multi_accepted': ma, 'streams': s, 'steady_combined_tok_s': steady, 'overall_tok_s': overall,
               'per_stream_tok_s': [rate(r) for r in recs], 'multi_steps': multi, 'combined_prefill': comb,
               'first_diff_from_solo': diffs, 'solo_mean_tok_s': solo_mean}
        report['together'].append(row)
        print(f"{s} together: {steady:.1f} tok/s combined while all ran ({steady / solo_mean:.2f}x solo's "
              f"{solo_mean:.1f}), {overall:.1f} tok/s submit-to-last; each {', '.join(f'{rate(r):.1f}' for r in recs)}; "
              f"multi steps {multi}, drafts in them {md} ({ma} accepted); prefill shared by {comb}; "
              f"greedy tokens differ from solo at "
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
    if 'probes' not in a.skip:
        report['probes'] = probes(path, tag, decode, a.answer)
    Path(a.out).write_text(json.dumps(report, indent=1) + '\n')
    print(f'saved: {a.out}')


def probe_op(rid, ids, priority=0, seed=31):
    return {'op': 'submit', 'request_id': rid, 'conversation_id': None, 'token_ids': ids, 'priority': priority,
            'sampling': {'temperature': 1.0, 'top_p': 1.0, 'max_tokens': 1, 'logprobs': 20, 'seed': seed}}


def probes(path, tag, decode, answer):
    """s0ma change 96: one-token probes (see the module docstring, 5)."""
    out = {}
    system = case_ids('chat-000')[:60] + case_ids('chat-001')[:60] + case_ids('chat-002')[:30]   # ~150
    state = case_ids('agentic-000')[1000:1250]                                                   # ~250
    sources = [('agentic-001', 500), ('agentic-001', 3000), ('mt-000', 200), ('mt-000', 1200), ('agentic-000', 4000),
               ('agentic-000', 6000), ('agentic-001', 6500), ('mt-000', 2000), ('agentic-000', 2500)]
    questions = [case_ids(n)[o:o + 70] for n, o in sources]

    def group(first, n):   # n probes; a token of the round's own first: never served from a kept conversation
        return [[first] + system + state + q for q in questions[:n]]

    def at_once(prompts, label, priority=0, seed=31):
        recs = together(path, [probe_op(f'{label}-{tag}-{i}', p, priority, seed + i) for i, p in enumerate(prompts)])
        return recs

    def detail(recs):
        return [r['finish']['stats'] for r in recs]

    print(f"probes: {len(system) + len(state) + len(questions[0]) + 1} tokens each (shared prefix "
          f"{len(system) + len(state) + 1}, question {len(questions[0])})", flush=True)
    # the curve: 1, 3, 6, 9 at once
    curve = []
    for k, n in enumerate((1, 3, 6, 9)):
        t0 = time.perf_counter()
        recs = at_once(group(4000 + k, n), f'pc{n}')
        last = max(r['stamps'][0] for r in recs) - t0
        st = detail(recs)
        comb = sorted({x.get('combined_prefill', 0) for x in st})
        curve.append({'probes': n, 'last_answer_s': last, 'combined_prefill': comb,
                      'probe': [bool(x.get('probe')) for x in st], 'prefill_ms': [x.get('prefill_ms') for x in st]})
        print(f"  {n} at once: last answer {last * 1000:.0f} ms after the first was sent; prefills shared by {comb}; "
              f"as probes: {sum(bool(x.get('probe')) for x in st)} of {n}", flush=True)
    out['curve'] = curve
    # exactness: each of 6 alone, then the same 6 together
    prompts = group(4100, 6)
    alone = [submit(path, probe_op(f'pa-{tag}-{i}', p, 0, 41 + i)) for i, p in enumerate(prompts)]
    both = at_once(prompts, 'pt', 0, 41)
    worst, sets, flips, tokens_same = 0.0, 0, 0, 0
    for x, y in zip(alone, both):
        if not x.get('events') or not y.get('events'):   # (an end-of-text first token: nothing to compare)
            continue
        tx = [(e['token_id'], e['logprob']) for e in x['events'][0]['top_logprobs']]
        ty = [(e['token_id'], e['logprob']) for e in y['events'][0]['top_logprobs']]
        dx, dy = dict(tx), dict(ty)
        sets += set(dx) != set(dy)
        flips += [t for t, _ in tx[:2]] != [t for t, _ in ty[:2]]
        tokens_same += x['tokens'] == y['tokens']
        for t in set(dx) & set(dy):
            worst = max(worst, abs(dx[t] - dy[t]))
    groups_together = sorted({s.get('combined_prefill', 0) for s in detail(both)})
    out['exact'] = {'max_abs_logprob_diff': worst, 'top20_sets_differ': sets, 'top2_order_flips': flips,
                    'same_token': tokens_same, 'groups_together': groups_together}
    print(f"  6 alone vs together (shared by {groups_together}): top-20 max |logprob diff| {worst:.2e}, "
          f"top-20 sets differ in {sets}, top-2 order flips {flips}, the same drawn token {tokens_same} of 6", flush=True)
    # a probe group during a main request's decode (both priority 0)
    res = {}
    m = threading.Thread(target=submit, args=(path, op(f'pm-{tag}', f'pm-{tag}', decode[0], answer, 0), res, 'm'))
    m.start()
    time.sleep(2.5)
    t0 = time.perf_counter()
    recs = at_once(group(4200, 6), 'pd')
    t1 = max(r['stamps'][0] for r in recs)
    m.join()
    ms = res['m']['stamps']
    inside = [x for x in ms if t0 <= x <= t1]
    before = [x for x in ms if x < t0]
    win = (len(inside) - 1) / (inside[-1] - inside[0]) if len(inside) > 2 and inside[-1] > inside[0] else 0.0
    gap = max((b - a for a, b in zip(ms, ms[1:]) if t0 - 0.2 <= a <= t1), default=0.0)
    out['during_decode'] = {'main_tok_s': rate(res['m']), 'main_tok_s_while_probes': win,
                            'main_longest_gap_s': gap, 'probes_last_answer_s': t1 - t0,
                            'main_tokens_before': len(before)}
    print(f"  6 probes during a main decode: answered {(t1 - t0) * 1000:.0f} ms after sending; the main stream "
          f"{rate(res['m']):.1f} tok/s overall, its longest pause around them {gap * 1000:.0f} ms", flush=True)
    # probes before a returning main turn: served from its kept conversation?
    turn = []
    for k, (name, extra) in enumerate((('agentic-001', False), ('agentic-000', True))):
        ids = [4300 + k] + case_ids(name, 6000)
        conv = f'pr-{tag}-{k}'
        first = submit(path, op(f'pr1-{tag}-{k}', conv, ids, 32, 0))
        if extra:
            at_once(group(4400, 6), 'pp')
        follow = ids + first['tokens'] + case_ids('chat-001')
        second = submit(path, op(f'pr2-{tag}-{k}', conv, follow, 8, 0))
        st = second['finish']['stats']
        turn.append({'probes_between': extra, 'prompt': len(follow), 'reused': st.get('reused_prompt_tokens'),
                     'loaded': st.get('conversation_loaded'), 'ttft_s': second['stamps'][0] - second['submitted']})
        print(f"  a returning main turn ({len(follow)} tokens) {'after 6 probes' if extra else 'alone'}: "
              f"{st.get('reused_prompt_tokens')} reused{' (loaded from RAM)' if st.get('conversation_loaded') else ''}, "
              f"first token {(second['stamps'][0] - second['submitted']) * 1000:.0f} ms", flush=True)
    out['returning_turn'] = turn
    # 16 probes beside workers holding every slot, and one waiting for a slot
    res = {}
    ws = [threading.Thread(target=submit, args=(path, op(f'sw-{tag}-{i}', f'sw-{tag}-{i}', decode[i % len(decode)],
                                                           64, 5), res, f'w{i}')) for i in range(5)]
    for t in ws:
        t.start()
        time.sleep(0.2)
    t0 = time.perf_counter()
    pool = group(4500, 9) + group(4501, 7)
    recs = at_once(pool, 'ps', 5)
    t1 = max(r['stamps'][0] for r in recs)
    for t in ws:
        t.join()
    st = detail(recs)
    out['stress'] = {'answered': len(recs), 'last_answer_s': t1 - t0,
                     'probes': sum(bool(x.get('probe')) for x in st),
                     'slots_taken': sum(1 for x in st if x.get('live_slot', -1) >= 0),
                     'workers_done': sum(len(res[f'w{i}']['tokens']) > 0 for i in range(5))}
    print(f"  16 probes beside 5 workers (every slot taken, one waiting): {len(recs)} answered, the last after "
          f"{(t1 - t0) * 1000:.0f} ms; {out['stress']['slots_taken']} took a live slot; workers done "
          f"{out['stress']['workers_done']} of 5", flush=True)
    return out


if __name__ == '__main__':
    main()
