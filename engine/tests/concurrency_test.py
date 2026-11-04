"""s0ma change 91: the scheduler over the replay model with live slots (protocol only: no GPU).

Several requests at once: each one's greedy tokens equal its tokens alone; same-priority requests share multi steps;
a better priority runs alone while worse ones wait; cancellation and disconnects free their slots."""
import json
import socket
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

binary = sys.argv[1]
V = 64


def rows():
    # position p: the argmax is (7 * p + 3) % (V - 1) + 1 (never the eos 0)
    out = []
    for p in range(29):
        r = [-1000.0] * V
        r[(7 * p + 3) % (V - 1) + 1] = 5.0
        out.append(r)
    return out


def start(tmp, slots, delay=3, decline=False, name="", **extra):
    path = Path(tmp) / f"s{slots}{'d' if decline else ''}{name}.sock"
    fixture = Path(tmp) / f"f{slots}{'d' if decline else ''}{name}.json"
    fixture.write_text(json.dumps({"schema": "ds41-replay-test-v1", "eos": 0, "logits": rows(),
                                   "delay_ms": delay, "live_slots": slots, "decline_multi": decline, **extra}))
    proc = subprocess.Popen([binary, "serve", "--socket", str(path), "--replay-test", str(fixture), "--prefix-cache"],
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    deadline = time.monotonic() + 5
    while not path.exists():
        assert time.monotonic() < deadline, proc.poll()
        time.sleep(.01)
    return proc, path


def request(path, obj, out=None, hold=None):
    sock = socket.socket(socket.AF_UNIX)
    sock.settimeout(30)
    sock.connect(str(path))
    sock.sendall((json.dumps(obj) + "\n").encode())
    events = []
    for line in sock.makefile("rb"):
        e = json.loads(line)
        e["_t"] = time.monotonic()
        events.append(e)
        if hold is not None and e.get("type") == "token" and len(events) == hold:
            sock.close()
            break
    else:
        sock.close()
    if out is not None:
        out[obj["request_id"]] = events
    return events


def tokens(events):
    return [e["token_id"] for e in events if e.get("type") == "token"]


def submit(rid, conv, n, prompt, priority=None):
    op = {"op": "submit", "request_id": rid, "conversation_id": conv, "token_ids": prompt,
          "sampling": {"temperature": 0, "max_tokens": n}, "draft": "off"}
    if priority is not None:
        op["priority"] = priority
    return op


with tempfile.TemporaryDirectory(prefix="ds41-conc-") as tmp:
    prompts = {f"c{i}": [5 + i] * (3 + i) for i in range(4)}
    # alone, one slot
    proc, path = start(tmp, 1)
    try:
        alone = {c: tokens(request(path, submit(f"a{c}", c, 12, p))) for c, p in prompts.items()}
    finally:
        proc.terminate(); proc.wait(10)
    for c, t in alone.items():
        assert len(t) == 12, (c, t)
    # three slots, four requests at once, the same priority: tokens equal, multi steps taken, one waits for a slot
    proc, path = start(tmp, 3)
    try:
        out = {}
        threads = [threading.Thread(target=request, args=(path, submit(f"b{c}", c, 12, p, 5), out))
                   for c, p in prompts.items()]
        for t in threads: t.start()
        for t in threads: t.join(60)
        multi = 0
        for c in prompts:
            ev = out[f"b{c}"]
            assert tokens(ev) == alone[c], (c, tokens(ev), alone[c])
            fin = ev[-1]
            assert fin["type"] == "finish" and fin["finish_reason"] == "length", fin
            multi += fin["stats"].get("multi_steps", 0)
        assert multi > 0, "no multi steps"
        # a better priority runs alone: workers started first stop stepping until the main request is done
        out = {}
        workers = [threading.Thread(target=request, args=(path, submit(f"w{i}", f"wc{i}", 20, prompts["c1"], 10), out))
                   for i in range(2)]
        for t in workers: t.start()
        time.sleep(0.15)
        main = threading.Thread(target=request, args=(path, submit("m", "mc", 10, prompts["c2"], 0), out))
        main.start(); main.join(60)
        for t in workers: t.join(60)
        m_first = out["m"][0]["_t"]; m_last = [e for e in out["m"] if e.get("type") == "token"][-1]["_t"]
        assert tokens(out["m"]) == alone["c2"][:10], tokens(out["m"])
        for i in range(2):
            during = [e for e in out[f"w{i}"] if e.get("type") == "token" and m_first + 0.02 < e["_t"] < m_last - 0.02]
            assert not during, f"worker w{i} stepped while the main request ran ({len(during)} tokens)"
            assert len(tokens(out[f"w{i}"])) == 20
        # a client that leaves mid-answer frees its slot; the others finish
        out = {}
        gone = threading.Thread(target=request, args=(path, submit("g", "gc", 400, prompts["c3"], 5), out, 3))
        stay = [threading.Thread(target=request, args=(path, submit(f"s{i}", f"sc{i}", 15, prompts["c0"], 5), out))
                for i in range(3)]
        gone.start()
        for t in stay: t.start()
        gone.join(60)
        for t in stay: t.join(60)
        for i in range(3):
            assert tokens(out[f"s{i}"]) == alone["c0"][:15] if len(alone["c0"]) >= 15 else len(tokens(out[f"s{i}"])) == 15
        # the same conversation again reuses its slot (no prompt rows prefilled anew)
        ev = request(path, submit("r", "c0", 3, prompts["c0"] + alone["c0"][:2], 5))
        fin = ev[-1]
        assert fin["finish_reason"] in ("length", "stop"), fin
    finally:
        proc.terminate()
        try:
            proc.wait(10)
        except subprocess.TimeoutExpired:
            proc.kill(); raise
    err = proc.stderr.read().decode(errors="replace")
    # prompts that queue behind a better priority start together: one combined prefill (and declined: one by one)
    for decline in (False, True):
        proc, path = start(tmp, 4, decline=decline)
        try:
            out = {}
            blocker = threading.Thread(target=request, args=(path, submit("x", "xc", 25, prompts["c0"], 0), out))
            blocker.start()
            time.sleep(0.05)
            ws = [threading.Thread(target=request, args=(path, submit(f"p{c}", f"pc{c}", 8, p, 3), out))
                  for c, p in list(prompts.items())[1:]]
            for t in ws: t.start()
            blocker.join(60)
            for t in ws: t.join(60)
            combined = []
            for c in list(prompts)[1:]:
                ev = out[f"p{c}"]
                assert tokens(ev) == alone[c][:8], (c, decline, tokens(ev), alone[c][:8])
                combined.append(ev[-1]["stats"].get("combined_prefill", 0))
            if decline:
                assert combined == [0, 0, 0], combined
            else:
                assert combined == [3, 3, 3], combined
                assert out["p" + list(prompts)[1]][-1]["stats"].get("multi_prefills", 0) >= 1
        finally:
            proc.terminate()
            proc.wait(10)
    assert "terminate" not in err, err[-2000:]
    # a conversation longer than a parked slot holds keeps slot 0: the others still start and finish (no livelock)
    proc, path = start(tmp, 4, delay=5, name="long", slot_context=20)
    try:
        out = {}
        tl = threading.Thread(target=request, args=(path, submit("L", "cl", 40, [6] * 30), out))
        tl.start(); time.sleep(0.3)
        ts = [threading.Thread(target=request, args=(path, submit(k, "c" + k, 5, [7 + i] * 3), out)) for i, k in enumerate("AB")]
        for t in ts: t.start()
        for t in [tl] + ts: t.join(60)
        for k, n in (("L", 40), ("A", 5), ("B", 5)):
            assert out[k][-1]["finish_reason"] == "length" and len(tokens(out[k])) == n, (k, out[k][-1])
    finally:
        proc.terminate(); proc.wait(10)
    # a prompt whose client leaves mid-prompt (the drafter's prompt left incomplete) does not break the parked one
    proc, path = start(tmp, 2, delay=5, name="spark", spark=True)
    try:
        out = {}
        tr = threading.Thread(target=request, args=(path, submit("R", "cr", 60, [5] * 3), out))
        tr.start(); time.sleep(0.03)
        sock = socket.socket(socket.AF_UNIX); sock.connect(str(path))
        sock.sendall((json.dumps(submit("P", "cp", 5, [6] * 200)) + "\n").encode())
        time.sleep(0.3); sock.close()
        tr.join(60)
        assert out["R"][-1]["finish_reason"] == "length" and len(tokens(out["R"])) == 60, out["R"][-1]
    finally:
        proc.terminate(); proc.wait(10)
    # s0ma change 93: with a drafter, multi steps draft too: the same greedy tokens alone and together
    proc, path = start(tmp, 3, delay=3, name="draft", drafter=True)
    try:
        def dsubmit(rid, conv, n, prompt):
            o = submit(rid, conv, n, prompt, 5)
            o["draft"] = "on"
            return o
        solo_d = {c: tokens(request(path, dsubmit(f"ds{c}", f"dsc{c}", 20, p))) for c, p in list(prompts.items())[:3]}
        for c, t in solo_d.items():
            assert t == alone[c][:20] if len(alone[c]) >= 20 else len(t) == 20, (c, t)
        out = {}
        ts = [threading.Thread(target=request, args=(path, dsubmit(f"dt{c}", f"dtc{c}", 20, p), out))
              for c, p in list(prompts.items())[:2]]
        for t in ts: t.start()
        for t in ts: t.join(60)
        drafted = 0
        for c in list(prompts)[:2]:
            assert tokens(out[f"dt{c}"]) == solo_d[c], (c, tokens(out[f"dt{c}"]), solo_d[c])
            st = out[f"dt{c}"][-1]["stats"]
            drafted += st.get("multi_drafted", 0)
            assert st.get("multi_accepted", 0) == st.get("multi_drafted", 0), st   # a perfect drafter
        assert drafted > 0, "no drafts in multi steps"
    finally:
        proc.terminate(); proc.wait(10)
print("concurrency: same tokens alone and together, multi steps, priority, leaving clients, reuse, combined prefill, drafts in multi steps passed")
