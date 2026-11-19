"""s0ma change 131: the monitor (GET /monitor, GET /monitor/data)."""

import json

import httpx

import ds41_api.encoding as enc
from ds41_api.backend import FinishEvent, SamplingParams, Usage
from ds41_api.monitor import Monitor

EOS = enc.eos_token


def test_finish_event_carries_engine_stats():
    ev = FinishEvent.from_dict({"type": "finish", "finish_reason": "stop", "usage": {"prompt_tokens": 3},
                                "error": None, "stats": {"cycles": 4}})
    assert ev.stats == {"cycles": 4}
    assert ev.to_dict()["stats"] == {"cycles": 4}
    assert "stats" not in FinishEvent.from_dict({"type": "finish", "finish_reason": "stop"}).to_dict()


def test_monitor_record_from_engine_stats():
    m = Monitor()
    m.started("abcdef0123456789", 100, SamplingParams(max_tokens=64), None, "my-conversation", 2)
    m.token("abcdef0123456789")
    stats = {"cycles": 10, "target_verify_ms": 250.0, "draft_ms": 23.0, "drafted": 30, "accepted": 21,
             "decode_tps": 81.5, "ttft_ms": 900.0, "prefill_ms": 800.0, "prefill_input_tokens": 4000,
             "reused_prompt_tokens": 96000, "prompt_cache": "live", "expert_host_hits": 7, "live_slots": 4,
             "decode_tier_counters": {"expert_vram_hits": 900, "expert_host_hits": 70, "expert_nvme_misses": 0,
                                      "expert_h2d_bytes": 10 ** 9}}
    m.finished("abcdef0123456789", FinishEvent("stop", Usage(prompt_tokens=100000, completion_tokens=29), None, stats))
    snap = m.snapshot("model-x")
    assert snap["active"] == []
    (r,) = snap["recent"]
    assert r["id"] == "abcdef01" and r["conversation"] != "my-conversation" and len(r["conversation"]) == 8
    assert r["verify_ms_step"] == 25.0 and r["draft_ms_step"] == 2.3 and r["tokens_per_step"] == 2.9
    assert r["acceptance"] == 0.7 and r["decode_tok_s"] == 81.5 and r["prefill_tok_s"] == 5000.0
    assert r["cached_tokens"] == 96000 and r["experts_ram"] == 70 and r["h2d_bytes"] == 10 ** 9
    assert snap["engine"]["live_slots"] == 4 and snap["totals"]["requests"] == 1


def test_monitor_page_and_data(server_factory):
    handle = server_factory()
    handle.script("Plain answer text." + EOS)
    resp = httpx.post(f"{handle.base}/v1/chat/completions",
                      json={"messages": [{"role": "user", "content": "a private question"}]}, timeout=30.0)
    assert resp.status_code == 200
    page = httpx.get(f"{handle.base}/monitor", timeout=10.0)
    assert page.status_code == 200 and "ds41d monitor" in page.text
    data = httpx.get(f"{handle.base}/monitor/data", timeout=10.0)
    assert data.status_code == 200
    d = data.json()
    assert d["totals"]["requests"] == 1 and len(d["recent"]) == 1
    r = d["recent"][0]
    assert r["prompt_tokens"] > 0 and r["completion_tokens"] > 0 and r["finish"] == "stop"
    blob = json.dumps(d)
    assert "private question" not in blob and "Plain answer" not in blob   # numbers only


def test_monitor_off(server_factory, monkeypatch):
    monkeypatch.setenv("DS41_MONITOR", "0")
    handle = server_factory()
    assert httpx.get(f"{handle.base}/monitor", timeout=10.0).status_code == 404
    assert httpx.get(f"{handle.base}/monitor/data", timeout=10.0).status_code == 404
