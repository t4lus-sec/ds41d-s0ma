"""s0ma change 131: live engine analytics while ds41d serves.

  GET /monitor        a page that polls /monitor/data every two seconds
  GET /monitor/data   JSON: the requests running now; the last 300 finished ones (token counts, timings, draft
                      acceptance, expert tiers); the engine's own counters; the saved conversations (counts and
                      bytes); the GPU (nvidia-smi) and host memory over the last 30 minutes

Numbers only: no prompt or completion text ever reaches the monitor, and conversation ids appear only as an
8-character hash. The API binds loopback unless started with --allow-remote. DS41_MONITOR=0 turns it off (the
routes answer 404 and nothing is recorded). The GPU sampler runs only while someone has the page open (a poll in
the last 60 s), so an unwatched server spends nothing on it.

The monitor wraps the engine backend inside the loop guard, so it sees each engine request, a loop-guard restart
included, with the stats object ds41d sends with every finish.
"""

import asyncio
import hashlib
import os
import shutil
import subprocess
import threading
import time
from collections import deque
from typing import Any, Callable, Dict, List, Optional, Sequence

from .backend import Backend, BackendStream, FinishEvent, SamplingParams, TokenEvent

KEEP = 300            # finished requests kept
SAMPLES = 900         # GPU/host samples kept (every 2 s: 30 minutes)
IDLE_S = 60.0         # the sampler stops this long after the last poll
CONVERSATIONS_S = 10  # the saved-conversation list is fetched at most this often

# engine stats keys that are engine-life totals or levels (shown as such); the rest are per request
ENGINE_KEYS = (
    "expert_vram_hits", "expert_host_hits", "expert_nvme_misses", "expert_h2d_bytes", "expert_nvme_bytes",
    "expert_spec_reads", "expert_spec_hits", "expert_spec_waits", "expert_spec_dropped",
    "engram_read_ms", "need_host_ms", "route_wait_ms", "forward_ms", "live_slots", "slot_swaps",
    "h1_cut_steps", "h1_rows_cut", "probe_prefills", "multi_prefills",
    "conversations_saved", "conversations_saved_bytes", "conversation_saves", "conversation_loads",
    "conversation_evictions", "conversations_ended", "conversations_on_disk", "conversations_disk_bytes",
    "conversation_disk_writes", "conversation_disk_loads",
)


def enabled() -> bool:
    return os.environ.get("DS41_MONITOR", "1") != "0"


def _short(conversation_id: Optional[str]) -> Optional[str]:
    return hashlib.sha1(conversation_id.encode("utf-8")).hexdigest()[:8] if conversation_id else None


def _num(d: Dict[str, Any], key: str) -> Optional[float]:
    v = d.get(key)
    return v if isinstance(v, (int, float)) and not isinstance(v, bool) else None


def _ratio(a: Optional[float], b: Optional[float], scale: float = 1.0) -> Optional[float]:
    return round(a / b * scale, 3) if a is not None and b else None


class Monitor:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.t0 = time.time()
        self.active: Dict[str, Dict[str, Any]] = {}
        self.done: deque = deque(maxlen=KEEP)
        self.engine: Dict[str, Any] = {}
        self.totals = {"requests": 0, "prompt_tokens": 0, "completion_tokens": 0, "errors": 0, "cancelled": 0}
        self.gpu: deque = deque(maxlen=SAMPLES)
        self.last_poll = 0.0
        self._sampler: Optional[threading.Thread] = None
        self.conversations: Optional[Dict[str, Any]] = None
        self._conversations_at = 0.0
        self._conversations_task: Optional[asyncio.Task] = None
        self.queued: Callable[[], int] = lambda: 0

    # ---- recording (called from the backend wrapper) ---------------------------------------------------------
    def started(self, rid: str, prompt_tokens: int, sampling: SamplingParams, images: Optional[Sequence[Any]],
                conversation_id: Optional[str], priority: Optional[int]) -> None:
        with self.lock:
            self.active[rid] = {
                "id": rid[:8], "t_start": time.time(), "prompt_tokens": prompt_tokens,
                "max_tokens": sampling.max_tokens, "images": len(images or []),
                "conversation": _short(conversation_id), "priority": priority,
                "tokens": 0, "t_first": None, "t_last": None,
            }

    def token(self, rid: str) -> None:
        now = time.time()
        with self.lock:
            a = self.active.get(rid)
            if a is not None:
                a["tokens"] += 1
                if a["t_first"] is None:
                    a["t_first"] = now
                a["t_last"] = now

    def abandoned(self, rid: str) -> None:
        with self.lock:
            self.active.pop(rid, None)

    def finished(self, rid: str, ev: FinishEvent) -> None:
        now = time.time()
        s = ev.stats or {}
        u = ev.usage
        with self.lock:
            a = self.active.pop(rid, None) or {"id": rid[:8], "t_start": now, "prompt_tokens": u.prompt_tokens,
                                               "images": 0, "conversation": None, "priority": None, "tokens": 0,
                                               "t_first": None, "t_last": None}
            cycles = _num(s, "cycles")
            completion = u.completion_tokens or a["tokens"]
            prefill_tokens = _num(s, "prefill_input_tokens")
            prefill_ms = _num(s, "prefill_ms") if _num(s, "prefill_ms") is not None else u.prefill_ms
            ttft = _num(s, "ttft_ms")
            if not ttft and a["t_first"]:
                ttft = (a["t_first"] - a["t_start"]) * 1000.0
            decode = _num(s, "decode_tps")
            if not decode and a["t_first"] and a["t_last"] and a["t_last"] > a["t_first"] and a["tokens"] > 1:
                decode = (a["tokens"] - 1) / (a["t_last"] - a["t_first"])
            drafted, accepted = _num(s, "drafted"), _num(s, "accepted")
            tiers = s.get("decode_tier_counters") if isinstance(s.get("decode_tier_counters"), dict) else {}
            rec = {
                "id": a["id"], "t_start": round(a["t_start"], 3), "t_end": round(now, 3),
                "wall_s": round(now - a["t_start"], 3), "conversation": a["conversation"], "priority": a["priority"],
                "images": a["images"], "probe": bool(s.get("probe")), "finish": ev.finish_reason,
                "error": bool(ev.error), "loop_guard": u.loop_guard,
                "prompt_tokens": u.prompt_tokens or a["prompt_tokens"],
                "cached_tokens": u.cached_prompt_tokens if u.cached_prompt_tokens is not None else _num(s, "reused_prompt_tokens"),
                "prompt_cache": u.prompt_cache or s.get("prompt_cache"),
                "prefill_tokens": prefill_tokens, "prefill_ms": prefill_ms,
                "prefill_tok_s": _ratio(prefill_tokens, prefill_ms, 1000.0),
                "ttft_ms": round(ttft, 1) if ttft else None,
                "completion_tokens": completion, "decode_tok_s": round(decode, 2) if decode else None,
                "drafted": drafted, "accepted": accepted, "acceptance": _ratio(accepted, drafted),
                "copy_drafted": _num(s, "copy_drafted"), "copy_accepted": _num(s, "copy_accepted"),
                "steps": cycles, "tokens_per_step": _ratio(completion, cycles),
                "verify_ms_step": _ratio(_num(s, "target_verify_ms"), cycles),
                "draft_ms_step": _ratio(_num(s, "draft_ms"), cycles),
                "multi_steps": _num(s, "multi_steps"),
                "experts_vram": _num(tiers, "expert_vram_hits"), "experts_ram": _num(tiers, "expert_host_hits"),
                "experts_nvme": _num(tiers, "expert_nvme_misses"), "h2d_bytes": _num(tiers, "expert_h2d_bytes"),
            }
            self.done.append(rec)
            t = self.totals
            t["requests"] += 1
            t["prompt_tokens"] += rec["prompt_tokens"] or 0
            t["completion_tokens"] += completion or 0
            t["errors"] += 1 if ev.finish_reason == "error" else 0
            t["cancelled"] += 1 if ev.finish_reason == "cancelled" else 0
            for key in ENGINE_KEYS:
                v = _num(s, key)
                if v is not None:
                    self.engine[key] = v
            if s:
                self.engine["at"] = round(now, 3)

    # ---- sampling ----------------------------------------------------------------------------------------------
    def _sample_loop(self) -> None:
        smi = shutil.which("nvidia-smi")
        fields = ("util", "mem_busy", "mem_used_mib", "mem_total_mib", "power_w", "power_limit_w", "temp_c", "sm_mhz")
        query = ("utilization.gpu,utilization.memory,memory.used,memory.total,power.draw,power.limit,"
                 "temperature.gpu,clocks.sm")
        while True:
            started = time.time()
            if started - self.last_poll > IDLE_S:
                time.sleep(2.0)
                continue
            sample: Dict[str, Any] = {"t": round(started, 3)}
            if smi:
                try:
                    out = subprocess.run([smi, f"--query-gpu={query}", "--format=csv,noheader,nounits"],
                                         capture_output=True, text=True, timeout=5).stdout.splitlines()
                    if out:
                        for name, raw in zip(fields, out[0].split(",")):
                            try:
                                sample[name] = float(raw.strip())
                            except ValueError:
                                sample[name] = None
                except (OSError, subprocess.SubprocessError):
                    pass
            try:
                mem = {}
                with open("/proc/meminfo") as fh:
                    for line in fh:
                        k, _, rest = line.partition(":")
                        if k in ("MemTotal", "MemAvailable"):
                            mem[k] = int(rest.split()[0]) * 1024
                sample["host_total"] = mem.get("MemTotal")
                sample["host_available"] = mem.get("MemAvailable")
                sample["load1"] = os.getloadavg()[0]
            except OSError:
                pass
            with self.lock:
                self.gpu.append(sample)
            time.sleep(max(0.2, 2.0 - (time.time() - started)))

    def _ensure_sampler(self) -> None:
        if self._sampler is None:
            self._sampler = threading.Thread(target=self._sample_loop, name="ds41-monitor-sampler", daemon=True)
            self._sampler.start()

    async def _fetch_conversations(self, backend: Backend) -> None:
        try:
            reply = await asyncio.wait_for(backend.list_conversations(), timeout=5.0)
        except Exception:
            return
        if not isinstance(reply, dict) or not reply.get("ok"):
            return
        ram = [c for c in reply.get("ram") or [] if isinstance(c, dict)]
        disk = [c for c in reply.get("disk") or [] if isinstance(c, dict)]
        ram.sort(key=lambda c: c.get("idle_s") or 0)
        self.conversations = {
            "ram_count": len(ram), "ram_tokens": sum(int(c.get("tokens") or 0) for c in ram),
            "ram_bytes": reply.get("ram_bytes"), "ram_budget": reply.get("ram_budget"),
            "disk_count": len(disk), "disk_bytes": sum(int(c.get("bytes") or 0) for c in disk),
            "ram": [{"conversation": _short(str(c.get("conversation_id") or "")), "tokens": c.get("tokens"),
                     "bytes": c.get("bytes"), "priority": c.get("priority"), "idle_s": c.get("idle_s")}
                    for c in ram[:20]],
            "at": round(time.time(), 3),
        }

    # ---- the data endpoint ---------------------------------------------------------------------------------------
    def poll(self, backend: Backend) -> None:
        self.last_poll = time.time()
        self._ensure_sampler()
        busy = self._conversations_task is not None and not self._conversations_task.done()
        if not busy and self.last_poll - self._conversations_at > CONVERSATIONS_S:
            self._conversations_at = self.last_poll
            self._conversations_task = asyncio.ensure_future(self._fetch_conversations(backend))

    def snapshot(self, model_id: str) -> Dict[str, Any]:
        now = time.time()
        with self.lock:
            active: List[Dict[str, Any]] = []
            for a in self.active.values():
                live = None
                if a["t_first"] and a["t_last"] and a["tokens"] > 1 and a["t_last"] > a["t_first"]:
                    live = round((a["tokens"] - 1) / (a["t_last"] - a["t_first"]), 2)
                active.append({
                    "id": a["id"], "conversation": a["conversation"], "priority": a["priority"],
                    "images": a["images"], "prompt_tokens": a["prompt_tokens"], "max_tokens": a.get("max_tokens"),
                    "tokens": a["tokens"], "elapsed_s": round(now - a["t_start"], 2),
                    "prefilling": a["t_first"] is None, "tok_s": live,
                })
            return {
                "model": model_id, "now": round(now, 3), "uptime_s": round(now - self.t0, 1),
                "queued": self.queued(), "active": active, "recent": list(self.done),
                "totals": dict(self.totals), "engine": dict(self.engine), "gpu": list(self.gpu),
                "conversations": self.conversations,
            }


class MonitorBackend(Backend):
    """The engine backend, with each request's start, tokens and finish recorded in a Monitor."""

    def __init__(self, inner: Backend, monitor: Monitor):
        self.inner = inner
        self.monitor = monitor

    async def submit(self, token_ids: Sequence[int], sampling: SamplingParams, images: Optional[Sequence[Any]] = None,
                     conversation_id: Optional[str] = None, request_id: Optional[str] = None,
                     priority: Optional[int] = None) -> BackendStream:
        stream = await self.inner.submit(token_ids, sampling, images=images, conversation_id=conversation_id,
                                         request_id=request_id, priority=priority)
        rid = stream.request_id
        self.monitor.started(rid, len(token_ids), sampling, images, conversation_id, priority)

        async def events():
            finished = False
            try:
                async for ev in stream:
                    if isinstance(ev, TokenEvent):
                        self.monitor.token(rid)
                    elif isinstance(ev, FinishEvent):
                        self.monitor.finished(rid, ev)
                        finished = True
                    yield ev
            finally:
                if not finished:
                    self.monitor.abandoned(rid)

        return BackendStream(rid, events())

    async def cancel(self, request_id: str) -> bool:
        return await self.inner.cancel(request_id)

    async def end_conversation(self, conversation_id: Optional[str] = None,
                               prefix: Optional[str] = None) -> Dict[str, Any]:
        return await self.inner.end_conversation(conversation_id, prefix)

    async def list_conversations(self) -> Dict[str, Any]:
        return await self.inner.list_conversations()

    async def close(self) -> None:
        await self.inner.close()
