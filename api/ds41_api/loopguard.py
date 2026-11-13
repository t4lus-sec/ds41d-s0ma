"""The loop guard (s0ma change 114): DeepSeek-V4.1-Flash's reasoning can collapse into a short cycle ("Let me run.
OK. Let me run.", "The **board**. The **`vm_exec`**.") that runs to the token cap or until sampling happens to break
it. Reported on many engines and quants; no setting prevents it. Two parts, both off with DS41_LOOP_GUARD=0:

1. **Generation.** LoopGuardBackend wraps the engine backend. While a request is reasoning, it watches the generated
   token ids: when, over the last DS41_LOOP_WINDOW (1024) positions, fewer than DS41_LOOP_DISTINCT (25%) of the
   DS41_LOOP_NGRAM (16)-token windows are distinct, the reasoning is in a cycle. The guard then ends it: it cancels
   the engine request and continues from the same tokens plus the end-of-reasoning token (the live slot holds them,
   so nothing is prefilled again but that one token), and the model goes on to its answer or tool call. A cycle in
   the answer itself, or a second one after the guard already ended the reasoning, cuts the request (finish_reason
   "length"). Either way the usage says so (usage.loop_guard, and ds41_loop_guard in the API's usage block).
   Normal text is far from the threshold: prose and code keep >90% of their 16-token windows distinct.

2. **History.** strip_reasoning_loops() cuts a cycle out of an earlier assistant turn's reasoning before the prompt
   is rendered, at the first repeat of the cycle. DeepSeek keeps all earlier reasoning when tools are defined, so
   without this a loop in the history pulls every later call toward the same cycle. The cut is a pure function of
   the text, so the same history always renders the same prompt (prompt caches hold).
"""

import os
import re
import uuid
from collections import Counter, deque
from typing import Any, AsyncIterator, Dict, List, Optional, Sequence, Tuple

from .backend import Backend, BackendStream, FinishEvent, SamplingParams, TokenEvent, Usage


def _env_int(name: str, default: int, lo: int, hi: int) -> int:
    try:
        v = int(os.environ.get(name) or default)
    except ValueError:
        return default
    return max(lo, min(hi, v))


def _env_float(name: str, default: float, lo: float, hi: float) -> float:
    try:
        v = float(os.environ.get(name) or default)
    except ValueError:
        return default
    return max(lo, min(hi, v))


def guard_enabled() -> bool:
    return os.environ.get("DS41_LOOP_GUARD", "1") != "0"


class CycleDetector:
    """Over the last `window` n-gram positions, the share of distinct n-grams; a cycle when it falls below `ratio`."""

    def __init__(self, ngram: int, window: int, ratio: float):
        self.n, self.w, self.ratio = ngram, window, ratio
        self.recent: deque = deque(maxlen=ngram)
        self.grams: deque = deque()
        self.counts: Counter = Counter()

    def reset(self) -> None:
        self.recent.clear()
        self.grams.clear()
        self.counts.clear()

    def push(self, token) -> bool:
        self.recent.append(token)
        if len(self.recent) < self.n:
            return False
        g = hash(tuple(self.recent))
        self.grams.append(g)
        self.counts[g] += 1
        if len(self.grams) > self.w:
            old = self.grams.popleft()
            c = self.counts[old] - 1
            if c:
                self.counts[old] = c
            else:
                del self.counts[old]
        return len(self.grams) >= self.w and len(self.counts) < self.ratio * self.w


class LoopGuardBackend(Backend):
    def __init__(self, inner: Backend, think_start_id: int, think_end_id: int):
        self.inner = inner
        self.think_start_id = think_start_id
        self.think_end_id = think_end_id
        self.ngram = _env_int("DS41_LOOP_NGRAM", 16, 4, 128)
        self.window = _env_int("DS41_LOOP_WINDOW", 1024, 64, 65536)
        self.ratio = _env_float("DS41_LOOP_DISTINCT", 0.25, 0.01, 0.95)
        self.current: Dict[str, str] = {}   # outer request id -> the engine request now generating
        self.fired = 0

    @classmethod
    def wrap(cls, backend: Backend, encoder: Any) -> Backend:
        """The guard around `backend` when the tokenizer has the reasoning tokens (DeepSeek's), else `backend`."""
        if not guard_enabled() or isinstance(backend, LoopGuardBackend):
            return backend
        from . import encoding as enc

        start = encoder.token_id(enc.thinking_start_token)
        end = encoder.token_id(enc.thinking_end_token)
        if start is None or end is None:
            return backend
        return cls(backend, start, end)

    async def submit(self, token_ids: Sequence[int], sampling: SamplingParams, images=None,
                     conversation_id: Optional[str] = None, request_id: Optional[str] = None,
                     priority: Optional[int] = None) -> BackendStream:
        rid = request_id or uuid.uuid4().hex
        kw = dict(conversation_id=conversation_id, priority=priority)
        first = await self.inner.submit(token_ids, sampling, images=images, request_id=rid, **kw)
        if sampling.max_tokens <= 1 or images or not token_ids or token_ids[-1] != self.think_start_id:
            return first   # probes, image prompts and chat mode go straight through
        self.current[rid] = first.request_id
        return BackendStream(rid, self._guard(rid, first, list(token_ids), sampling, kw))

    async def _drain(self, stream: BackendStream, generated: List[int], out: List[TokenEvent]) -> Optional[FinishEvent]:
        """After a cancel: the tokens still in flight (kept, and passed on), then the finish."""
        async for ev in stream:
            if isinstance(ev, TokenEvent):
                generated.append(ev.token_id)
                out.append(ev)
            elif isinstance(ev, FinishEvent):
                return ev
        return None

    async def _guard(self, rid: str, stream: BackendStream, prompt: List[int], sampling: SamplingParams,
                     kw: Dict[str, Any]) -> AsyncIterator[Any]:
        det = CycleDetector(self.ngram, self.window, self.ratio)
        generated: List[int] = []
        reasoning = True
        ended_reasoning = False
        first_usage: Optional[Usage] = None
        note: Optional[str] = None
        try:
            while True:
                fired = False
                async for ev in stream:
                    if isinstance(ev, FinishEvent):
                        usage = self._merge(first_usage, ev.usage, len(generated), note)
                        yield FinishEvent(ev.finish_reason, usage, ev.error)
                        return
                    generated.append(ev.token_id)
                    yield ev
                    if reasoning and ev.token_id == self.think_end_id:
                        reasoning = False
                        det.reset()
                        continue
                    if det.push(ev.token_id):
                        fired = True
                        break
                if not fired:
                    return   # the stream ended without a finish event; the caller reports it
                self.fired += 1
                await self.inner.cancel(stream.request_id)
                late: List[TokenEvent] = []
                fin = await self._drain(stream, generated, late)
                for ev in late:
                    yield ev
                if first_usage is None and fin is not None:
                    first_usage = fin.usage
                left = sampling.max_tokens - len(generated) - 1
                if not reasoning or ended_reasoning or left < 1 or (fin is not None and fin.finish_reason == "error"):
                    note = (note + "; " if note else "") + f"cycle in the {'reasoning' if reasoning else 'answer'} " \
                        f"at token {len(generated)}: cut"
                    usage = self._merge(first_usage, None, len(generated), note)
                    yield FinishEvent("length", usage, None)
                    return
                # end the reasoning and go on from the same tokens: the live slot holds them
                note = f"cycle in the reasoning at token {len(generated)}: reasoning ended"
                ended_reasoning = True
                reasoning = False
                det.reset()
                end = TokenEvent(self.think_end_id, 0.0, [])
                generated.append(self.think_end_id)
                yield end
                cont = SamplingParams(temperature=sampling.temperature, top_p=sampling.top_p, max_tokens=left,
                                      stop_token_ids=sampling.stop_token_ids, seed=sampling.seed,
                                      logprobs=sampling.logprobs)
                stream = await self.inner.submit(prompt + generated, cont, request_id=rid + "-g1", **kw)
                self.current[rid] = stream.request_id
        finally:
            self.current.pop(rid, None)

    @staticmethod
    def _merge(first: Optional[Usage], last: Optional[Usage], generated: int, note: Optional[str]) -> Usage:
        """One usage for the whole request: the first engine request's prompt and cache figures, every token
        generated, and what the guard did."""
        base = first or last or Usage()
        if first is None:
            u = base
        else:
            u = Usage(**{k: getattr(first, k) for k in first.__dataclass_fields__})
            if last is not None:
                for k in ("draft_tokens", "accepted_draft_tokens"):
                    a, b = getattr(first, k), getattr(last, k)
                    if a is not None or b is not None:
                        setattr(u, k, (a or 0) + (b or 0))
        u.completion_tokens = generated
        if note:
            u.loop_guard = note
        return u

    async def cancel(self, request_id: str) -> bool:
        return await self.inner.cancel(self.current.get(request_id, request_id))

    async def end_conversation(self, conversation_id=None, prefix=None):
        return await self.inner.end_conversation(conversation_id, prefix)

    async def list_conversations(self):
        return await self.inner.list_conversations()

    async def close(self) -> None:
        await self.inner.close()

    def __getattr__(self, name):   # anything else a backend offers (tests' FakeBackend introspection)
        return getattr(self.inner, name)


_WORD = re.compile(r"\S+")


def cut_cycle(text: str, ngram: int = 8, window: int = 300, ratio: float = 0.3, repeats: int = 3) -> Optional[int]:
    """Where a cycle starts in `text` (a character offset), or None: over word n-grams, the first `window` whose
    distinct share falls below `ratio`; the cut goes at the first n-gram in it that occurs `repeats` times or more
    from there on (the cycle's first lap)."""
    words = [(m.start(), m.group()) for m in _WORD.finditer(text)]
    if len(words) < window + ngram:
        return None
    grams = [hash(tuple(w for _, w in words[i:i + ngram])) for i in range(len(words) - ngram + 1)]
    counts: Counter = Counter()
    for j, g in enumerate(grams):
        counts[g] += 1
        if j >= window:
            old = grams[j - window]
            counts[old] -= 1
            if not counts[old]:
                del counts[old]
        if j + 1 >= window and len(counts) < ratio * window:
            start = j + 1 - window
            rest = Counter(grams[start:])
            for i in range(start, len(grams)):
                if rest[grams[i]] >= repeats:
                    return words[i][0]
            return words[start][0]
    return None


def strip_reasoning_loops(messages: List[Dict[str, Any]]) -> int:
    """Cut cycles out of earlier assistant turns' reasoning (in place); returns how many were cut."""
    if not guard_enabled():
        return 0
    cut = 0
    for m in messages:
        if not isinstance(m, dict) or m.get("role") != "assistant":
            continue
        for key in ("reasoning_content", "reasoning"):
            text = m.get(key)
            if isinstance(text, str) and len(text) > 2000:
                at = cut_cycle(text)
                if at is not None:
                    m[key] = text[:at].rstrip()
                    cut += 1
    return cut
