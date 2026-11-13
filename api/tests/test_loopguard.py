"""s0ma change 114: the loop guard. A reasoning cycle is ended and the request goes on from the same tokens plus the
end-of-reasoning token; a cycle in the answer cuts the request; normal text is untouched; a cycle in an earlier
turn's reasoning is cut out of the history, the same way every time."""

import random

import pytest

pytest.importorskip("tokenizers")

import ds41_api.encoding as enc  # noqa: E402
from ds41_api.backend import FinishEvent, SamplingParams, TokenEvent  # noqa: E402
from ds41_api.encoder import Encoder  # noqa: E402
from ds41_api.fake_backend import FakeBackend  # noqa: E402
from ds41_api.loopguard import CycleDetector, LoopGuardBackend, cut_cycle, strip_reasoning_loops  # noqa: E402
from ds41_api.parser import StreamParser  # noqa: E402
from ds41_api.schemas import ChatCompletionRequest  # noqa: E402

from test_specials import make_tokenizer  # noqa: E402


@pytest.fixture()
def setup(monkeypatch):
    monkeypatch.setenv("DS41_LOOP_WINDOW", "256")
    encoder = Encoder(make_tokenizer())
    fake = FakeBackend(tokenizer=encoder.tokenizer)
    guard = LoopGuardBackend.wrap(fake, encoder)
    assert isinstance(guard, LoopGuardBackend)
    return encoder, fake, guard


def ids(encoder, text):
    return encoder._plain_ids(text)


def prose(n_words, seed=1):
    r = random.Random(seed)
    words = "the a model reads files and writes code then checks tests before it answers with care".split()
    return " ".join(r.choice(words) + str(r.randrange(1000)) for _ in range(n_words))


async def run(guard, encoder, prompt_ids, max_tokens=100000):
    stream = await guard.submit(prompt_ids, SamplingParams(max_tokens=max_tokens,
                                stop_token_ids=(encoder.token_id(enc.eos_token),)), request_id="r1")
    detok = encoder.detokenizer()
    parser = StreamParser("thinking", specials=detok.specials)
    finish = None
    async for ev in stream:
        if isinstance(ev, TokenEvent):
            parser.feed(detok.feed(ev.token_id))
        elif isinstance(ev, FinishEvent):
            finish = ev
    parser.feed(detok.flush())
    return parser.message(), finish


def thinking_prompt(encoder):
    return ids(encoder, "<prompt>") + [encoder.token_id(enc.thinking_start_token)]


async def test_reasoning_cycle_is_ended_and_the_answer_follows(setup):
    encoder, fake, guard = setup
    think_end, eos = encoder.token_id(enc.thinking_end_token), encoder.token_id(enc.eos_token)
    good = ids(encoder, prose(80))
    cycle = ids(encoder, "Let me run. OK. Let me run the tool now. ") * 200
    fake.add_script(good + cycle + [think_end] + ids(encoder, "never reached") + [eos])
    fake.add_script(ids(encoder, "The answer.") + [eos])
    prompt = thinking_prompt(encoder)
    msg, fin = await run(guard, encoder, prompt)
    assert msg["content"] == "The answer."
    assert msg["reasoning_content"].startswith(prose(80)[:40])
    assert "never reached" not in msg["reasoning_content"]
    assert fin.finish_reason == "stop" and "reasoning ended" in (fin.usage.loop_guard or "")
    # the continuation is the same tokens plus the end of reasoning
    second = fake.submitted[1]["token_ids"]
    assert second[:len(prompt)] == prompt and second[-1] == think_end
    gen = second[len(prompt):-1]
    assert gen == (good + cycle)[:len(gen)] and len(gen) < len(good) + len(cycle)
    assert fake.submitted[1]["request_id"] == "r1-g1"
    assert fin.usage.completion_tokens == len(gen) + 1 + len(ids(encoder, "The answer.")) + 0


async def test_normal_reasoning_and_code_pass_untouched(setup):
    encoder, fake, guard = setup
    think_end, eos = encoder.token_id(enc.thinking_end_token), encoder.token_id(enc.eos_token)
    code = "".join(f"    x{i} = load(rows[{i}], width={i % 7})\n" for i in range(120))
    fake.add_script(ids(encoder, prose(700)) + ids(encoder, code) + [think_end] + ids(encoder, "Done.") + [eos])
    msg, fin = await run(guard, encoder, thinking_prompt(encoder))
    assert msg["content"] == "Done." and fin.usage.loop_guard is None and len(fake.submitted) == 1


async def test_answer_cycle_cuts_the_request(setup):
    encoder, fake, guard = setup
    think_end = encoder.token_id(enc.thinking_end_token)
    fake.add_script(ids(encoder, prose(50)) + [think_end] + ids(encoder, "No. I can't. I will end. ") * 300)
    msg, fin = await run(guard, encoder, thinking_prompt(encoder))
    assert fin.finish_reason == "length" and "answer" in fin.usage.loop_guard
    assert len(fake.submitted) == 1


async def test_second_cycle_after_ending_the_reasoning_cuts(setup):
    encoder, fake, guard = setup
    cycle = ids(encoder, "Emit. OK. Let me write. ") * 300
    fake.add_script(ids(encoder, prose(40)) + cycle)
    fake.add_script(cycle)
    msg, fin = await run(guard, encoder, thinking_prompt(encoder))
    assert fin.finish_reason == "length" and "reasoning ended" in fin.usage.loop_guard and "cut" in fin.usage.loop_guard
    assert len(fake.submitted) == 2


async def test_probes_and_chat_mode_go_straight_through(setup):
    encoder, fake, guard = setup
    fake.add_script(ids(encoder, "x"))
    s = await guard.submit(thinking_prompt(encoder), SamplingParams(max_tokens=1))
    assert type(s).__name__ == "BackendStream" and s.request_id == fake.submitted[-1]["request_id"]
    chat = ids(encoder, "<p>") + [encoder.token_id(enc.thinking_end_token)]
    fake.add_script(ids(encoder, "Let me run. ") * 400)
    msg, fin = await run(guard, encoder, chat)
    assert fin.usage.loop_guard is None   # chat mode: no reasoning to guard


def test_detector_threshold():
    d = CycleDetector(16, 256, 0.25)
    r = random.Random(3)
    assert not any(d.push(r.randrange(50000)) for _ in range(5000))
    d.reset()
    assert any(d.push(t) for t in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12] * 100)


def test_history_cycle_cut_and_stable():
    good = prose(400, seed=7)
    loop = " ".join(["The **roles+minimal**. The **`vm_exec`**. The **board**. Let's run."] * 120)
    msgs = [{"role": "user", "content": "go"},
            {"role": "assistant", "reasoning_content": good + "\n\n" + loop, "content": "",
             "tool_calls": [{"id": "c", "type": "function", "function": {"name": "f", "arguments": "{}"}}]},
            {"role": "tool", "tool_call_id": "c", "content": "ok"},
            {"role": "assistant", "reasoning_content": good, "content": "fine"}]
    a = [dict(m) for m in msgs]
    b = [dict(m) for m in msgs]
    assert strip_reasoning_loops(a) == 1 and strip_reasoning_loops(b) == 1
    assert a[1]["reasoning_content"] == b[1]["reasoning_content"]
    kept = a[1]["reasoning_content"]
    assert kept.startswith(good) and kept.count("The **board**") <= 1
    assert a[3]["reasoning_content"] == good   # normal reasoning untouched
    assert cut_cycle(good) is None


def test_history_cut_through_the_encoder(monkeypatch):
    encoder = Encoder(make_tokenizer())
    loop = " ".join(["Emit. OK. Let me go. Let me write."] * 200)
    req = ChatCompletionRequest(messages=[
        {"role": "user", "content": "go"},
        {"role": "assistant", "reasoning_content": prose(300) + " " + loop,
         "tool_calls": [{"id": "c", "type": "function", "function": {"name": "f", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "c", "content": "ok"},
    ], tools=[{"type": "function", "function": {"name": "f", "parameters": {"type": "object"}}}])
    prompt, _ = encoder.build_prompt(req)
    assert prompt.count("Let me write.") <= 1
    monkeypatch.setenv("DS41_LOOP_GUARD", "0")
    prompt2, _ = encoder.build_prompt(req)
    assert prompt2.count("Let me write.") >= 100
