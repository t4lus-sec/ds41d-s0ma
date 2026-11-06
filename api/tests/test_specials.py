"""s0ma change 110: special strings in a request's text stay text; the model's own control tokens stay control.

DeepSeek-V4.1's tokenizer matches its special strings anywhere in a string, so before this change a tool result or
a reasoning trace that mentioned "</think>" put the real end-of-reasoning token into the next prompt, and the parser
ended the reasoning at the characters "</think>" written inside it. A small byte-level tokenizer with the same
special strings stands in for the real one.
"""

import pytest

tokenizers = pytest.importorskip("tokenizers")
from tokenizers import Tokenizer, decoders, models, pre_tokenizers  # noqa: E402

import ds41_api.encoding as enc  # noqa: E402
from ds41_api.encoder import Encoder  # noqa: E402
from ds41_api.parser import StreamParser  # noqa: E402
from ds41_api.schemas import ChatCompletionRequest  # noqa: E402
from ds41_api.specials import CONTROL, MARK  # noqa: E402

TH_END = enc.thinking_end_token
EOS = enc.eos_token
D = enc.dsml_token


def make_tokenizer() -> Tokenizer:
    alphabet = sorted(pre_tokenizers.ByteLevel.alphabet())
    tok = Tokenizer(models.BPE(vocab={c: i for i, c in enumerate(alphabet)}, merges=[]))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False)
    tok.decoder = decoders.ByteLevel()
    tok.add_special_tokens(CONTROL)
    return tok


@pytest.fixture(scope="module")
def encoder() -> Encoder:
    return Encoder(make_tokenizer())


def ids_of(encoder: Encoder, s: str):
    return encoder.tokenizer.encode(s, add_special_tokens=False).ids


def count(ids, encoder: Encoder, special: str) -> int:
    return list(ids).count(encoder.tokenizer.token_to_id(special))


def test_plain_requests_tokenize_as_before(encoder):
    req = ChatCompletionRequest(messages=[{"role": "user", "content": "hello <b>there</b>"}])
    prompt, _ = encoder.build_prompt(req)
    assert MARK not in prompt
    assert encoder.encode_prompt(prompt) == ids_of(encoder, prompt)


def test_special_strings_in_request_text_are_plain(encoder):
    noisy = f"see {TH_END} and {EOS} and <{D} calls> and <think> in a file"
    req = ChatCompletionRequest(
        messages=[
            {"role": "user", "content": noisy},
            {"role": "assistant", "content": noisy, "reasoning_content": "I read " + TH_END + " there",
             "tool_calls": [{"id": "c1", "type": "function",
                             "function": {"name": "f", "arguments": '{"x": "' + EOS + '"}'}}]},
            {"role": "tool", "tool_call_id": "c1", "content": noisy},
            {"role": "user", "content": "again"},
        ],
        tools=[{"type": "function", "function": {"name": "f", "description": "writes " + TH_END,
                                                 "parameters": {"type": "object"}}}],
    )
    clean = ChatCompletionRequest(
        messages=[
            {"role": "user", "content": "x"},
            {"role": "assistant", "content": "x", "reasoning_content": "I read there",
             "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "f", "arguments": '{"x": "y"}'}}]},
            {"role": "tool", "tool_call_id": "c1", "content": "x"},
            {"role": "user", "content": "again"},
        ],
        tools=[{"type": "function", "function": {"name": "f", "description": "writes",
                                                 "parameters": {"type": "object"}}}],
    )
    noisy_ids = encoder.encode_prompt(encoder.build_prompt(req)[0])
    clean_ids = encoder.encode_prompt(encoder.build_prompt(clean)[0])
    # the control tokens come from the template alone: as many as in the same conversation without the strings
    for s in (TH_END, EOS, D, "<think>", enc.USER_SP_TOKEN, enc.ASSISTANT_SP_TOKEN):
        assert count(noisy_ids, encoder, s) == count(clean_ids, encoder, s), s
    # and the characters are there, as text
    text = encoder.tokenizer.decode(noisy_ids, skip_special_tokens=True)
    assert text.count("see </think> and") >= 3
    assert "\\u" not in text and MARK not in text


def test_literal_marker_character_round_trips(encoder):
    req = ChatCompletionRequest(messages=[{"role": "user", "content": "a" + MARK + "b " + TH_END}])
    ids = encoder.encode_prompt(encoder.build_prompt(req)[0])
    text = encoder.tokenizer.decode(ids, skip_special_tokens=False)
    assert "a" + MARK + "b " + TH_END in text
    assert count(ids, encoder, TH_END) == 0


def stream(encoder: Encoder, pieces, thinking_mode="thinking"):
    """pieces: (text, real) pairs: real ones go through the tokenizer's special matching (the model's control
    tokens), the others as plain text (the model writing the characters)."""
    ids = []
    for text, real in pieces:
        ids += ids_of(encoder, text) if real else encoder._plain_ids(text)
    detok = encoder.detokenizer()
    parser = StreamParser(thinking_mode, specials=detok.specials)
    deltas = []
    for t in ids:
        deltas += parser.feed(detok.feed(t))
    deltas += parser.feed(detok.flush())
    deltas += parser.finish_truncated()
    return parser, deltas


def test_reasoning_ends_only_at_the_control_token(encoder):
    parser, deltas = stream(encoder, [
        ("templates close with ", False), (TH_END, False), (" and the reply ends with " + EOS + ".", False),
        (TH_END, True), ("Use ", True), (TH_END, False), (".", True), (EOS, True),
    ])
    msg = parser.message()
    assert msg["reasoning_content"] == "templates close with </think> and the reply ends with " + EOS + "."
    assert msg["content"] == "Use </think>."
    assert "".join(d.get("reasoning_content", "") for d in deltas) == msg["reasoning_content"]
    assert "".join(d.get("content", "") for d in deltas) == msg["content"]
    assert parser.state == "done"


def test_tool_call_only_from_the_control_tokens(encoder):
    literal_call = "\n\n<" + D + " calls>\n"
    real_call = enc.tool_calls_template.format(
        dsml_token=D, tc_block_name=enc.tool_calls_block_name,
        tool_calls=enc.tool_call_template.format(
            dsml_token=D, tool_call_tag_name=enc.tool_call_tag_name, name="f",
            arguments=enc.encode_arguments_to_dsml({"name": "f", "arguments": {"x": "a"}})))
    parser, _ = stream(encoder, [
        (TH_END, True), ("The block starts", True), (literal_call, False), ("in a file.", True),
        ("\n\n" + real_call, True),
        (EOS, True),
    ])
    msg = parser.message()
    assert msg["content"].startswith("The block starts" + literal_call + "in a file.")
    assert len(msg["tool_calls"]) == 1 and msg["tool_calls"][0]["function"]["name"] == "f"


def test_literal_marker_in_output(encoder):
    parser, _ = stream(encoder, [("x" + MARK + "y", False), (TH_END, True), ("ok", True), (EOS, True)])
    assert parser.message()["reasoning_content"] == "x" + MARK + "y"
    assert parser.message()["content"] == "ok"
