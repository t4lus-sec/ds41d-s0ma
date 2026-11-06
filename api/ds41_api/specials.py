"""Special tokens as control only (s0ma change 110).

DeepSeek-V4.1's control strings (</think>, ｜DSML｜, <｜end▁of▁sentence｜>, <｜User｜>, ...) are single special tokens,
and the tokenizer turns the same characters into those ids wherever they appear in a prompt string. So a file, a
tool result or the model's own reasoning that merely mentions "</think>" became the real end-of-reasoning token in
the next prompt, and on the way out the parser, reading detokenized text, could not tell the model's end of
reasoning from the characters "</think>" written inside it.

This module keeps the two apart:
- Prompts: every string that comes from a request (messages, tools, response_format, context) is escaped before
  the reference encoder renders it: each special string becomes a two-character marker (MARK + a private-use code),
  and a literal MARK becomes MARK + LITERAL. Encoder.encode_prompt tokenizes the template as before (its control
  strings stay ids) and each marker as plain text.
- Output: the detokenizer turns the control ids into markers (and escapes a literal MARK), the parser's delimiters
  are the markers, and everything the parser hands out is unmarked again.

A prompt with no special string in its request text renders and tokenizes exactly as before.
"""

import re
from typing import Any, Dict, Iterable, List, Optional

from . import encoding as enc

MARK = ""
LITERAL = MARK + ""
_FIRST_CODE = 0xE100
_LAST_CODE = 0xE0FF + 0x0E00   # codes 0xE100..0xEEFF: 3,584 strings

# the encoder's own control strings, in a fixed order (their markers never change)
CONTROL: List[str] = list(dict.fromkeys([
    enc.bos_token, enc.eos_token, enc.thinking_start_token, enc.thinking_end_token, enc.dsml_token,
    enc.USER_SP_TOKEN, enc.ASSISTANT_SP_TOKEN, enc.LATEST_REMINDER_SP_TOKEN, enc.IMAGE_PLACEHOLDER,
    *enc.DS_TASK_SP_TOKENS.values(),
]))

_MARKER_RE = re.compile(MARK + ".", re.DOTALL)


class Specials:
    """The special strings of one tokenizer and their markers."""

    def __init__(self, strings: Iterable[str]):
        rest = sorted({s for s in strings if s and s not in CONTROL})
        order = [s for s in CONTROL] + rest
        if _FIRST_CODE + len(order) > _LAST_CODE:
            raise ValueError(f"{len(order)} special strings: more than the markers can name")
        self.order = order
        self.mark: Dict[str, str] = {s: MARK + chr(_FIRST_CODE + i) for i, s in enumerate(order)}
        self._unmark: Dict[str, str] = {v: k for k, v in self.mark.items()}
        self._unmark[LITERAL] = MARK
        # candidates by their first two characters, longest first
        self._by2: Dict[str, List[str]] = {}
        for s in sorted(order, key=len, reverse=True):
            self._by2.setdefault(s[:2], []).append(s)
        firsts = {s[0] for s in order} | {MARK}
        self._first = re.compile("[" + "".join(re.escape(c) for c in sorted(firsts)) + "]")

    def escape(self, text: str) -> str:
        """Request text -> text whose special strings are markers (a literal MARK escaped)."""
        out: List[str] = []
        last = 0
        for m in self._first.finditer(text):
            i = m.start()
            if i < last:
                continue
            if text[i] == MARK:
                out.append(text[last:i])
                out.append(LITERAL)
                last = i + 1
                continue
            for s in self._by2.get(text[i:i + 2], ()):
                if text.startswith(s, i):
                    out.append(text[last:i])
                    out.append(self.mark[s])
                    last = i + len(s)
                    break
        if not out:
            return text
        out.append(text[last:])
        return "".join(out)

    def escape_tree(self, obj: Any) -> Any:
        """Every string in a JSON-like value escaped (dict keys too)."""
        if isinstance(obj, str):
            return self.escape(obj)
        if isinstance(obj, list):
            return [self.escape_tree(v) for v in obj]
        if isinstance(obj, tuple):
            return tuple(self.escape_tree(v) for v in obj)
        if isinstance(obj, dict):
            return {(self.escape(k) if isinstance(k, str) else k): self.escape_tree(v) for k, v in obj.items()}
        return obj

    def string_of(self, marker: str) -> Optional[str]:
        """The special string (or MARK) a two-character marker stands for."""
        return self._unmark.get(marker)

    def unmark(self, text: str) -> str:
        if MARK not in text:
            return text
        return _MARKER_RE.sub(lambda m: self._unmark.get(m.group(0), m.group(0)), text)

    def unmark_tree(self, obj: Any) -> Any:
        if isinstance(obj, str):
            return self.unmark(obj)
        if isinstance(obj, list):
            return [self.unmark_tree(v) for v in obj]
        if isinstance(obj, dict):
            return {k: self.unmark_tree(v) for k, v in obj.items()}
        return obj
