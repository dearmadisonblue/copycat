from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Callable, Optional, Protocol

from .error import ModelProtocolError


@dataclass(frozen=True)
class ModelTurn:
    answer: str
    thinking: Optional[str] = None
    raw: Optional[str] = None
    streamed: bool = False
    prompt_tokens: Optional[int] = None


class ModelBackend(Protocol):
    def generate(
        self,
        *,
        prompt: str,
        stack: str,
        module_catalog: str,
    ) -> ModelTurn:
        ...


@dataclass(frozen=True)
class ModelOK:
    code: str


@dataclass(frozen=True)
class ModelError:
    message: str


_PROTOCOL_ELEMENT = re.compile(
    r"<(?P<tag>OK|ERROR)>(?P<payload>.*?)</(?P=tag)>",
    re.IGNORECASE | re.DOTALL,
)

_PROTOCOL_TAG = re.compile(
    r"</?(?:OK|ERROR)\b[^>]*>",
    re.IGNORECASE,
)


def parse_model_answer(answer: str) -> ModelOK | ModelError:
    """Extract one case-insensitive OK or ERROR element from a model answer."""
    matches = list(_PROTOCOL_ELEMENT.finditer(answer))
    tag_markers = list(_PROTOCOL_TAG.finditer(answer))

    if len(matches) != 1 or len(tag_markers) != 2:
        raise ModelProtocolError(
            answer,
            "Expected exactly one <OK>...</OK> or <ERROR>...</ERROR> "
            "element (tag names are case-insensitive). Text outside that "
            "single element is allowed.",
        )

    match = matches[0]
    tag = match.group("tag").upper()
    normalized = f"<{tag}>{match.group('payload')}</{tag}>"

    try:
        root = ET.fromstring(normalized)
    except ET.ParseError as exc:
        raise ModelProtocolError(
            answer,
            "The protocol element is not well-formed XML.",
        ) from exc

    if root.attrib or list(root):
        raise ModelProtocolError(
            answer,
            "Protocol elements may not have attributes or child elements.",
        )

    payload = root.text or ""

    if tag == "OK":
        return ModelOK(payload.strip())

    message = payload.strip()
    if not message:
        raise ModelProtocolError(
            answer,
            "<ERROR> must contain a useful error message.",
        )
    return ModelError(message)


class StubModel:
    """Deterministic backend for parser/evaluator tests and examples."""

    def __init__(self, response: str | Callable[[str, str, str], str]):
        self.response = response
        self.calls: list[tuple[str, str, str]] = []

    def generate(
        self,
        *,
        prompt: str,
        stack: str,
        module_catalog: str,
    ) -> ModelTurn:
        self.calls.append((prompt, stack, module_catalog))

        if callable(self.response):
            answer = self.response(prompt, stack, module_catalog)
        else:
            answer = self.response

        return ModelTurn(answer=answer, thinking="stub")


COPYCAT_SYSTEM_PROMPT = r"""
You are the program-synthesis engine for Copycat, a tiny concatenative stack language.

You will receive:
1. the current Copycat data stack, written from bottom to top;
2. a natural-language task.

Your job is to synthesize the smallest Copycat program that performs the task
when executed immediately against that stack.

RETURN FORMAT
Your final response must contain exactly ONE protocol element:

<OK>Copycat code</OK>

or, only if you cannot produce a valid program:

<ERROR>a short, useful explanation</ERROR>

The protocol tag names are case-insensitive. Text outside that one element is
allowed, but do not emit a second OK or ERROR element. Do not put attributes on
the protocol element. If XML metacharacters occur inside the payload, escape
them correctly.

COPYCAT SYNTAX
Natural numbers:
  0
  1
  42

Strings:
  "hello"
  "two words"

Quotations:
  [d]
  [1 2 f]

Primitive words are single lowercase letters. User-defined words, when present,
are lowercase kebab-case names longer than one character.

MODULE WORDS
An invocation may include a catalog of additional module words. Each entry gives
the word name followed by documentation explaining how to use it. The word's
implementation is deliberately hidden. Treat documented module words as available
Copycat operations and rely on their descriptions and examples.

PRIMITIVES
a  apply
  Remove the top quotation and execute its contents.
  Example: 1 [d] a  ==>  1 1

b  abstract
  Wrap the top value in a quotation.
  Example: 1 b  ==>  [1]

c  catenate
  Concatenate the top two quotations.
  Example: [1] [2] c  ==>  [1 2]

d  copy
  Duplicate the top data-stack value.
  Example: 1 d  ==>  1 1

e  drop
  Remove the top data-stack value.
  Example: 1 2 e  ==>  1

f  swap
  Exchange the top two data-stack values.
  Example: 1 2 f  ==>  2 1

r / s  reset / shift
  Delimited-control primitives. Do not use them unless the task actually
  requires continuation capture.

MODEL FORMS
Copycat also has {natural language} model forms, but DO NOT emit model forms
inside generated code in this version.

IMPORTANT
The generated code executes immediately with the current stack already present.
Do not reproduce existing stack values unless the task requires copying them.
Prefer the shortest valid program.

EXAMPLES

Current data stack:
1 2

Task:
swap the top two values

Final answer:
<OK>f</OK>


Current data stack:
"hello"

Task:
duplicate the top value

Final answer:
<OK>d</OK>


Current data stack:
(empty)

Task:
put the number 7 on the stack

Final answer:
<OK>7</OK>


Current data stack:
1

Task:
remove the value

Final answer:
<OK>e</OK>
""".strip()


class Gemma4Backend:
    MODEL_ID = "google/gemma-4-E2B-it"

    def __init__(
        self,
        model,
        processor,
        *,
        system_prompt: str = COPYCAT_SYSTEM_PROMPT,
        max_new_tokens: int = 8_192,
        stream_output: bool = True,
    ):
        self.model = model
        self.processor = processor
        self.system_prompt = system_prompt
        self.max_new_tokens = max_new_tokens
        self.stream_output = stream_output
        self.last_turn: Optional[ModelTurn] = None

    @classmethod
    def load(
        cls,
        model_id: str = MODEL_ID,
        *,
        system_prompt: str = COPYCAT_SYSTEM_PROMPT,
        max_new_tokens: int = 8_192,
        stream_output: bool = True,
    ) -> "Gemma4Backend":
        import torch
        from transformers import AutoModelForMultimodalLM, AutoProcessor

        processor = AutoProcessor.from_pretrained(model_id)

        model = AutoModelForMultimodalLM.from_pretrained(
            model_id,
            dtype=torch.float16,
            device_map="auto",
            attn_implementation="sdpa",
        )
        model.eval()

        return cls(
            model,
            processor,
            system_prompt=system_prompt,
            max_new_tokens=max_new_tokens,
            stream_output=stream_output,
        )

    def generate(
        self,
        *,
        prompt: str,
        stack: str,
        module_catalog: str,
    ) -> ModelTurn:
        import torch
        from transformers import TextStreamer

        visible_stack = stack if stack else "(empty)"

        messages = [
            {
                "role": "system",
                "content": (
                    f"{self.system_prompt}\n\n"
                    "AVAILABLE MODULE WORDS\n"
                    f"{module_catalog}"
                ),
            },
            {
                "role": "user",
                "content": (
                    "Current data stack (bottom to top):\n"
                    f"{visible_stack}\n\n"
                    "Task:\n"
                    f"{prompt}"
                ),
            },
        ]

        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            add_generation_prompt=True,
            enable_thinking=True,
        ).to(self.model.device)

        input_len = inputs["input_ids"].shape[-1]
        streamer = None

        if self.stream_output:
            tokenizer = getattr(self.processor, "tokenizer", self.processor)
            streamer = TextStreamer(
                tokenizer,
                skip_prompt=True,
                skip_special_tokens=False,
            )
            print("\n--- Gemma live stream: thinking followed by final answer ---")

        with torch.inference_mode():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=True,
                temperature=1.0,
                top_p=0.95,
                top_k=64,
                streamer=streamer,
            )

        if self.stream_output:
            print("--- End Gemma live stream ---")

        raw = self.processor.decode(
            outputs[0][input_len:],
            skip_special_tokens=False,
        )

        try:
            parsed = self.processor.parse_response(
                raw,
                prefix=inputs["input_ids"],
            )
        except TypeError:
            parsed = self.processor.parse_response(raw)

        answer = parsed.get("content", "")
        thinking = parsed.get("thinking")

        if not isinstance(answer, str):
            answer = str(answer)

        if thinking is not None and not isinstance(thinking, str):
            thinking = str(thinking)

        turn = ModelTurn(
            answer=answer,
            thinking=thinking,
            raw=raw,
            streamed=self.stream_output,
            prompt_tokens=input_len,
        )
        self.last_turn = turn
        return turn
