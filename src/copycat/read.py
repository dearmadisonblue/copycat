from __future__ import annotations

import json
import re

from lark import Lark, Transformer
from lark.exceptions import UnexpectedCharacters, UnexpectedInput, UnexpectedToken

from .error import ParseError
from .term import Annotate, Latent, Number, Quote, Sequence, Span, String, Word


# Single-letter names are reserved for the kernel primitives. User-defined words
# are lowercase kebab-case and contain at least two characters. Annotation names
# use the same lowercase kebab-case spelling but do not reserve single letters.
_WORD_PATTERN = r"(?:[a-z](?=[a-z0-9-])[a-z0-9]*(?:-[a-z0-9]+)*|[abcdefrs])"
_USER_WORD_RE = re.compile(r"[a-z](?=[a-z0-9-])[a-z0-9]*(?:-[a-z0-9]+)*\Z")
_ANNOTATION_PATTERN = r"@[a-z][a-z0-9]*(?:-[a-z0-9]+)*"

_GRAMMAR = rf"""
start: term*

?term: quotation
     | MODEL           -> model
     | ESCAPED_STRING  -> string
     | NUMBER          -> number
     | ANNOTATION      -> annotation
     | WORD            -> word

quotation: "[" term* "]"

MODEL.3: /\{{[^}}]*\}}/s
NUMBER: /0|[1-9][0-9]*/
ANNOTATION.2: /{_ANNOTATION_PATTERN}/
WORD: /{_WORD_PATTERN}/

%import common.ESCAPED_STRING
%import common.WS
%ignore WS
"""

_PARSER = Lark(_GRAMMAR, parser="lalr", propagate_positions=True)


def _is_user_word_name(name: str) -> bool:
    return _USER_WORD_RE.fullmatch(name) is not None


class _BuildAST(Transformer):
    def __init__(self, source: str, source_name: str):
        super().__init__()
        self.source = source
        self.source_name = source_name

    def _span(self, token) -> Span:
        return Span.from_token(token, self.source, self.source_name)

    def word(self, children):
        (token,) = children
        return Word(str(token), span=self._span(token))

    def annotation(self, children):
        (token,) = children
        return Annotate(str(token)[1:], span=self._span(token))

    def number(self, children):
        (token,) = children
        return Number(int(str(token)), span=self._span(token))

    def string(self, children):
        (token,) = children
        return String(json.loads(str(token)), span=self._span(token))

    def model(self, children):
        (token,) = children
        text = str(token)
        return Latent(text[1:-1], span=self._span(token))

    def quotation(self, children):
        return Quote(Sequence(tuple(children)))

    def start(self, children):
        return Sequence(tuple(children))


def _eof_location(source: str) -> tuple[int, int]:
    lines = source.splitlines()
    if not lines:
        return 1, 1
    return len(lines), len(lines[-1]) + 1


def _friendly_parse_error(
    exc: UnexpectedInput,
    source: str,
    source_name: str,
) -> ParseError:
    raw_expected = tuple(
        sorted(
            getattr(exc, "expected", ())
            or getattr(exc, "allowed", ())
            or ()
        )
    )

    terminal_names = {
        "ANNOTATION": "annotation @name",
        "ESCAPED_STRING": "string",
        "LSQB": "'['",
        "RSQB": "']'",
        "MODEL": "model invocation {...}",
        "NUMBER": "natural number",
        "WORD": "word",
        "$END": "end of input",
    }
    expected = tuple(terminal_names.get(name, name) for name in raw_expected)

    if isinstance(exc, UnexpectedCharacters):
        pos = exc.pos_in_stream
        char = source[pos : pos + 1]

        if char == "{":
            message = "Unclosed model invocation. Add a matching '}'."
        elif char == '"':
            message = "Invalid or unterminated string literal."
        elif char == "@":
            message = "Invalid annotation. Use @ followed by a lowercase kebab-case name."
        elif char and char.isalpha() and char.islower():
            message = (
                f"Unexpected word character {char!r}. Single-letter names are "
                "reserved for the primitives a-f, r, and s."
            )
        else:
            shown = repr(char) if char else "end of input"
            message = f"Unexpected character {shown}."

        return ParseError(
            message, source, source_name, exc.line, exc.column, expected
        )

    if isinstance(exc, UnexpectedToken):
        token = exc.token

        if token.type == "$END":
            line, column = _eof_location(source)
            if "RSQB" in raw_expected:
                message = "Unclosed quotation. Add a matching ']'."
            else:
                message = "Unexpected end of input."

            return ParseError(
                message, source, source_name, line, column, expected
            )

        if token.type == "RSQB":
            message = "Unexpected ']'. There is no matching '[' for this bracket."
        else:
            message = f"Unexpected token {str(token)!r}."

        return ParseError(
            message, source, source_name, exc.line, exc.column, expected
        )

    return ParseError(
        "Could not parse input.",
        source,
        source_name,
        exc.line,
        exc.column,
        expected,
    )


def read(code: str, *, source_name: str = "<input>") -> Sequence:
    """Parse Copycat source into an AST."""
    try:
        tree = _PARSER.parse(code)
    except UnexpectedInput as exc:
        raise _friendly_parse_error(exc, code, source_name) from None

    return _BuildAST(code, source_name).transform(tree)
