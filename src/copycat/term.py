from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Optional


@dataclass(frozen=True)
class Span:
    source: str
    source_name: str
    start: int
    end: int
    line: int
    column: int
    end_line: int
    end_column: int

    @classmethod
    def from_token(cls, token, source: str, source_name: str) -> "Span":
        return cls(
            source=source,
            source_name=source_name,
            start=token.start_pos,
            end=token.end_pos,
            line=token.line,
            column=token.column,
            end_line=token.end_line,
            end_column=token.end_column,
        )

    def context(self) -> str:
        lines = self.source.splitlines() or [""]
        line_index = min(max(self.line - 1, 0), len(lines) - 1)
        text = lines[line_index]
        width = max(1, (self.end - self.start) if self.line == self.end_line else 1)
        width = min(width, max(1, len(text) - self.column + 2))
        return f"{text}\n{' ' * (self.column - 1)}{'^' * width}"


@dataclass(frozen=True, kw_only=True)
class Term:
    span: Optional[Span] = field(default=None, compare=False, repr=False)


@dataclass(frozen=True)
class Word(Term):
    name: str

    def __str__(self) -> str:
        return self.name


@dataclass(frozen=True)
class Annotation(Term):
    name: str

    def __str__(self) -> str:
        return f"@{self.name}"


@dataclass(frozen=True)
class Quote(Term):
    body: Term

    def __str__(self) -> str:
        return f"[{self.body}]"


@dataclass(frozen=True)
class Sequence(Term):
    body: tuple[Term, ...]

    def __str__(self) -> str:
        return " ".join(str(child) for child in self.body)


@dataclass(frozen=True)
class String(Term):
    value: str

    def __str__(self) -> str:
        return json.dumps(self.value, ensure_ascii=False)


@dataclass(frozen=True)
class Number(Term):
    value: int

    def __str__(self) -> str:
        return str(self.value)


@dataclass(frozen=True)
class Model(Term):
    prompt: str

    def __str__(self) -> str:
        return "{" + self.prompt + "}"
