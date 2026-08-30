from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .object import Object


class CopycatError(Exception):
    """Base class for errors intended for humans and future synthesis loops."""


@dataclass
class ParseError(CopycatError):
    message: str
    source: str
    source_name: str
    line: int
    column: int
    expected: tuple[str, ...] = ()

    def __str__(self) -> str:
        lines = self.source.splitlines() or [""]
        line_index = min(max(self.line - 1, 0), len(lines) - 1)
        text = lines[line_index]
        caret = " " * max(0, self.column - 1) + "^"

        parts = [
            f"Parse error in {self.source_name} at line {self.line}, column {self.column}:",
            self.message,
            "",
            text,
            caret,
        ]
        if self.expected:
            parts.extend(["", "Expected: " + ", ".join(self.expected)])
        return "\n".join(parts)


@dataclass
class EvaluationError(CopycatError):
    message: str
    operation: Optional[Object] = None

    def __str__(self) -> str:
        if self.operation is None or self.operation.span is None:
            return f"Evaluation error: {self.message}"

        span = self.operation.span
        return (
            f"Evaluation error in {span.source_name} at line {span.line}, "
            f"column {span.column}:\n{self.message}\n\n{span.context()}"
        )


@dataclass
class ModelProtocolError(CopycatError):
    answer: str
    message: str

    def __str__(self) -> str:
        return (
            "Model protocol error: "
            + self.message
            + "\n\nModel answer:\n"
            + self.answer
        )


@dataclass
class ModelReportedError(CopycatError):
    prompt: str
    message: str

    def __str__(self) -> str:
        return f"Model reported an error for {{{self.prompt}}}: {self.message}"


@dataclass
class GeneratedCodeError(CopycatError):
    prompt: str
    code: str
    cause: Exception

    def __str__(self) -> str:
        return (
            f"Generated code for {{{self.prompt}}} could not be used.\n\n"
            f"Generated code:\n{self.code}\n\n"
            f"Cause:\n{self.cause}"
        )
