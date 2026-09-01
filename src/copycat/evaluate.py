from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

from .error import (
    CopycatError,
    EvaluationError,
    GeneratedCodeError,
    ModelReportedError,
)
from .model import ModelBackend, ModelError, parse_model_answer
from .module import Module
from .read import read
from .term import Annotate, Latent, Number, Quote, Sequence, String, Term, Word


@dataclass(frozen=True)
class EvaluationSnapshot:
    """An immutable view of an evaluation at one point in time."""

    steps: int
    gas: int
    hand: Optional[Term]
    sink: tuple[Term, ...]
    data: tuple[Term, ...]
    code: tuple[Term, ...]
    is_active: bool
    value: Term


Primitive = Callable[["Evaluate"], None]


class Evaluate:
    """A controllable Copycat evaluation."""

    __slots__ = (
        "_code",
        "_completion_reported",
        "_data",
        "_gas",
        "_hand",
        "_model_backend",
        "_module",
        "_primitives",
        "_program",
        "_sink",
        "_steps",
        "_verbose",
    )

    def __init__(
        self,
        program: Term,
        module: Optional[Module] = None,
        *,
        gas: int = 1_000_000,
        model_backend: Optional[ModelBackend] = None,
        verbose: bool = True,
    ) -> None:
        active_module = module if module is not None else Module()
        if not isinstance(active_module, Module):
            raise TypeError("Evaluate expects module to be a Module instance.")

        self._program = program
        self._module = active_module
        self._model_backend = model_backend
        self._verbose = verbose
        self._gas = gas
        self._steps = 0
        self._hand: Optional[Term] = None
        self._code = [program]
        self._data: list[Term] = []
        self._sink: list[Term] = []
        self._completion_reported = False
        self._primitives: dict[str, Primitive] = {
            "a": op_app,
            "b": op_abstract,
            "c": op_catenate,
            "d": op_copy,
            "e": op_drop,
            "f": op_swap,
            "r": op_mark,
            "s": op_jump,
        }

        if self._verbose:
            print("=== Copycat evaluation ===")
            print(f"Program: {program or '(empty program)'}")
            print(f"Initial gas: {gas}")

    @property
    def module(self) -> Module:
        return self._module

    @property
    def model_backend(self) -> Optional[ModelBackend]:
        return self._model_backend

    @property
    def verbose(self) -> bool:
        return self._verbose

    @property
    def gas(self) -> int:
        return self._gas

    @property
    def steps(self) -> int:
        return self._steps

    @property
    def hand(self) -> Optional[Term]:
        return self._hand

    @property
    def sink(self) -> tuple[Term, ...]:
        return tuple(self._sink)

    @property
    def data(self) -> tuple[Term, ...]:
        """The data stack, ordered bottom-to-top."""
        return tuple(self._data)

    @property
    def code(self) -> tuple[Term, ...]:
        """The remaining code, ordered next-to-execute first."""
        return tuple(reversed(self._code))

    @property
    def value(self) -> Term:
        return Sequence(tuple(self._sink + self._data + list(reversed(self._code))))

    @property
    def is_active(self) -> bool:
        return self._gas > 0 and bool(self._code)

    @property
    def next_term(self) -> Optional[Term]:
        return self._code[-1] if self._code else None

    def snapshot(self) -> EvaluationSnapshot:
        return EvaluationSnapshot(
            steps=self._steps,
            gas=self._gas,
            hand=self._hand,
            sink=self.sink,
            data=self.data,
            code=self.code,
            is_active=self.is_active,
            value=self.value,
        )

    def _tick(self) -> Term:
        self._gas -= 1
        self._steps += 1
        self._hand = self._code.pop()
        return self._hand

    def _trace(self, phase: str, term: Term) -> None:
        if not self._verbose:
            return

        snapshot = self.snapshot()
        sink = str(Sequence(snapshot.sink)) or "(empty)"
        data = str(Sequence(snapshot.data)) or "(empty)"
        remaining = str(Sequence(snapshot.code)) or "(empty)"
        print(
            f"\n[Copycat step {snapshot.steps} — {phase}] "
            f"{type(term).__name__}: {term}"
        )
        print(f"  gas remaining: {snapshot.gas}")
        print(f"  sink: {sink}")
        print(f"  data (bottom to top): {data}")
        print(f"  remaining code: {remaining}")

    def _report_completion(self) -> None:
        if self._completion_reported:
            return
        self._completion_reported = True
        if self._verbose:
            print("\n=== Evaluation complete ===")
            print(f"Result: {self.value or '(empty)'}")

    def _dispatch(self, term: Term) -> None:
        match term:
            case Word(name):
                if binding := self._primitives.get(name):
                    binding(self)
                elif name in self._module:
                    self._code.append(self._module.parsed(name))
                else:
                    self._residualize(
                        f"Undefined word {name!r}.",
                        stop=True,
                    )

            case Annotate(name):
                if name == "eq":
                    if len(self._data) < 2:
                        raise EvaluationError(
                            "@eq needs 2 values on the data stack.",
                            term,
                        )
                    left, right = self._data[-2:]
                    if left != right:
                        raise EvaluationError(
                            f"@eq assertion failed: {left} != {right}.",
                            term,
                        )

            case Quote(_) | Number(_) | String(_):
                self._data.append(term)

            case Sequence(body):
                self._code.extend(reversed(body))

            case Latent(prompt):
                _run_model_effect(self, term, prompt)

            case _:
                raise EvaluationError(
                    f"Unknown runtime object {term!r}.",
                    term,
                )

    def step(self) -> EvaluationSnapshot:
        if not self.is_active:
            raise StopIteration("Evaluation is not active.")

        term = self._tick()
        self._trace("before", term)
        self._dispatch(term)
        self._trace("after", term)

        if not self.is_active:
            self._report_completion()

        return self.snapshot()

    def run(self) -> Term:
        while self.is_active:
            self.step()
        self._report_completion()
        return self.value

    def _thunk(self) -> None:
        self._sink.extend(self._data)
        self._data = []
        if self._hand is not None:
            self._sink.append(self._hand)

    def _stop(self) -> None:
        self._thunk()
        self._gas = 0

    def _residualize(self, message: str, *, stop: bool = False) -> None:
        if self._verbose:
            action = "stop" if stop else "residualize"
            print(f"\n[Copycat {action}] {message}")

        if stop:
            self._stop()
        else:
            self._thunk()


def stack_string(data: list[Term]) -> str:
    """Representation shown to the model, ordered bottom-to-top."""
    return str(Sequence(tuple(data)))


def op_copy(state: Evaluate) -> None:
    if not state._data:
        state._residualize("d needs 1 value on the data stack, but the stack is empty.")
        return
    state._data.append(state._data[-1])


def op_drop(state: Evaluate) -> None:
    if not state._data:
        state._residualize("e needs 1 value on the data stack, but the stack is empty.")
        return
    state._data.pop()


def op_swap(state: Evaluate) -> None:
    if len(state._data) < 2:
        state._residualize(
            f"f needs 2 values on the data stack, but found {len(state._data)}."
        )
        return
    state._data[-2], state._data[-1] = state._data[-1], state._data[-2]


def op_abstract(state: Evaluate) -> None:
    if not state._data:
        state._residualize("b needs 1 value on the data stack, but the stack is empty.")
        return
    state._data[-1] = Quote(state._data[-1])


def op_app(state: Evaluate) -> None:
    if not state._data:
        state._residualize(
            "a needs a quotation on top of the data stack.",
            stop=True,
        )
        return

    block = state._data[-1]

    if not isinstance(block, Quote):
        state._residualize(
            f"a expected a quotation on top of the stack, but found {block}.",
            stop=True,
        )
        return

    state._data.pop()
    state._code.append(block.body)


def _catenate_objects(first: Term, second: Term) -> Sequence:
    first_items = first.body if isinstance(first, Sequence) else (first,)
    second_items = second.body if isinstance(second, Sequence) else (second,)
    return Sequence(tuple(first_items) + tuple(second_items))


def op_catenate(state: Evaluate) -> None:
    if len(state._data) < 2:
        state._residualize(
            f"c needs 2 quotations on the data stack, but found {len(state._data)}."
        )
        return

    first, second = state._data[-2], state._data[-1]

    if not isinstance(first, Quote) or not isinstance(second, Quote):
        state._residualize(
            f"c expected two quotations, but found {first} and {second}."
        )
        return

    state._data[-2:] = [Quote(_catenate_objects(first.body, second.body))]


def op_jump(state: Evaluate) -> None:
    if not state._data:
        state._residualize(
            "s needs a handler quotation on top of the data stack.",
            stop=True,
        )
        return

    handler = state._data[-1]

    if not isinstance(handler, Quote):
        state._residualize(
            f"s expected a handler quotation, but found {handler}.",
            stop=True,
        )
        return

    buffer: list[Term] = []
    index = 1
    mark_found = False

    while index <= len(state._code):
        point = state._code[-index]

        if isinstance(point, Word) and point.name == "r":
            mark_found = True
            break

        buffer.append(point)
        index += 1

    if not mark_found:
        state._residualize(
            "s could not find a matching r in the continuation.",
            stop=True,
        )
        return

    continuation = Quote(Sequence(tuple(buffer)))
    state._code = state._code[:-index]
    state._data.pop()
    state._data.append(continuation)
    state._code.append(handler.body)


def op_mark(state: Evaluate) -> None:
    state._thunk()


def _run_model_effect(
    state: Evaluate,
    term: Latent,
    prompt: str,
) -> None:
    if state.model_backend is None:
        raise EvaluationError(
            "This program performs a model effect, "
            "but no model backend was supplied.",
            term,
        )

    visible_stack = stack_string(state._data)
    if state.verbose:
        print("\n=== Model effect ===")
        print(f"Task: {prompt}")
        print(f"Stack (bottom to top): {visible_stack or '(empty)'}")

    turn = state.model_backend.generate(
        prompt=prompt,
        stack=visible_stack,
        module_catalog=str(state.module),
    )

    if state.verbose and turn.prompt_tokens is not None:
        print(f"Prompt tokens: {turn.prompt_tokens:,}")

    if state.verbose and turn.thinking and not turn.streamed:
        print("\n--- Model thinking ---")
        print(turn.thinking)

    if state.verbose:
        print("\n--- Model final answer ---")
        print(turn.answer)

    reply = parse_model_answer(turn.answer)

    if isinstance(reply, ModelError):
        raise ModelReportedError(prompt, reply.message)

    if state.verbose:
        print("\n--- Generated Copycat ---")
        print(reply.code or "(empty program)")

    try:
        generated = read(reply.code, source_name="<model output>")
    except CopycatError as exc:
        raise GeneratedCodeError(
            prompt,
            reply.code,
            exc,
        ) from exc

    state._code.append(generated)


def evaluate(
    program: Term,
    module: Optional[Module] = None,
    *,
    gas: int = 1_000_000,
    model_backend: Optional[ModelBackend] = None,
    verbose: bool = True,
) -> Term:
    return Evaluate(
        program,
        module=module,
        gas=gas,
        model_backend=model_backend,
        verbose=verbose,
    ).run()


def normalize(
    source: str,
    *,
    gas: int = 1_000_000,
    model_backend: Optional[ModelBackend] = None,
    module: Optional[Module] = None,
    verbose: bool = True,
) -> str:
    return str(
        evaluate(
            read(source),
            module=module,
            gas=gas,
            model_backend=model_backend,
            verbose=verbose,
        )
    )
