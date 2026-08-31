from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping, Optional

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


Primitive = Callable[["State"], None]


@dataclass
class State:
    gas: int
    code: list[Term]
    data: list[Term]
    sink: list[Term]
    module: Module
    primitives: Mapping[str, Primitive]
    model_backend: Optional[ModelBackend]
    verbose: bool
    hand: Optional[Term] = None
    step: int = 0

    @property
    def value(self) -> Term:
        return Sequence(tuple(self.sink + self.data + list(reversed(self.code))))

    @property
    def is_active(self) -> bool:
        return self.gas > 0 and bool(self.code)

    def tick(self) -> Term:
        self.gas -= 1
        self.step += 1
        self.hand = self.code.pop()
        return self.hand

    def trace(self, phase: str, term: Term) -> None:
        if not self.verbose:
            return

        sink = str(Sequence(tuple(self.sink))) or "(empty)"
        data = str(Sequence(tuple(self.data))) or "(empty)"
        remaining = str(Sequence(tuple(reversed(self.code)))) or "(empty)"
        print(
            f"\n[Copycat step {self.step} — {phase}] "
            f"{type(term).__name__}: {term}"
        )
        print(f"  gas remaining: {self.gas}")
        print(f"  sink: {sink}")
        print(f"  data (bottom to top): {data}")
        print(f"  remaining code: {remaining}")

    def thunk(self) -> None:
        self.sink.extend(self.data)
        self.data = []
        if self.hand is not None:
            self.sink.append(self.hand)

    def stop(self) -> None:
        self.thunk()
        self.gas = 0

    def residualize(self, message: str, *, stop: bool = False) -> None:
        if self.verbose:
            action = "stop" if stop else "residualize"
            print(f"\n[Copycat {action}] {message}")

        if stop:
            self.stop()
        else:
            self.thunk()


def stack_string(data: list[Term]) -> str:
    """Representation shown to the model, ordered bottom-to-top."""
    return str(Sequence(tuple(data)))


def op_copy(state: State) -> None:
    if not state.data:
        state.residualize("d needs 1 value on the data stack, but the stack is empty.")
        return
    state.data.append(state.data[-1])


def op_drop(state: State) -> None:
    if not state.data:
        state.residualize("e needs 1 value on the data stack, but the stack is empty.")
        return
    state.data.pop()


def op_swap(state: State) -> None:
    if len(state.data) < 2:
        state.residualize(
            f"f needs 2 values on the data stack, but found {len(state.data)}."
        )
        return
    state.data[-2], state.data[-1] = state.data[-1], state.data[-2]


def op_abstract(state: State) -> None:
    if not state.data:
        state.residualize("b needs 1 value on the data stack, but the stack is empty.")
        return
    state.data[-1] = Quote(state.data[-1])


def op_app(state: State) -> None:
    if not state.data:
        state.residualize(
            "a needs a quotation on top of the data stack.",
            stop=True,
        )
        return

    block = state.data[-1]

    if not isinstance(block, Quote):
        state.residualize(
            f"a expected a quotation on top of the stack, but found {block}.",
            stop=True,
        )
        return

    state.data.pop()
    state.code.append(block.body)


def _catenate_objects(first: Term, second: Term) -> Sequence:
    first_items = first.body if isinstance(first, Sequence) else (first,)
    second_items = second.body if isinstance(second, Sequence) else (second,)
    return Sequence(tuple(first_items) + tuple(second_items))


def op_catenate(state: State) -> None:
    if len(state.data) < 2:
        state.residualize(
            f"c needs 2 quotations on the data stack, but found {len(state.data)}."
        )
        return

    first, second = state.data[-2], state.data[-1]

    if not isinstance(first, Quote) or not isinstance(second, Quote):
        state.residualize(
            f"c expected two quotations, but found {first} and {second}."
        )
        return

    state.data[-2:] = [Quote(_catenate_objects(first.body, second.body))]


def op_jump(state: State) -> None:
    if not state.data:
        state.residualize(
            "s needs a handler quotation on top of the data stack.",
            stop=True,
        )
        return

    handler = state.data[-1]

    if not isinstance(handler, Quote):
        state.residualize(
            f"s expected a handler quotation, but found {handler}.",
            stop=True,
        )
        return

    buffer: list[Term] = []
    index = 1
    mark_found = False

    while index <= len(state.code):
        point = state.code[-index]

        if isinstance(point, Word) and point.name == "r":
            mark_found = True
            break

        buffer.append(point)
        index += 1

    if not mark_found:
        state.residualize(
            "s could not find a matching r in the continuation.",
            stop=True,
        )
        return

    continuation = Quote(Sequence(tuple(buffer)))
    state.code = state.code[:-index]
    state.data.pop()
    state.data.append(continuation)
    state.code.append(handler.body)


def op_mark(state: State) -> None:
    state.thunk()


def _run_model_effect(
    state: State,
    term: Latent,
    prompt: str,
) -> None:
    if state.model_backend is None:
        raise EvaluationError(
            "This program performs a model effect, "
            "but no model backend was supplied.",
            term,
        )

    visible_stack = stack_string(state.data)
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

    state.code.append(generated)


def evaluate(
    program: Term,
    module: Optional[Module] = None,
    *,
    gas: int = 1_000_000,
    model_backend: Optional[ModelBackend] = None,
    verbose: bool = True,
) -> Term:
    primitives: dict[str, Primitive] = {
        "a": op_app,
        "b": op_abstract,
        "c": op_catenate,
        "d": op_copy,
        "e": op_drop,
        "f": op_swap,
        "r": op_mark,
        "s": op_jump,
    }

    active_module = module if module is not None else Module()
    if not isinstance(active_module, Module):
        raise TypeError("evaluate expects module to be a Module instance.")

    state = State(
        code=[program],
        data=[],
        sink=[],
        gas=gas,
        module=active_module,
        primitives=primitives,
        model_backend=model_backend,
        verbose=verbose,
    )

    if verbose:
        print("=== Copycat evaluation ===")
        print(f"Program: {program or '(empty program)'}")
        print(f"Initial gas: {gas}")

    while state.is_active:
        term = state.tick()
        state.trace("before", term)

        match term:
            case Word(name):
                if binding := state.primitives.get(name):
                    binding(state)
                elif name in state.module:
                    state.code.append(state.module.parsed(name))
                else:
                    state.residualize(
                        f"Undefined word {name!r}.",
                        stop=True,
                    )

            case Annotate(name):
                if name == "eq":
                    if len(state.data) < 2:
                        raise EvaluationError(
                            "@eq needs 2 values on the data stack.",
                            term,
                        )
                    left, right = state.data[-2:]
                    if left != right:
                        raise EvaluationError(
                            f"@eq assertion failed: {left} != {right}.",
                            term,
                        )

            case Quote(_) | Number(_) | String(_):
                state.data.append(term)

            case Sequence(body):
                state.code.extend(reversed(body))

            case Latent(prompt):
                _run_model_effect(state, term, prompt)

            case _:
                raise EvaluationError(
                    f"Unknown runtime object {term!r}.",
                    term,
                )

        state.trace("after", term)

    if verbose:
        print("\n=== Evaluation complete ===")
        print(f"Result: {state.value or '(empty)'}")

    return state.value


def run(
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
