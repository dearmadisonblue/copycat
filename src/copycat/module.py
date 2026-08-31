from __future__ import annotations

import warnings
from collections.abc import Iterator, Mapping
from os import PathLike
from types import MappingProxyType
from zipfile import ZIP_DEFLATED, ZipFile

from .term import Sequence, String
from .read import _is_user_word_name, read


__all__ = [
    "Module",
    "ModuleDocumentationWarning",
]

_Path = str | PathLike[str]
_SMOKE_TEST_GAS = 100_000
_TEST_PREFIX = "test-"
_DOC_SUFFIX = "-doc"


class ModuleDocumentationWarning(UserWarning):
    """A module word has missing or unusable documentation."""


def _is_test_word(name: str) -> bool:
    return name.startswith(_TEST_PREFIX)


def _is_doc_word(name: str) -> bool:
    return name.endswith(_DOC_SUFFIX)


def _validate_sources(sources: Mapping[str, str]) -> None:
    for name, source in sources.items():
        if not isinstance(name, str):
            raise TypeError("Module word names must be strings.")
        if not _is_user_word_name(name):
            raise ValueError(
                f"Invalid module word name {name!r}. User-defined words must "
                "be lowercase kebab-case and longer than one character."
            )
        if not isinstance(source, str):
            raise TypeError(f"Module body for {name!r} must be source text.")


def _documentation_text(program: Sequence) -> str | None:
    if len(program.body) == 1 and isinstance(program.body[0], String):
        return program.body[0].value
    return None


def _format_catalog(
    ordinary_words: list[str],
    documentation: Mapping[str, str],
) -> str:
    if not ordinary_words:
        return "(none)"

    entries = []
    for name in ordinary_words:
        description = documentation.get(name, "(undocumented)")
        entries.append(f"{name}\n  {description}")
    return "\n\n".join(entries)


class Module(Mapping[str, str]):
    """An immutable module with cached syntax and documentation."""

    __slots__ = ("_catalog", "_documentation", "_parsed", "_sources")

    def __setattr__(self, name: str, value: object) -> None:
        if hasattr(self, name):
            raise AttributeError("Module instances are immutable.")
        object.__setattr__(self, name, value)

    def __init__(self, sources: Mapping[str, str] | None = None) -> None:
        copied_sources = dict(sources or {})
        _validate_sources(copied_sources)

        parsed = {
            name: read(source, source_name=f"<word {name}>")
            for name, source in copied_sources.items()
        }
        ordinary_words = sorted(
            name
            for name in copied_sources
            if not _is_test_word(name) and not _is_doc_word(name)
        )

        documentation: dict[str, str] = {}
        for name in ordinary_words:
            doc_name = name + _DOC_SUFFIX
            doc_program = parsed.get(doc_name)
            if doc_program is None:
                warnings.warn(
                    f"Module word {name!r} is undocumented; expected {doc_name!r}.",
                    ModuleDocumentationWarning,
                    stacklevel=2,
                )
                continue

            text = _documentation_text(doc_program)
            if text is None:
                warnings.warn(
                    f"Documentation word {doc_name!r} must contain exactly one string; "
                    f"{name!r} will be presented as undocumented.",
                    ModuleDocumentationWarning,
                    stacklevel=2,
                )
                continue
            documentation[name] = text

        for doc_name in sorted(
            name
            for name in copied_sources
            if _is_doc_word(name) and not _is_test_word(name)
        ):
            documented_name = doc_name[: -len(_DOC_SUFFIX)]
            if documented_name not in copied_sources:
                warnings.warn(
                    f"Documentation word {doc_name!r} has no corresponding "
                    "module word.",
                    ModuleDocumentationWarning,
                    stacklevel=2,
                )

        self._sources = MappingProxyType(copied_sources)
        self._parsed = MappingProxyType(parsed)
        self._documentation = MappingProxyType(documentation)
        self._catalog = _format_catalog(ordinary_words, documentation)

        _run_smoke_tests(self)

    def __getitem__(self, name: str) -> str:
        return self._sources[name]

    def __iter__(self) -> Iterator[str]:
        return iter(self._sources)

    def __len__(self) -> int:
        return len(self._sources)

    def __repr__(self) -> str:
        return f"Module({dict(self._sources)!r})"

    def __str__(self) -> str:
        return self._catalog

    def parsed(self, name: str) -> Sequence:
        """Return the cached syntax object for a module word."""
        return self._parsed[name]

    def documentation(self, name: str) -> str | None:
        """Return decoded documentation for a word, when available."""
        return self._documentation.get(name)

    @staticmethod
    def load(path: _Path) -> Module:
        """Load, cache, document-check, and smoke-test a .module ZIP archive."""
        sources: dict[str, str] = {}

        with ZipFile(path, "r") as archive:
            for info in archive.infolist():
                name = info.filename
                if info.is_dir() or "/" in name or "\\" in name:
                    raise ValueError(
                        f"Invalid module entry {name!r}. "
                        "Module entries must be flat files."
                    )
                if name in sources:
                    raise ValueError(f"Duplicate module entry {name!r}.")

                try:
                    source = archive.read(info).decode("utf-8")
                except UnicodeDecodeError as exc:
                    raise ValueError(
                        f"Module entry {name!r} is not valid UTF-8 source text."
                    ) from exc

                sources[name] = source

        return Module(sources)

    def save(self, path: _Path) -> None:
        """Serialize this module as a ZIP archive of extensionless sources."""
        with ZipFile(path, "w", compression=ZIP_DEFLATED) as archive:
            for name in sorted(self):
                archive.writestr(name, self[name].encode("utf-8"))


def _run_smoke_tests(module: Module) -> None:
    tests = sorted(name for name in module if _is_test_word(name))
    if not tests:
        return

    # Import lazily to avoid a module/evaluator import cycle.
    from .evaluate import evaluate

    for name in tests:
        evaluate(
            module.parsed(name),
            module=module,
            gas=_SMOKE_TEST_GAS,
            verbose=False,
        )


EMPTY_MODULE = Module()
