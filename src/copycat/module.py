from __future__ import annotations

import warnings
from collections.abc import Iterator, Mapping, MutableMapping
from os import PathLike
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
        _validate_source(name, source)


def _validate_source(name: str, source: str) -> None:
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


def _is_ordinary_word(name: str) -> bool:
    return not _is_test_word(name) and not _is_doc_word(name)


def _ordinary_words(sources: Mapping[str, str]) -> list[str]:
    return sorted(name for name in sources if _is_ordinary_word(name))


def _refresh_word_documentation(
    name: str,
    sources: Mapping[str, str],
    parsed: Mapping[str, Sequence],
    documentation: dict[str, str],
    *,
    warning_stacklevel: int,
) -> None:
    if name not in sources or not _is_ordinary_word(name):
        documentation.pop(name, None)
        return

    doc_name = name + _DOC_SUFFIX
    doc_program = parsed.get(doc_name)
    if doc_program is None:
        documentation.pop(name, None)
        warnings.warn(
            f"Module word {name!r} is undocumented; expected {doc_name!r}.",
            ModuleDocumentationWarning,
            stacklevel=warning_stacklevel,
        )
        return

    text = _documentation_text(doc_program)
    if text is None:
        documentation.pop(name, None)
        warnings.warn(
            f"Documentation word {doc_name!r} must contain exactly one string; "
            f"{name!r} will be presented as undocumented.",
            ModuleDocumentationWarning,
            stacklevel=warning_stacklevel,
        )
        return

    documentation[name] = text


def _build_documentation(
    sources: Mapping[str, str],
    parsed: Mapping[str, Sequence],
) -> dict[str, str]:
    documentation: dict[str, str] = {}

    for name in _ordinary_words(sources):
        _refresh_word_documentation(
            name,
            sources,
            parsed,
            documentation,
            warning_stacklevel=4,
        )

    for doc_name in sorted(
        name
        for name in sources
        if _is_doc_word(name) and not _is_test_word(name)
    ):
        documented_name = doc_name[: -len(_DOC_SUFFIX)]
        if documented_name not in sources:
            warnings.warn(
                f"Documentation word {doc_name!r} has no corresponding "
                "module word.",
                ModuleDocumentationWarning,
                stacklevel=3,
            )

    return documentation


def _refresh_documentation(
    changed_names: set[str],
    sources: Mapping[str, str],
    parsed: Mapping[str, Sequence],
    documentation: dict[str, str],
) -> None:
    affected_words: set[str] = set()

    for name in changed_names:
        if _is_test_word(name):
            continue
        if _is_doc_word(name):
            documented_name = name[: -len(_DOC_SUFFIX)]
            if _is_ordinary_word(documented_name):
                affected_words.add(documented_name)
            if name in sources and documented_name not in sources:
                warnings.warn(
                    f"Documentation word {name!r} has no corresponding "
                    "module word.",
                    ModuleDocumentationWarning,
                    stacklevel=4,
                )
        else:
            affected_words.add(name)
            doc_name = name + _DOC_SUFFIX
            if name not in sources and doc_name in sources:
                warnings.warn(
                    f"Documentation word {doc_name!r} has no corresponding "
                    "module word.",
                    ModuleDocumentationWarning,
                    stacklevel=4,
                )

    for name in sorted(affected_words):
        _refresh_word_documentation(
            name,
            sources,
            parsed,
            documentation,
            warning_stacklevel=5,
        )


class Module(MutableMapping[str, str]):
    """A mutable module with cached syntax and documentation."""

    __slots__ = ("_catalog", "_documentation", "_parsed", "_sources")

    def __init__(self, sources: Mapping[str, str] | None = None) -> None:
        copied_sources = dict(sources or {})
        _validate_sources(copied_sources)

        parsed = {
            name: read(source, source_name=f"<word {name}>")
            for name, source in copied_sources.items()
        }
        documentation = _build_documentation(copied_sources, parsed)

        self._sources = copied_sources
        self._parsed = parsed
        self._documentation = documentation
        self._catalog = _format_catalog(
            _ordinary_words(copied_sources),
            documentation,
        )

        _run_smoke_tests(self)

    @classmethod
    def _from_cached_state(
        cls,
        sources: dict[str, str],
        parsed: dict[str, Sequence],
        documentation: dict[str, str],
        catalog: str,
    ) -> Module:
        module = cls.__new__(cls)
        module._sources = sources
        module._parsed = parsed
        module._documentation = documentation
        module._catalog = catalog
        return module

    def _adopt(self, module: Module) -> None:
        self._sources = module._sources
        self._parsed = module._parsed
        self._documentation = module._documentation
        self._catalog = module._catalog

    def _updated(self, updates: Mapping[str, str]) -> Module:
        sources = self._sources.copy()
        parsed = self._parsed.copy()
        documentation = self._documentation.copy()
        changed_names: set[str] = set()

        for name, source in updates.items():
            _validate_source(name, source)
            if name in sources and sources[name] == source:
                continue
            program = read(source, source_name=f"<word {name}>")
            sources[name] = source
            parsed[name] = program
            changed_names.add(name)

        if not changed_names:
            return self

        _refresh_documentation(
            changed_names,
            sources,
            parsed,
            documentation,
        )
        catalog = _format_catalog(_ordinary_words(sources), documentation)
        candidate = self._from_cached_state(
            sources,
            parsed,
            documentation,
            catalog,
        )
        _run_smoke_tests(candidate)
        return candidate

    def __getitem__(self, name: str) -> str:
        return self._sources[name]

    def __setitem__(self, name: str, source: str) -> None:
        self._adopt(self._updated({name: source}))

    def __delitem__(self, name: str) -> None:
        if name not in self._sources:
            raise KeyError(name)

        sources = self._sources.copy()
        parsed = self._parsed.copy()
        documentation = self._documentation.copy()
        del sources[name]
        del parsed[name]
        _refresh_documentation(
            {name},
            sources,
            parsed,
            documentation,
        )
        catalog = _format_catalog(_ordinary_words(sources), documentation)
        candidate = self._from_cached_state(
            sources,
            parsed,
            documentation,
            catalog,
        )
        _run_smoke_tests(candidate)
        self._adopt(candidate)

    def update(self, *args, **kwargs: str) -> None:
        updates = dict(*args, **kwargs)
        self._adopt(self._updated(updates))

    def clear(self) -> None:
        if not self._sources:
            return
        self._adopt(self._from_cached_state({}, {}, {}, "(none)"))

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
