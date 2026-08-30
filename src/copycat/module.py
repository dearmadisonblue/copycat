from __future__ import annotations

from os import PathLike
from typing import Mapping
from zipfile import ZIP_DEFLATED, ZipFile

from .read import _is_user_word_name, read


__all__ = ["load_module", "save_module"]

_Path = str | PathLike[str]
_SMOKE_TEST_GAS = 100_000


def _validate_dictionary(dictionary: Mapping[str, str]) -> None:
    for name, source in dictionary.items():
        if not isinstance(name, str):
            raise TypeError("Dictionary word names must be strings.")
        if not _is_user_word_name(name):
            raise ValueError(
                f"Invalid dictionary word name {name!r}. User-defined words must "
                "be lowercase kebab-case and longer than one character."
            )
        if not isinstance(source, str):
            raise TypeError(f"Dictionary body for {name!r} must be source text.")
        read(source, source_name=f"<word {name}>")


def _run_smoke_tests(dictionary: dict[str, str]) -> None:
    # Import lazily to avoid a module/evaluator import cycle.
    from .evaluate import evaluate

    for name in sorted(word for word in dictionary if word.startswith("test-")):
        evaluate(
            read(name, source_name=f"<smoke test {name}>"),
            dictionary=dictionary,
            gas=_SMOKE_TEST_GAS,
            strict=True,
            verbose=False,
        )


def load_module(path: _Path) -> dict[str, str]:
    """Load and smoke-test a .module ZIP archive."""
    dictionary: dict[str, str] = {}

    with ZipFile(path, "r") as archive:
        for info in archive.infolist():
            name = info.filename
            if info.is_dir() or "/" in name or "\\" in name:
                raise ValueError(
                    f"Invalid module entry {name!r}. Module entries must be flat files."
                )
            if name in dictionary:
                raise ValueError(f"Duplicate module entry {name!r}.")

            try:
                source = archive.read(info).decode("utf-8")
            except UnicodeDecodeError as exc:
                raise ValueError(
                    f"Module entry {name!r} is not valid UTF-8 source text."
                ) from exc

            dictionary[name] = source

    _validate_dictionary(dictionary)
    _run_smoke_tests(dictionary)
    return dictionary


def save_module(dictionary: Mapping[str, str], path: _Path) -> None:
    """Serialize a dictionary as a .module ZIP archive of extensionless source files."""
    _validate_dictionary(dictionary)

    with ZipFile(path, "w", compression=ZIP_DEFLATED) as archive:
        for name in sorted(dictionary):
            archive.writestr(name, dictionary[name].encode("utf-8"))
