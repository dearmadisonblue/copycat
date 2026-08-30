# Copycat

Copycat is a small concatenative language in which a model invocation is a
first-class program-synthesis effect. The deterministic runtime stays in
control: `{natural language}` asks a backend for Copycat code, and that code
continues immediately against the current data stack.

## Installation

Install the core language in editable mode:

```bash
python -m pip install -e .
```

Install the optional Gemma backend and notebook test dependencies when needed:

```bash
python -m pip install -e ".[gemma,test]"
```

## Usage

```python
from copycat import run

assert run("1 2 f", verbose=False) == "2 1"
```

Copycat dictionaries are mutable mappings from lowercase word names to source
text. User-defined names must be longer than one character; single-letter names
are reserved for the kernel primitives `a`-`f`, `r`, and `s`.

```python
dictionary = {
    "duplicate": "d",
    "duplicate-doc": '"Duplicate the top value."',
}

assert run("1 duplicate", dictionary=dictionary, verbose=False) == "1 1"
```

Dictionaries can be serialized as `.module` ZIP archives. Each archive member is
an extensionless UTF-8 text file whose filename is the word name and whose
contents are its Copycat source body.

```python
from copycat import load_module, save_module

save_module(dictionary, "example.module")
dictionary = load_module("example.module")
```

See [`notebooks/copycat.ipynb`](notebooks/copycat.ipynb) for reader,
evaluator, dictionary, model-effect, and live Gemma examples.
