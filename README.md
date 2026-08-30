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

Copycat modules are immutable mappings from lowercase word names to source text.
Their source is parsed and cached when the module is constructed. User-defined
names must be longer than one character; single-letter names are reserved for
the kernel primitives `a`-`f`, `r`, and `s`.

```python
from copycat import Module

module = Module({
    "duplicate": "d",
    "duplicate-doc": (
        '"duplicate (x -- x x): Duplicate the top value. '
        'Example: 1 duplicate ==> 1 1."'
    ),
})

assert run("1 duplicate", module=module, verbose=False) == "1 1"
```

Documentation uses the conventional `-doc` suffix. A documentation word must
contain exactly one string. Missing or unusable documentation produces a warning
and appears as `(undocumented)` to model backends; syntax errors remain fatal.
Model invocations receive a cached catalog containing module word names and
documentation, but never their definitions or `test-*` words.

Annotations use `@name` syntax and have identity semantics. The built-in `@eq`
annotation asserts that the top two values are deeply equal without removing
them from the stack.

```python
assert run("1 1 @eq", strict=True, verbose=False) == "1 1"
```

Modules can be serialized as `.module` ZIP archives. Each archive member is
an extensionless UTF-8 text file whose filename is the word name and whose
contents are its Copycat source body. Every `test-*` word is run automatically
when a module is loaded; a failed assertion prevents the module from loading.

```python
from copycat import load_module, save_module

save_module(module, "example.module")
module = load_module("example.module")
```

The repository includes `modules/data.module`, a documented and smoke-tested
module of products, sums, options, booleans, and Scott-encoded lists. In Colab
or another remote environment it can be fetched directly from:

```text
https://raw.githubusercontent.com/dearmadisonblue/copycat/main/modules/data.module
```

See [`notebooks/copycat.ipynb`](notebooks/copycat.ipynb) for reader, evaluator,
module, model-effect, and live Gemma examples.
