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
from copycat import normalize

assert normalize("1 2 f", verbose=False) == "2 1"
```

Use `Evaluate` when evaluation needs to be inspected or advanced one machine
transition at a time:

```python
from copycat import Evaluate, read

evaluation = Evaluate(read("1 2 f"), verbose=False)
snapshot = evaluation.step()

assert snapshot.steps == 1
assert str(evaluation.run()) == "2 1"
```

Copycat modules are mutable mappings from lowercase word names to source text.
Source is parsed and cached when it is added or changed. User-defined names must
be longer than one character; single-letter names are reserved for the kernel
primitives `a`-`f`, `r`, and `s`.

```python
from copycat import Module

module = Module({
    "duplicate": "d",
    "duplicate-doc": (
        '"duplicate (x -- x x): Duplicate the top value. '
        'Example: 1 duplicate ==> 1 1."'
    ),
})

assert normalize("1 duplicate", module=module, verbose=False) == "1 1"

module["duplicate"] = "d d"
assert normalize("1 duplicate", module=module, verbose=False) == "1 1 1"
```

Assignments, deletions, and batch updates refresh the affected documentation and
the model catalog, then run the module's smoke tests. A failed mutation leaves the
module unchanged.

Documentation uses the conventional `-doc` suffix. A documentation word must
contain exactly one string. Missing or unusable documentation produces a warning
and appears as `(undocumented)` to model backends; syntax errors remain fatal.
Model invocations receive a cached catalog containing module word names and
documentation, but never their definitions or `test-*` words.

Annotations use `@name` syntax and have identity semantics. The built-in `@eq`
annotation asserts that the top two values are deeply equal without removing
them from the stack.

```python
assert normalize("1 1 @eq", verbose=False) == "1 1"
```

Modules are stored as directories of extensionless UTF-8 text files. Each
filename is a word name and its contents are the Copycat source body. Saving
synchronizes the directory with the module, removing files for deleted words.
ZIP archives containing the same flat file layout can also be loaded as a
portable transport format. Every `test-*` word is run automatically when a
module is loaded; a failed assertion prevents the module from loading.

```python
from copycat import Module

module.save("example")
module = Module.load("example")
module = Module.load("example.zip")
```

The repository includes `modules/data`, a documented and smoke-tested module
of products, sums, options, booleans, and Scott-encoded lists. Clone the
repository to install the package and load the module from the same revision:

```bash
git clone https://github.com/dearmadisonblue/copycat.git
python -m pip install -e copycat
```

```python
from copycat import Module

data_module = Module.load("copycat/modules/data")
```

See [`notebooks/copycat.ipynb`](notebooks/copycat.ipynb) for reader, evaluator,
module, model-effect, and live Gemma examples.
