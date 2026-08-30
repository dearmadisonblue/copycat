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

assert run("1 2 Swap", verbose=False) == "2 1"
```

See [`notebooks/copycat.ipynb`](notebooks/copycat.ipynb) for reader,
evaluator, model-effect, and live Gemma examples.
