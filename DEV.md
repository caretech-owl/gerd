# Development Guide

## Basics

To get started on development you need to install [uv](https://docs.astral.sh/uv/getting-started/).
You can use `pip`, `pipx` or `conda` to do so:

```shell
pip install uv
```

Next intall the package and all dependencies with `uv sync`.

```shell
# cd <gerd_project_root>
uv sync
```

After that, it should be possible to run scripts without further issues:

```shell
python examples/hello.py
```

To add a new *runtime* dependency, just run `uv add`:

```sh
uv add langchain
```

To add a new *development* dependency, run `uv add` with the `--dev` flag:

```sh
uv add mypy --dev
```

## Further tools

### Poe Task Runner

See [pyproject.toml](pyproject.toml) for task runner configurations.
You can run most of the tools mentioned above with a (shorter) call to `poe`.

```shell
uv run poe lint  # do some linting (with mypy)
```

### PyTest

Test case are run via pytest. Tests can be found in the [tests folder](./tests).
Tests will not be run via pre-commit since they might be too complex to be done before commits.
To run the standard set of tests use the `poe` task `test`:

```shell
uv run poe test
```

More excessive testing can be trigger with `test_manual` which will NOT mock calls to the used models:

```shell
uv run poe test_manual
```

### Ruff

Ruff is used for linting and code formatting.
Ruff follows `black` styling [ref](https://docs.astral.sh/ruff/faq/#is-the-ruff-linter-compatible-with-black).
Ruff will be run automatically before commits when pre-commit hooks are installed.
To run `ruff` manually, use uv:

```sh
uv run ruff check gerd
```

There is a [VSCode extension](https://marketplace.visualstudio.com/items?itemName=charliermarsh.ruff) that handles formatting and linting.

### MyPy

MyPy does static type checking.
It will not be run automatically.
To run MyPy manually use uv with the folder to be checked:

```shell
uv run mypy gerd
```

### Pre-commit hooks

These hooks are used to check linting and run tests before commit changes to prevent faulty commits.
This should not include long running actions (such as tests) since committing should be fast.
To install pre-commit hooks, execute this *once*:

```shell
uv run pre-commit install
```

## Implemented GUIs

### Run Frontend 

Either run Generate Frontend:

```shell
uv run poe gen
```

or QA Frontend:

```shell
uv run poe qa
```

The backend is chosen via `config.yaml`.


## CI/CD and Distribution

### GitHub Actions

GitHub Actions can be found under [github/workflows](./.github/workflows/) and will run linting.
In its current config it will only be executed when a PR for `main` is created or when a special `dev-gha` branch is created.
It will also trigger actions when commits are pushed to `main` directly but this should be avoided.

### GitHub Issue Templates

#### Bugs

#### Feature 

#### Use Case
