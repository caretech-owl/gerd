# Development Guide

## Tools

### Poetry

#### Getting started

Setup your environment and install poetry (e.g. via `pip`):

```sh
pip install poetry
```

Next intall the package and all dependencies, including `dev` depenendies:

```sh
poetry install --with dev
```

To add a new *runtime* dependency, just run `poetry add`:

```sh
poetry add langchain
```

To add a new *development* dependency, run `poetry add` with a group specifier:

```sh
poetry add mypy --group dev
```

Poetry will install dependencies in its own virtual environment.
You need to prepend `poetry run` to any Python command you'd like to execute:

```sh
poetry run python main.py
poetry run ruff team_red
```

### PyTest

Test case are run via pytest. Tests can be found in the [tests folder](./tests).
Tests will not be run via pre-commit since they might be too complex to be done before commits.

### Ruff

Ruff is used for linting and code formatting.
Ruff follows `black` styling [ref](https://docs.astral.sh/ruff/faq/#is-the-ruff-linter-compatible-with-black).
Ruff will be run automatically before commits when pre-commit hooks are installed.
To run `ruff` manually, use poetry syntax:

```sh
poetry run ruff check team_red
```

### MyPy

MyPy does static type checking.
It will not be run automatically.
To run MyPy manually use poetry syntax with the folder to be checked:

```shell
poetry run mypy team_red
```

### Pre-commit hooks

These hooks are used to check linting and run tests before commit changes to prevent faulty commits.
This should not include long running actions (such as tests) since committing should be fast.
To install pre-commit hooks, execute this *once*:

```shell
poetry run pre-commit install
```

### Poe Task Runner

See [pyproject.toml](pyproject.toml) for task runner configurations.

#### Run Frontend 

Either run Generate Frontend:

```python
poetry poe frontend gen
```

or QA Frontend:

```
poetry poe frontend qa
```

The backend is chosen via `config.yaml`.
Currently only the direct backend is implemented.

### GitHub Actions

GitHub Actions can be found under [github/workflows](./.github/workflows/) and will run linting.
In its current config it will only be executed when a PR for `main` is created or when a special `dev-gha` branch is created.
It will also trigger actions when commits are pushed to `main` directly but this should be avoided.

### GitHub Issue Templates

#### Bugs

#### Feature 

#### Use Case

