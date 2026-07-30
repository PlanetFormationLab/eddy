# Contributing to `eddy`

Contributions are welcome — bug reports, fixes, new methods, and
documentation improvements alike. For anything substantial, please
open an [issue](https://github.com/PlanetFormationLab/eddy/issues/new)
first so we can discuss the design.

## Development setup

```bash
git clone https://github.com/PlanetFormationLab/eddy.git
cd eddy
pip install -e ".[test,docs]"
```

This installs `eddy` in editable mode together with the testing and
documentation extras. Python 3.10+ is required.

If your editor uses Pylance or Pyright (e.g. VS Code), append
`--config-settings editable_mode=compat`:

```bash
pip install -e ".[test,docs]" --config-settings editable_mode=compat
```

setuptools' default PEP 660 editable install uses a `sys.meta_path`
finder hook that Pylance can't follow, which surfaces as spurious
`reportMissingImports` warnings on `from eddy import ...`. The `compat`
mode writes the old-style `.pth` entry that adds the source tree to
`sys.path` directly, which static analyzers understand. Editable
behaviour is otherwise identical.

If you plan to submit a pull request, consider enabling the pre-commit
hooks so style/lint checks run on every commit:

```bash
pip install pre-commit
pre-commit install
```

## Running the test suite

```bash
pytest -v
```

The smoke tests under `tests/` reuse FITS files from
`docs/tutorials/` as fixtures. The data files are gitignored; the
tutorials' download cells (or the URLs in `.github/workflows/ci.yml`)
will fetch them on demand. Tests that need a missing fixture are
skipped rather than failed locally.

Mark slow tests with `@pytest.mark.slow` so they can be excluded with
`pytest -m "not slow"`.

## Code style

We run [`ruff`](https://docs.astral.sh/ruff/) for linting and basic
style. The config lives in `pyproject.toml` under `[tool.ruff]`; run

```bash
ruff check .
ruff check . --fix     # apply auto-fixable suggestions
```

CI rejects PRs with lint errors. The rule set is intentionally minimal
(pycodestyle `E`/`W` + pyflakes `F`); we don't enforce import sorting
or naming conventions.

## Branching and commits

- Branch from `master` for new work. Use descriptive branch names
  (e.g. `fix-rotationmap-pa-wrap`).
- Keep commits focused and self-describing. The body should explain
  *why*, not just *what*.
- Reference the relevant issue number in the commit message or PR
  description when applicable.

## Pull requests

- Make sure `pytest` and `ruff check .` pass locally before opening
  the PR.
- New user-visible behaviour (new arguments, deprecated signatures,
  changed defaults) should add an entry to `CHANGELOG.md`.
- Tutorials in `docs/tutorials/` are executed notebooks. If your
  change affects an existing tutorial's output, re-execute it with
  `jupyter nbconvert --execute --inplace docs/tutorials/<name>.ipynb`
  before committing.

## Questions

For anything else, open an issue or contact Richard Teague directly.
