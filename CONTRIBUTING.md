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

## Making a release

Releases are cut manually from `master`. The version lives in two
places and both must agree, or the built wheel and `eddy.__version__`
will disagree at runtime:

- `pyproject.toml` (`version = "..."`)
- `eddy/__init__.py` (`__version__ = "..."`)

1. **Open a release PR from a branch**, containing the version bump in
   both files and the `CHANGELOG.md` heading change: rename the
   accumulated `## [Unreleased]` section to `## [X.Y.Z] - YYYY-MM-DD`.
   Every user-visible change should already have an entry there from
   the PR that introduced it.

2. **Check it locally** before merging:

   ```bash
   ruff check .
   pytest -v
   python -c "import eddy; print(eddy.__version__)"
   ```

   Building the docs needs `pandoc` on `PATH` (a system package, not a
   pip one) because `nbsphinx` shells out to it for the tutorials:

   ```bash
   pip install -e ".[docs]"
   sphinx-build -b html docs docs/_build/html
   ```

3. **Merge the PR**, then tag the merge commit on `master` and push the
   tag. Tags carry a leading `v` and are annotated:

   ```bash
   git checkout master && git pull
   git tag -a vX.Y.Z -m "eddy X.Y.Z"
   git push origin vX.Y.Z
   ```

4. **Build and upload.** There is no publish workflow; this is a local
   `twine` step. Build from a clean tree so the sdist does not pick up
   stray files, and keep the version's artifacts in `dist/`:

   ```bash
   pip install --upgrade build twine
   python -m build
   twine check dist/astro_eddy-X.Y.Z*
   twine upload dist/astro_eddy-X.Y.Z*
   ```

   Upload the two artifacts for the new version only — passing a bare
   `dist/*` re-submits every previous release and fails. A version
   number cannot be reused on PyPI even after deletion, so check
   `twine check` output before uploading.

5. **Create the GitHub release** from the tag, pasting that version's
   `CHANGELOG.md` section as the body:

   ```bash
   gh release create vX.Y.Z --title "eddy X.Y.Z" --notes-file -
   ```

6. **Confirm** the new version resolves and Read the Docs has built the
   tag:

   ```bash
   pip index versions astro-eddy
   ```

Steps 3-5 have been missed before: `3.0.1` and `3.1.0` have no git tag,
and `3.1.0` was never uploaded to PyPI at all (it is in the changelog
but absent from the release history). Work through the list in order.

## Questions

For anything else, open an issue or contact Richard Teague directly.
