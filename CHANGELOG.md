# Changelog

## Unreleased

### Changed

- Code linted and formatted with ruff (configuration in `pyproject.toml`);
  CI runs `ruff check` and `ruff format --check` on pull requests.

### Fixed

- `ValueError` raised for an unknown independent variable name now includes
  the name (the message was missing its f-string prefix).
- Exceptions re-raised inside `except` blocks now chain the original error.

### Changed

- v0.0.0a2
  Switched from using `setup.cfg` to using `pyproject.toml` for packaging. 
  Switched from Azure Pipelines to GitHub Actions for test and release.