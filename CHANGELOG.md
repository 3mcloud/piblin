# Changelog

## Unreleased

### Fixed

- Package metadata declared the licence as "Proprietary" while the LICENSE
  file is MIT. The licence is now the SPDX expression `MIT` with
  `license-files = ["LICENSE"]`.

### Changed

- v0.0.0a2
  Switched from using `setup.cfg` to using `pyproject.toml` for packaging. 
  Switched from Azure Pipelines to GitHub Actions for test and release.
