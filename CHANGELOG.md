# Changelog

## Unreleased

### Added

- `file_reader` argument for `read_file`, `read_files`, `read_directory`,
  `read_directories` and the matching `DataCollectionReader` methods. Pass a
  `FileReader` subclass, or a list tried in order, to read files with readers
  from other packages or to choose one reader for a shared extension. Readers
  are otherwise still selected from the file extension.

### Changed

- v0.0.0a2
  Switched from using `setup.cfg` to using `pyproject.toml` for packaging. 
  Switched from Azure Pipelines to GitHub Actions for test and release.
