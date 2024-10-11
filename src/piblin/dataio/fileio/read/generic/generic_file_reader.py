from abc import ABC
import piblin.dataio.fileio.read.file_reader as file_reader


class GenericFileReader(file_reader.FileReader, ABC):
    """Dummy class for generic file readers."""
    ...
