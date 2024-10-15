from typing import Dict
from abc import ABC, abstractmethod
import piblin.dataio.fileio.read.file_reader as file_reader


class MultifileReader(file_reader.FileReader, ABC):
    """Dummy class for multifile readers."""

    @abstractmethod
    def supported_pattern(self):
        """The pattern of files supported by this multifile reader.
        A pattern is encoded as a set of file extensions mapped to the
        minimum number of which may be present.
        """
        raise NotImplementedError

    def supports_pattern(self, pattern: Dict):
        """Determine whether this multifile reader supports the given pattern.
        To support a given pattern this reader must have the same set of file
        extensions in its supported pattern and require fewer of each kind of
        file than is specified in the pattern.
        """
        if pattern.keys() != self.supported_pattern().keys():
            return False

        else:
            for key, value in pattern.items():
                if value < self.supported_pattern()[key]:
                    return False

        return True

    @abstractmethod
    def data_from_filelist(self, filelist):
        raise NotImplementedError
