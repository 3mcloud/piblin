from abc import ABC, abstractmethod
from typing import List
import piblin.dataio.fileio.read.file_reader as file_reader


class InvalidSummaryFileError(Exception):
    """Raised when a summary file and its summarized files cannot be parsed."""
    ...


class SummaryFileReader(file_reader.FileReader, ABC):
    """Reader for files which can trigger reading of multiple other files.
    Certain individual files can contain lists of related files which are
    to be read together to form one or more datasets to be organized into a
    single measurement set rather than treated as containing individual
    measurement sets. For example, large-area imaging scans often produce
    a single large image composed of many individual images stored in
    single files. Also, volumetric datasets are commonly stored as slices,
    where each layer of the volume is a single image file.
    In order to handle such cases, when cralds attempts to read a directory,
    it has to check for the presence of summary files first, and remove their
    dependent files from the list of files to be read. The summary file reader
    will also be responsible for reading all of the dependent files and
    assembling the appropriate datasets and measurement sets to return to the
    data collection readers.
    """

    # TODO - this requires the file contents, not just the path
    @abstractmethod
    def is_valid(self, filepath: str) -> bool:
        """Determine whether the summary file at the given path is valid.

        Parameters
        ----------
        filepath : str
            The path to the summary file to check for validity.

        Returns
        -------
        bool
            Whether or not this file is a valid summary file.

        Raises
        ------
        InvalidSummaryFileError
            If the summary file is not valid.

        Notes
        -----
        A summary file can be orphaned from the files that it summarizes
        so it is necessary to check that the summarized files are present
        before reading the data.
        """
        ...

    @abstractmethod
    def get_summarized_filenames(self, filepath: str) -> List[str]:
        """Create a list of summarized filenames for the given summary file.
        Parameters
        ----------
        filepath : str
            The path to the summary file to extract dependent filenames from.
        Returns
        -------
        list of str
            The paths to the files whose reading is triggered by this file.
        """
        ...
