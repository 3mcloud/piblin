"""Definition of abstract base class for file readers.

Concrete implementations of the FileReader abstract base class convert
filenames into sets of measurements. Such an implementation must provide a
read mode, so that the file can be imported, a list of supported extensions,
and an implementation of `data_from_file_contents()` which carries out the
conversion specific to the file format of interest.

Classes
-------
FileReader - Abstract class for file readers.
"""
from typing import ClassVar, Set, Union

import piblin.data.datasets.abc.dataset as dataset
import piblin.data.data_collections.measurement as measurement
import piblin.data.data_collections.measurement_set as measurement_set
import piblin.dataio.fileio.read.file_reader as file_reader


class ExtReader(file_reader.FileReader):  # change the class name to match the extension being supported
    """Example class for file readers.

    All FileReader subclasses must implement the two abstract properties
    default_mode and supported_extension, and the class method
    data_from_file_contents.

    Methods
    -------
    data_from_file_contents - convert file contents to appropriate dataset(s).
    """
    supported_extensions: ClassVar[Set[str]]  # the extensions of the file formats this class reads

    def __init__(self):
        super().__init__()

    @property
    def default_mode(self) -> str:
        return ""  # the mode to pass to python's open function; likely "r" (text) or "rb" (binary)

    @property
    def encoding(self) -> Union[str, None]:
        """The default encoding to use to read the file."""
        return None

    @classmethod
    def _data_from_file_contents(cls, file_contents, file_location=None, **read_kwargs):
        """Convert file contents to appropriate dataset(s).

        This is an abstract method of FileReader so must be overridden in
        subclasses. Needs to raise a specific exception if read fails.

        Parameters
        ----------
        file_contents : buffer or list of str
        file_location : str
            The directory containing the file.

        Returns
        -------
        piblin.data.data_collections.measurement_set.MeasurementSet
            The set of measurements in the file.
        """
        conditions = {}
        details = {}

        dataset_ = dataset.Dataset([])

        measurements = [measurement.Measurement.from_single_dataset(dataset_, conditions, details)]

        measurement_set_ = measurement_set.MeasurementSet(measurements)  # return this measurement set instance
        raise NotImplementedError("This is a template class not for use.")

