"""Definition of abstract base class for file readers.

Concrete implementations of the FileReader abstract base class convert filenames into sets of measurements. Such an
implementation must provide a read method, so that the file can be imported, a set of supported extensions, and an
implementation of `data_from_file_contents()` which carries out the conversion specific to the file format of interest.

Classes
-------
FileReader - Abstract class for file readers.

See Also
--------
piblin.fileio.FileParsingException
    Raised when a cralds file reader fails to convert a file to a measurement set.
"""
from typing import ClassVar, Union, Set
from abc import ABC, abstractmethod
import os

import piblin.data.data_collections.measurement_set as measurement_set


class FileReader(ABC):
    """Abstract class for file readers.

    All FileReader subclasses must implement the abstract property `default_mode` (returning a string) and the class
    method `_data_from_file_contents` (returning a measurement set). The class variable `supported_extensions` is also
    required to enable autodetect.
    The rest of the functionality can typically rely only on the methods of this ABC, although some file readers will
    need to override other methods. The _read_file_contents method assumes that python's open builtin can be used to
    read in the file contents, and so in cases where this is not true (e.g. an external package uses the filename to
    create an object directly) this method is not needed and the default_mode and encoding properties will not be used.
    The data_from_filepath method should only be overridden if it is also called on the superclass to ensure that file
    metadata is added to all measurements.

    The public API of this class consists of its properties and the methods data_from_filepath and supports_extension,
    as listed below.

    Methods
    -------
    data_from_file_contents -> MeasurementSet
        Convert file contents to appropriate measurement(s).
    supports_extension -> bool
        Determine whether this file reader supports the given extension.
    """
    supported_extensions: ClassVar[Set[str]]

    def __init__(self):
        super().__init__()

    @classmethod
    def supports_extension(cls, extension: str) -> bool:
        """Determine whether this file reader supports the given extension.

        Parameters
        ----------
        extension : str
            The extension with which to determine compatibility.

        Returns
        -------
        bool
            Whether or not this file reader supports the given extension.
        """
        for supported_extension in cls.supported_extensions:
            if supported_extension.lower() == extension.lower():
                return True
        return False

    @property
    @abstractmethod
    def default_mode(self) -> str:
        """The default mode in which to read the file.

        All FileReader subclasses will by default attempt to use the builtin method "open" to read the file contents,
        with the "mode" parameter set to the value in this property. The default implementation matches the default of
        the "open" builtin - to open for reading in text mode.

        Returns
        -------
        str, default "r"
            The default mode in which to read the file.
        """
        return "r"

    @property
    def encoding(self) -> Union[str, None]:
        """The default encoding to use to read the file.

        All FileReader subclasses will by default attempt to use the builtin method "open" to read the file contents,
        with the "encoding" parameter set to the value in this property. The default implementation matches the default
        of the "open" builtin - to open with the default encoding for the current locale.
        An encoding can only be specified if the read method is text, i.e. it cannot be specified for binary mode.

        Returns
        -------
        str or None, default None
            The default encoding to use to read the file.
        """
        return None

    def data_from_filepath(self,
                           filepath: str,
                           **read_kwargs) -> measurement_set.MeasurementSet:
        """Convert the file at the given path into appropriate measurement(s).

        Parameters
        ----------
        filepath : str
            The path to the file to convert to a measurement set.

        Returns
        -------
        piblin.data.data_collections.measurement_set.MeasurementSet
            The set of measurements in the file.
        """
        file_contents = self._read_file_contents(filepath, **read_kwargs)

        file_location = os.path.split(filepath)[0]
        file_name = os.path.split(filepath)[1]

        file_measurements = self._data_from_file_contents(
            file_contents=file_contents,
            file_location=file_location,
            file_name=file_name,
            **read_kwargs
        )

        file_location, file_name = os.path.split(filepath)
        for measurement in file_measurements:
            measurement.add_detail("source_disk_directory", file_location)
            measurement.add_detail("source_file_name", file_name)
            measurement.add_detail("source_file_name_head", os.path.splitext(file_name)[0])
            measurement.add_detail("source_file_path", filepath)
            for dataset in measurement.datasets:
                dataset.source = filepath

        return file_measurements

    def _read_file_contents(self, filepath: str, **read_kwargs):
        """Read the contents of a file into a single object.

        This method wraps the builtin "open" method to allow for default behaviour to be specified for specific
        parameters of that method, currently the values of "mode" and "encoding". This method also allows for the
        initial read of the file contents from the file at the given path to be overridden so that approaches other
        than python's builtin `open` method can be used. This is particularly useful when a 3rd party package is used
        to read files, as these packages often return structured data rather than the contents of a file, which can then
        be converted to the cralds data structures in the `_data_from_file_contents` method.

        Parameters
        ----------
        filepath : str
            The path to the file to be read.
        """
        if self.encoding is None:
            with open(filepath, mode=self.default_mode) as file:
                return file.read()
        else:
            with open(filepath, mode=self.default_mode, encoding=self.encoding) as file:
                return file.read()

    @classmethod
    @abstractmethod
    def _data_from_file_contents(cls,
                                 file_contents,
                                 file_location=None,
                                 file_name=None,
                                 **read_kwargs) -> measurement_set.MeasurementSet:
        """Convert file contents to an appropriate set of measurements.

        This is an abstract method so must be overridden in subclasses. It needs to raise a specific exception if read
        fails.

        Parameters
        ----------
        file_contents : buffer or list of str
            The contents of the file to convert to measurements.
        file_location : str
            Path to the file to be read.
        file_name : str
            The name of the file to be read.

        Returns
        -------
        piblin.data.data_collections.MeasurementSet
            The set of measurements in the file.
        """
        pass
