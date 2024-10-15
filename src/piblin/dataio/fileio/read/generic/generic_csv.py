"""Reading of spectra in generic .csv format.

Classes
-------
CsvReader - Reader for files in the .csv format.
"""
import numpy as np

import piblin.data.datasets.abc.split_datasets.one_dimensional_dataset as one_d_dataset
import piblin.data.data_collections.measurement as measurement
import piblin.data.data_collections.measurement_set as measurement_set
import piblin.dataio.fileio.read.generic.generic_file_reader as generic_file_reader


class CsvReader(generic_file_reader.GenericFileReader):
    """Reader for files in the .csv format."""
    supported_extensions = {"csv"}

    def __init__(self):
        super().__init__()

    @property
    def default_mode(self) -> str:
        return ""

    def _read_file_contents(self, filepath: str, **read_kwargs):
        """Read the contents of a file into a single object.

        This method wraps the builtin "open" method to allow for default
        behaviour to be specified for specific parameters of that method,
        currently the values of "mode" and "encoding".

        Parameters
        ----------
        filepath : str
            The path to the file to be read.
        """
        dtype = read_kwargs.get("dtype", None)
        comments = read_kwargs.get("comments", None)
        delimiter = read_kwargs.get("delimiter", ",")
        converters = read_kwargs.get("converters", None)
        skiprows = read_kwargs.get("skiprows", 0)
        usecols = read_kwargs.get("usecols", None)
        unpack = read_kwargs.get("unpack", False)
        ndmin = read_kwargs.get("ndmin", 0)
        encoding = read_kwargs.get("encoding", None)
        max_rows = read_kwargs.get("max_rows", None)

        return np.loadtxt(fname=str(filepath),
                          dtype=dtype,
                          comments=comments,
                          delimiter=delimiter,
                          converters=converters,
                          skiprows=skiprows,
                          usecols=usecols,
                          unpack=unpack,
                          ndmin=ndmin,
                          encoding=encoding,
                          max_rows=max_rows)

    @classmethod
    def _data_from_file_contents(cls, file_contents, file_location=None, **read_kwargs):
        """Read a measurement set from the contents of a csv file.

        Parameters
        ----------
        file_contents : numpy.ndarray
            The contents of the csv file as a numpy array.
        as_rows : bool
            Whether to read the csv as containing rows of data.
        single_x_column : bool
            Whether each dataset has specific x values.

        Returns
        -------
        MeasurementSet
            The set of measurements contained in the csv file.
        """
        as_rows = read_kwargs.get("as_rows", False)
        single_x_column = read_kwargs.get("single_x_column", False)
        column_index_as_metadata = read_kwargs.get("column_index_as_metadata", False)

        if as_rows:
            file_contents = file_contents.T

        num_columns = file_contents.shape[1]

        measurements = []
        if single_x_column:
            x_values = file_contents[:, 0]
            for y_column_index in range(1, num_columns):

                if column_index_as_metadata:
                    conditions = {"index": y_column_index}
                else:
                    conditions = {}

                measurements.append(measurement.Measurement.from_single_dataset(
                    one_d_dataset.OneDimensionalDataset.create(x_values=x_values,
                                                               y_values=file_contents[:, y_column_index]),
                    conditions=conditions))
        else:
            for x_column_index in range(0, num_columns, 2):
                x_values = file_contents[:, x_column_index]
                y_values = file_contents[:, x_column_index + 1]
                if column_index_as_metadata:
                    conditions = {"index": x_column_index / 2}
                else:
                    conditions = {}

                measurements.append(measurement.Measurement.from_single_dataset(
                    one_d_dataset.OneDimensionalDataset.create(x_values=x_values,
                                                               y_values=y_values),
                    conditions=conditions))

        return measurement_set.MeasurementSet(measurements)
