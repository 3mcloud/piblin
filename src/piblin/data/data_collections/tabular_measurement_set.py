import csv
from typing import List
import numpy as np
import piblin.data.datasets.abc.split_datasets.zero_dimensional_dataset as zero_dimensional_dataset
import piblin.data.data_collections.measurement
import piblin.data.data_collections.measurement_set
import piblin.data.data_collections.experiment_set


class TabularMeasurementSet(object):
    """Tabular set of measurements.

    The data of a tabular dataset is stored in a list of lists.
    Each list corresponds to a row in a table.

    Parameters
    ----------
    data : list of list
        Two-dimensional array of data.
        Rows represent samples, columns are variables.
    column_headers : List of str
        Identifiers for each column/variable.
    n_metadata_columns : int
        The number of columns corresponding to metadata.
    dataset_types : list of class
        The classes for each dataset in the data columns.
        If no data is passed, the data is read as a set of
        unrelated scalar values.
    dataset_end_indices : list of int
        The column index at which each dataset ends.
        If no indices are passed, a single dataset is assumed
        using the type provided in the previous parameter. If no
        type is provided, uses a 1D dataset.
        If multiple types are provided, should warn user.
    """
    @property
    def num_rows(self):
        return len(self.data)

    @property
    def num_cols(self):
        return len(self.data[0])

    @property
    def n_columns(self):
        if len(self.data) == 0:
            return 0
        else:
            return len(self.data[0])

    @property
    def n_rows(self):
        return len(self.data)

    @property
    def n_datasets(self):
        return len(self.dataset_types)

    def __init__(self,
                 data: List[List],
                 n_metadata_columns: int = 0,
                 column_headers: List[str] = None,
                 dataset_types: List = None,
                 dataset_end_indices: List[int] = None):

        super().__init__()

        self._data = data
        self._dataset_end_indices = dataset_end_indices

        for row in data[1:]:
            if len(row) != self.n_columns:
                raise ValueError("Flat data must be a 2-dimensional array")

        if n_metadata_columns > self.n_columns:
            raise ValueError("Number of metadata columns greater than number of columns.")
        else:
            self._n_metadata_columns = n_metadata_columns
            self._n_data_columns = self.n_columns - self._n_metadata_columns

        if column_headers is None:
            self._column_headers = self.__generate_column_headers(self._n_metadata_columns,
                                                                  self._n_data_columns)
        else:
            if len(column_headers) != self.n_columns:
                raise ValueError("Number of column headers different to number of columns.", len(column_headers), self._n_columns)
            else:
                self._column_headers = column_headers

        if dataset_types is None:
            self._dataset_types = [zero_dimensional_dataset.ZeroDimensionalDataset] * self._n_data_columns
            self._dataset_end_indices = list(range(1, self.n_columns))
        else:
            self._dataset_types = dataset_types
            if len(dataset_types) == 1:
                self._dataset_end_indices = [self.n_columns - self._n_metadata_columns]
            elif dataset_end_indices is None:
                raise ValueError("dataset end indices required for multiple datasets")

    @staticmethod
    def __generate_column_headers(n_metadata_columns,
                                  n_data_columns):
        """Generate default column headers if none are provided.

        Returns
        -------
        list of str
        """
        metadata_headers = [f"metadata_var_{i}" for i in range(n_metadata_columns)]
        data_headers = [f"data_var_{i}" for i in range(n_data_columns)]
        return metadata_headers + data_headers

    @staticmethod
    def parse_string(str_):
        """Convert a string to a bool, int, double or just return the string.

        Metadata strings by definition only contain str parts, however these
        strings can represent any python object and need to be type-converted
        when the conversion to a metadata dictionary is performed. This
        implementation only covers bool, int and float values.

        Parameters
        ----------
        str_ : str
            The string to be converted to another type.

        Returns
        -------
        value
            The string value converted to the appropriate type.
        """
        if str_ == "False":
            return False
        elif str_ == "True":
            return True

        try:
            value = int(str_)
        except ValueError:
            try:
                value = float(str_)
            except ValueError:
                value = str_

        return value

    def to_measurement_set(self, merge_redundant: bool = True):
        """Convert this tabular dataset to a measurement set.

        All values in the metadata columns have to be converted
        into key-value pairs and provided as the measurement
        conditions. There is no way to separate conditions from
        details so the assumption is made that all values in the
        flat data are important. When this assumption fails, the
        user can later edit the returned measurement set's metadata
        directly.

        The column headers for the metadata columns are simply used
        directly as dictionary keys for the corresponding variable
        values.
        Each dataset class is responsible for flattening its x-values
        into column headers and so must be responsible for deflattening
        them back into x-values. Each Dataset subclass must therefore
        have a static method for de-stringifying its labels.
        If we cannot get x-values then by definition we have a collection
        of scalars, I think.

        Returns
        -------
        MeasurementSet
        """
        measurements = []
        for i, row in enumerate(self.data):  # over measurements

            conditions = {}

            for j, variable_value in enumerate(row[0:self.n_metadata_columns]):
                conditions[self.column_headers[j]] = variable_value

            datasets = []
            str_data_values = row[self.n_metadata_columns:]
            data_values = []
            for value in str_data_values:
                data_values.append(self.parse_string(value))

            dataset_start_index = 0
            for dataset_end_index, dataset_type in zip(self._dataset_end_indices, self.dataset_types):

                column_start_index = dataset_start_index + self._n_metadata_columns
                column_end_index = dataset_end_index + self.n_metadata_columns
                # x_label, y_label, x_values = dataset_type.decode_column_labels(
                #     self.column_headers[column_start_index:column_end_index])
                #
                # y_values = dataset_type.unflatten_dependent_variables(data_values[dataset_start_index: dataset_end_index])
                #
                # dataset = dataset_type(dependent_variable_data=y_values,
                #                        independent_variable_data=x_values,
                #                        independent_variable_names=x_label,
                #                        dependent_variable_name=y_label)

                dataset = dataset_type.unflatten(self.column_headers[column_start_index:column_end_index],
                                                 np.array(data_values[dataset_start_index: dataset_end_index]))

                datasets.append(dataset)
                dataset_start_index = dataset_end_index

            measurements.append(piblin.data.data_collections.measurement.Measurement(datasets,
                                                                                          conditions=conditions,
                                                                                          details={}))

        return piblin.data.data_collections.measurement_set.MeasurementSet(measurements, merge_redundant=merge_redundant)

    def _row_to_measurement(self, row: List[object]):
        """Convert a single row of flat data to a measurement.

        Parameters
        ----------
        row : list of object
            The row corresponding to a single measurement.

        Returns
        -------
        Measurement
            The measurement corresponding to the given row.
        """
        conditions = self._row_metadata_to_dict(row[0: self.n_metadata_columns])
        datasets = []

        return piblin.data.data_collections.measurement.Measurement(datasets=datasets,
                                                                         conditions=conditions,
                                                                         details={})

    def _row_data_to_datasets(self):
        """Convert a single row of data to datasets."""
        ...

    def _row_metadata_to_dict(self, row_metadata):
        """Convert a single row of metadata to a conditions dictionary.

        Parameters
        ----------
        row_metadata : list of object
            The row corresponding to conditions of a single measurement.

        Returns
        -------
        metadata_dict : dict
            The dictionary corresponding to the given list.
        """
        metadata_dict = {}
        for metadata_index, metadata_value in enumerate(row_metadata[0: self.n_metadata_columns]):
            metadata_dict[self.column_headers[metadata_index]] = metadata_value
        return metadata_dict

    def to_experiment_set(self):
        """Convert this tabular data to an experiment set."""
        return piblin.data.data_collections.experiment_set.ExperimentSet.from_measurement_set(self.to_measurement_set())

    # some of these may be private variables - reassess

    @property
    def n_metadata_columns(self):
        return self._n_metadata_columns

    @property
    def n_data_columns(self):
        return self._n_data_columns

    @property
    def dataset_types(self):
        return self._dataset_types

    @property
    def metadata_columns(self):
        return self.data[0:self.n_metadata_columns]

    @property
    def dataset_columns(self):
        return np.array(self.data)[:, self.n_metadata_columns:]

    @property
    def data(self):
        return self._data

    @property
    def column_headers(self):
        return self._column_headers

    def to_csv(self, filepath):

        with open(filepath, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(self.column_headers)
            writer.writerows(self.data)

    def to_mixing_studio(self):
        """Return a URL to a webapp for statistical modelling.

        Use the Mixing Studio API to add a model and get a URL to it.

        Returns
        -------
        url: str
        """
        ...

    def __str__(self):
        """Create a human-readable representation of this tabular dataset."""
        str_rep = self.__class__.__name__ + "\n"
        str_rep += "-" * (len(str_rep) - 1) + "\n\n"

        str_rep += f"Number of Rows: {self.n_columns}\n"
        str_rep += f"Number of Cols: {self.n_rows}"
        str_rep += f", of which the first {self._n_metadata_columns} contain metadata\n"

        if self.n_rows != 0:

            header_str = "\n| "
            for column_header in self.column_headers:
                header_str += f"{column_header} | "
            header_str = header_str[:-1] + "\n"

            str_rep += header_str + "-" * (len(header_str) - 1) + "\n"

            data_str = "| "
            for row in range(self.n_rows):
                for column in range(self.n_columns):
                    data_str += f"{self.data[row][column]} | "
                data_str = data_str[:-1] + "\n| "

            str_rep += data_str[:-3]

        if self.n_datasets > 0:

            str_rep += "\n\nDataset Breakdown\n------------------\n"
            for dataset_type, dataset_end_index in zip(self.dataset_types, self._dataset_end_indices):
                str_rep += f"{dataset_type.__name__} ({dataset_end_index})\n"

        return str_rep


class ConsistentTabularMeasurementSet(TabularMeasurementSet):
    """A tabular measurement set where each row contains the same number and type of datasets."""
    ...


class TidyTabularMeasurementSet(ConsistentTabularMeasurementSet):
    """A consistent rabular measurement set where datasets have the same number of points."""
    ...
