from typing import List, Tuple

import numpy as np
import numpy.typing
import piblin.data.data_collections.measurement as measurement_
import piblin.data.data_collections.consistent_measurement_set as consistent_measurement_set


class TidyMeasurementSet(consistent_measurement_set.ConsistentMeasurementSet):
    """A measurement set with consistent, tidy measurements.

    The fact that this class is tidy means that the dataset_independent_variables
    and dataset_lengths properties are single lists in comparison to the
    measurement set and consistent measurement set classes.

    Parameters
    ----------
    measurements : list of Measurement
        The measurements of this tidy measurement set.
    """
    def __init__(self,
                 measurements: List[measurement_.Measurement] = None,
                 merge_redundant: bool = True):

        super().__init__(measurements=measurements,
                         merge_redundant=merge_redundant)

        if not self.is_tidy:
            raise ValueError("Measurements are not tidy.")

    @property
    def dataset_lengths(self) -> List[int]:
        """The number of data points in each dataset of this tidy measurement set."""
        if not self.measurements:
            return []
        return self.measurements[0].dataset_lengths

    # # should create a tabular dataset from your args then convert
    # @classmethod
    # def from_flat_data(cls,
    #                    flat_data: np.ndarray,
    #                    dataset_types: List[type],
    #                    dataset_end_indices: List[int],
    #                    dataset_x_values: List[np.ndarray]):
    #     """Create a tidy measurement set from flat data.
    #
    #     Parameters
    #     ----------
    #     flat_data : numpy.ndarray
    #         The data to convert to a set of measurements.
    #     dataset_types : list of type
    #         The type of each dataset in all measurements.
    #     dataset_end_indices : list of int
    #         The index of the final point in each dataset for all measurements.
    #     dataset_x_values : list of numpy.ndarray
    #         The independent variable values of each dataset for all measurements.
    #
    #     Returns
    #     -------
    #     ConsistentMeasurementSet
    #         A measurement set created from the flat data.
    #     """
    #     measurements = []
    #     for i, row in enumerate(flat_data):
    #
    #         measurements.append(piblin.data.data_collections.measurement.Measurement.from_flat_data(row,
    #                                                                                                      dataset_end_indices,
    #                                                                                                      dataset_types,
    #                                                                                                      dataset_x_values))
    #
    #     return cls(measurements)

    @property
    def mean(self) -> measurement_.Measurement:
        """The mean measurement of this experiment's repetitions.

        Returns
        -------
        measurement : Measurement
            The mean measurement of this experiment's repetitions.
        """
        return self.compute_mean()

    def compute_mean(self) -> measurement_.Measurement:
        """Compute the mean measurement of this consistent measurement set.

        Computing the mean measurement requires that the measurement set be tidy.
        Therefore, this method may require interpolation.
        """
        return NotImplemented

    def __compute_mean_experiment(self) -> measurement_.Measurement:
        """Compute the mean experiment over the repetitions."""
        if len(self.measurements) == 1:
            return self.measurements[0]

        if not self.is_tidy:
            self.to_tidy_measurement_set()

        _, flat_data = self.flatten_datasets()

        mean_data = flat_data.mean(axis=0)

        mean_datasets = []
        dataset_index = 0
        dataset_start_index = 0
        for dataset_length, dataset_type in zip(self.dataset_lengths,
                                                self.dataset_types):

            mean_dataset = dataset_type(mean_data[dataset_start_index:dataset_start_index + dataset_length],
                                        self.datasets[0][dataset_index].independent_variable_data,
                                        dependent_variable_name=self.repetitions[0][dataset_index].dependent_variable_name)
            mean_datasets.append(mean_dataset)

            dataset_start_index = dataset_length
            dataset_index += 1

        return measurement_.Measurement(mean_datasets,
                                        self.equal_shared_conditions,
                                        self.equal_shared_details)

    @property
    def data_variation(self):
        return None

    def __compute_data_variation(self):
        """Compute a measure of the variation in the replicate datasets.

        Returns
        -------
        list
        """
        _, flat_data = self.flatten_datasets()

        var_data = flat_data.std(axis=0)

        var_datasets = []
        dataset_index = 0
        dataset_start_index = 0
        for dataset_length, dataset_type in zip(self.dataset_lengths,
                                                self.dataset_types):

            var_dataset = var_data[dataset_start_index:dataset_start_index + dataset_length]
            var_datasets.append(var_dataset)

            dataset_start_index = dataset_length
            dataset_index += 1

        return var_datasets

    def flatten_datasets(self) -> Tuple[List[str],
                                        List[np.ndarray]]:
        """Flatten the datasets of this tidy measurement set.

        The tidy measurement set is consistent and by definition has datasets in each measurement
        that share identical independent variable values. As such, a single set of data headers
        can be produced that correctly describe each individual measurement's flattened data.
        This method creates a single set of headers and data rows from the measurements' datasets
        by appending.

        Returns
        -------
        dataset_column_headers : List of str
            The shared (across datasets) headers of this tidy measurement set.
        dataset_rows : List of np.ndarray
            The datasets, each flattened to a single numpy array of the appropriate dtype.
        """
        dataset_column_headers = []
        for dataset in self.measurements[0].datasets:
            column_labels: List[str]
            column_labels, _ = dataset.flatten()
            dataset_column_headers.extend(column_labels)

        dataset_rows = []
        for measurement in self.measurements:

            dataset_values: List[object] = []
            for dataset in measurement.datasets:
                data: np.ndarray
                _, data = dataset.flatten()
                dataset_values.extend(data)

            dataset_values: np.ndarray = np.asarray(dataset_values)
            dataset_rows.append(dataset_values)

        return dataset_column_headers, dataset_rows
