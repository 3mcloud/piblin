import numpy as np
import numpy.typing

import piblin.data.datasets.abc.split_datasets as split_datasets
import piblin.data.datasets.abc.split_datasets.one_dimensional_dataset as one_dimensional_dataset
import piblin.data.datasets.abc.split_datasets.two_dimensional_dataset as two_dimensional_dataset
import piblin.data.datasets.abc.split_datasets.zero_dimensional_dataset as zero_dimensional_dataset
import piblin.data.datasets.abc.unambiguous_datasets.unambiguous_dataset as unambiguous_dataset


class DatasetFactory:
    @staticmethod
    def from_split_data(
        dependent_variable_data: numpy.typing.ArrayLike,
        dependent_variable_names: list[str] = None,
        dependent_variable_units: list[str] = None,
        independent_variable_data: list[numpy.typing.ArrayLike] = None,
        independent_variable_names: list[str] = None,
        independent_variable_units: list[str] = None,
        source: str = None,
    ):
        """Create the appropriate dataset from split data."""
        import piblin.data.datasets.abc.split_datasets.zero_dimensional_dataset as zero_dimensional_dataset

        dependent_variable_data = np.array(dependent_variable_data)

        if independent_variable_data is not None:
            number_of_independent_dimensions = len(independent_variable_data)
        else:
            number_of_independent_dimensions = dependent_variable_data.ndim

        if number_of_independent_dimensions == 0:
            return zero_dimensional_dataset.ZeroDimensionalDataset(
                dependent_variable_data=dependent_variable_data,
                dependent_variable_names=dependent_variable_names,
                dependent_variable_units=dependent_variable_units,
                independent_variable_data=independent_variable_data,
                independent_variable_names=independent_variable_names,
                independent_variable_units=independent_variable_units,
                source=source,
            )

        if number_of_independent_dimensions == 1:
            return one_dimensional_dataset.OneDimensionalDataset(
                dependent_variable_data=dependent_variable_data,
                dependent_variable_names=dependent_variable_names,
                dependent_variable_units=dependent_variable_units,
                independent_variable_data=independent_variable_data,
                independent_variable_names=independent_variable_names,
                independent_variable_units=independent_variable_units,
                source=source,
            )

        elif number_of_independent_dimensions == 2:
            return two_dimensional_dataset.TwoDimensionalDataset(
                dependent_variable_data=dependent_variable_data,
                dependent_variable_names=dependent_variable_names,
                dependent_variable_units=dependent_variable_units,
                independent_variable_data=independent_variable_data,
                independent_variable_names=independent_variable_names,
                independent_variable_units=independent_variable_units,
                source=source,
            )

        else:
            return split_datasets.SplitDataset(
                dependent_variable_data=dependent_variable_data,
                dependent_variable_names=dependent_variable_names,
                dependent_variable_units=dependent_variable_units,
                independent_variable_data=independent_variable_data,
                independent_variable_names=independent_variable_names,
                independent_variable_units=independent_variable_units,
                source=source,
            )

    @staticmethod
    def from_data(
        data: set[tuple[tuple, tuple]],
        variable_names: tuple[tuple[str], tuple[str]] = None,
        variable_units: tuple[tuple[str], tuple[str]] = None,
        source: str = None,
    ):
        """Given a set of data, produce the least redundant representation.

        Parameters
        ----------
        data
        variable_names
        variable_units
        source
        """

        unambiguous_dataset_ = unambiguous_dataset.UnambiguousDataset(
            data=data, variable_names=variable_names, variable_units=variable_units, source=source
        )

        if unambiguous_dataset_.number_of_independent_dimensions == 0:
            return zero_dimensional_dataset.ZeroDimensionalDataset(
                dependent_variable_data=unambiguous_dataset_.dependent_variable_data,
                dependent_variable_names=unambiguous_dataset_.dependent_variable_names,
                dependent_variable_units=unambiguous_dataset_.dependent_variable_units,
                independent_variable_data=unambiguous_dataset_.independent_variable_data,
                independent_variable_names=unambiguous_dataset_.independent_variable_names,
                independent_variable_units=unambiguous_dataset_.independent_variable_units,
                source=source,
            )

        elif unambiguous_dataset_.number_of_independent_dimensions == 1:
            return one_dimensional_dataset.OneDimensionalDataset(
                dependent_variable_data=unambiguous_dataset_.dependent_variable_data,
                dependent_variable_names=unambiguous_dataset_.dependent_variable_names,
                dependent_variable_units=unambiguous_dataset_.dependent_variable_units,
                independent_variable_data=unambiguous_dataset_.independent_variable_data,
                independent_variable_names=unambiguous_dataset_.independent_variable_names,
                independent_variable_units=unambiguous_dataset_.independent_variable_units,
                source=source,
            )

        elif unambiguous_dataset_.number_of_independent_dimensions == 2:
            return two_dimensional_dataset.TwoDimensionalDataset(
                dependent_variable_data=unambiguous_dataset_.dependent_variable_data,
                dependent_variable_names=unambiguous_dataset_.dependent_variable_names,
                dependent_variable_units=unambiguous_dataset_.dependent_variable_units,
                independent_variable_data=unambiguous_dataset_.independent_variable_data,
                independent_variable_names=unambiguous_dataset_.independent_variable_names,
                independent_variable_units=unambiguous_dataset_.independent_variable_units,
                source=source,
            )

        else:
            return split_datasets.SplitDataset(
                dependent_variable_data=unambiguous_dataset_.dependent_variable_data,
                dependent_variable_names=unambiguous_dataset_.dependent_variable_names,
                dependent_variable_units=unambiguous_dataset_.dependent_variable_units,
                independent_variable_data=unambiguous_dataset_.independent_variable_data,
                independent_variable_names=unambiguous_dataset_.independent_variable_names,
                independent_variable_units=unambiguous_dataset_.independent_variable_units,
                source=source,
            )
