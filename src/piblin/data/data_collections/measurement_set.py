"""
Module: MeasurementSet - Collections of scientific measurements.
"""
import copy
import collections.abc
from typing import Callable, Tuple, List, Set, Dict, Union

import numpy as np
import numpy.typing
from matplotlib import pyplot as plt
import matplotlib.axes
import matplotlib.figure
import matplotlib.colors
import piblin.data.data_collections.measurement as measurement_


class MeasurementSet(collections.abc.MutableSequence):
    """Indexed measurements with no pre-defined metadata relationships.

    The MeasurementSet is an indexed list of Measurement instances, and
    places condition and detail metadata into sets based on whether
    their keys are shared, and whether shared keys have equal or varying
    values across the set of measurements.
    The measurement set allows easy viewing and editing of the metadata
    associated with measurements so that condition and detail information
    can be added or removed, and supports conversion between the two types
    of metadata.
    A measurement set has two further properties which depend on the
    datasets of its measurements: consistency and tidiness. A consistent
    measurement set collects measurements with the same number and type
    of datasets. A tidy measurement set is a consistent measurement set
    whose measurements' datasets at a given index share independent
    variable values.

    Parameters
    ----------
    measurements : list of piblin.data.measurement.Measurement
        The measurements to be collected.

    Attributes
    ----------
    measurements -> list of piblin.data.measurement.Measurement
        The collected measurements.
    num_measurements -> int
        The number of collected measurements.
    datasets -> list of list of dataset
        The datasets of the set of measurements.
    num_datasets -> list of int
        The number of datasets in each measurement of the set.
    dataset_types -> list of list of type
        The types of datasets in each measurement of the set.
    dataset_lengths -> list of list of int  # length = num points?
        The lengths of datasets in each measurement of the set.
    dataset_independent_variables() -> list of np.ndarray
        The independent variables of each measurement of the set.
    is_tidy -> bool
        Determine if the measurement set is tidy.
    is_consistent -> bool
        Determine if this measurement set is consistent.
    all_condition_names -> Set of str
        All condition names present in at least one measurement.
    shared_condition_names -> set of str
        The condition names shared by all measurements.
    unshared_condition_names -> set of str
        The condition names not shared by all measurements.
    equal_shared_condition_names -> set of str
        The shared condition names with the same value in all measurements.
    varying_shared_condition_names -> set of str
        The condition names shared by all measurements with varying values.
    equal_shared_conditions -> dict
        The conditions shared by all measurements with equal values.
    varying_shared_conditions -> dict
        The conditions shared by all measurements with varying values.
    all_detail_names -> set of str
        All detail names present in at least one measurement.
    shared_detail_names -> set of str
        The detail names shared by all measurements.
    unshared_detail_names -> set of str
        The detail names not shared by all measurements.
    equal_shared_detail_names -> set of str
        The detail names with the same value in all measurements.
    varying_shared_detail_names -> set of str
        The detail names shared by all measurements without the same value
        in all measurements.
    equal_shared_details -> dict
        The details shared by all measurements with the same value.

    Methods
    -------
    from_single_measurement(Measurement) -> MeasurementSet
        Create a measurement set from a single measurement.
    from_measurement_sets([MeasurementSet]) -> MeasurementSet
        Create a single measurement set by combining multiple sets.
    from_flat_data(np.ndarray, [[type]], [[int]], [[[object]]]) ->
    MeasurementSet
        Create a measurement set from flat data.
    to_consistent_measurement_set() -> ConsistentMeasurementSet
        Convert this measurement set to a consisntent measurement set.
    to_tidy_measurement_set([Interpolate]) -> TidyMeasurementSet
        Convert this measurement set to a tidy measurement set.
    to_tabular_dataset() -> TabularMeasurementSet
        Convert this measurement set to a tabular dataset.
    are_repetitions() -> bool
        Determine whether the list of measurements are repetitions.
    are_not_repetitions -> bool
        Determine whether the list of measurements are not repetitions.
    condition_to_detail -> None
        Convert a condition to a detail.
    conditions_to_details -> None
        Convert a set of conditions to details for all measurements.
    detail_to_condition -> None
        Convert a detail to a condition.
    details_to_conditions(list of str) -> None
        Convert a set of details to conditions for all measurements.
    add_equal_shared_condition() -> None
        Add a condition with a specified value to all measurements.
    add_equal_shared_conditions() -> None
        Add conditions with specified values to all measurements.
    add_varying_shared_condition() -> None
        Add a condition with different values for each measurement.
    add_varying_shared_conditions() -> None
        Add conditions with different values for each measurement.
    add_condition() -> None
        Add a condition to all measurements in the collection.
    add_detail() -> None
        Add a detail to all measurements in the collection.
    add_equal_shared_detail() -> None
        Add a detail with a specified value to all measurements.
    add_equal_shared_details() -> None
        Add details with specified values to all measurements.
    add_varying_shared_detail() -> None
        Add a detail with a specific value for each measurement.
    add_varying_shared_details() -> None
        Add details with varying values for each measurement.
    remove_condition() -> None
        Remove the condition with the specified name from all measurements.
    remove_conditions()
        Remove conditions with the specified name from all measurements.
    remove_detail() -> None
        Remove the detail with the specified name from all measurements.
    remove_details()
        Remove details with the specified name from all measurements.
    flatten()
        Convert the set of measurements to a flat dataset.
    flatten_datasets()
        Convert the measurement data to a set of rows.
    flatten_metadata()
        Convert the measurement metadata to a set of rows.
    datasets_at_index() -> list of dataset
        The dataset at a given index from each measurement.
    delete_dataset_of_measurement_at_index(int, int) -> None
        Delete the dataset at a specified index in a specified measurement.
    delete_all_datasets_at_index(int)
        Delete the dataset at the specified index from all measurements.
    remove_measurements_with_condition_name() -> None
        Remove measurements with the specified condition name.
    remove_measurements_without_condition_name() -> None
        Remove measurements without the specified condition name.
    remove_measurements_with_condition() -> None
        Remove measurements with the specified condition.
    remove_measurements_without_condition() -> None
        Remove measurements without the specified condition.
    remove_measurements_by_condition_test() -> None
        Remove measurements by testing a condition value.
    remove_measurements_by_conditions_test() -> None
        Remove measurements by testing multiple condition values.
    visualize() -> Figure, Axes
        Create a visual representation of this set of measurements.
    produce_color_map()
        Create a color map varying across measurements.
    compute_figure_size()
        Compute an appropriate size for the complete figure.
    validate_title()
        Assess and potentially create a title for the complete figure.
    validate_figsize()
        Assess and potentially create a size for the complete figure.
    produce_color_maps()
        Produce a color map for each measurement.
    visualize_on_single_plot()
        Visualize all measurements on a single plot per dataset.
    visualize_on_multiple_plots()
        Visualize the measurement datasets on individual plots.
    """
    def __init__(self,
                 measurements: List[measurement_.Measurement] = None,
                 merge_redundant: bool = True):

        if measurements is None:
            self.__measurements = []
        else:

            if not isinstance(measurements, list):
                raise ValueError(f"Measurements are not a list: "
                                 f"{measurements}")

            if not measurements:
                self.__measurements = []
            elif len(measurements) == 1:
                self.__measurements = measurements
            else:
                if merge_redundant:
                    self.__measurements = \
                        self.__combine_measurements(measurements)
                else:
                    self.__measurements = measurements

    @classmethod
    def from_single_measurement(cls,
                                measurement: measurement_.Measurement) -> \
            "MeasurementSet":
        """Create a measurement set from one measurement.

        Parameters
        ----------
        measurement : Measurement
            The measurement from which to create a measurement set.
        """
        return cls([measurement])

    @classmethod
    def from_measurement_sets(
            cls,
            measurement_sets: List["MeasurementSet"],
            merge_redundant: bool = True,
            ) -> "MeasurementSet":
        """Create a single combined measurement set from multiple.

        Parameters
        ----------
        measurement_sets : list of MeasurementSet
            The collections of measurements to combine.

        Returns
        -------
        MeasurementSet
            The combined collection of measurements.
        """
        all_measurements = []
        for measurement_set in measurement_sets:
            all_measurements.extend(measurement_set.measurements)

        return cls(all_measurements, merge_redundant=merge_redundant)

    @classmethod
    def __combine_measurements(cls, measurements: List[measurement_.Measurement]) -> List[measurement_.Measurement]:
        """Combine a set of measurements.

        Given any set of measurements, relationships may exist between them
        that allow them to be combined into a smaller set of measurements.
        By determining the unique sets of conditions across the
        measurements, new measurements can be created that are labelled
        with those conditions but combine datasets from multiple original
        measurements. The simplest example is shown below, where two
        initial measurements

        conditions = {label: A}, datasets = [a]
        conditions = {label: A}, datasets = [b]

        can be combined into a single measurement without loss of
        information.

        conditions = {label: A}, datasets = [a, b]

        This combination process does not guarantee that details will be
        equal in the combined measurements. If a detail differs between
        measurements to be combined, that detail is not included.

        Parameters
        ----------
        measurements : List of Measurement
            The measurements to be combined.

        Returns
        -------
        combined_measurements : List of Measurement
            The measurements after combination.
        """
        unique_sets_of_conditions = []
        for measurement in measurements:
            measurement_conditions = []
            for key, value in measurement.conditions.items():
                measurement_conditions.append((key, value))

            measurement_conditions = tuple(measurement_conditions)
            if measurement_conditions not in unique_sets_of_conditions:
                unique_sets_of_conditions.append(measurement_conditions)

        unique_dicts_of_conditions = []
        for unique_set_of_conditions in unique_sets_of_conditions:
            unique_dicts_of_conditions.append(dict(unique_set_of_conditions))

        # combine all measurements with each unique set of conditions by combining their datasets and details
        combined_measurements = []
        for unique_condition in unique_dicts_of_conditions:

            datasets = []
            combined_details = {}
            conflicting_detail_names = set()
            for measurement in measurements:
                if measurement.conditions == unique_condition:

                    datasets.extend(measurement.datasets)

                    if not combined_details:
                        combined_details = measurement.details
                    else:
                        for key, value in measurement.details.items():
                            if key not in combined_details and key not in conflicting_detail_names:
                                combined_details[key] = value
                            elif combined_details[key] != value:
                                conflicting_detail_names.add(key)
                                print(f"Warning: Detail with name {key} has differing values for merged measurements."
                                      f"\n\tConsider using the `merge_redundant=False` parameter to the read function.")
                                del combined_details[key]

            combined_measurements.append(measurement_.Measurement(datasets=datasets,
                                                                  conditions=unique_condition,
                                                                  details=combined_details))

        return combined_measurements

    @classmethod
    def combine(cls, measurement_sets: List["MeasurementSet"], merge_redundant: bool = True) \
            -> "MeasurementSet":
        """Alias for from_measurement_sets."""
        return cls.from_measurement_sets(measurement_sets, merge_redundant=merge_redundant)

    def __add__(self, other):
        return self.combine([self, other])

    def combine_zerod_datasets(self, x_name: str, y_name: str, merge_datasets: bool = True) -> None:
        """Perform combination of 0D datasets on a per-measurement basis.

        Parameters
        ----------
        x_name : str
            The name of the 0D dataset to use as the source of x-values.
        y_name : str
            The name of the 0D dataset to use as the source of y-values.
        merge_datasets : bool
            Whether to merge the resulting single-point datasets to
            a single multipoint dataset.
        """
        for measurement in self.measurements:
            measurement.combine_zerod_datasets(x_name=x_name,
                                               y_name=y_name,
                                               merge_datasets=merge_datasets)

    def combine_multi_zerod_datasets(self,
                                     x_name: str,
                                     y_names: List[str] = None,
                                     merge_datasets: bool = True) -> None:
        """Perform combination of 0D datasets on a per-measurement basis.

        Parameters
        ----------
        x_name : str
            The name of the 0D dataset to use as the source of x-values.
        y_names : List of str
            The names of the 0D datasets to use as the source of y-values.
        merge_datasets : bool
            Whether to merge the resulting single-point datasets to
            a single multipoint dataset.
        """
        for measurement in self.measurements:
            measurement.combine_multi_zerod_datasets(x_name=x_name,
                                                     y_names=y_names,
                                                     merge_datasets=merge_datasets)
    def combine_oned_datasets(self, x_name: str, y_name: str):
        """Combine multiple one-dimensional datasets.

        This procedure is done at the measurement level so this method
        simply iterates over measurements.

        Parameters
        ----------
        x_name : str
            The name of the independent variable.
        y_name : str
            The name of the dependent variable.
        """

        for measurement in self.measurements:
            measurement.combine_oned_datasets(x_name=x_name, y_name=y_name)

    def split_by_dataset_indices(self,
                                 dataset_indices: List[Set[int]]) \
            -> List["MeasurementSet"]:
        """Split this measurement set up by dataset indices.

        This split works by splitting each measurement in the set by
        dataset index, and then combining the resulting measurements
        into new sets.
        """
        for measurement_indices in dataset_indices:
            for dataset_index in measurement_indices:
                if dataset_index >= max(self.num_datasets):
                    raise IndexError(
                        f"Provided index not present in any measurement: "
                        f"{dataset_index}"
                    )

        split_measurements = [measurement.split_by_dataset_indices(
            dataset_indices=dataset_indices,
            conditions_to_remove=None,
            conditions_to_add=None,
            allow_empty_measurements=True)
            for measurement in self.measurements]

        measurement_sets = []
        for i in range(len(dataset_indices)):
            measurements = [measurement[i]
                            for measurement in split_measurements]
            measurement_sets.append(type(self)(measurements=measurements))

        return measurement_sets

    def split_by_condition_name(self,
                                condition_name) -> List["MeasurementSet"]:

        unique_values: Set = set()
        for measurement in self.measurements:
            unique_values.add(measurement.conditions[condition_name])
        list(unique_values).sort()

        measurement_sets = []
        for unique_value in unique_values:
            measurements = \
                [measurement
                 for measurement in self.measurements
                 if measurement.conditions[condition_name] == unique_value]

            measurement_sets.append(
                self.__class__(measurements=measurements,
                               merge_redundant=False)
            )

        return measurement_sets

    def split_by_dataset_independent_variable_name(
            self,
            independent_variable_name: str) \
            -> Tuple["MeasurementSet", "MeasurementSet"]:
        """Split this measurement set into two measurement sets by dataset
        independent variable name.

        This method iterates over each measurement in the set, splitting it
        into a pair of measurements, one containing datasets that have
        the given independent variable name, and one containing those that
        do not. Both of the resulting measurements will have the same
        metadata. These measurements are collected in two lists and each
        is combined into a measurement set.

        Parameters
        ----------
        independent_variable_name : str
            The name of the independent variable by which to split.

        Returns
        -------
        Tuple of MeasurementSet
            The measurement sets with and without the specified
            independent variable name.
        """

        measurements_with_independent_variable_name: \
            List[measurement_.Measurement] = []

        measurements_without_independent_variable_name: \
            List[measurement_.Measurement] = []

        for measurement in self.measurements:
            measurement_with_independent_variable_name, \
                measurement_without_independent_variable_name = \
                measurement.split_by_dataset_independent_variable_name(
                    independent_variable_name=independent_variable_name,
                    allow_empty_measurements=True
                )

            if measurement_with_independent_variable_name.datasets:
                measurements_with_independent_variable_name.append(
                    measurement_with_independent_variable_name
                )

            if measurement_without_independent_variable_name.datasets:
                measurements_without_independent_variable_name.append(
                    measurement_without_independent_variable_name
                )

        return \
            self.__class__(
                measurements=measurements_with_independent_variable_name,
                merge_redundant=False
            ), \
            self.__class__(
                measurements=measurements_without_independent_variable_name,
                merge_redundant=False
            )

    def split_by_dataset_independent_variable_names(
            self,
            independent_variable_names: List[str]
    ) -> Tuple["MeasurementSet", "MeasurementSet"]:
        """Split this measurement set into two measurement sets by dataset
        independent variable names.

        This method iterates over each measurement in the set, splitting it
        into a pair of measurements, one containing datasets that have
        the given independent variable names, and one containing those that
        do not. Both of the resulting measurements will have the same
        metadata. These measurements are collected in two lists and each
        is combined into a measurement set.

        Parameters
        ----------
        independent_variable_names : List of str
            The name of the independent variable by which to split.

        Returns
        -------
        Tuple of MeasurementSet
            The measurement sets with and without the specified
            independent variable names.
        """
        measurements_with_names = []
        measurements_without_names = []
        for measurement in self.measurements:
            measurement_with_names, \
                measurement_without_names = \
                measurement.split_by_dataset_independent_variable_names(
                    independent_variable_names
                )

            if measurement_with_names.datasets:
                measurements_with_names.append(
                    measurement_with_names
                )

            if measurement_without_names.datasets:
                measurements_without_names.append(
                    measurement_without_names
                )

        return self.__class__(measurements=measurements_with_names), \
            self.__class__(measurements=measurements_without_names)

    def split_by_dataset_dependent_variable_name(
            self,
            dependent_variable_name: str) -> List["Measurement"]:
        """Split this measurement into multiple measurements based on
        dataset dependent variable names.

        Parameters
        ----------
        dependent_variable_name : str

        Returns
        -------
        List of MeasurementSet
            The resulting list of measurement sets after the split.
        """
        all_matching_measurements = []
        all_non_matching_measurements = []
        for measurement in self.measurements:
            matching_measurements, non_matching_measurements = \
                measurement.split_by_dataset_dependent_variable_name(
                    dependent_variable_name
                )

            all_matching_measurements.append(matching_measurements)
            all_non_matching_measurements.append(non_matching_measurements)

        return [self.__class__(measurements=all_matching_measurements,
                               merge_redundant=False),
                self.__class__(measurements=all_non_matching_measurements,
                               merge_redundant=False)]

    def combine_datasets(self, dataset_indices=None) -> None:
        """Combine the datasets at the given indices for all measurements.

        Parameters
        ----------
        dataset_indices : List of int
            The indices of the datasets to combine.
        """
        for measurement in self.measurements:
            measurement.combine_datasets(dataset_indices=dataset_indices)

    # need to create tabular dataset from your flat data then convert
    # @classmethod
    # def from_flat_data(cls,
    #                    flat_data: np.ndarray,
    #                    dataset_types: List[List[type]],
    #                    dataset_end_indices: List[List[int]],
    #                    dataset_x_values: List[List[np.ndarray]]) -> "MeasurementSet":
    #     """Create a measurement set from flat data.
    #
    #     Parameters
    #     ----------
    #     flat_data : numpy.ndarray
    #         The data to convert to a set of measurements.
    #     dataset_types : list of list of type
    #         The type of each dataset in the measurements.
    #     dataset_end_indices : list of list of int
    #         The index of the final point in each dataset.
    #     dataset_x_values : list of list of numpy.ndarray
    #         The independent variable values of each dataset.
    #
    #     Returns
    #     -------
    #     MeasurementSet
    #         A measurement set created from the flat data.
    #
    #     Notes
    #     -----
    #     This method only supports creation from flat data, i.e. no metadata is added
    #     to the measurement set.
    #     """
    #     measurements = []
    #     for i, row in enumerate(flat_data):
    #
    #         measurements.append(measurement_.Measurement.from_flat_data(row,
    #                                                                     dataset_end_indices[i],
    #                                                                     dataset_types[i],
    #                                                                     dataset_x_values[i]))
    #
    #     return cls(measurements)

    def to_consistent_measurement_set(
            self,
            dataset_indices_to_retain: List[int] = None,
            make_copy=True):
        """Convert the set of measurements to a consistent measurement set.

        If this measurement set is not already consistent, datasets will
        need to be removed until it becomes so. Each dataset index is
        checked for consistency, and if it is found to not be consistent,
        it is removed from all measurements. This is repeated until all
        remaining datasets are consistent.

        Parameters
        ----------
        dataset_indices_to_retain : list of int
            The indices of the datasets to retain if consistent.
        make_copy : bool
            Whether to alter the measurement set in place or create a
            duplicate.

        Returns
        -------
        ConsistentMeasurementSet
            The consistent measurement set converted from this measurement
            set.
        """
        from piblin.data.data_collections.consistent_measurement_set \
            import ConsistentMeasurementSet

        if self.num_measurements == 0:
            return ConsistentMeasurementSet(self.measurements)

        if dataset_indices_to_retain is None:
            dataset_indices_to_retain = range(self.max_num_datasets)

        delete_datasets_at_index = [
            not self._has_consistent_dataset_type_at_index(dataset_index)
            for dataset_index in dataset_indices_to_retain
        ]

        if make_copy:
            edited_measurement_set = copy.deepcopy(self)
        else:
            edited_measurement_set = self

        num_datasets_deleted = 0
        for dataset_index in range(self.max_num_datasets):
            if dataset_index not in dataset_indices_to_retain:
                edited_measurement_set.remove_all_datasets_at_index(
                    dataset_index - num_datasets_deleted
                )
                num_datasets_deleted += 1

        for delete_dataset_at_index, \
                dataset_index in zip(delete_datasets_at_index,
                                     dataset_indices_to_retain):
            if delete_dataset_at_index:
                edited_measurement_set.remove_all_datasets_at_index(
                    dataset_index - num_datasets_deleted
                )
                num_datasets_deleted += 1

        return \
            ConsistentMeasurementSet(edited_measurement_set.measurements, merge_redundant=False)

    def to_tidy_measurement_set(
            self,
            interpolation_transforms: List["interpolate_.Interpolate1D"] = None
    ):
        """Convert this set of measurements to a tidy measurement set.

        This method first needs to ensure that the measurement set is
        consistent, which it does by conversion. Following this, each
        dataset at a given index must have the same independent variable
        values across all measurements. This is achieved by interpolation.
        If interpolation old_transforms are not provided, a linear
        interpolation is created based on the measurements present in
        the set.

        Parameters
        ----------
        interpolation_transforms : list of Interpolate
            The interpolation old_transforms to use on the datasets.
        """
        from piblin.data.data_collections.tidy_measurement_set \
            import TidyMeasurementSet

        if self.is_tidy:
            return TidyMeasurementSet(self.measurements,
                                      merge_redundant=False)

        consistent_measurement_set = self.to_consistent_measurement_set()

        # this should be in consistent measurement set class
        if consistent_measurement_set.is_tidy:
            return TidyMeasurementSet(
                consistent_measurement_set.measurements,
                merge_redundant=False
            )

        # for each dataset in each of the consistent measurements
        dataset_index_range = range(
            consistent_measurement_set.num_datasets
        )
        for dataset_index in dataset_index_range:

            # make sure there is an interpolation transform that can be applied
            if not self._has_common_independent_variables_at_index(dataset_index):
                if interpolation_transforms is None:
                    if consistent_measurement_set.datasets_at_index(dataset_index)[0].number_of_independent_dimensions == 1:

                        import piblin.transform.modify.interpolate as interpolate_
                        interpolate = interpolate_.Interpolate1D.from_measurement_set_datasets(consistent_measurement_set,
                                                                                               dataset_index)
                    else:
                        raise NotImplementedError("Auto-interpolation is only implemented for 1D datasets.")
                else:
                    interpolate = interpolation_transforms[dataset_index]

                # apply it to the current dataset for all measurements
                for measurement in consistent_measurement_set.measurements:
                    interpolate.apply_to(measurement.datasets[dataset_index],
                                         make_copy=False)

        return TidyMeasurementSet(consistent_measurement_set.measurements,
                                  merge_redundant=False)

    def flatten(self,
                force_tidiness: bool = False,
                include_unshared_conditions: bool = False,
                include_equal_conditions: bool = False,
                default_value: object = None) \
            -> Tuple[List[str], List[List[object]]]:
        """Convert the set of measurements to a flat dataset.

        In most cases of interest, a flattened measurement set will be
        created from a consistent, tidy measurement set which can have a
        single common set of column headers and then a row for each
        measurement's metadata and data values. The production of a tabular
        measurement set from a tidy measurement set is done in the tidy
        measurement set class. If the measurement set is not consistent
        or consistent but not tidy, there is no single set of column
        headers and the flat dataset is created with a set of headers
        for every measurement in the set. There are two cases of
        relevance then, whether the measurement set is tidy or not. If
        it is tidy, the flattening can just proceed as normal. If not tidy,
        a flag can be used to attempt to force tidiness, or the measurement
        set can just be written with column headers per measurement. In
        the latter case a list of per-row headers and per-row data values
        is returned rather than a cralds data collection class.

        Parameters
        ----------
        force_tidiness : bool
            Whether to make this measurement set tidy before flattening.
        include_unshared_conditions : bool
            Whether to include conditions not defined for all measurements.
            Default is a tidy dataset which has specific values for all
            entries.
        include_equal_conditions : bool
            Whether to include conditions which are equal for all
            measurements. Default is a tidy dataset which does not
            include redundant columns.
        default_value : object
            The value to set for missing conditions.
            Used only if include_unshared_conditions is set true.

        Returns
        -------

        """
        if (not self.is_tidy and force_tidiness) or self.is_tidy:

            tidy_measurement_set = self.to_tidy_measurement_set()

            metadata_headers, metadata_values = \
                tidy_measurement_set.flatten_metadata(
                    include_unshared_conditions=include_unshared_conditions,
                    include_equal_conditions=include_equal_conditions,
                    default_value=default_value
                )

            data_header, data_values = \
                tidy_measurement_set.flatten_datasets()

            column_headers = metadata_headers
            column_headers.extend(data_header)

            flat_data = []
            for metadata_value, data_value in zip(metadata_values,
                                                  data_values):

                all_data = metadata_value + list(data_value)

                if len(column_headers) != len(all_data):
                    raise ValueError(
                        f"Error flattening untidy measurement set: "
                        f"{len(column_headers)} Column Headers"
                        f"{len(all_data)} Data Columns"
                    )

                flat_data.append(all_data)

            return column_headers, flat_data

        elif not self.is_tidy and not force_tidiness:

            metadata_headers, metadata_values = self.flatten_metadata(
                include_unshared_conditions=include_unshared_conditions,
                include_equal_conditions=include_equal_conditions,
                default_value=default_value
            )

            data_headers, data_values = self.flatten_datasets()

            flat_headers = []
            flat_data = []
            for metadata_value, \
                    data_header, \
                    data_value in zip(metadata_values,
                                      data_headers,
                                      data_values):
                column_headers = metadata_headers + data_header

                all_data = metadata_value + list(data_value)

                if len(column_headers) != len(all_data):
                    raise ValueError(
                        f"Error flattening untidy measurement set: "
                        f"{len(column_headers)} {len(all_data)}"
                    )

                flat_headers.append(column_headers)
                flat_data.append(all_data)

            return flat_headers, flat_data

    def flatten_datasets(self,
                         force_tidiness=True) \
            -> Tuple[List[numpy.typing.NDArray], List[List[object]]]:
        """Flatten the datasets of this measurement set.

        It is unclear if this is useful to anyone, but it is kept here for
        comparison to the same method for a tidy measurement set.
        Because there is no guarantee on the lengths of datasets, this
        method will compute a set of column labels for every measurement
        in the set.
        """
        if force_tidiness:
            measurement_set = self.to_tidy_measurement_set()
        else:
            measurement_set = self

        all_headers = []
        all_values = []
        for measurement in measurement_set.measurements:
            dataset_headers = []
            dataset_values = []
            for dataset in measurement.datasets:
                column_labels, data = dataset.flatten()
                dataset_headers.extend(column_labels)
                dataset_values.extend(data)

            all_headers.append(dataset_headers)
            all_values.append(dataset_values)

        return all_headers, all_values

    def flatten_metadata(self,
                         include_unshared_conditions: bool = False,
                         include_equal_conditions: bool = False,
                         default_value: object = None) \
            -> Tuple[List[str], List[List[object]]]:
        """Convert the measurement metadata to a set of rows.

        Each measurement can be flattened by a call to measurement.flatten.
        However, this call will simply return a row for that measurement with
        all of its conditions included. When combining two measurements with
        different condition names, this creates a problem.
        We need to use some of the condition sets defined in this class to
        specify the number of columns in the complete spreadsheet, then
        call flatten specifically requesting the metadata that we want.
        We also don't need to get the column names back from the measurements.
        We instead impose them here.
        The measurements are responsible for flattening themselves.
        Several measurements can only be reliably flattened if they have the
        same types of datasets and can be placed on common axes. That is,
        there are conditions on a set of measurements which must be met before
        it can be flattened and those can be checked herein and fixed if not
        met.

        Parameters
        ----------
        include_unshared_conditions : bool
            Whether to include all metadata, resulting in some empty cells.
            Set false by default, which results in a complete flat dataset.
        include_equal_conditions : bool
            Whether to include equal condition metadata, resulting in redundant columns.
            Set false by default, resulting in no redundant data being included.
        default_value : object
            The value to set for missing conditions.
            Used only if include_unshared_conditions is set true.

        Returns
        -------
        column_headers : list of str
            Identifiers for each column of the metadata array.
        numpy.ndarray
            The flattened metadata array.
        """
        if include_unshared_conditions and include_equal_conditions:
            column_headers = self.all_condition_names
        elif include_unshared_conditions and not include_equal_conditions:
            column_headers = self.all_condition_names - self.equal_shared_condition_names
        elif not include_unshared_conditions and include_equal_conditions:
            column_headers = self.shared_condition_names
        else:
            column_headers = self.varying_shared_condition_names

        column_headers = list(column_headers)

        rows = []
        for measurement in self.measurements:
            _, row = measurement.flatten_metadata(
                condition_names=column_headers,
                default_value=default_value
            )
            rows.append(row)

        return column_headers, rows

    def to_tabular_measurement_set(
            self,
            include_unshared_conditions: bool = False,
            include_equal_conditions: bool = False,
            default_value: object = None
    ) -> "piblin.data.TabularMeasurementSet":
        """Convert this set of a measurements to a tabular dataset.

        By default this method will create a *clean* flat dataset,
        i.e. one for which the value of every variable is known for
        every sample, and that value differs for at least 2 samples.
        These criteria can be broken individually by setting the
        appropriate parameter to true. Including unshared conditions
        will result in an empty cell for each sample which does not have
        a value for the corresponding key. Including equal conditions
        will add redundant columns where all cells are filled, but the
        variable value is the same for all.

        This method does not support specifying the inclusion of
        particular conditions or *any* details; it is expected that the
        user will edit this information using the methods of the
        measurement set directly.

        Parameters
        ----------
        include_unshared_conditions : bool
            Whether to include all metadata, resulting in some empty cells.
            Set false by default, which results in a complete flat dataset.
        include_equal_conditions : bool
            Whether to include equal condition metadata, resulting in
            redundant columns.
            Set false by default, resulting in no redundant data being
            included.
        default_value : object
            The value to use for missing metadata.
        """
        n_metadata_columns = self._num_metadata_keys(
            include_unshared_conditions=include_unshared_conditions,
            include_equal_conditions=include_equal_conditions
        )

        headers, columns = self.flatten(
            force_tidiness=True,
            include_unshared_conditions=include_unshared_conditions,
            include_equal_conditions=include_equal_conditions,
            default_value=default_value
        )

        import piblin.data.data_collections.tabular_measurement_set
        return piblin.data.data_collections.tabular_measurement_set.TabularMeasurementSet(
            data=columns,
            n_metadata_columns=n_metadata_columns,
            column_headers=headers,
            dataset_types=self.dataset_types,
            dataset_end_indices=self.dataset_lengths
        )

    def to_xarray(self, merge_dependent_data: bool = False):
        """Convert MeasurementSet to list of xarray Datasets

        return tabular_dataset_.TabularMeasurementSet(data=columns,
                                                      n_metadata_columns=n_metadata_columns,
                                                      column_headers=headers,
                                                      dataset_types=self.dataset_types,
                                                      dataset_end_indices=self.dataset_lengths)

        Parameters
        ----------
        merge_dependent_data : bool, optional
            If True, dependent data in each Measurement with the same dependent varible name
            are stacked along a new 'dataset' dimension, by default False

        Returns
        -------
        List[xarray.Dataset]
            List containing a xarray.Dataset for each Measurement in MeasurementSet
        """
        measurements = []
        for n, measurement in enumerate(self.measurements):
            ds = measurement.to_xarray(
                merge_dependent_data=merge_dependent_data)
            ds = ds.assign_coords(measurement=n)
            measurements.append(ds)
        return measurements

    @classmethod
    def from_xarray(cls, xr_data) -> "MeasurementSet":
        """Create a measurement set from an xarray dataset."""
        import xarray as xr
        from piblin.data.data_collections.measurement import Measurement

        if not isinstance(xr_data, (list, tuple)) or not all(
                isinstance(ds, xr.Dataset) for ds in xr_data):
            raise ValueError(
                "`xr_data` should be a list of xarray Datasets."
            )

        measurements = []
        for ds in xr_data:
            if "measurement" in ds.coords:
                ds = ds.drop_vars("measurement")
            ds = Measurement.from_xarray(ds)
            measurements.append(ds)

        return cls(measurements)

    @property
    def measurements(self) -> List[measurement_.Measurement]:
        """The collected measurements."""
        return self.__measurements

    @measurements.setter
    def measurements(self, measurements: List[measurement_.Measurement]):
        self.__measurements = measurements

    @property
    def num_measurements(self) -> int:
        """The number of collected measurements."""
        return len(self.measurements)

    @property
    def datasets(self) -> List[List[object]]:
        return [measurement.datasets for measurement in self.measurements]

    @property
    def num_datasets(self) -> List[int]:
        """The number of datasets in each measurement of the set."""
        return [len(measurement.datasets)
                for measurement in self.measurements]

    @property
    def max_num_datasets(self) -> int:
        return max(self.num_datasets)

    @property
    def dataset_types(self) -> List[List[type]]:
        """The types of datasets in each measurement of this set."""
        return [measurement.dataset_types
                for measurement in self.measurements]

    @property
    def dataset_lengths(self) -> List[List[int]]:
        """The lengths of datasets in each measurement of the set."""
        return [measurement.dataset_lengths
                for measurement in self.measurements]

    @property
    def dataset_independent_variable_data(self) -> List[List[np.ndarray]]:
        return [measurement.dataset_independent_variable_data
                for measurement in self.measurements]

    @property
    def is_tidy(self) -> bool:
        """Determine if this measurement set is tidy."""
        return self.is_consistent & \
            self._has_common_independent_variables()

    @property
    def is_consistent(self) -> bool:
        """Determine if this measurement set is consistent.

        Returns
        -------
        bool
            Whether this measurement set is consistent.
        """
        return self._has_consistent_num_datasets() & \
            self._has_consistent_dataset_types() & \
            self._has_consistent_dataset_dependent_variable_names() & \
            self._has_consistent_dataset_dependent_variable_units() & \
            self._has_consistent_dataset_independent_variable_names() & \
            self._has_consistent_dataset_independent_variable_units()

    @property
    def consistency_str(self):
        """Create a string description of consistency properties."""
        return \
            f"Consistent number of datasets?:\t" \
            f"{self._has_consistent_num_datasets()}" \
            f"\nConsistent types of dataset?:\t" \
            f"{self._has_consistent_dataset_types()}" \
            f"\nConsistent dep. var. names?:\t" \
            f"{self._has_consistent_dataset_dependent_variable_names()}" \
            f"\nConsistent dep. var. units?:\t" \
            f"{self._has_consistent_dataset_dependent_variable_units()}" \
            f"\nConsistent ind. var. names?:" \
            f"\t{self._has_consistent_dataset_independent_variable_names()}" \
            f"\nConsistent ind. var. units?:\t" \
            f"{self._has_consistent_dataset_independent_variable_units()}" \
            f"\n"

    def _has_consistent_num_datasets(self) -> bool:
        """Check if the measurements have a consistent number of datasets.

        By definition, an empty measurement set has a consistent number
        of datasets. A measurement set with a single measurement also has
        a consistent number of datasets as no comparison measurement
        exists.

        Returns
        -------
        bool
            Whether the number of datasets of each measurement is
            consistent.
        """
        if self.num_measurements <= 1:
            return True

        num_datasets = len(self.measurements[0].datasets)
        for measurement in self.measurements[1:]:
            if len(measurement.datasets) != num_datasets:
                return False
        return True

    def _has_consistent_dataset_types(self) -> bool:
        """Check whether this set of measurements have consistent dataset
        types.

        By definition, an empty measurement set has consistent dataset
        types.

        Returns
        -------
        bool
            Whether the type of datasets in each measurement is consistent.
        """
        if self.num_measurements == 0:
            return True

        dataset_types = [type(dataset)
                         for dataset in self.measurements[0].datasets]

        for measurement in self.measurements[1:]:
            if [type(dataset) for dataset in measurement.datasets] != \
                    dataset_types:
                return False

        return True

    def _has_consistent_dataset_dependent_variable_names(self) -> bool:
        """Check whether measurements have consistent dataset dependent variable names.

        Returns
        -------
        bool
            Whether the dependent variable names of the datasets in each
            measurement is consistent.
        """
        if self.num_measurements == 0:
            return True

        dataset_dependent_variable_names = \
            [dataset.dependent_variable_names
             for dataset in self.measurements[0].datasets]

        for measurement in self.measurements[1:]:
            if [dataset.dependent_variable_names
                for dataset in measurement.datasets] != \
                    dataset_dependent_variable_names:
                return False

        return True

    def _has_consistent_dataset_dependent_variable_units(self) -> bool:
        """Check whether this set of measurements have consistent dataset dependent variable units.

        Returns
        -------
        bool
            Whether the dependent variable units of the datasets in each
            measurement is consistent.
        """
        if self.num_measurements == 0:
            return True

        dataset_dependent_variable_units = \
            [dataset.dependent_variable_units
             for dataset in self.measurements[0].datasets]

        for measurement in self.measurements[1:]:
            if [dataset.dependent_variable_units
                for dataset in measurement.datasets] != \
                    dataset_dependent_variable_units:
                return False

        return True

    def _has_consistent_dataset_independent_variable_names(self) -> bool:
        """Check whether this set of measurements have consistent dataset independent variable names.

        Returns
        -------
        bool
            Whether the independent variable names of the datasets in each
            measurement is consistent.
        """
        if self.num_measurements == 0:
            return True

        dataset_independent_variable_names = \
            [dataset.independent_variable_names
             for dataset in self.measurements[0].datasets]

        for measurement in self.measurements[1:]:
            if [dataset.independent_variable_names
                for dataset in measurement.datasets] != \
                    dataset_independent_variable_names:
                return False

        return True

    def _has_consistent_dataset_independent_variable_units(self) -> bool:
        """Check whether this set of measurements have consistent dataset independent variable units.

        Returns
        -------
        bool
            Whether the independent variable units of the datasets in each
            measurement is consistent.
        """
        if self.num_measurements == 0:
            return True

        dataset_independent_variable_units = \
            [dataset.independent_variable_units
             for dataset in self.measurements[0].datasets]

        for measurement in self.measurements[1:]:
            if [dataset.independent_variable_units
                for dataset in measurement.datasets] != \
                    dataset_independent_variable_units:
                return False

        return True

    def _has_consistent_dataset_type_at_index(self, dataset_index: int) -> bool:
        """Check whether this set of measurements have consistent dataset types at a given index.

        Parameters
        ----------
        dataset_index : int
            The index of the dataset to check for consistency.

        Returns
        -------
        bool
            Whether the type of datasets are the given index are
            consistent.
        """
        for measurement in self.measurements:
            if dataset_index >= measurement.num_datasets:
                return False

        dataset_type = type(self.measurements[0].datasets[dataset_index])
        for measurement in self.measurements[1:]:
            if type(measurement.datasets[dataset_index]) != dataset_type:
                return False

        return True

    def _has_common_independent_variables(self) -> bool:
        """Determine whether this measurement set's datasets share independent variables.

        To be "tidy", a measurement set must first be consistent, i.e. all
        measurements must have the same number and types of datasets.
        Tidiness refers to the ability to turn the measurement set into
        a rectangular array, which requires all datasets at a given
        index to not only be the same type, but also to share the exact
        same set of independent variables.

        Returns
        -------
        bool
            True iff this measurement set's datasets share independent
            variables.
        """
        if self.num_measurements == 0:
            return True

        if not self.is_consistent:
            return False

        for dataset_index in range(0, self.measurements[0].num_datasets):
            if not self._has_common_independent_variables_at_index(dataset_index):
                return False

        return True

    def _has_common_independent_variables_at_index(self, dataset_index: int) -> bool:
        """Determine whether one of this measurement set's datasets shares independent variables.

        Parameters
        ----------
        dataset_index : int
            Index of the dataset to check.

        Returns
        -------
        bool
            True iff this measurement set's datasets at the given index share independent variables.
        """
        for measurement in self.measurements[1:]:
            dataset = measurement.datasets[dataset_index]
            for independent_index in range(dataset.number_of_independent_dimensions):
                if not np.array_equal(dataset.independent_variable_data[independent_index],
                                      self.measurements[0].datasets[dataset_index].independent_variable_data[independent_index]):
                    return False

        return True

    def remove_dataset_of_measurement_at_index(self, measurement_index, dataset_index) -> None:
        """Delete the dataset at the specified index in the specified measurement.

        Parameters
        ----------
        measurement_index : int
            The index of the measurement.
        dataset_index : int
            The index of the dataset.
        """
        if measurement_index > self.num_measurements - 1:
            raise ValueError(f"Measurement index out of range: {measurement_index}")

        if dataset_index > self.measurements[measurement_index].num_datasets - 1:
            raise ValueError(f"Dataset index out of range: {dataset_index}")

        del self.measurements[measurement_index].datasets[dataset_index]

    def remove_all_datasets_at_index(self, index: int) -> None:
        """Delete the dataset from the given index from all measurements.

        Because there are no restrictions on number of datasets per measurement,
        this method needs to have behaviour defined for cases where data is missing.

        Parameters
        ----------
        index : int
            The index of the dataset to remove from all measurements.
        """
        for measurement in self.measurements:
            if measurement.num_datasets < index - 1:
                raise IndexError("No dataset at required index.")

            measurement.datasets = measurement.datasets[0:index] + measurement.datasets[index+1:]

    def _has_consistent_dataset_lengths(self) -> bool:
        """Check whether this set of measurements have consistent dataset end indices.

        Returns
        -------
        bool
            Whether the dataset_end_indices of each measurement are consistent.
        """
        dataset_end_indices = self.measurements[0].dataset_lengths
        for measurement in self.measurements[1:]:
            if measurement.dataset_lengths != dataset_end_indices:
                return False
        return True

    def _num_metadata_keys(self,
                           include_unshared_conditions: bool,
                           include_equal_conditions: bool) -> int:
        """Determine the number of metadata values given options.

        Parameters
        ----------
        include_unshared_conditions : bool
            Whether to include conditions not present in every measurement.
        include_equal_conditions : bool
            Whether to include conditions that are the same in every measurement.

        Returns
        -------
        int
            The number of metadata values for the chosen settings.
        """

        if include_unshared_conditions and include_equal_conditions:
            return len(self.all_condition_names)
        elif include_unshared_conditions and not include_equal_conditions:
            return len(self.all_condition_names - self.equal_shared_condition_names)
        elif not include_unshared_conditions and include_equal_conditions:
            return len(self.shared_condition_names)
        else:  # default - clean dataset
            return len(self.varying_shared_condition_names)

    @property
    def all_condition_names(self) -> Set[str]:
        """All condition names present in at least one measurement.

        Returns
        -------
        set
            The names of all conditions defined for at least one measurement.
        """
        return self.__compute_all_condition_names()

    def __compute_all_condition_names(self) -> Set[str]:
        """Determine names of conditions defined for at least one measurement.

        Returns
        -------
        names : Set of str
        names : Set of str
            Names of conditions defined for at least one measurement.
        """
        if self.num_measurements == 0:
            return set()

        return set.union(*[measurement.condition_names for measurement in self.measurements])

    @property
    def shared_condition_names(self) -> Set[str]:
        """The condition names shared by all measurements."""
        return self.__compute_shared_condition_names()

    def __compute_shared_condition_names(self) -> Set[str]:
        """Determine the names of conditions defined for all measurements.

        In order for a particular condition to be used to organize a set of
        measurements into experiments or sets thereof, it must be present in
        the condition metadata of all measurements in that set. The set of
        shared conditions is therefore required for organization.

        Returns
        -------
        shared_names : Set of str
            Names of conditions defined for all measurements.
        """
        if self.num_measurements == 0:
            return set()

        return set.intersection(*[measurement.condition_names for measurement in self.measurements])

    @property
    def unshared_condition_names(self) -> Set[str]:
        """The condition names not shared by all measurements."""
        return self.__compute_unshared_condition_names()

    def __compute_unshared_condition_names(self) -> Set[str]:
        """Find condition metadata names not present in all measurements.

        In addition to seeing shared and equal conditions, a user may be
        interested in conditions defined only for certain measurements so that
        values may be filled in for other measurements, or removed. The names of
        these conditions are defined for at least one but strictly fewer than
        num_measurements measurements.

        Returns
        -------
        Set of str
            Condition names present in at least one but not all measurements.
        """
        return self.all_condition_names.difference(self.shared_condition_names)

    @property
    def equal_shared_condition_names(self) -> Set[str]:
        """The shared condition names with the same value in all measurements."""
        return self.__compute_equal_shared_condition_names()

    def __compute_equal_shared_condition_names(self) -> Set[str]:
        """Determine the set of condition names equal for all measurements.

        If a condition name is present for all measurements and has the same
        value for each of them, then it cannot describe a physical property
        which is responsible for variation across an experiment.

        Returns
        -------
        equal_names : Set of str
            The set of conditions equal for all measurements.
        """
        equal_shared_condition_names = set()
        for shared_condition_name in self.shared_condition_names:

            if all([measurement.conditions[shared_condition_name] ==
                    self.measurements[0].conditions[shared_condition_name]
                    for measurement in self.measurements]):

                equal_shared_condition_names.add(shared_condition_name)

        return equal_shared_condition_names

    @property
    def varying_shared_condition_names(self) -> Set[str]:
        """The condition names shared by all measurements with varying values."""
        return self.__compute_varying_shared_condition_names()

    def __compute_varying_shared_condition_names(self) -> Set[str]:
        """Compute the set of shared conditions with varying values.

        These conditions are the ones of interest for comparing measurements.
        They are defined for all measurements but do not have the same value
        in each measurement, and therefore potentially describe factors
        responsible for variation between the measurement datasets.

        Returns
        -------
        Set of str
            The set of shared condition names with unequal values.
        """
        return self.shared_condition_names.difference(self.equal_shared_condition_names)

    @property
    def equal_shared_conditions(self) -> Dict[str, object]:
        """The condition names shared by all measurements with equal values."""
        return self.__compute_equal_shared_conditions()

    def __compute_equal_shared_conditions(self) -> Dict[str, object]:
        """Create a dictionary of shared, equal conditions.

        Returns
        -------
        equal_conditions : dict (str -> value)
            A dictionary containing names and values of equal conditions.
        """
        return {name: self.measurements[0].conditions[name]
                for name in self.equal_shared_condition_names}

    @property
    def conditions(self) -> Dict[str, object]:
        return self.equal_shared_conditions

    @property
    def varying_shared_conditions(self) -> Dict[str, List[object]]:
        """The conditions shared by all measurements with varying values."""
        return self.__compute_varying_shared_conditions()

    def __compute_varying_shared_conditions(self) -> Dict[str, List[object]]:
        """Create a dictionary of shared, varying conditions.

        Returns
        -------
        varying_conditions : dict (str -> list)
            A dictionary containing names and all values of shared conditions.
        """
        return {name: [measurement.conditions[name] for measurement in self.measurements]
                for name in self.varying_shared_condition_names}

    @property
    def has_varying_shared_conditions(self) -> bool:
        """Whether this measurement set has varying shared conditions."""
        return len(self.varying_shared_condition_names) != 0

    @property
    def all_detail_names(self) -> Set[str]:
        """All detail names present in at least one measurement."""
        return self.__compute_all_detail_names()

    def __compute_all_detail_names(self) -> Set[str]:
        """Determine all detail names present in at least one measurement.

        Returns
        -------
        names : Set of str
            The set of all detail names present in this measurementset.
        """
        if self.num_measurements == 0:
            return set()

        return set.union(*[measurement.detail_names for measurement in self.measurements])

    @property
    def shared_detail_names(self) -> Set[str]:
        """The detail names shared by all measurements."""
        return self.__compute_shared_detail_names()

    def __compute_shared_detail_names(self) -> Set[str]:
        """Determine the set of detail names shared by all measurements.

        Returns
        -------
        shared_details : Set of str
        """
        if self.num_measurements == 0:
            return set()

        return set.intersection(*[measurement.detail_names for measurement in self.measurements])

    @property
    def unshared_detail_names(self) -> Set[str]:
        """The detail names not shared by all measurements."""
        return self.__compute_unshared_detail_names()

    def __compute_unshared_detail_names(self) -> Set[str]:
        """Find condition metadata names not present in all measurements.

        In addition to seeing shared and equal conditions, a user may be
        interested in conditions defined only for certain measurements so that
        values may be filled in for other measurements.

        Returns
        -------
        Set of str
        """
        return self.all_detail_names.difference(self.shared_detail_names)

    @property
    def equal_shared_detail_names(self) -> Set[str]:
        """The detail names with the same value in all measurements."""
        return self.__compute_equal_shared_detail_names()

    def __compute_equal_shared_detail_names(self) -> Set[str]:
        """Determine the set of condition names equal for all measurements.

        Returns
        -------
        equal_condition_names : Set of str
            The set of conditions equal for all measurements.
        """
        equal_condition_names = set()
        for condition_name in self.__compute_shared_detail_names():

            is_equal = True
            value = self.measurements[0].details[condition_name]

            for measurement in self.measurements[1:]:

                if measurement.details[condition_name] != value:
                    is_equal = False
                    break

            if is_equal:
                equal_condition_names.add(condition_name)

        return equal_condition_names

    @property
    def varying_shared_detail_names(self) -> Set[str]:
        """The detail names shared by all measurements without the same value."""
        return self.__compute_varying_shared_detail_names()

    def __compute_varying_shared_detail_names(self) -> Set[str]:
        """Compute the names of shared details with differing values across measurements.

        Returns
        -------
        Set of str
            The names of shared details with differing values across measurements.
        """
        return self.shared_detail_names - self.equal_shared_detail_names

    @property
    def details(self) -> Dict[str, object]:
        return self.equal_shared_details

    @property
    def equal_shared_details(self) -> Dict[str, object]:
        """The details shared by all measurements with the same value."""
        return self.__compute_equal_shared_details()

    def __compute_equal_shared_details(self) -> Dict[str, object]:
        """Create a dictionary of common and equal details.

        Returns
        -------
        equal_details : dict
            A dictionary containing names and values of equal details.
        """
        equal_details = {}
        for key in self.__compute_equal_shared_detail_names():
            equal_details[key] = self.measurements[0].details[key]

        return equal_details

    def are_repetitions(self) -> bool:
        """Determine whether the list of measurements are repetitions.

        Returns
        -------
        bool
            Whether this list of measurements are repetitions.
        """
        return measurement_.Measurement.are_repetitions(self.measurements)

    def are_not_repetitions(self) -> bool:
        """Determine whether the list of measurements aren't repetitions.

        Returns
        -------
        bool
            Whether this list of measurements are not repetitions.
        """
        return not self.are_repetitions()

    def condition_to_detail(self, condition_name: str) -> None:
        """Convert a condition to a detail for all measurements.

        Parameters
        ----------
        condition_name : str
            The name of the condition to convert to a detail.

        Notes
        -----
        This method is silent if the condition name is not present in the
        conditions dictionary of any measurement.
        """
        for measurement in self.measurements:
            measurement.condition_to_detail(condition_name)

    def conditions_to_details(self, condition_names: List[str]) -> None:
        """Convert a set of conditions to details for all measurements.

        Parameters
        ----------
        condition_names : list of str
            The names of the conditions to convert to details.
        """
        for name in condition_names:
            self.condition_to_detail(name)

    def detail_to_condition(self, detail_name: str) -> None:
        """Convert a detail to a condition for all measurements.

        Parameters
        ----------
        detail_name : str
            The name of the detail to convert to a condition.

        Notes
        -----
        This method is silent if the detail name is not present in the
        details dictionary of any measurement.
        """
        for measurement in self.measurements:
            measurement.detail_to_condition(detail_name)

    def details_to_conditions(self, detail_names: List[str]) -> None:
        """Convert a set of details to conditions for all measurements.

        Parameters
        ----------
        detail_names : list of str
            The names of the details to convert to conditions.
        """
        for name in detail_names:
            self.detail_to_condition(name)

    def add_equal_shared_condition(self, name: str, value: object = None):
        """Add a condition with a specified value to all measurements.

        Parameters
        ----------
        name : str
            The name of the key to assign the value to.
        value : object
            The value of the dict for the given key. Default is None.
        """
        for measurement in self.measurements:
            measurement.add_condition(name, value)

    def add_equal_shared_conditions(self, names: List[str], values: List[object]) -> None:
        """Add conditions with specified values to all measurements.

        Parameters
        ----------
        names : list of str
            The names for the conditions.
        values : list of variable
            The values of the conditions.
        """
        for measurement in self.measurements:
            measurement.add_conditions(names, values)

    def add_varying_shared_condition(self, name: str, values: List[object]) -> None:
        """Add a condition with different values for each measurement.

        Parameters
        ----------
        name : str
            The name of the condition to add.
        values : list of variable
            The value of the condition for each measurement.
        """
        if len(values) != self.num_measurements:
            raise ValueError("Incorrect number of values for new condition.")

        for measurement, value in zip(self.measurements, values):
            measurement.add_condition(name, value)

    def add_varying_shared_conditions(self, names: List[str], values: List[List[object]]) -> None:
        """Add conditions with different values for each measurement.

        Parameters
        ----------
        names : list of str
            The names of the conditions to add.
        values : list of list
            The values of each condition.
            The number of lists must be equal to the number of
            names provided. Each list must have length equal to the
            number of measurements.
        """
        if len(names) != len(values):
            raise ValueError("Number of names not equal to number of value lists.")

        for name, value_list in zip(names, values):
            self.add_varying_shared_condition(name, value_list)

    def remove_condition(self, name: str) -> None:
        """Remove a condition from all measurements.

        Parameters
        ----------
        name : str
            The name of the key of the entry to remove.
        """
        for measurement in self.measurements:
            measurement.remove_condition(name)

    def remove_conditions(self, names: List[str]) -> None:
        """Remove conditions from all measurements.

        Parameters
        ----------
        names : list of str
            The names of the conditions to remove.
        """
        for measurement in self.measurements:
            measurement.remove_conditions(names)

    def add_equal_shared_detail(self, name: str, value: object = None) -> None:
        """Add a detail with a specified value to all measurements.

        Parameters
        ----------
        name : str
            The name of the key to assign the value to.
        value : object
            The value of the dict for the given key. Default is None.
        """
        for measurement in self.measurements:
            measurement.add_detail(name, value)

    def add_equal_shared_details(self, names: List[str], values: List[object]) -> None:
        """Add details with specified values to all measurements.

        Parameters
        ----------
        names : list of str
            The names of the details to add.
        values : list of variable
            The values of the details to add.
        """
        for measurement in self.measurements:
            measurement.add_details(names, values)

    def add_varying_shared_detail(self, name: str, values: List[object]) -> None:
        """Add a detail with a specific value for each measurement.

        Parameters
        ----------
        name : str
            The name of the detail to add.
        values : list of variable
            The values of the detail to add.
        """
        for measurement, value in zip(self.measurements, values):
            measurement.add_detail(name, value)

    def add_varying_shared_details(self, names: List[str], values: List[List[object]]) -> None:
        """Add details with varying values for each measurement.

        Parameters
        ----------
        names : list of str
            The names of the details to add.
        values : list of list
            A list of lists of varying values to add.
        """
        for name, value_list in zip(names, values):
            self.add_varying_shared_detail(name, value_list)

    def remove_detail(self, name: str) -> None:
        """Remove a detail from all measurements.

        Parameters
        ----------
        name : str
            The name of the key of the entry to remove.
        """
        for measurement in self.measurements:
            measurement.remove_detail(name)

    def remove_details(self, names: List[str]) -> None:
        """Remove details from all measurements.

        Parameters
        ----------
        names : List of str
            The names of the details to be removed.
        """
        for name in names:
            self.remove_detail(name)

    def remove_measurement_at_index(self, index: int) -> None:
        """Remove the measurement at the given index.

        Parameters
        ----------
        index : int
            The index of the measurement to remove.
        """
        del self.measurements[index]

    def remove_measurements(self, measurements: List[measurement_.Measurement]):
        """Remove specific measurements from this set.

        Parameters
        ----------
        measurements : List of Measurement
            The measurements to remove from this set.
        """
        for measurement_to_remove in measurements:
            self.measurements.remove(measurement_to_remove)

    def remove_measurements_with_condition_name(self, name: str) -> None:
        """Remove measurements from this set which have a specified condition name.

        Parameters
        ----------
        name : str
            The name of the condition.
        """
        measurements_to_remove = [measurement for measurement in self if measurement.has_condition_name(name)]
        self.remove_measurements(measurements_to_remove)

    def remove_measurements_without_condition_name(self, name: str) -> None:
        """Remove measurements from this set which do not have a specified condition name.

        Parameters
        ----------
        name : str
            The name of the condition.
        """
        measurements_to_remove = [measurement for measurement in self if not measurement.has_condition_name(name)]
        self.remove_measurements(measurements_to_remove)

    def remove_measurements_with_condition(self, name: str, value: object) -> None:
        """Remove measurements from this set which have a specified condition.

        Parameters
        ----------
        name : str
            The name of the condition.
        value : object
            The value of the condition which will result in removal of a measurement.
        """
        measurements_to_remove = \
            [measurement for measurement in self.measurements if measurement.conditions[name] == value]
        self.remove_measurements(measurements_to_remove)

    def remove_measurements_without_condition(self, name: str, value: object) -> None:
        """Remove measurements from this set which do not have a specified condition.

        Parameters
        ----------
        name : str
            The name of the condition.
        value : variable
            The value of the condition.
        """
        measurements_to_remove = \
            [measurement for measurement in self.measurements if measurement.conditions[name] != value]
        self.remove_measurements(measurements_to_remove)

    def remove_measurements_by_condition_test(self, name: str, test: Callable[[object], bool]) -> None:
        """Remove measurements from this set with a test on a condition's values.

        The measurement-editing test methods above are only able to check for
        presence and equality. Tests applied to measurements may be much more
        complex, in which case a function can be used to test conditions of the
        measurements of the set.

        For example, if a user wishes to remove any measurements that have the
        condition "a" with positive values,

        measurement_set.remove_measurements_by_condition_test("a", lambda value_of_a: value_of_a >= 0.0)

        will achieve the stated goal.

        Parameters
        ----------
        name : str
            The name of the condition.
        test : function(variable) -> bool
            A test that returns true if a measurement is to be removed.
        """
        measurements_to_remove = \
            [measurement for measurement in self.measurements if test(measurement.conditions[name])]

        self.remove_measurements(measurements_to_remove)

    def remove_measurements_by_conditions_test(self, names: list, test: Callable[[List[object]], bool]) -> None:
        """Remove measurements from this set with a test on conditions' values.

        More complex tests require multiple conditions. This can be achieved
        by passing a lambda that takes a list as its argument.

        lambda values: values[0] >= 0.0 and values[1] == "red"

        Parameters
        ----------
        names : list of str
            The names of the conditions to test.
        test : function(list)
            A test that returns true if a measurement is to be removed.
        """
        measurements_to_remove = []
        for measurement in self.measurements:
            if measurement.has_condition_names(names):
                values = [measurement.conditions[name] for name in names]
                if test(values):
                    measurements_to_remove.append(measurement)

        self.remove_measurements(measurements_to_remove)

    def condition_label(self, include_name: bool = True) -> str:
        """A label composed from this measurement set's equal shared conditions."""
        condition_label: str = ""
        for name, value in self.equal_shared_conditions.items():
            if include_name:
                condition_label += f"{name}={value}, "
            else:
                condition_label += f"{value}, "

        return condition_label[:-2]

    def produce_color_map(self):
        """Create a color map varying across measurements.

        This needs to provide a color map per measurement as
        there is no guarantee of consistent dataset length or
        type for this class.
        This may only be defined for a consistent measurement set?
        Possible that plotting all datasets at an index onto a given plot
        is only possible if they are all the same dataset type. Otherwise
        we have to remove the on_single_plot argument from the base MeasurementSet
        and just accept that each dataset will go onto one plot. Then their dataset
        requested colours can just be used.
        Yes this makes sense.
        """
        return self.measurements[0].datasets[0].create_color_map(len(self.measurements))

    def compute_figure_size(self,
                            expand_datasets: bool,
                            expand_measurements: bool) -> Tuple[int, int]:
        """Compute an appropriate size for the complete figure.

        If a single plot has been requested, then the figure will
        be the same size as a single measurement from the set, so
        the largest measurement will be used.
        If multiple plots have been requested, the size will be a
        sum over the sizes of the set of measurements.

        Returns
        -------
        Tuple of int
            The size of the complete figure.
        """
        figsize = [0, 0]

        for measurement in self.measurements:
            measurement_figsize = measurement.compute_matplotlib_figure_size(expand_datasets=expand_datasets)

            if measurement_figsize[0] > figsize[0]:
                figsize[0] = measurement_figsize[0]

            if expand_measurements:
                figsize[1] += measurement_figsize[1]
            else:
                if measurement_figsize[1] > figsize[1]:
                    figsize[1] = measurement_figsize[1]

        return tuple(figsize)

    def _validate_title(self, title: str) -> str:
        """Assess and potentially create a title for the complete figure.

        Parameters
        ----------
        title : str
            The title for the complete figure.
        """
        if title is None:
            title = f"{self.__class__.__name__}: Varying "
            for name in self.varying_shared_condition_names:
                title += f"{name}, "

        return title[:-2]

    def _validate_figsize(self,
                          figsize: Tuple[int],
                          expand_datasets: bool,
                          expand_measurements: bool) -> Tuple[int]:
        """Assess and potentially create a size for the complete figure."""
        if figsize is None:
            figsize = self.compute_figure_size(expand_datasets, expand_measurements)

        return figsize

    def _setup_fig_and_axes(self,
                            axes: Union[matplotlib.axes.Axes, List[matplotlib.axes.Axes]] = None,
                            expand_datasets: bool = True,
                            expand_measurements: bool = True,
                            total_figsize: tuple = None,
                            title: str = None) -> Tuple[matplotlib.figure.Figure,
                                                        Union[matplotlib.axes.Axes, List[matplotlib.axes.Axes]]]:
        """Validate/prepare a figure and axes for plotting this measurement set.

        Parameters
        ----------
        axes : matplotlib.axes.Axes or list of matplotlib.axes.Axes
            The axes on which to plot this measurement set.
            Default is to create appropriate axes.
        expand_datasets : bool
            Whether to plot each measurement's datasets on their own axes object.
            Default is to do this, as multiple y-axes is a special case.
        expand_measurements : bool
            Whether to plot each measurement on its own row of axes objects.
        total_figsize
            If a figure must be created, this is its size.
            Default None means the size will be computed.
        title
            If a figure must be created, this is its title.

        Returns
        -------

        """
        title = self._validate_title(title)

        if axes is None:  # create

            total_figsize = self._validate_figsize(total_figsize,
                                                   expand_datasets,
                                                   expand_measurements)

            if expand_measurements:

                if expand_datasets:
                    fig, axes = plt.subplots(self.num_measurements,
                                             self.max_num_datasets,
                                             figsize=total_figsize)

                elif not expand_datasets:
                    fig, axes = plt.subplots(self.num_measurements,
                                             1,
                                             figsize=total_figsize)

            elif not expand_measurements:

                if expand_datasets:
                    fig, axes = plt.subplots(1,
                                             self.max_num_datasets,
                                             figsize=total_figsize)
                elif not expand_datasets:
                    fig, axes = plt.subplots(1,
                                             1,
                                             figsize=total_figsize)
            fig.suptitle(title)

        else:
            if self.num_datasets == 1:
                fig = axes.get_figure()
            else:
                fig = axes.flat[0].get_figure()

        fig.set_tight_layout({"rect": [0, 0.03, 1, 0.9]})
        return fig, axes

    def produce_color_maps(self) -> list:
        """Produce a color map for each measurement.

        Returns
        -------
        list
        """
        color_maps = []
        for dataset_type in self.dataset_types:
            color_maps.append(iter(dataset_type.DEFAULT_COLOR_MAP(np.linspace(0, 1, self.num_measurements))))

        return color_maps

    def visualize_on_multiple_plots(self,
                                    fig: matplotlib.figure.Figure,
                                    axes: List[matplotlib.axes.Axes],
                                    expand_datasets: bool = True,
                                    expand_measurements: bool = True,
                                    **plot_kwargs) -> None:
        """Visualize the measurement datasets on individual plots.

        Parameters
        ----------
        fig : matplotlib.figure.Figure
            The figure on which the measurements are to be plotted.
        axes : list of matplotlib.axes.Axes
            The set of axes on which to plot the measurements.
        expand_datasets
            Whether to plot each dataset on its own axes object.
            Default is to do this, as multiple y-axes is a special case.
        expand_measurements : bool
            Whether to plot each measurement on its own row of axes objects.
        """
        if self.num_measurements == 1:
            self.measurements[0].visualize(axes=axes,
                                           expand_datasets=expand_datasets,
                                           include_text=False,
                                           **plot_kwargs)
            return

        if expand_measurements:

            if expand_datasets:  # num_measurements x num_datasets

                for i, (axis, measurement) in enumerate(zip(axes, self.measurements)):
                    measurement.visualize(axes=axis,
                                          expand_datasets=expand_datasets,
                                          include_text=False,
                                          **plot_kwargs)

                    if expand_datasets and measurement.num_datasets < self.max_num_datasets:
                        if len(axis) != measurement.num_datasets:
                            for dataset_axis in axis[measurement.num_datasets:]:
                                fig.delaxes(dataset_axis)

                return

            elif not expand_datasets:  # num_measurements x 1

                for i, (axis, measurement) in enumerate(zip(axes, self.measurements)):
                    measurement.visualize(axes=axis,
                                          expand_datasets=expand_datasets,
                                          include_text=False,
                                          dataset_colors=self.dataset_colors[i],
                                          **plot_kwargs)

        elif not expand_measurements:

            if expand_datasets:  # 1 x num_measurements

                for i, measurement in enumerate(self.measurements):
                    measurement.visualize(axes=axes,
                                          expand_datasets=expand_datasets,
                                          include_text=False,
                                          dataset_colors=self.dataset_colors[i],
                                          **plot_kwargs)

            elif not expand_datasets:  # 1 x 1

                for i, measurement in enumerate(self.measurements):
                    measurement.visualize(axes=axes,
                                          expand_datasets=expand_datasets,
                                          include_text=False,
                                          dataset_colors=self.dataset_colors[i],
                                          **plot_kwargs)
                return

    @property
    def dataset_colors(self) -> List[List[np.ndarray]]:
        """Determine the colors for plotting this measurement set's datasets.

        For an inconsistent measurement set the package defaults to using the
        single default color for each dataset type.
        Each member of the list will be a list whose length is equal to the
        number of datasets in the corresponding measurement.
        """
        colors = []
        for measurement in self.measurements:
            dataset_colors = []
            for dataset in measurement.datasets:
                dataset_colors.append(matplotlib.colors.to_rgba_array(dataset.DEFAULT_COLOR)[0])
            colors.append(dataset_colors)

        return colors

    def _has_collapsible_measurements(self) -> bool:
        """Determine whether this measurement set has collapsible measurements.

        Returns
        -------
        bool
            Whether this measurement set has collapsible measurements.
        """
        dataset_types = self.dataset_types[0]
        for measurement in self.measurements[1:]:
            if measurement.dataset_types != dataset_types:
                return False
        return True

    def _has_collapsible_datasets(self) -> bool:
        """Determine whether this measurement set has collapsible datasets.

        Returns
        -------
        bool
            Whether this measurement set has collapsible datasets.
        """
        for measurement in self.measurements:
            if not measurement.has_collapsible_datasets():
                return False
        return True

    def visualize(self,
                  axes: matplotlib.axes.Axes or List[matplotlib.axes.Axes] = None,
                  expand_datasets: bool = True,
                  expand_measurements: bool = True,
                  include_text: bool = True,
                  figure_title: str = None,
                  total_figsize: Tuple[int] = None,
                  **plot_kwargs) -> Tuple[matplotlib.figure.Figure,
                                          matplotlib.axes.Axes]:
        """Create a visual representation of this set of measurements.

        This method should tabulate metadata independent of the dataset types.
        Then need to make a display of each measurement's dataset.
        Plot defaults need to come from the dataset; colormap, size.

        Parameters
        ----------
        axes
            One or more axes on which to plot this measurement set's datasets.
        expand_datasets : bool
            Whether to plot each measurement's datasets on their own axes object.
            Default is to do this, as multiple y-axes is a special case.
        expand_measurements : bool
            Whether to plot each measurement on its own row of axes objects.
        include_text : bool
            Whether to display the str representation of the measurement set.
            Default is to include this information.
        figure_title : str
            A title for the figure.
            Default is to create a title from metadata.
        total_figsize : tuple
            The size (in inches) of the complete figure.

        Returns
        -------
        fig : matplotlib.figure.Figure
            A matplotlib figure containing the axes.
        axes : matplotlib.axes.Axes
            Matplotlib axes containing the plotted measurement set.
        """
        if include_text:
            print(str(self))

        if not expand_measurements:
            if not self._has_collapsible_measurements():
                print("Warning: Measurements are not collapsible.")
                expand_measurements = True

        if not expand_datasets:
            if not self._has_collapsible_datasets():
                print("Warning: Datasets are not collapsible.")
                expand_datasets = True

        fig, axes = self._setup_fig_and_axes(axes=axes,
                                             expand_datasets=expand_datasets,
                                             expand_measurements=expand_measurements,
                                             total_figsize=total_figsize,
                                             title=figure_title)

        self.visualize_on_multiple_plots(fig=fig,
                                         axes=axes,
                                         expand_datasets=expand_datasets,
                                         expand_measurements=expand_measurements,
                                         **plot_kwargs)

        return fig, axes

    def __conditions_str(self) -> str:
        """Create a human-readable representation of the conditions.

        Returns
        -------
        str_rep : str
            Human-readable representation of the conditions.
        """
        str_rep = ""
        if self.varying_shared_condition_names:
            str_rep += "Differing Conditions\n--------------------\n"
            for i, measurement in enumerate(self.measurements):
                str_rep += "Measurement " + str(i) + "\t"
                for key in self.varying_shared_condition_names:
                    str_rep += key + "=" + \
                               str(measurement.conditions[key]) + "\t"
                str_rep += "\n"

            str_rep += "\n"

        if self.equal_shared_conditions:
            str_rep += "Equal Conditions\n----------------\n"
            for key, value in self.equal_shared_conditions.items():
                str_rep += key + " = " + str(value) + "\n"

            str_rep += "\n"

        if self.unshared_condition_names:
            str_rep += "Unshared Conditions\n-------------------\n"
            for key in self.unshared_condition_names:
                for i, measurement in enumerate(self.measurements):
                    if key in measurement.conditions.keys():
                        str_rep += "Measurement " + str(i) + "\t" + key + \
                                   " = " + \
                                   str(measurement.conditions[key]) + "\n"

            str_rep += "\n"

        return str_rep

    def __details_str(self) -> str:
        """Create a human-readable representation of the details.

        Returns
        -------
        str_rep : str
            Human-readable representation of the conditions.
        """
        str_rep = ""
        if self.varying_shared_detail_names:
            str_rep += "Differing Details\n-----------------\n"
            for i, measurement in enumerate(self.measurements):
                str_rep += "Measurement " + str(i) + "\t"
                for key in self.varying_shared_detail_names:
                    str_rep += key + "=" + str(measurement.details[key]) + "\t"
                str_rep += "\n"

            str_rep += "\n"

        if self.equal_shared_details:
            str_rep += "Equal Details\n-------------\n"
            for key, value in self.equal_shared_details.items():
                str_rep += key + " = " + str(value) + "\n"

            str_rep += "\n"

        if self.unshared_detail_names:
            str_rep += "Unshared Details\n----------------\n"
            for key in self.unshared_detail_names:
                for i, measurement in enumerate(self.measurements):
                    if key in measurement.details.keys():
                        str_rep += "Measurement " + str(i) + "\t" + key + \
                                   " = " + \
                                   str(measurement.details[key]) + "\n"
        return str_rep

    def __len__(self) -> int:
        return len(self.measurements)

    def __getitem__(self, position) -> measurement_.Measurement or "MeasurementSet":
        if isinstance(position, slice):
            return MeasurementSet(self.measurements[position], merge_redundant=False)
        return self.measurements[position]

    def __delitem__(self, index) -> None:
        del self.measurements[index]

    def __setitem__(self, index, value) -> None:
        self.measurements[index] = value

    def insert(self, index: int, value) -> None:
        self.measurements.insert(index, value)

    def __str__(self) -> str:
        """Create a human-readable representation of this object.

        Directly reporting the data for all measurements is not very useful
        to the user as relationships between the measurements are not obvious.
        This is available to the user via for measurement in measurement_set:
        print(measurement).
        Here we instead focus on reporting summarized information.
        """
        str_rep = f"{self.__class__.__name__}\n"
        underline = len(self.__class__.__name__) * "-"
        str_rep += f"{underline}\n\n"
        # str_rep += self.__conditions_str()
        # str_rep += self.__details_str()

        str_rep += "Measurements\n------------\n"
        for i, measurement in enumerate(self.measurements):
            str_rep += str(i) + " " + measurement.one_line_str() + "\n"

        return str_rep + "\n"

    def __repr__(self) -> str:
        """Return eval()-able representation of this instance."""
        str_rep = self.__class__.__name__ + "("
        str_rep += "measurements=["
        for measurement in self.measurements:
            str_rep += repr(measurement) + ", "

        return str_rep[:-2] + "])"

    def __eq__(self, other) -> bool:
        """Return True is self is equal to other."""
        if self is other:
            return True

        if len(self) != len(other):
            return False

        for measurement in self.measurements:
            if measurement not in other.measurements:
                return False

        return True

    def __cmp__(self):
        """Comparison based on magnitudes is purposely not defined."""
        return NotImplemented

    def to_pandas(self, with_cralds_class: bool=True, concat: bool = False):
        """Converts MeasurementSet into a Pandas DataFrame.

        Parameters
        ----------
        with_cralds_class : bool, optional
            If True, a column corresponding to the source cralds Dataset class is
            added to the DataFrame. This simiplifies conversion back into a cralds
            data structure.
        concat : bool, optional
            If True, lists of DataFrames and concatenated into a single DataFrame.

        Returns
        -------
        converted_data: Union[pd.DataFrame, List[pd.DataFrame]]
            DataFrame or list of DataFrames based on source data.

        Raises
        ------
        ImportError:
            Raised if `pandas` is not installed in environment.
        """
        from piblin.dataio.pandas import _to_pandas

        return _to_pandas(self, with_cralds_class, concat)
