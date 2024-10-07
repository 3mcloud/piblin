from typing import List, Tuple, Dict
import matplotlib.pyplot as plt
import matplotlib.axes
import matplotlib.figure
import matplotlib.colors
import numpy as np

import piblin.data.data_collections.measurement as measurement_
import piblin.data.data_collections.measurement_set as measurement_set


class InconsistentMeasurementsError(Exception):
    ...
    """Raised when inconsistent measurements are present."""


class ConsistentMeasurementSet(measurement_set.MeasurementSet):
    """A set of measurements with consistent datasets.

    To be consistent, a set of measurements must have the same
    number and types of dataset. This simplifies the implementation
    of several properties inherited from the superclass but restricts the
    manipulation of the measurements to changes that do not remove the
    consistency.
    All metadata properties are not affected by the notion of consistency so
    their implementations are all directly inherited from the base class.
    Any method that removes measurements from an already consistent measurement
    set by using conditions will not affect the consistency.

    Parameters
    ----------
    measurements : list of piblin.data.measurement.Measurement
        The measurements to be collected.

    Attributes
    ----------
    measurements -> list of piblin.data.measurement.Measurement
        The collected measurements. Inherited directly.
    num_measurements -> int
        The number of collected measurements. Inherited directly.
    datasets -> list of list of dataset
        The datasets of the set of measurements. Inherited directly.
    num_datasets -> int
        The number of datasets in each measurement of the set.
        A consistent measurement set has a constant number of datasets across all measurements.
        This property is therefore an int instead of a list of ints.
    dataset_types -> list of type
        The types of datasets in each measurement of the set.
        A consistent measurement set has a shared set of types across all measurements.
        This property is therefore simply a list of types.
    dataset_lengths -> list of list of int
        The lengths of datasets in each measurement of the set. Not directly inherited.
    dataset_independent_variables() -> list of np.ndarray
        The independent variables of each measurement of the set.
    is_tidy -> bool
        Determine if the measurement set is tidy. Directly inherited.
    is_consistent -> bool
        Determine if this measurement set is consistent. Directly inherited.
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
        The detail names shared by all measurements without the same value in all measurements.
    equal_shared_details -> dict
        The details shared by all measurements with the same value.

    Methods
    -------
    from_single_measurement(Measurement) -> ConsistentMeasurementSet
        Create a measurement set from a single measurement. Directly inherited.
    from_measurement_set(MeasurementSet) -> ConsistentMeasurementSet
    from_measurement_sets([MeasurementSet]) -> ConsistentMeasurementSet
        Create a single measurement set by combining multiple measurement sets. Directly inherited.
    from_flat_data(np.ndarray, [[type]], [[int]], [[[object]]]) -> MeasurementSet
        Create a measurement set from flat data. TODO - may have different signature
    to_tidy_measurement_set([Interpolate]) -> TidyMeasurementSet
        Convert this consistent measurement set to a tidy measurement set. Currently directly inherited.
    to_tabular_dataset() -> TabularMeasurementSet
        Convert this measurement set to a tabular dataset. Currently directly inherited.
   are_repetitions() -> bool
        Determine whether the list of measurements are repetitions. Directly inherited.
    are_not_repetitions -> bool
        Determine whether the list of measurements are not repetitions. Directly inherited.
    condition_to_detail(str) -> None
        Convert a condition to a detail.
    conditions_to_details([str]) -> None
        Convert a set of conditions to details for all measurements.
    detail_to_condition(str) -> None
        Convert a detail to a condition.
    details_to_conditions(list of str) -> None
        Convert a set of details to conditions for all measurements.
    add_equal_shared_condition(str, object) -> None
        Add a condition with a specified value to all measurements.
    add_equal_shared_conditions([str], [object]) -> None
        Add conditions with specified values to all measurements.
    add_varying_shared_condition(str, [object]) -> None
        Add a condition with different values for each measurement.
    add_varying_shared_conditions([str], [[object]]) -> None
        Add conditions with different values for each measurement.
    add_condition(str, object) -> None
        Add a condition to all measurements in the collection.
    add_detail(str, object) -> None
        Add a detail to all measurements in the collection.
    add_equal_shared_detail(str, object) -> None
        Add a detail with a specified value to all measurements.
    add_equal_shared_details([str], [object]) -> None
        Add details with specified values to all measurements.
    add_varying_shared_detail(str, [object]) -> None
        Add a detail with a specific value for each measurement.
    add_varying_shared_details([str], [[object]]) -> None
        Add details with varying values for each measurement.
    remove_condition(str) -> None
        Remove the condition with the specified name from all measurements.
    remove_conditions([str])
        Remove conditions with the specified name from all measurements.
    remove_detail(str) -> None
        Remove the detail with the specified name from all measurements.
    remove_details([str])
        Remove details with the specified name from all measurements.
    flatten()
        Convert the set of measurements to a flat dataset.
    flatten_datasets()
        Convert the measurement data to a set of rows.
    flatten_metadata()
        Convert the measurement metadata to a set of rows.
    datasets_at_index() -> list of dataset
        The dataset at a given index from each measurement.
    delete_dataset_of_measurement_at_index(int, int) -> None  # TODO
        Delete the dataset at the specified index in the specified measurement.
        This operation can remove consistency of the measurement set.
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

        super().__init__(measurements=measurements,
                         merge_redundant=merge_redundant)

        if not self.is_consistent:
            try:
                self.measurements = self.enforce_consistency(measurements)
            except ValueError:
                raise InconsistentMeasurementsError("Measurements are not consistent." + self.consistency_str)

    def _determine_unique_condition_combinations(self) -> List[Dict[str, object]]:
        """Determine the unique combinations of conditions among the measurements in this set.

        When determining replicate relationships among measurements, the set of unique combinations
        of conditions is required in order to organize the measurements more highly than they are
        organized in the measurement set. The two examples in current use are determining sets of replicates
        prior to creating experiment objects out of the measurements, and finding repetitions with different
        datasets that can be combined into consistent measurements.

        Returns
        -------
        List of Dict
            The unique combinations of condition values in this measurement set.
        """
        # which conditions are present in all measurements?
        shared_condition_names = \
            self.shared_condition_names

        # cache these for performance reasons
        measurement_conditions = [measurement.conditions for measurement in self.measurements]

        # what are the unique combinations of values for these conditions?
        unique_combinations: List[Dict] = []
        for i, measurement in enumerate(self.measurements):
            shared_conditions = {}
            # which measurement conditions are shared across the measurement set
            for condition_name in measurement.condition_names:
                if condition_name in shared_condition_names:
                    shared_conditions[condition_name] = \
                        measurement_conditions[i][condition_name]

            if unique_combinations:
                unique = True
                for unique_combination in unique_combinations:
                    if unique_combination == shared_conditions:
                        unique = False
                        break

                if unique:
                    unique_combinations.append(shared_conditions)
            else:
                unique_combinations.append(shared_conditions)

        return unique_combinations

    def enforce_consistency(self, measurements: List[measurement_.Measurement]):
        """Can potentially use the replicate relationship to force consistency onto a measurement set.
        the basic idea is that any set of measurements that shares the same condition metadata but has
        different dataset number, type, names or units can have those datasets placed into a single
        measurement with a single list of those datasets and the equal condition metadata.
        To do this: 1) determine all sets of replicates (c.f. experiment set), concatenate their datasets
        and create a new measurement for each set. Then create a set of all new measurements and check it for
        consistency. If it's ok, set this class' measurement set to one made from those measurements.
        Look at _update_experiments!
        """
        shared_condition_names = self.shared_condition_names

        unique_combinations = self._determine_unique_condition_combinations()

        # cache these for performance reasons
        measurement_conditions = [measurement.conditions for measurement in self.measurements]

        new_measurements = []
        for unique_combination in unique_combinations:

            combine_measurements = []

            # create a list of measurements with the given conditions
            for i, measurement in enumerate(self.measurements):

                shared_conditions = {}

                for condition_name in measurement.condition_names:
                    if condition_name in shared_condition_names:
                        shared_conditions[condition_name] = \
                            measurement_conditions[i][condition_name]

                if shared_conditions == unique_combination:
                    combine_measurements.append(measurement)

            all_datasets = []
            for measurement in combine_measurements:
                all_datasets.extend(measurement.datasets)

            combined_measurement = measurement_.Measurement(datasets=all_datasets,
                                                            conditions=unique_combination)

            new_measurements.append(combined_measurement)

        new_measurement_set = measurement_set.MeasurementSet(new_measurements)
        if not new_measurement_set.is_consistent:
            raise ValueError
        else:
            return new_measurements

    @property
    def num_datasets(self) -> int:
        """The number of datasets in each consistent measurement.

        In a consistent measurement set this is guaranteed to be a single integer.
        """
        if self.num_measurements == 0:
            return 0

        return self.measurements[0].num_datasets

    @property
    def dataset_types(self) -> List[type]:
        """The types of dataset in each consistent measurement.

        In a consistent measurement set this is guaranteed to be a list of types.
        """
        if self.num_measurements == 0:
            return []

        return self.measurements[0].dataset_types

    @property
    def dataset_dimensionalities(self) -> List[int]:
        """The number of independent variables of each dataset in each consistent measurement."""
        if self.num_measurements == 0:
            return []

        return self.measurements[0].dataset_dimensionalities

    @property
    def dataset_dependent_variable_names(self) -> List[List[str]]:
        """The dependent variable names of each dataset in each consistent measurement."""
        if self.num_measurements == 0:
            return []

        return self.measurements[0].dataset_dependent_variable_names

    @property
    def dataset_dependent_variable_units(self) -> List[List[str]]:
        """The dependent variable units of each dataset in each consistent measurement."""
        if self.num_measurements == 0:
            return []

        return self.measurements[0].dataset_dependent_variable_units

    @property
    def dataset_independent_variable_names(self) -> List[List[str]]:
        """The independent variable names of each dataset in each consistent measurement."""
        if self.num_measurements == 0:
            return []

        return self.measurements[0].dataset_independent_variable_names

    @property
    def dataset_independent_variable_units(self) -> List[List[str]]:
        """The independent variable units of each dataset in each consistent measurement."""
        if self.num_measurements == 0:
            return []

        return self.measurements[0].dataset_independent_variable_names

    def datasets_at_index(self, index: int):
        """The datasets of each measurement in this measurement set at the given index."""
        return [measurement.datasets[index] for measurement in self.measurements]

    @classmethod
    def from_measurement_set(cls,
                             measurement_set_: measurement_set.MeasurementSet,
                             dataset_indices_to_retain: List[int] = None,
                             make_copy=True
                             ) -> "ConsistentMeasurementSet":
        """Create a consistent measurement set from a measurement set.

        A measurement set may be partially consistent, allowing it to be stripped of
        inconsistent datasets to become consistent.
        This method wraps the corresponding method from MeasurementSet.

        Parameters
        ----------
        measurement_set_ : MeasurementSet
            The measurement set from which to create a consistent measurement set.
        dataset_indices_to_retain : list of int
            The indices of the datasets to retain if consistent.
        make_copy : bool
            Whether to alter the measurement set in place or create a duplicate.

        """
        ms = measurement_set_.to_consistent_measurement_set(dataset_indices_to_retain=dataset_indices_to_retain,
                                                            make_copy=make_copy)

        return cls(ms.measurements)

    def _has_consistent_dataset_type_at_index(self, dataset_index):
        """Check whether this set of measurements have consistent dataset types at a given index.

        This implementation is simpler than for the superclass due to the
        guaranteed constant number of datasets across all measurements.

        Parameters
        ----------
        dataset_index : int
            The index of the dataset to check for consistency.

        Returns
        -------
        bool
            Whether the type of datasets are the given index are consistent.
        """
        if dataset_index >= self.num_datasets:
            return False

        dataset_type = type(self.measurements[0].datasets[dataset_index])
        for measurement in self.measurements[1:]:
            if type(measurement.datasets[dataset_index]) != dataset_type:
                return False

        return True

    def remove_dataset_of_measurement_at_index(self, measurement_index, dataset_index) -> None:
        """Delete the dataset at the specified index in the specified measurement.

        Applying this method will remove consistency for almost all measurement sets that
        will exist (exceptions being single-measurement measurement sets).

        Parameters
        ----------
        measurement_index : int
            The index of the measurement.
        dataset_index : int
            The index of the dataset.
        """
        if self.num_measurements == 1:
            super().remove_dataset_of_measurement_at_index(measurement_index, dataset_index)
        else:
            raise NotImplementedError("Cannot currently delete single measurement from consistent measurement set.")

    @property
    def max_num_datasets(self) -> int:
        return self.num_datasets

    def _has_collapsible_measurements(self) -> bool:
        return True

    @property
    def dataset_colormaps(self) -> List[matplotlib.colors.Colormap]:
        """Create color maps for plotting this measurement set's datasets.

        Each dataset column in the consistent measurement set will share a
        dataset type due to the guarantee of consistency in this measurement
        set. Each dataset class provides a default color map that can be used
        to generate distinct colors for multiple datasets of the same type.
        """
        return [dataset.DEFAULT_COLOR_MAP for dataset in self.measurements[0].datasets]

    @property
    def dataset_colors(self,
                       dataset_colormaps: List[matplotlib.colors.Colormap] = None) -> List[List[np.ndarray]]:
        """Get a set of per-dataset colors for this consistent measurement set.

        Returns
        -------
        A per-dataset list of per-measurement colors drawn from default color maps.
        """
        if dataset_colormaps is None:
            dataset_colormaps = self.dataset_colormaps

        return [dataset_colormaps[i](np.linspace(0,
                                                 1,
                                                 self.num_measurements))
                for i in range(self.num_datasets)]
