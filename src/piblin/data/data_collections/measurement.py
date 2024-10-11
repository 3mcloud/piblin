"""Abstraction of the concept of a scientific measurement.

Classes
-------
ExistingMetadataError - Raised when requested new metadata keys are already present.
MissingMetadataError - Raised when metadata keys are not present.
DuplicateMetadataError - Raised when duplicate metadata keys are detected.
DifferingConditionError - Raised when differing conditions are detected.
IncompatibleListError - Raised when two lists are of inappropriate unequal length.
MultipleMeasurementError - Raised when creating from a multi-measurement tabular set.
InvalidDatasetIndexError - Raised when a dataset index is not valid
Measurement - A scientific measurement made on a sample under controlled conditions.
"""
import copy
import importlib
import itertools
from typing import Set, List, Tuple, Dict, Union
import matplotlib.pyplot as plt
import matplotlib.axes
import matplotlib.figure
import numpy as np
import piblin.data.datasets.abc.dataset as dataset_
import piblin.data.datasets.abc.split_datasets.zero_dimensional_dataset as zero_dimensional_dataset_
import piblin.data.datasets.abc.split_datasets.one_dimensional_dataset as one_dimensional_dataset_
import piblin.data
from piblin.data.data_collections import dict_repr, dict_str


class ExistingMetadataError(Exception):
    """Raised when requested new metadata keys are already present."""


class MissingMetadataError(Exception):
    """Raised when metadata keys are not present."""


class DuplicateMetadataError(Exception):
    """Raised when duplicate metadata keys are detected."""


class DifferingConditionError(Exception):
    """Raised when differing conditions are detected."""


class IncompatibleListError(Exception):
    """Raised when two lists are of inappropriate unequal length."""


class MultipleMeasurementError(Exception):
    """Raised when creating from a multi-measurement tabular set."""


class InvalidDatasetIndexError(Exception):
    """Raised when a dataset index is not valid."""


class Measurement(object):
    """A scientific measurement made on a sample under specific conditions.

    A measurement is a process performed on a sample under a set of
    conditions. Each condition has a name and a corresponding value. A
    *sample* also has properties, which are treated as measurement
    conditions herein. The measurement produces one or more datasets,
    the nature of which depends on the experimental procedure. Two
    measurements performed under the same conditions which produce the
    same number and types of datasets are repetitions.
    Metadata information about the measurement which are not part of the
    conditions are stored as details. Conditions and details are
    interchangeable programmatically.

    Parameters
    ----------
    datasets : list of Dataset
        The datasets collected as a result of the measurement.
    conditions : dict of (str, object)
        Experimental conditions under which the measurement was made,
        and sample properties.
    details : dict of (str, object)
        Measurement information or sample properties that are not
        considered experimental conditions.

    Attributes
    ----------
    datasets -> list of dataset_.Dataset
        The datasets collected as a result of the measurement.
    num_datasets -> int
        The number of datasets collected as a result of the measurement.
    dataset_types -> list of Class
        The types of dataset collected as a result of the measurement.
    dataset_lengths -> list of int
        The lengths of datasets collected as a result of the measurement.
    dataset_independent_variable_data -> list of np.ndarray
        The independent variable data of each dataset collected as a
        result of the measurement.
    conditions -> dict of str: object
        Experimental conditions under which the measurement was made.
    condition_names -> list of str
        Names of the experimental conditions recorded for this measurement.
    details -> dict of str: object
        Experimental details of the measurement.
    detail_names -> list of str
        Names of the experimental details recorded for this measurement.

    Methods
    -------
    from_single_dataset(Dataset) -> Measurement
        Create a measurement from a single dataset.
    from_flat_data() -> Measurement
        Create a measurement from flat data.
    from_measurements(list of Measurement, bool, bool) -> Measurement
        Create a measurement from multiple existing measurements.
    add_condition(str, object)
        Add a condition to this measurement.
    add_conditions (list of str, list of object)
        Add multiple conditions to this measurement.
    remove_condition(str)
        Remove the condition with the specified name from this measurement.
    remove_conditions(list of str)
        Remove conditions with the specified names from this measurement.
    update_condition(str, object, bool)
        Update a condition of this measurement.
    update_conditions(list of str, list of object, bool)
        Update multiple conditions of this measurement.
    add_detail(str, object)
        Add a detail to this measurement.
    add_details(list of str, list of object)
        Add multiple details to this measurement.
    remove_detail(str)
        Remove the detail with the specified name from this measurement.
    remove_details(list of str)
        Remove details with the specified names from this measurement.
    update_detail(str, object, bool)
        Update a detail of this measurement.
    update_details(list of str, list of object, bool)
        Update multiple details of this measurement.
    condition_to_detail(str)
        Convert the condition with the specified name to a detail.
    conditions_to_details(list of str)
        Convert conditions with the specified names to details.
    detail_to_condition(str)
        Convert the detail with the specified name to a condition.
    details_to_conditions(list of str)
        Convert details with the specified names to conditions.
    has_condition_name(str) -> bool
        Determine whether this measurement has the specified condition
        name.
    has_condition_names(list of str) -> bool
        Determine whether this measurement has multiple specified
        condition names.
    is_replicate_of(Measurement) -> bool
        Determine whether this measurement is a replicate of another.
    is_not_replicate_of(Measurement) -> bool
        Determine whether this measurement is not a replicate of another.
    has_equal_conditions_to(Measurement) -> bool
        Determine whether this measurement has equal conditions to another.
    has_equal_shared_conditions_to(Measurement) -> bool
        Determine whether this measurement has equal values for its
        shared conditions to another.
    are_repetitions(list of Measurement) -> bool
        Determine if a list of measurements are repetitions.
    are_not_repetitions(list of Measurement) -> bool
        Determine if a list of measurements are not repetitions.
    has_detail_name() -> bool
        Determine if this measurement has the specified detail name.
    has_detail_names() -> bool
        Determine if this measurement has multiple specfied detail names.
    flatten(list of str, list of str, object, list of int)
        Convert this measurement into a single column/row.
    flatten_metadata(list of str, list of str, object)
        Convert the specified subset of this measurement's metadata into a
        single column/row.
    flatten_datasets()
        Turn the specified datasets of this measurement into a single
        column/row.
    visualize()
        Visualize this measurement's datasets graphically.
    compute_matplotlib_figure_size()
        Compute an appropriate size for a matplotlib visualization of
        this measurement.
    one_line_str() -> str
        Create a one-line string representation of this measurement.
    """

    MATPLOTLIB_FIGURE_SPACING: int = 4
    """The default spacing between dataset plots for a measurement."""

    __INDENT = "    "
    __DATASET_PRINT_MAX = 0

    def __init__(self,
                 datasets: List[dataset_.Dataset] = None,
                 conditions: Dict[str, object] = None,
                 details: Dict[str, object] = None):

        self.__validate_metadata(conditions, details)

        if not datasets:
            self._datasets = []
        else:
            self._datasets = datasets

        if conditions is None:
            self._conditions = {}
        else:
            self._conditions = conditions

        if details is None:
            self._details = {}
        else:
            self._details = details

    @staticmethod
    def __validate_metadata(conditions: Dict[str, object],
                            details: Dict[str, object]) -> None:
        """Validate the metadata provided to the initializer.

        Both conditions and details are provided to the initializer,
        but as their individual entries are interchangeable, they cannot
        share keys, as the switch from a condition to a detail (or vice
        versa) would be less well-defined. A key in either the
        conditions or details dictionary should map to a single
        real-world piece of metadata.

        Parameters
        ----------
        conditions : dict
            Experimental conditions provided to the initializer.
        details : dict
            Experimental details provided to the initializer.

        Raises
        ------
        DuplicateMetadataError
            If the same name appears in the keys of the conditions and
            details dictionaries.
        """
        if conditions is None or details is None:
            return

        for condition_name in conditions.keys():
            if condition_name in details.keys():
                raise DuplicateMetadataError(
                    f"Duplicate key present in conditions and details: "
                    f"{condition_name}"
                )

    @classmethod
    def from_single_dataset(
            cls,
            dataset: dataset_.Dataset = None,
            conditions: Dict[str, object] = None,
            details: Dict[str, object] = None) -> "Measurement":
        """Create a measurement from a single dataset.

        Parameters
        ----------
        dataset : dataset_.Dataset
            The single dataset from which to create the measurement.
        conditions : dict
            The conditions under which the dataset was measured.
        details : dict
            The details of the measurement.

        Returns
        -------
        Measurement
            The measurement containing the single dataset.
        """
        if dataset is None:
            datasets = []
        else:
            datasets = [dataset]

        return cls(datasets, conditions, details)

    @staticmethod
    def from_tabular_measurement_set(
            tabular_measurement_set: "piblin.data.TabularMeasurementSet"
    ) -> "Measurement":
        """Create a measurement from a tabular measurement set."""
        if tabular_measurement_set.num_rows > 1:
            raise MultipleMeasurementError(
                "Cannot convert multi-row table to single measurement."
            )

        return tabular_measurement_set.to_measurement_set()[0]

    def to_tabular_measurement_set(
            self,
            condition_names: List[str] = None,
            detail_names: List[str] = None,
            default_value: object = None
    ) -> "piblin.data.TabularMeasurementSet":
        """Convert this measurement to a single-row table.

        Parameters
        ----------
        condition_names : list of str
            Conditions to include in the table.
            Default of None includes all condition metadata.
        detail_names : list of str
            Details to include in the table.
            Default of None includes no detail metadata.
        default_value : object
            The default value for missing metadata names.

        Returns
        -------
        piblin.data.TabularMeasurementSet
            A single-row tabular measurement set with this measurement's
            metadata and datasets.
        """
        column_headers, \
            data = self.flatten(condition_names=condition_names,
                                detail_names=detail_names,
                                default_value=default_value)

        if condition_names is None:
            n_metadata_columns = len(self.condition_names)
        else:
            n_metadata_columns = len(condition_names)

        if detail_names is not None:
            n_metadata_columns += len(detail_names)

        import piblin.data.data_collections.tabular_measurement_set

        return piblin.data.data_collections.tabular_measurement_set.TabularMeasurementSet(
            data=[data],
            n_metadata_columns=n_metadata_columns,
            column_headers=column_headers,
            dataset_types=self.dataset_types,
            dataset_end_indices=self.dataset_lengths
        )

    def combine_datasets(self, dataset_indices: List[int] = None) -> None:
        """Combine some of this measurement's datasets into single dataset.

        The driving use-case for this functionality is from X-Ray
        diffraction, where datasets over regions of the x-value must be
        collected separately and hence exist in separate files.

        Parameters
        ----------
        dataset_indices : list of int
            The indices of the datasets of this measurement to combine.
            Default of None will attempt to combine all datasets.
        """
        if dataset_indices is None:
            dataset_indices = set(range(self.num_datasets + 1))

        datasets_to_combine = [dataset
                               for i, dataset in enumerate(self.datasets)
                               if i in dataset_indices]

        new_dataset = \
            datasets_to_combine[0].from_datasets(datasets_to_combine)

        new_datasets = []
        for dataset_index, dataset in enumerate(self.datasets):
            if dataset_index not in dataset_indices:
                new_datasets.append(dataset)

        new_datasets.append(new_dataset)

        self.datasets = new_datasets

    def combine_oned_datasets(self, x_name: str, y_name: str):
        """Combine multiple one-dimensional datasets.

        Find all datasets in this measurement that have the matching provided
        independent (x) and dependent (y) variable names and combine them into
        a single one-dimensional dataset.

        Parameters
        ----------
        x_name : str
            The name of the independent variable.
        y_name : str
            The name of the dependent variable.
        """
        combination_datasets = [dataset for dataset in self.datasets if
                                isinstance(dataset, one_dimensional_dataset_.OneDimensionalDataset) and
                                dataset.x_name == x_name and
                                dataset.y_name == y_name]

        # TODO - assert units match across all x and y

        for combination_dataset in combination_datasets:
            self.datasets.remove(combination_dataset)

        self.datasets.append(
            one_dimensional_dataset_.OneDimensionalDataset.from_datasets(
                combination_datasets
            )
        )

    def combine_multiple_zerod_datasets(
            self,
            x_name: List[str],
            y_names: List[str],
    ) -> None:
        """Combine zero-dimensional datasets with the provided names.

        Parameters
        ----------
        x_name
            The name of the zero-dimensional dataset to treat as x-values.
        y_names
            The names of zero-dimensional datasets to treat as y-values.
        """
        raise NotImplementedError()

    def combine_zerod_datasets(self, x_name: str, y_name: str, merge_datasets: bool = True) -> None:
        """Combine 0D datasets with the given names into 1D datasets.

        Parameters
        ----------
        x_name : str
            The name of the dependent variable of the dataset from which to source x-values.
        y_name : str
            The name of the dependent variable of the dataset from which to source y-values.
        merge_datasets : bool
            Whether to merge the resulting single-point datasets to
            a single multipoint dataset.
        """
        x_dataset_indices: List[int]
        try:
            x_dataset_indices = self.index_of(x_name)
            if isinstance(x_dataset_indices, int):
                x_dataset_indices = [x_dataset_indices]
        except ValueError as value_error:
            raise value_error

        y_dataset_indices: List[int]
        try:
            y_dataset_indices = self.index_of(y_name)
            if isinstance(y_dataset_indices, int):
                y_dataset_indices = [y_dataset_indices]
        except ValueError as value_error:
            raise value_error

        if len(x_dataset_indices) != len(y_dataset_indices):
            raise ValueError(f"Cannot create 1D datasets from two sets of 0D datasets of different size: "
                             f"{x_name}: {len(x_dataset_indices)} datasets, "
                             f"{y_name}: {len(y_dataset_indices)} datasets")
        elif len(x_dataset_indices) > 1:
            print(f"Warning: pairing of {len(x_dataset_indices)} x- and y-values in "
                  f"{y_name}=f({x_name}) performed based on dataset order only.")

        x_datasets = [self.datasets[x_dataset_index]
                      for x_dataset_index in x_dataset_indices]

        y_datasets = [self.datasets[y_dataset_index]
                      for y_dataset_index in y_dataset_indices]

        self.remove_datasets_at_indices(x_dataset_indices + y_dataset_indices)

        combined_datasets = (
            one_dimensional_dataset_.OneDimensionalDataset.from_multiple_zerod_datasets(
                x_datasets=x_datasets,
                y_datasets=y_datasets,
                merge_datasets=merge_datasets
            )
        )

        if isinstance(combined_datasets, list):
            self.datasets.extend(combined_datasets)
        else:
            self.datasets.append(combined_datasets)

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
        x_dataset_indices: List[int]
        try:
            x_dataset_indices = self.index_of(x_name)
            if isinstance(x_dataset_indices, int):
                x_dataset_indices = [x_dataset_indices]
        except ValueError as value_error:
            raise value_error

        x_datasets = [self.datasets[x_dataset_index]
                      for x_dataset_index in x_dataset_indices]

        if y_names is None:
            y_names = set([dataset.dependent_variable_name
                           for dataset in self.datasets
                           if dataset.dependent_variable_name != x_name])

        all_y_dataset_indices = []
        for y_name in y_names:
            y_dataset_indices: List[int]
            try:
                y_dataset_indices = self.index_of(y_name)
                if isinstance(y_dataset_indices, int):
                    y_dataset_indices = [y_dataset_indices]
            except ValueError as value_error:
                raise value_error

            all_y_dataset_indices.extend(y_dataset_indices)

            y_datasets = [self.datasets[y_dataset_index]
                          for y_dataset_index in y_dataset_indices]

            combined_datasets = (
                one_dimensional_dataset_.OneDimensionalDataset.from_multiple_zerod_datasets(
                    x_datasets=x_datasets,
                    y_datasets=y_datasets,
                    merge_datasets=merge_datasets
                )
            )

            if isinstance(combined_datasets, list):
                self.datasets.extend(combined_datasets)
            else:
                self.datasets.append(combined_datasets)

        self.remove_datasets_at_indices(x_dataset_indices + all_y_dataset_indices)

    @classmethod
    def from_measurements(
            cls,
            measurements: List["Measurement"],
            allow_missing_conditions: bool = False,
            keep_partial_conditions: bool = False
    ) -> "Measurement":
        """Create a measurement from multiple measurements.

        A set of measurements will each have datasets of their own as well
        as condition and detail metadata. If, for example,
        two measurements have been made on the same sample, they may
        share some condition metadata and thus be able to be combined,
        either by keeping only shared conditions or directly merging all
        conditions. To evaluate this, first need to determine that no
        condition whose name is shared across any pair of the
        measurements has a different value in any pair of measurements.
        That is, for all condition names shared by two or more
        measurements, every measurement with that condition must have
        the same value for it. Whether it matters if a condition
        is missing should be an option, as this technically gives a
        measurement a value for that condition of None.

        The conditions of the new measurement are those that are considered
        shared after the above procedure has been carried out. A further
        question remains; what to do with the conditions that were not
        shared? The simplest approach is to just not include them. If
        they are kept, their values can be retained, and they can be
        added to the conditions or details of the new consolidated
        measurement.

        This code is likely very important when combining data from
        different techniques.

        If two or more measurements have equal condition metadata then
        they can be combined into a single measurement with one set of
        conditions and a set of combined datasets. This differs from
        combining replicates into experiments in that there are no
        requirements on the measurements to have the same datasets.
        Each dataset can still be processed independently.

        Parameters
        ----------
        measurements : list of Measurement
            The measurements to be combined to a single measurement.
        allow_missing_conditions : bool
            Whether to allow measurements which are partial repetitions to
            be combined.
        keep_partial_conditions : bool
            Whether to retain conditions not already shared by all
            measurements.

        Returns
        -------
        Measurement
            A single measurement combining all given measurements.
        """
        def validate(measurements_: List["Measurement"],
                     allow_missing_conditions_: bool) -> bool:
            """Check validity of input arguments.

            first need to determine that no condition whose name is
            shared across any pair of the measurements has a different
            value in any pair of measurements. That is, for all
            condition names shared by two or more measurements,
            every measurement with that condition must have the same value
            for it.

            Parameters
            ----------
            measurements_ : list of Measurement
                The measurements to validate.
            allow_missing_conditions_ : bool
                Whether to allow measurements which are partial repetitions
                 to be combined.

            Returns
            -------
            bool
                Whether the measurements are valid for combination.
            """
            if allow_missing_conditions_:
                for measurement_, other_measurement in zip(measurements_,
                                                           measurements_):
                    if measurement_ is other_measurement:
                        continue
                    else:
                        if not measurement_._has_equal_shared_conditions_to(
                                other_measurement
                        ):
                            return False

            else:
                for i in range(len(measurements_)-1):
                    if not measurements_[i]._has_equal_conditions_to(
                            measurements_[i+1]
                    ):
                        return False

            return True

        if not measurements:
            return Measurement([], {}, {})

        if not validate(measurements, allow_missing_conditions):
            raise DifferingConditionError(
                "Cannot combine measurements with unequal conditions."
            )

        if allow_missing_conditions:
            combined_conditions = {}
            if keep_partial_conditions:
                for measurement in measurements:
                    combined_conditions.update(measurement.conditions)
            else:
                all_condition_names = {
                    str(condition_name)
                    for measurement in measurements
                    for condition_name in measurement.condition_names
                }

                for condition_name in all_condition_names:
                    use_condition = True
                    for measurement in measurements:
                        if condition_name not in \
                                measurement.condition_names:
                            use_condition = False
                            break
                    if use_condition:
                        combined_conditions[condition_name] = \
                            measurements[0].conditions[condition_name]
        else:
            combined_conditions = measurements[0].conditions

        datasets = []
        for measurement in measurements:
            datasets.extend(measurement.datasets)

        return cls(datasets, combined_conditions, {})

    def to_measurements(
            self,
            dataset_indices: List[Set[int]] = None,
            conditions_to_remove: List[Set[str]] = None,
            conditions_to_add: List[dict] = None,
            allow_empty_measurements: bool = True) -> List["Measurement"]:
        """Split this measurement into multiple measurements.

        Because this procedure starts with a single measurement, all
        datasets will initially have equal conditions and details. In
        the simplest (default) case, these can be propagated to all the
        child measurements when splitting.

        Parameters
        ----------
        dataset_indices : List of Set of int
            The groups of dataset indices to keep in the child
            measurements.
        conditions_to_remove : List of Set of str
            Any conditions of the original measurement to not keep in the
            child measurements.
        conditions_to_add : List of Set of str
            Any conditions to add to the child measurements.
        allow_empty_measurements : bool
            Whether to permit dataset indices that are not valid for this
            measurement.

        Returns
        -------
        measurements : List of Measurement
            The measurements created by splitting this measurement.
        """
        if dataset_indices is None:
            dataset_indices = [{i} for i in range(self.num_datasets)]

        if conditions_to_remove is None:
            conditions_to_remove = [{}] * len(dataset_indices)

        if conditions_to_add is None:
            conditions_to_add = [{}] * len(dataset_indices)

        measurements = []
        for measurement_index, \
                measurement_dataset_indices in enumerate(dataset_indices):

            if not allow_empty_measurements:
                for index in measurement_dataset_indices:
                    if index >= self.num_datasets or index < 0:
                        raise IndexError(
                            f"Dataset index {index} not valid. "
                            f"Number of datasets = {self.num_datasets}"
                        )

            datasets = []
            for dataset_index in measurement_dataset_indices:
                if dataset_index < self.num_datasets:
                    datasets.append(self.datasets[dataset_index])

            conditions = copy.deepcopy(self.conditions)
            for condition in conditions_to_remove[measurement_index]:
                conditions.pop(condition)
            if conditions_to_add[measurement_index]:
                conditions.update(conditions_to_add[measurement_index])

            measurements.append(Measurement(datasets=datasets,
                                            conditions=conditions,
                                            details=self.details))

        return measurements

    def to_xarray(self, merge_dependent_data: bool = False):
        """Convert measurement to xarray Dataset.

        Parameters
        ----------
        merge_dependent_data : bool, optional
            If True, dependent data with the same dependent varible name
            are stacked along a new 'dataset' dimension, by default False

        Returns
        -------
        xarray.Dataset
            Dataset containing all measurement data. Conditions and
            details contained in Dataset attrs.
        """
        import xarray as xr
        from copy import deepcopy
        attrs = {}
        attrs["conditions"] = deepcopy(self.conditions)
        attrs["details"] = deepcopy(self.details)

        if merge_dependent_data:

            def create_xr_dataset(datasets):
                data_arrays = {}
                for n, dataset in enumerate(datasets):
                    da = dataset.to_xarray()
                    da.attrs["dependent_variable_name"] = \
                        dataset.dependent_variable_name
                    da = da.assign_coords(dataset=n)
                    dep_name = dataset.dependent_variable_name
                    if dep_name in data_arrays:
                        da = xr.concat([data_arrays[dep_name], da],
                                       "dataset")
                        da.name = dep_name
                        data_arrays[dep_name] = da
                    else:
                        da.name = dep_name
                        data_arrays[dep_name] = da
                return xr.merge(data_arrays.values())

        else:

            def create_xr_dataset(datasets):
                data_arrays = {}
                for n, dataset in enumerate(datasets):
                    da = dataset.to_xarray()
                    da.attrs["dependent_variable_name"] = \
                        dataset.dependent_variable_name
                    org_dep_name = dataset.dependent_variable_name
                    dep_name = org_dep_name
                    n = 2
                    while dep_name in data_arrays:
                        dep_name = f"{org_dep_name}_{n}"
                        n += 1
                    da.name = dep_name
                    data_arrays[dep_name] = da
                return xr.merge(data_arrays.values())

        ds = create_xr_dataset(self.datasets)
        ds.attrs.update(attrs)
        return ds

    @classmethod
    def from_xarray(cls, xr_data) -> "Measurement":
        """Convert xarray.Dataset into cralds Measurement.

        Parameters
        ----------
        xr_data : xarray.Dataset
            xarray Dataset object containing measurement data.

        Returns
        -------
        Measurement
            Measurement representation of xarray Dataset.
        """
        conditions = xr_data.attrs.get("conditions", {})
        details = xr_data.attrs.get("details", {})

        cralds_datasets = []
        for data_variable in xr_data.data_vars:
            dep_data = xr_data[data_variable]
            if "dataset" in xr_data.coords:
                data_arrays = [
                    dep_data.sel(dataset=n).drop("dataset")
                    for n in dep_data.coords["dataset"]
                ]
            else:
                data_arrays = [dep_data]
            for da in data_arrays:
                cralds_module_str, _, cralds_class_str = \
                    da.attrs["cralds_cls"].rpartition('.')
                cralds_class = \
                    getattr(importlib.import_module(cralds_module_str),
                            cralds_class_str)
                cralds_datasets.append(cralds_class.from_xarray(da))

        return cls(datasets=cralds_datasets,
                   conditions=conditions,
                   details=details)

    def split(self, classifier: callable):
        """Split this measurement using a callable.

        Parameters
        ----------
        classifier : callable
            A callable that takes a measurement and returns a set of
            measurements.
        """
        raise NotImplementedError(
            "Generic split by callable is to be implemented."
        )

    def split_by_dataset_indices(
            self,
            dataset_indices: List[Set[int]] = None,
            conditions_to_remove: List[Set[str]] = None,
            conditions_to_add: List[dict] = None,
            allow_empty_measurements: bool = True) -> List["Measurement"]:
        """Alias for to_measurements."""
        return self.to_measurements(
            dataset_indices=dataset_indices,
            conditions_to_remove=conditions_to_remove,
            conditions_to_add=conditions_to_add,
            allow_empty_measurements=allow_empty_measurements
        )

    def split_datasets_by_callable(self, classifier: callable) -> List[dataset_.Dataset]:
        """Split all datasets of this measurement set with a classifier.

        Parameters
        ----------
        classifier : callable
            The classifier to use when splitting datasets.

        Returns
        -------
        List of Dataset
        """
        return list(itertools.chain([dataset.split_by_callable(classifier)
                                    for dataset in self.datasets]))

    def split_by_condition_names(self,
                                 condition_name_groups: Set[Set[str]]) -> \
            List["Measurement"]:
        """Split this measurement by subsets of condition names.

        I cannot imagine a use case for this, so it is not tested or
        carefully considered.

        Parameters
        ----------
        condition_name_groups : Set of Set
            One set per output measurement, containing a set of condition
            names to keep.

        Returns
        -------
        measurements: List of Measurement
        """
        measurements = []
        for condition_name_group in condition_name_groups:
            conditions = {key: value
                          for key, value in self.conditions.items()
                          if key in condition_name_group}

            measurements.append(Measurement(datasets=self.datasets,
                                            conditions=conditions,
                                            details=self.details))

        return measurements

    def split_by_dataset_independent_variable_name(
            self,
            independent_variable_name: str,
            allow_empty_measurements: bool = True
    ) -> Tuple["Measurement", "Measurement"]:
        """Split this measurement by dataset independent variable name.

        This split is achieved by taking the set of datasets in this
        measurement, and determining whether each has an independent
        variable with the provided name. The datasets are thus split into
        two sets, and then two new measurements are created with equal
        condition and detail metadata to the original measurement.
        Either or both of the resulting measurements may contain no
        datasets, either if this measurement already has no datasets,
        or if it has one or more datasets which are all without an
        independent variable with the specified name (resulting in the
        measurement with the name having no datasets), or if it has one or
        more datasets that all have an independent variable with the
        specified name (resulting in the measurement without the name
        having no datasets). The boolean parameter causes these cases to
        result in an exception.
        Note that this method does not need to know the values of the
        independent variable with the specified name because it is only
        splitting on the name. Splitting by independent variable value is
        implemented in the method
        split_by_dataset_independent_variable_value.

        Parameters
        ----------
        independent_variable_name : str
            The name of the independent variable to split this measurement
            on.
        allow_empty_measurements : bool
            Whether to allow a split to result in an empty measurement.
            True by default.

        Returns
        -------
        Tuple of Measurement
            The pair of measurements whose datasets have or do not have
            the specified independent variable name.
        """
        datasets_with_name: List[dataset_.Dataset] = []
        datasets_without_name: List[dataset_.Dataset] = []
        for dataset in self.datasets:
            if independent_variable_name in \
                    dataset.independent_variable_names:
                datasets_with_name.append(dataset)
            else:
                datasets_without_name.append(dataset)

        measurement_with_name = Measurement(datasets=datasets_with_name,
                                            conditions=self.conditions,
                                            details=self.details)

        if not measurement_with_name.datasets \
                and not allow_empty_measurements:
            raise ValueError(
                f"Splitting measurement by independent_variable_name "
                f"{independent_variable_name} "
                f"will result in an empty measurement.")

        measurement_without_name = Measurement(
            datasets=datasets_without_name,
            conditions=self.conditions,
            details=self.details
        )

        if not measurement_without_name.datasets and \
                not allow_empty_measurements:
            raise ValueError(
                f"Splitting measurement by independent_variable_name "
                f"{independent_variable_name} "
                f"will result in an empty measurement.")

        return measurement_with_name, measurement_without_name

    def split_by_dataset_independent_variable_names(
            self,
            independent_variable_names: List[str]
    ) -> Tuple["Measurement", "Measurement"]:
        """Split this measurement by dataset independent variable names.

        This split is achieved by taking the set of datasets in this
        measurement, and determining whether each has independent
        variables with the provided names. The datasets are thus split
        into two sets, and two new measurements are created sharing the
        same condition and detail metadata.
        Either or both of the resulting measurements may contain no
        datasets.

        Parameters
        ----------
        independent_variable_names : List of str

        Returns
        -------
        Tuple of Measurement
            The pair of measurements whose datasets have or do not have
            the specified independent variable name.
        """
        datasets_with_names: List[dataset_.Dataset] = []
        datasets_without_names: List[dataset_.Dataset] = []

        for dataset in self.datasets:

            if np.all([name in dataset.independent_variable_names
                       for name in independent_variable_names]):
                datasets_with_names.append(dataset)
            else:
                datasets_without_names.append(dataset)

        measurement_with_names = \
            Measurement(datasets=datasets_with_names,
                        conditions=self.conditions,
                        details=self.details)

        measurement_without_names = \
            Measurement(datasets=datasets_without_names,
                        conditions=self.conditions,
                        details=self.details)

        return measurement_with_names, measurement_without_names

    def split_by_dataset_dependent_variable_name(
            self, dependent_variable_name: str) -> Tuple["Measurement", "Measurement"]:
        """Split this measurement by dataset dependent variable names.

        Parameters
        ----------
        dependent_variable_name : str

        Returns
        -------
        List of Measurement
            The resulting list of measurements after the split.
        """
        matching_datasets = []
        non_matching_datasets = []

        for dataset in self.datasets:
            if dataset.dependent_variable_name == dependent_variable_name:
                matching_datasets.append(dataset)
            else:
                non_matching_datasets.append(dataset)

        return (
            Measurement(conditions=self.conditions,
                        details=self.details,
                        datasets=matching_datasets),
            Measurement(conditions=self.conditions,
                        details=self.details,
                        datasets=non_matching_datasets)
        )

    def split_by_dataset_independent_variable_value(
            self,
            dataset_independent_variable_name: str,
            return_measurement_without_name: bool = False
    ):
        """Split this measurement by independent variable value.

        This method splits this measurement into a set of measurements
        based on the values of the specified independent variable in
        their datasets. First it splits the measurement up into a
        measurement containing datasets that have the specified name and a
        measurement containing datasets that don't have the specified
        name. After this, it checks the datasets of the measurement that
        do have the name, and (if defined) extracts a single value of the
        independent variable for each and adds it to the conditions of
        that measurement. The result of this split is a set of
        measurements, each of which contains datasets that share the
        same value of the independent variable with the specified name.

        Parameters
        ----------
        dataset_independent_variable_name : str
            The name of the independent variable whose values are to be
            split on.
        return_measurement_without_name : bool
            Whether to include the measurement with datasets that do not
            have the independent variable or not. Default is to not
            return this measurement.
        """
        measurement_with_name, \
            measurement_without_name = \
            self.split_by_dataset_independent_variable_name(
                independent_variable_name=dataset_independent_variable_name
            )

        all_values = []
        for dataset in measurement_with_name.datasets:

            independent_variable_index = \
                dataset.independent_variable_index(
                    dataset_independent_variable_name
                )

            if not np.all(
                dataset.independent_variable_data[independent_variable_index] == dataset.independent_variable_data[independent_variable_index][0]
            ):
                raise ValueError(
                    "more than one unique value for ind var in a dataset"
                )

            all_values.append(
                (dataset,
                 dataset.independent_variable_data[independent_variable_index][0]
                 )
            )

        # there will be one measurement per unique value potentially with multiple datasets
        measurements = []
        unique_values = set([value[1] for value in all_values])
        for unique_value in unique_values:
            datasets = []
            for pair in all_values:
                if pair[1] == unique_value:
                    datasets.append(pair[0])

            new_conditions = {dataset_independent_variable_name: unique_value}
            new_conditions.update(self.conditions)

            measurements.append(Measurement(datasets=datasets,
                                            conditions=new_conditions,
                                            details=self.details))

        if return_measurement_without_name:
            return measurement_without_name, measurements
        else:
            return measurements

    def split_by_dataset_independent_variable_values(
            self,
            independent_variable_names: List[str],
            return_measurement_without_names: bool = False
    ):
        """Split this measurement by independent variable values.

        This method splits this measurement into a set of measurements
        based on the values of the specified independent variables in
        their datasets. This is useful when a measurement has multiple
        datasets that have the same independent variable names but
        different independent variable values for those names. First it
        splits the measurement up into a measurement containing datasets
        that have the specified names and a measurement containing
        datasets that don't have the specified names. After this,
        it checks the datasets of the measurement that do have the names,
        and (if defined) extracts the values of the independent variables
        for each and adds it to the conditions of that measurement. The
        result of this split is a set of measurements, each of which
        contains datasets that share the same value of the independent
        variable with the specified name.

        Parameters
        ----------
        independent_variable_names : str
            The name of the independent variable whose values are to be
            split on.
        return_measurement_without_names : bool
            Whether to include the measurement with datasets that do not
            have the independent variable or not. Default is to not
            return this measurement.
        """
        measurement_with_names, measurement_without_names = \
            self.split_by_dataset_independent_variable_names(
                independent_variable_names=independent_variable_names
            )

        # only care about those datasets that have the right independent
        # variable names
        all_independent_variable_values = []  # per dataset, what independent variable values are there?
        for dataset in measurement_with_names.datasets:

            measurement_values = []
            # for x, then for y
            independent_variable_indices = [dataset.independent_variable_index(name) for name in
                                            independent_variable_names]

            measurement_values = tuple([dataset.independent_variable_data[index][0] for index in independent_variable_indices])

            all_independent_variable_values.append((dataset, measurement_values))

        # there will be one measurement per unique value potentially with multiple datasets in each
        measurements = []
        unique_values = set([independent_variable_value[1] for independent_variable_value in all_independent_variable_values])
        for unique_value in unique_values:
            datasets = []
            for pair in all_independent_variable_values:
                associated_dataset = pair[0]
                independent_variable_value = pair[1]

                if independent_variable_value == unique_value:
                    datasets.append(associated_dataset)

            new_conditions = {}
            for independent_variable_name, value in zip(independent_variable_names, unique_value):
                new_conditions[independent_variable_name] = value

            # new_conditions = {independent_variable_names: unique_value}
            new_conditions.update(self.conditions)

            measurements.append(Measurement(datasets=datasets,
                                            conditions=new_conditions,
                                            details=self.details))

        if return_measurement_without_names:
            return measurement_without_names, measurements
        else:
            return measurements

    @classmethod
    def combine(cls,
                measurements: List["Measurement"],
                allow_missing_conditions: bool = False,
                keep_partial_conditions: bool = False) -> "Measurement":
        """Alias for from_measurements."""
        return cls.from_measurements(
            measurements=measurements,
            allow_missing_conditions=allow_missing_conditions,
            keep_partial_conditions=keep_partial_conditions
        )

    def remove_dataset_at_index(self, index: int) -> None:
        """Remove the dataset at the specified index.

        Parameters
        ----------
        index : int
            The index of the dataset to remove.
        """
        try:
            del self.datasets[index]
        except IndexError:
            if self.num_datasets > 0:
                indices_str = f"Valid indices: 0, {self.num_datasets - 1}"
            else:
                indices_str = ""

            raise IndexError(
                f"Measurement has no dataset at index {index}.\n" +
                indices_str
            )

    def remove_datasets_at_indices(self, indices: List[int]) -> None:
        """Remove datasets at specified indices.

        Parameters
        ----------
        indices : List of int
            The indices of the datasets to remove.
        """
        for index in sorted(indices, reverse=True):
            self.remove_dataset_at_index(index)

    def remove_dataset(self, dataset: dataset_.Dataset) -> None:
        """Remove a specified dataset from this measurement.

        Parameters
        ----------
        dataset : dataset_.Dataset
            The dataset to remove from this measurement.
        """
        try:
            self.datasets.remove(dataset)
        except ValueError:
            raise ValueError(f"Measurement does not contain specified "
                             f"dataset: {dataset}")

    def remove_datasets_with_name(self, dependent_variable_name: str) -> None:
        """Remove datasets with a specified dependent variable name.

        Parameters
        ----------
        dependent_variable_name : str
            The name of the dependent variable for which to remove datasets.
        """
        indices_of_datasets_to_remove = self.index_of(dependent_variable_name)
        if isinstance(indices_of_datasets_to_remove, list):
            for index in indices_of_datasets_to_remove:
                self.remove_dataset_at_index(index)
        else:
            self.remove_dataset_at_index(indices_of_datasets_to_remove)


    @property
    def datasets(self) -> List[dataset_.Dataset]:
        """The datasets collected as a result of the measurement."""
        return self._datasets

    @datasets.setter
    def datasets(self, datasets: List[dataset_.Dataset]) -> None:
        """Set the datasets collected as a result of this measurement."""
        self._datasets = datasets

    def index_of(self, dependent_variable_name: str) -> Union[int, List[int]]:
        """Return index of a dataset in this measurement with given name.

        This method allows retrieval of datasets from a measurement by
        their dependent variable name. This is often more convenient than
        having to inspect indices to find the correct dataset. As a
        measurement may contain multiple datasets with the same dependent
        variable name, this method returns the lowest-valued index.

        Parameters
        ----------
        dependent_variable_name : str
            The name of the dependent variable of interest.

        Returns
        -------
        i : int or List of int
            The lowest-valued index of a dataset with the given
            dependent variable name.
        """
        indices = []
        for i, dataset in enumerate(self.datasets):
            if dataset.dependent_variable_name == dependent_variable_name:
                indices.append(i)

        if len(indices) == 1:
            return indices[0]
        elif len(indices) > 0:
            return indices
        else:

            valid_values = [dataset.dependent_variable_name
                            for dataset in self.datasets]
            raise ValueError(f"No dataset with name {dependent_variable_name} "
                             f"in measurement.\nValid values: {valid_values}")

    @property
    def num_datasets(self) -> int:
        """Number of datasets collected as a result of the measurement."""
        return len(self.datasets)

    @property
    def dataset_types(self) -> List[type]:
        """Types of dataset collected as a result of the measurement."""
        return [type(dataset) for dataset in self.datasets]

    @property
    def dataset_lengths(self) -> List[int]:
        """Lengths of datasets collected as a result of the measurement."""
        lengths = []
        lengths.extend([len(dataset.flatten_dependent_variables())
                        for dataset in self.datasets])
        return lengths

    @property
    def dataset_dependent_variable_names(self) -> List[List[str]]:
        return [dataset.dependent_variable_names
                for dataset in self.datasets]

    @property
    def dataset_dependent_variable_units(self) -> List[List[str]]:
        return [dataset.dependent_variable_units
                for dataset in self.datasets]

    @property
    def dataset_independent_variable_data(self) -> List[np.ndarray]:
        return [dataset.independent_variable_data
                for dataset in self.datasets]

    @property
    def dataset_independent_variable_names(self) -> List[List[str]]:
        return [dataset.independent_variable_names
                for dataset in self.datasets]

    @property
    def dataset_independent_variable_units(self) -> List[List[str]]:
        return [dataset.independent_variable_units
                for dataset in self.datasets]

    @property
    def dataset_dimensionalities(self) -> List[int]:
        """The number of independent variables of each dataset."""
        return [dataset.number_of_independent_dimensions
                for dataset in self.datasets]

    @property
    def conditions(self) -> Dict[str, object]:
        """Experimental conditions under which the measurement was made."""
        return copy.deepcopy(self._conditions)

    @property
    def condition_names(self) -> Set[str]:
        """Names of the conditions recorded for this measurement."""
        return set(self.conditions.keys())

    def has_condition_name(self, condition_name: str) -> bool:
        """Determine whether this measurement has the given condition.

        Parameters
        ----------
        condition_name : str
            The name of the condition to check for.

        Returns
        -------
        bool
            Whether this measurement has a condition with the given name.
        """
        return condition_name in self.condition_names

    def has_condition_names(self, condition_names: List[str]) -> bool:
        """Determine whether this measurement has the given conditions.

        Parameters
        ----------
        condition_names : list of str
            The names of the conditions to check for.
        """
        for condition_name in condition_names:
            if not self.has_condition_name(condition_name):
                return False

        return True

    def add_condition(self, name: str, value: object = None) -> None:
        """Add a condition to this measurement.

        If a condition with the same name is already present in this
        measurement, an error will be raised unless the existing value
        is the same as the new value. This behaviour prevents this method
        from updating a condition, which should be done with update_condition.

        Parameters
        ----------
        name : str
            The name for the condition.
        value : object
            The value of the condition.
        """
        if name in self.detail_names:
            raise ExistingMetadataError(
                "Detail already exists with same name as new detail.",
                name
            )

        if name in self.condition_names or name in self.detail_names:
            if self._conditions[name] != value:
                raise ExistingMetadataError(
                    "Condition with requested name and different value "
                    "already exists."
                )

        self._conditions[name] = value

    def add_conditions(self, names: List[str], values: List[object]) -> None:
        """Add conditions to this measurement.

        Parameters
        ----------
        names : list of str
            The names for the conditions.
        values : list of object
            The values of the conditions.
        """
        if len(names) != len(values):
            raise IncompatibleListError(
                "Incompatible lengths of metadata name, value lists."
            )

        for name, value in zip(names, values):
            self.add_condition(name, value)

    def remove_condition(self, name: str) -> None:
        """Remove a condition from this measurement.

        Parameters
        ----------
        name : str
            The name of the condition to remove.
        """
        if name in self.condition_names:
            del self._conditions[name]

    def remove_conditions(self, names: List[str]) -> None:
        """Remove conditions from this measurement.

        Parameters
        ----------
        names : list of str
            The names of the conditions to remove.
        """
        for name in names:
            self.remove_condition(name)

    def remove_conditions_equal_to(self, value: object):
        """Remove conditions with the given value.

        Parameters
        ----------
        value : object
            The value for which to remove conditions.
        """
        for name, existing_value in self.conditions.items():
            if existing_value == value:
                self.remove_condition(name)

    def rename_metadata(self, old_name: str, new_name: str) -> None:
        """Replace an existing metadata name with a new metadata name.

        Parameters
        ----------
        old_name : str
            The existing metadata name to be replaced.
        new_name : str
            The metadata name with which to replace the existing metadata name.
        """
        if old_name in self.condition_names:
            self.add_condition(new_name, self.conditions[old_name])
            self.remove_condition(old_name)

        elif old_name in self.detail_names:
            self.add_detail(new_name, self.details[old_name])
            self.remove_detail(old_name)

        else:
            raise MissingMetadataError(
                "Cannot rename non-existent metadata."
            )

    def update_condition(self,
                         name: str,
                         value: object = None,
                         name_must_exist: bool = False) -> None:
        """Update a condition of this measurement.

        Parameters
        ----------
        name : str
            The name of the condition.
        value : object
            The value of the condition.
        name_must_exist : bool
            Whether the name must already be in use for a condition.

        Raises
        ------
        MissingMetadataError
            If the name must already exist but does not.
        ExistingMetadataError
            If the name is already in use for a detail.
        """
        if name_must_exist and name not in self.condition_names:
            raise MissingMetadataError(
                "Cannot update value for missing name."
            )
        elif name in self.detail_names:
            raise ExistingMetadataError(
                "Condition name is already a detail."
            )

        self._conditions[name] = value

    def update_conditions(self,
                          names: List[str],
                          values: List[object],
                          name_must_exist: bool = False) -> None:
        """Update conditions of this measurement.

        Parameters
        ----------
        names : list of str
            The names of the conditions.
        values : list of object
            The values of the conditions.
        name_must_exist : bool
            Whether the names must each already be in use for a condition.

        Raises
        ------
        IncompatibleListError
            If the length of the names and values lists differ.

        """
        if len(names) != len(values):
            raise IncompatibleListError(
                "Incompatible lengths of metadata name, value lists."
            )

        for name, value in zip(names, values):
            self.update_condition(name, value, name_must_exist)

    @property
    def details(self) -> Dict[str, object]:
        """Details that are not considered experimental conditions."""
        return copy.deepcopy(self._details)

    @property
    def detail_names(self) -> Set[str]:
        """Names of the details recorded for this measurement."""
        return set(self.details.keys())

    def has_detail_name(self, name: str) -> bool:
        """Determine whether this measurement has the given detail.

        Parameters
        ----------
        name : str
            The name of the detail to check for.

        Returns
        -------
        bool
            Whether this measurement has the given detail.
        """
        return name in self.details.keys()

    def has_detail_names(self, names: List[str]) -> bool:
        """Determine whether this measurement has the given details.

        Parameters
        ----------
        names : list of str
            The names of the details to check for.

        Returns
        -------
        bool
            Whether this measurement has all given details.
        """
        for name in names:
            if not self.has_detail_name(name):
                return False
        return True

    def add_detail(self, name: str, value: object) -> None:
        """Add a detail to this measurement.

        Parameters
        ----------
        name : str
            The name for the detail.
        value : object
            The value of the detail.
        """
        if name in self.condition_names:
            raise ExistingMetadataError(
                "Condition already exists with same name as new detail."
            )

        if name in self.detail_names:
            if self._details[name] != value:
                raise ExistingMetadataError(
                    "Detail with requested name already exists."
                )

        self._details[name] = value

    def add_details(self, names: List[str], values: List[object]) -> None:
        """Add details to this measurement.

        Parameters
        ----------
        names : list of str
            The names for the details.
        values : list of object
            The values of the details.
        """
        if len(names) != len(values):
            raise IncompatibleListError(
                "Incompatible lengths of metadata name, value lists."
            )

        for name, value in zip(names, values):
            self.add_detail(name, value)

    def remove_detail(self, name: str) -> None:
        """Remove a detail from this measurement.

        Parameters
        ----------
        name : str
            The name of the detail to remove.
        """
        del self._details[name]

    def remove_details(self, names: List[str]) -> None:
        """Remove details from this measurement.

        Parameters
        ----------
        names : list of str
            The names of the details to remove.
        """
        for name in names:
            self.remove_detail(name)

    def remove_details_equal_to(self, value: object):
        """Remove conditions with the given value.

        Parameters
        ----------
        value : object
            The value for which to remove conditions.
        """
        for name, existing_value in self.details.items():
            if existing_value == value:
                self.remove_detail(name)

    def update_detail(self,
                      name: str,
                      value: object,
                      name_must_exist: bool = False) -> None:
        """Update a detail of this measurement.

        Parameters
        ----------
        name : str
            The name of the detail.
        value : object
            The value of the detail.
        name_must_exist : bool
            Whether the name must already be in use for a detail.

        Raises
        ------
        MissingMetadataError
            If the name must already exist but does not.
        ExistingMetadataError
            If the name is already in use for a condition.
        """
        if name_must_exist and name not in self.detail_names:
            raise MissingMetadataError(
                "Cannot update value for missing name."
            )
        elif name in self.condition_names:
            raise ExistingMetadataError(
                "Detail name is already a condition."
            )

        self._details[name] = value

    def update_details(self,
                       names: List[str],
                       values: List[object],
                       name_must_exist: bool = False) -> None:
        """Update details of this measurement,

        Parameters
        ----------
        names : list of str
            The names of the details.
        values : list of object
            The values of the details.
        name_must_exist : bool
            Whether the names must each already be in use for a detail.

        Raises
        ------
        IncompatibleListError
            If the length of the names and values lists differ.

        """
        if len(names) != len(values):
            raise IncompatibleListError(
                f"Incompatible lengths of metadata name, value lists. "
                f"Received {len(names)} names and {len(values)} values."
            )

        for name, value in zip(names, values):
            self.update_detail(name, value, name_must_exist)

    def is_replicate_of(self, other: "Measurement") -> bool:
        """Determine whether this measurement is a replicate of another.

        One measurement is defined as a replicate of another iff both have
        identical experimental conditions, and share the same type of
        datasets (in list order). Instead of using class this should use
        some property of Dataset subclasses (likely dimensionality).

        Parameters
        ----------
        other : Measurement
            Another measurement to be compared to this measurement.

        Returns
        -------
        bool
            True iff this Measurement and the other are repetitions.

        Notes
        -----
        This function relies on the implementation of the equality
        operator for dictionaries. Two dict objects are equal iff their
        sorted (key, value) lists compare equal.
        """
        if not self._has_equal_conditions_to(other):
            return False

        if len(self.datasets) != len(other.datasets):
            return False

        for my_dataset, \
                other_dataset in zip(self.datasets, other.datasets):
            if type(my_dataset) != type(other_dataset):
                return False

        return True

    def is_not_replicate_of(self, other: "Measurement") -> bool:
        """Determine if this measurement is not a replicate of another.

        One measurement is not a replicate of another if they have any
        differing experimental conditions.

        Parameters
        ----------
        other : Measurement
            Another measurement to be compared to this measurement.

        Returns
        -------
        bool
            True iff this measurement and the other are not repetitions.
        """
        return not self.is_replicate_of(other)

    def _has_equal_conditions_to(self, other: "Measurement") -> bool:
        """Determine if this measurement has equal conditions to another.

        Parameters
        ----------
        other : Measurement
            The measurement to compare conditions to.

        Returns
        -------
        bool
            Whether this measurement has equal conditions to the other.
            This means the same set of condition names with equal
            corresponding values.
        """
        return self.conditions == other.conditions

    def _has_equal_shared_conditions_to(self, other: "Measurement") -> bool:
        """Determine if this measurement has equal shared conditions.

        Parameters
        ----------
        other : Measurement
            The other measurement to compare to.

        Returns
        -------
        bool
            Whether this measurement's shared conditions with another are
            equal.
        """
        shared_condition_names = self.__shared_condition_names_with(other)
        for shared_condition_name in shared_condition_names:
            if self.conditions[shared_condition_name] != \
                    other.conditions[shared_condition_name]:
                return False

        return True

    def __shared_condition_names_with(self,
                                      other: "Measurement") -> Set[str]:
        """Determine condition names shared between this and another.

        Parameters
        ----------
        other : Measurement
            Another measurement to be compared to this measurement.

        Returns
        -------
        set of str
            The set of condition names shared by this measurement and
            the other.
        """
        return set(self.condition_names) & set(other.condition_names)

    def condition_to_detail(self, condition_name: str) -> None:
        """Change a condition to a detail by providing its name.

        Parameters
        ----------
        condition_name : str
            The name of the condition to convert to a detail.
        """
        self.__move_metadata(condition_name,
                             self._conditions,
                             self._details)

    def conditions_to_details(self, condition_names: List[str]) -> None:
        """Change a list of conditions to details by providing their names.

        Parameters
        ----------
        condition_names : list of str
            The names of conditions to convert to details.
        """
        for condition_name in condition_names:
            self.condition_to_detail(condition_name)

    def detail_to_condition(self, detail_name: str) -> None:
        """Change a detail to a condition by providing its name.

        Parameters
        ----------
        detail_name : str
            The name of the detail to convert to a condition.
        """
        self.__move_metadata(detail_name, self._details, self._conditions)

    def details_to_conditions(self, detail_names: List[str]) -> None:
        """Change a list of details fo conditions by providing their names.

        Parameters
        ----------
        detail_names : list of str
            The names of details to be converted to conditions.
        """
        for detail_name in detail_names:
            self.detail_to_condition(detail_name)

    @staticmethod
    def __move_metadata(condition_name: str,
                        from_: Dict[str, object],
                        to: Dict[str, object]) -> None:
        """Move a piece of metadata from one dict to another.

        Parameters
        ----------
        condition_name : str
            The name of the metadata to move between dictionaries.
        from_, to : dict
            The dictionaries to move metadata from and to.

        Notes
        -----
        The default behavior for this function when a key is missing is to
        silently do nothing.
        """
        if condition_name in from_.keys():
            to[condition_name] = from_[condition_name]
            from_.pop(condition_name)

    @staticmethod
    def are_repetitions(measurements: List["Measurement"]) -> bool:
        """Determine if a list of measurements are repetitions.

        Two measurements are repetitions iff their conditions are equal.
        For a set of measurements, each pair must be repetitions for the
        whole set to be repetitions.

        Parameters
        ----------
        measurements : list of measurement.Measurement
            The iterable of measurements to be tested.

        Returns
        -------
        bool
            True iff all measurements in the list are repetitions.

        Notes
        -----
        The replicate relationship is transitive, so if A is a
        replicate of B, and B is a replicate of C, then A is a replicate
        of C. This means the function only needs check pairwise
        relationships between successive measurements.
        """
        n = len(measurements)
        for i in range(n-1):
            if not measurements[i].is_replicate_of(measurements[i+1]):
                return False

        return True

    @staticmethod
    def are_not_repetitions(measurements: List["Measurement"]) -> bool:
        """Determine if a list of measurements are not repetitions.

        Parameters
        ----------
        measurements : iterable of measurement.Measurement
            The iterable of measurements to be tested.

        Returns
        -------
        bool
            True iff all measurements in the list are not repetitions.
        """
        return not Measurement.are_repetitions(measurements)

    def flatten(self,
                condition_names: List[str] = None,
                detail_names: List[str] = None,
                default_value: object = None) -> \
            Tuple[List[str], List[object]]:
        """Turn this measurement into a single column/row.

        The measurement condition metadata keys act as headers for the
        first columns of the data. Thereafter, each dataset must be
        flattened in turn and columns concatenated. A set of condition
        names to include can be provided as an argument which is useful
        when flattening multiple measurements. Any provided condition
        names which are not present for this measurement result in null
        entries in the flattened data.

        Parameters
        ----------
        condition_names : list of str
            Conditions to include in the flattened data.
            Default of None includes all condition metadata.
        detail_names : list of str
            Details to include in the flattened data
            Default of None includes no detail metadata.
        default_value : object
            The default value for missing metadata names.

        Returns
        -------
        headers : list of str
            The headers of the data columns.
        list of object
            The values of the metadata and data for the measurement.
            As the metadata values can be any object, the return type must
            be a python list.
        """
        headers: List[str]
        data: List[object]
        headers, data = self.flatten_metadata(condition_names,
                                              detail_names,
                                              default_value)

        dataset_headers, flat_datasets = self.flatten_datasets()

        headers.extend(dataset_headers)
        data.extend(flat_datasets)

        return headers, data

    def flatten_metadata(self,
                         condition_names: List[str] = None,
                         detail_names: List[str] = None,
                         default_value: object = None) -> \
            Tuple[List[str], List[object]]:
        """Turn the specified condition_values into a single column/row.

        If condition names are provided, must only include the
        corresponding subset of conditions, otherwise include all. If no
        details names are provided, no details are included. The default
        value will be used to fill in any missing condition_values.

        Parameters
        ----------
        condition_names: List of str
            Conditions to include in the flattened condition_values.
            Default of None includes all condition_values.
        detail_names : List of str
            Details to include in the flattened condition_values.
            Default of None includes no detail condition_values.
        default_value : object
            The default value for missing condition_values names.

        Returns
        -------
        flat_condition_names : List of str
            The names of the given condition keys.
        condition_values : List of object
            The values of the given condition keys.
            Must be a list as the condition_values can be any type.
        """
        if condition_names is None:
            flat_condition_names = list(self.condition_names)
        else:
            flat_condition_names = list(condition_names)

        condition_values: List[object] = \
            [self.conditions.get(name, default_value)
             for name in flat_condition_names]

        if detail_names:
            for name in detail_names:
                condition_values.append(self.details.get(name,
                                                         default_value))
                flat_condition_names.append(name)

        return flat_condition_names, condition_values

    def flatten_datasets(self) -> Tuple[List[str], List[object]]:
        """Turn this measurement's datasets into a single column/row.

        This method relies on the implementation of the flatten method
        within each individual dataset of the measurement, which must
        share a return type with this method.

        Returns
        -------
        column_headers : List of str
            The headers of the dataset columns.
        flat_data : List of object
            The measurement datasets flattened to a single list.
        """
        column_headers = []
        flat_data = []

        for dataset in self.datasets:
            dataset_headers, data = dataset.flatten()
            column_headers.extend(dataset_headers)
            flat_data.extend(list(data))

        return column_headers, flat_data

    def has_collapsible_datasets(self):
        """Determine whether this measurement has collapsible datasets.

        To be collapsible, all the datasets of a measurement must have
        the same number of independent dimensions (i.e. the same
        dimensionality) and the same independent variable units.

        Returns
        -------
        bool
            Whether this measurement has collapsible datasets.
        """
        for dataset in self.datasets[1:]:
            if dataset.number_of_independent_dimensions != \
                    self.datasets[0].number_of_independent_dimensions:
                return False
            if dataset.independent_variable_units != \
                    self.datasets[0].independent_variable_units:
                return False

        return True

    def _has_common_dependent_variable_unit(self) -> bool:
        """Determine whether the datasets share the same dependent unit.

        Returns
        -------
        bool
            Whether this measurement's datasets share a dependent unit.
        """
        dependent_variable_unit = self.datasets[0].dependent_variable_unit
        for dataset in self.datasets[1:]:
            if dataset.dependent_variable_unit != dependent_variable_unit:
                return False
        return True

    def condition_label(self, include_name: bool = True) -> str:
        """A label composed of this measurement's conditions.

        Parameters
        ----------
        include_name : bool
            Whether to include the condition name in the label.
        """
        condition_label: str = ""
        for name, value in self.conditions.items():
            if include_name:
                condition_label += f"{name}={value}, "
            else:
                condition_label += f"{value}, "

        return condition_label[:-2]

    def __validate_visualize_flags(
            self,
            expand_datasets: bool,
            allow_multiple_y_axes: bool
    ) -> Tuple[bool, bool]:
        """Validate the flags provided to the visualize method.

        Certain combinations of flags are not possible depending on the
        nature of the datasets contained by a measurement. This method
        checks for these cases, sets flags to appropriate values and issues
        warnings to the user when flags will be ignored.

        For a measurement with a single dataset, expansion and use of
        multiple y-axes is irrelevant so the less computationally taxing
        path is taken.

        A measurement's datasets may be "collapsible" or not, in that it
        may or may not be possible to visualize them on a single axes in
        any meaningful manner. If they are not collapsible, then the
        datasets of the measurement must be expanded

        Parameters
        ----------
        expand_datasets : bool
            Whether to expand the datasets in the visualization.
        allow_multiple_y_axes : bool
            Whether to use multiple y-axes in the visualization.

        Returns
        -------
        expand_datasets : bool
            Whether to expand the datasets in the visualization.
        allow_multiple_y_axes : bool
            Whether to use multiple y-axes in the visualization.
        """
        if self.num_datasets == 1:
            if not expand_datasets or not allow_multiple_y_axes:
                print("Warning: expand_datasets and allow_multiple_y_axes "
                      "are ignored for single-dataset measurements.")
                expand_datasets = True
                allow_multiple_y_axes = True

        if not expand_datasets:

            if not self.has_collapsible_datasets():
                expand_datasets = True
                allow_multiple_y_axes = False
                type_str = ""
                for dataset_type in self.dataset_types:
                    type_str += f"{dataset_type.__name__}, "
                print(f"Warning: Must expand datasets: {type_str[:-2]} "
                      f"due to mismatch in dimensionality. "
                      f"Ignoring expand_datasets argument")

            if not allow_multiple_y_axes and not self._has_common_dependent_variable_unit():
                allow_multiple_y_axes = True
                print(f"Warning: Different dependent variable units "
                      f"require multiple axes. "
                      f"Ignoring allow_multiple_y_axes")

        elif expand_datasets:
            if not allow_multiple_y_axes:
                print(f"Warning: allow_multiple_y_axes=False is ignored "
                      f"when datasets are expanded")

        return expand_datasets, allow_multiple_y_axes


    def visualize(self,
                  axes: Union[matplotlib.axes.Axes, List[matplotlib.axes.Axes]] = None,
                  expand_datasets: bool = True,
                  allow_multiple_y_axes: bool = True,
                  include_text: bool = True,
                  figure_title: str = None,
                  total_figsize: Tuple[int] = None,
                  dataset_colors: List[np.ndarray] = None,
                  **axes_plotting_kwargs) -> Tuple[matplotlib.figure.Figure,
                                                   Union[matplotlib.axes.Axes,
                                                         List[matplotlib.axes.Axes]]]:
        """Visualize this measurement's datasets.

        Create a textual representation of the conditions and details of
        this measurement, along with a visual representation of its
        dataset.

        Parameters
        ----------
        axes : matplotlib.axes.Axes or list of matplotlib.axes.Axes
            One or more matplotlib axes on which to plot this measurement's
            datasets.
            The default of None results in creation of a figure and axes.
        expand_datasets : bool
            Whether to plot each dataset on its own axes object.
            Default is to do this, as multiple y-axes is a special case.
        allow_multiple_y_axes : bool
            Whether to allow the use of multiple y-axes.
            Default is to allow this.
        include_text : bool
            Whether to include a table of metadata in the output.
            Default is to include this component of the measurement.
        figure_title : str
            A title for the plot, overriding the metadata title.
        total_figsize : tuple
            A tuple of 2 numbers setting the total figure size.
        dataset_colors : list of Color
            List of colors for the plotted datasets.

        Returns
        -------
        fig : matplotlib.figure.Figure
            A matplotlib figure visualizing this measurement.
        axes : matplotlib.axes.Axes or list of matplotlib.axes.Axes
            The axes of the matplotlib figure.
        """
        if include_text:
            print(str(self))

        expand_datasets, allow_multiple_y_axes = \
            self.__validate_visualize_flags(
                expand_datasets=expand_datasets,
                allow_multiple_y_axes=allow_multiple_y_axes
            )

        fig: matplotlib.figure.Figure
        axes: Union[matplotlib.axes.Axes, List[matplotlib.axes.Axes]]

        fig, axes = self._setup_fig_and_axes(
            axes=axes,
            expand_datasets=expand_datasets,
            total_figsize=total_figsize,
            figure_title=figure_title
        )

        if expand_datasets:  # plot each dataset on its own axes object using default aesthetics for each dataset

            if self.num_datasets == 1 and isinstance(axes, matplotlib.axes.Axes):
                axes = [axes]

            for dataset, axis in zip(self.datasets, axes):
                dataset.visualize(axes=axis,
                                  include_text=False,
                                  **axes_plotting_kwargs)

        elif not expand_datasets:  # plot all datasets on a single axes object, either sharing or not sharing a y-axis

            # because datasets are sharing a plot, we need to differentiate them aesthetically
            # here we have num_datasets things to distinguish
            # we want to respect defaults if they are sufficient to distinguish the datasets visually
            # because datasets now share an axes, we have to label them somehow
            # because they share all metadata they must be differentiated based on dataset properties
            # aesthetics and a legend is one approach
            # directly labelling data marks is another
            # the right way to differentiate depends on whether we have multiple y-axes or not
            assert isinstance(axes, matplotlib.axes.Axes)

            if "color" not in axes_plotting_kwargs.keys():
                if dataset_colors is not None:
                    axes_plotting_kwargs["color"] = dataset_colors[0]

            lines = axes.get_lines()
            if lines:
                line = lines[0]
                # axes.yaxis.label.set_color(line.get_color())
                # axes.tick_params(axis='y', colors=list(line.get_color()))
            else:
                ...
                # color = axes.collections[0].get_facecolors()[0]
                # axes.yaxis.label.set_color(color)
                # axes.tick_params(axis='y', colors=color)

            # the first dataset goes on the normal axes
            self.datasets[0].visualize(axes=axes,
                                       include_text=False,
                                       **axes_plotting_kwargs)

            if not allow_multiple_y_axes:

                for dataset in self.datasets[1:]:
                    dataset.visualize(axes=axes,
                                      include_text=False,
                                      **axes_plotting_kwargs)

            elif allow_multiple_y_axes:
                # if we have multiple y-axes, the color of the axis and marks can be matched up, then the y-axis labels
                # describe the datasets. this will require a color per dataset which has to be applied to the dataset
                # viz and the axis if the datasets have different default colors, use those. If they don't, need to generate.

                twin_axes = []
                for i in range(self.num_datasets - 1):
                    t = axes.twinx()
                    twin_axes.append(t)

                i = 1
                for dataset in self.datasets[1:]:

                    if "color" not in axes_plotting_kwargs.keys():
                        if dataset_colors is not None:
                            axes_plotting_kwargs["color"] = dataset_colors[i]

                    dataset.visualize(axes=twin_axes[i - 1],
                                      include_text=False,
                                      **axes_plotting_kwargs)

                    shift = 1 + ((i - 1) * 0.25)
                    twin_axes[i - 1].spines.right.set_position(("axes", shift))
                    twin_axes[i - 1].spines['right'].set_visible(True)
                    twin_axes[i - 1].spines['right'].set_color("k")

                    min_val = np.amin(dataset.dependent_variable_data)
                    max_val = np.amax(dataset.dependent_variable_data)

                    twin_axes[i - 1].set_ylim(min_val - min_val * 0.1,
                                              max_val + max_val * 0.1)

                    i += 1

                if axes.get_lines():
                    axes_legend_handles = [axes.get_lines()[0]]
                    for twin_axis in twin_axes:
                        line = twin_axis.get_lines()[0]
                        axes_legend_handles.append(line)
                        # twin_axis.yaxis.label.set_color(line.get_color())
                        # twin_axis.tick_params(axis='y', colors=list(line.get_color()))

                    # axes.legend(handles=axes_legend_handles)

            # axes.legend(handles=axes_legend_handles)

        if not isinstance(axes, list):
            axes = [axes]

        if self.num_datasets == 1 and isinstance(self.datasets[0],
                                                 zero_dimensional_dataset_.ZeroDimensionalDataset):
            axes[0].set_xticks([])
            axes[0].spines['bottom'].set_visible(False)

        return fig, axes

    def _default_figure_title(self) -> str:
        return f"{self.__class__.__name__}: {self.condition_label()}"

    def _setup_fig_and_axes(self,
                            axes: Union[matplotlib.axes.Axes, List[matplotlib.axes.Axes]] = None,
                            expand_datasets: bool = True,
                            total_figsize: Tuple[int] = None,
                            figure_title: str = None) -> Tuple[matplotlib.figure.Figure,
                                                               Union[matplotlib.axes.Axes,
                                                                     List[matplotlib.axes.Axes]]]:

        """Setup a matplotlib figure and axes for plotting a measurement.

        Parameters
        ----------
        axes : matplotlib.axes.Axes or list of matplotlib.axes.Axes
            The axes on which to plot this measurement.
            Default is to create a new instance.
        expand_datasets : bool
            Whether all datasets will be plotted on a single axes object
        total_figsize : tuple
            The total size (in inches) of the matplotlib figure.
            Default of None computes the size from dataset defaults.
        figure_title : str
            The overall title for the figure.
            Default is to use a basic overall title.

        Returns
        -------
        axes : list of matplotlib.axes.Axes or matplotlib.axes.Axes
            The axes on which to plot this measurement.
        fig : matplotlib.figure.Figure
            The figure containing the axes to be plotted.

        Notes
        -----
        If no axes has been provided, the appropriate object must
        be created, sized and titled. If one has been provided,
        it is validated, then its parent figure is extracted and
        returned without any sizing or titling imposed.
        """
        fig: matplotlib.figure.Figure

        if axes is None:  # the axes or list of axes must be created

            if total_figsize is None:
                total_figsize = \
                    self.compute_matplotlib_figure_size(expand_datasets=
                                                        expand_datasets)

            if not expand_datasets:
                fig, axes = plt.subplots(1,
                                         1,
                                         figsize=total_figsize)
            else:
                if self.num_datasets > 0:
                    fig, axes = plt.subplots(1,
                                             self.num_datasets,
                                             figsize=total_figsize)
                else:
                    fig = plt.figure()
                    axes = []

            if figure_title is not None:
                fig.suptitle(figure_title)
            else:
                fig.suptitle(self._default_figure_title())

        else:  # the axes or list of axes already exists

            # if len(self.datasets) != 1 and expand_datasets:
            #     if len(axes) <= len(self.datasets):
            #         raise ValueError("Incorrect number of axes for plotting measurement datasets: ", len(axes))

            try:
                fig = axes.get_figure()
            except AttributeError:
                fig = axes[0].get_figure()

        fig.set_tight_layout({"rect": [0, 0.03, 1, 0.9]})

        return fig, axes

    def compute_matplotlib_figure_size(self,
                                       expand_datasets: bool = True) -> \
            Tuple[int]:
        """Compute an ideal figure size for this measurement.

        Returns
        -------
        Tuple of int
            The ideal figure size for this measurement.

        Notes
        -----
        The first value in the figsize tuple gives the horizontal extent
        of the figure, and so is obtained by summing over the default
        horizontal extents of each dataset in the measurement.
        The second value in the figsize tuple gives the vertical extent,
        which is set to the largest default vertical extent among the
        datasets.
        In addition, vertical spacers are added to separate the datasets.
        An additional horizontal spacer is added to make room for the
        figure title above the individual axes titles.
        """
        prepare_figsize: List[int] = [0, 0]

        if expand_datasets:

            for dataset in self.datasets:
                prepare_figsize[0] += dataset.DEFAULT_FIGURE_SIZE[0]
                if dataset.DEFAULT_FIGURE_SIZE[1] > prepare_figsize[1]:
                    prepare_figsize[1] = dataset.DEFAULT_FIGURE_SIZE[1]

        else:
            prepare_figsize = list(self.datasets[0].DEFAULT_FIGURE_SIZE)

        return tuple(prepare_figsize)

    def one_line_str(self) -> str:
        """Return a single-line human-readable representation.

        This representation of an instance is used by the measurement
        set class to comparatively list each measurement in a set.

        Returns
        -------
        str_rep : str
            A single-line string representation of this measurement.
        """
        if len(self.condition_names) == 0:
            str_rep = "conditions={}, "
        else:
            str_rep = "conditions={"
            for key, value in self.conditions.items():
                str_rep += str(key) + ":" + str(value) + ", "
            str_rep = str_rep[:-2] + "}, "

        if len(self.detail_names) == 0:
            str_rep += "details={}, "
        else:
            str_rep += "details={"
            for key, value in self.details.items():
                str_rep += str(key) + ":" + str(value) + ", "
            str_rep = str_rep[:-2] + "}, "

        if self.num_datasets == 0:
            return str_rep + "datasets=[]"
        else:
            str_rep += "datasets=["
            for i, dataset in enumerate(self.datasets):
                str_rep += f"{i}: {dataset.one_line_description}, "

            return str_rep[:-2] + "]"

    def one_line_dataset_string(self) -> str:
        """Single-line human-readable representation of the datasets.

        This representation of an instance is used by the experiment
        class to describe the datasets in the context of multiple
        replicate measurements' metadata.
        """
        str_rep = "datasets=["
        if self.num_datasets == 0:
            return str_rep + "]"
        else:
            for i, dataset in enumerate(self.datasets):
                str_rep += f"{dataset.one_line_description}, "

            return str_rep[:-2] + "]"

    def __str__(self) -> str:
        """Return a human-readable representation of this instance."""

        def metadata_dict_str(dict_: Dict[str, object], title: str) -> str:
            """Return human-readable representation of a metadata dict.

            Parameters
            ----------
            dict_ : dict
                The metadata to convert to a human-readable representation.
            title : str
                The title to use for the human-readable representation.

            Returns
            -------
            dict_str_rep : str
                Human-readable representation of the metadata dict.
            """
            dict_str_rep = ""
            if dict_:
                dict_str_rep += dict_str(dict_=dict_,
                                         title=title,
                                         whitespace=self.__INDENT)
            else:
                dict_str_rep += title + ": None defined"
            return dict_str_rep + "\n"

        str_rep = "Measurement\n-----------\n\n"

        if len(self.datasets) == 0:
            str_rep += "Datasets: None defined"
        else:
            str_rep += f"Datasets ({self.num_datasets} total):\n\n"
            dataset_str = ""
            for dataset in self.datasets:
                if self.num_datasets > self.__DATASET_PRINT_MAX:
                    dataset_str += self.__INDENT + \
                                   dataset.one_line_description
                else:
                    dataset_str += self.__INDENT + str(dataset)

                dataset_str = dataset_str + "\n"

            str_rep += dataset_str

        str_rep += "\n"
        str_rep += metadata_dict_str(self.conditions, "Conditions")
        str_rep += metadata_dict_str(self.details, "Details")

        return str_rep

    def __repr__(self) -> str:
        """Create an eval()-able string representation of this measurement.

        Due to the potential large size of higher-dimensional
        dataset subclass string representations, there is no guarantee
        that the result of this method will actually be eval()-able.

        Returns
        -------
        str
            An eval()-able string representation of this measurement.
        """
        str_rep = self.__class__.__name__ + "("
        str_rep += "datasets=["
        for dataset in self.datasets:
            str_rep += repr(dataset) + ","
        str_rep += "]\n"

        if self.conditions:
            str_rep += ", conditions=" + dict_repr(self.conditions)

        if self.details:
            str_rep += ", details=" + dict_repr(self.details)

        return str_rep + ")"

    def __eq__(self, other: "Measurement") -> bool:
        """Return True is self is equal to other.

        Parameters
        ----------
        other : Measurement
            The measurement to check for equality.

        Returns
        -------
        bool
            True iff this measurement is equal to the other measurement.

        Notes
        -----
        Equality between measurements depends on the equality between
        python dicts, and between datasets as defined in this package.
        Equal dataset order is not a requirement for equality of
        measurements, however they must have the same set of datasets.
        Note that this is a difference in the equality and replicate
        relationships between measurements. Details are also included when
        assessing equality of measurements.
        """
        if self is other:
            return True

        for dataset in self.datasets:
            if dataset not in other.datasets:
                return False

        for dataset in other.datasets:
            if dataset not in self.datasets:
                return False

        if self.conditions != other.conditions:
            return False

        if self.details != other.details:
            return False

        return True

    def __cmp__(self):
        """Comparison based on magnitudes is purposely not defined."""
        return NotImplemented
