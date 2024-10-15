"""New implementation of an experiment which is now a consistent measurement set,
inheriting all the appropriate methods from the parent class.

An experiment must be consistent by definition.
Further restrictions can be determined by defining an experiment as a set of repetitions.
A set of measurements are repetitions if for each of their shared condition names, they
all have the same corresponding condition value.

In practice these are usually created by an experiment set which handles the change in
replicate relationships as metadata is edited.
"""
from typing import Tuple
import copy

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing
import matplotlib
import matplotlib.axes
import matplotlib.figure
import piblin.data.datasets.abc.split_datasets.histogram as histogram
import piblin.data.data_collections.measurement as measurement
import piblin.data.data_collections.summary as summary
import piblin.data.data_collections.consistent_measurement_set as consistent_measurement_set
from typing import List, Dict, Set, Union


class ReplicateError(Exception):
    """Raised when measurement parameters are not repetitions."""


class ConsistentDatasetError(Exception):
    """Raised when experimental datasets are not consistent."""


class MetadataEditingError(Exception):
    """Raised when a metadata change would change replicate relationships."""


class Experiment(consistent_measurement_set.ConsistentMeasurementSet):
    """A collection of consistent measurements performed under equal conditions.

    An experiment must be consistent, and adds a requirement
    on metadata to ensure that all of its measurements are
    repetitions.
    The concept of replacing a set of measurements with their
    mean is also defined for an experiment. This requires that
    the experiment be tidy. Also the mean experiment will be a
    measurement set composed of one measurement, which itself
    only has conditions and details which were equal for all
    measurements of the original measurement set.
    Along with the mean, a measure of variation can be defined
    for a particular dataset type and computed for an experiment.
    The ability to compute a statistical representation of
    multiple measurements also affects visualization as this can
    be shown in place of the complete set of measurements.

    Parameters
    ----------
    measurements : list of piblin.data.measurement.Measurement
        The set of replicate measurements of this experiment.

    Attributes
    ----------
    measurements -> list of piblin.data.measurement.Measurement
        The collected measurements.
    num_measurements -> int
        The number of collected measurements.
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
    repetitions -> list of Measurement
        The replicate measurements of this experiment.
        Alias for measurements.
    conditions -> dict
        The conditions under which this experiment was carried out.
        Alias for equal_shared_conditions.
    condition_names -> set of str
        The names of conditions under which this experiment was carried out.
        Alias for equal_shared_condition_names.
    average_measurement -> Measurement
        A single measurement representing the average of this experiment.
    variation_measurement -> Measurement
        A single measurement representing the variation of this experiment.

    Methods
    -------
    from_single_measurement(Measurement) -> ConsistentMeasurementSet
        Create an experiment from an existing single measurement.
    from_measurement_set(MeasurementSet) -> ConsistentMeasurementSet
        Create an experiment from an existing measurement set.
    from_measurement_sets([MeasurementSet]) -> ConsistentMeasurementSet
        Create an experiment by combining multiple measurement sets.
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
        Convert a set of conditions to details for all repetitions.
    detail_to_condition(str) -> None
        Convert a detail to a condition.
        A detail must be shared and equal across all repetitions to be converted.
    details_to_conditions(list of str) -> None
        Convert a set of details to conditions for all measurements.
    add_equal_shared_condition(str, object) -> None
        Add a condition with a specified value to all measurements.
    add_equal_shared_conditions([str], [object]) -> None
        Add conditions with specified values to all measurements.
    add_varying_shared_condition(str, [object]) -> None
        Add a condition with different values for each measurement.
        This is not possible for an experiment.
    add_varying_shared_conditions([str], [[object]]) -> None
        Add conditions with different values for each measurement.
        This is not possible for an experiment.
    add_condition(str, object) -> None
        Add a condition to all measurements in the collection.
    add_conditions(list of str, list of object) -> None
        Add conditions to all measurements in the collection.
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
                 measurements: List[measurement.Measurement] = None,
                 merge_redundant: bool = True):

        super().__init__(measurements=measurements,
                         merge_redundant=merge_redundant)

        if self.are_not_repetitions():
            raise ReplicateError("Cannot create experiment: measurements are not repetitions.")

    def to_measurement_set(self,
                           dataset_indices_to_retain: List[int] = None,
                           make_copy=True):
        """Cast this experiment to a measurement set."""
        return self.to_consistent_measurement_set(dataset_indices_to_retain=dataset_indices_to_retain,
                                                  make_copy=make_copy)

    def datasets_to_histogram(self,
                              dataset_indices: List[int] = None,
                              bins: Union[int, list, str] = 'auto',
                              histogram_range: Tuple[float, float] = None,
                              weights: np.typing.ArrayLike = None,
                              density: bool = None) -> None:
        """Convert this experiment to a histogram.

        An experiment contains multiple replicate measurements, each of which can have multiple datasets.
        If we are treating our experiment as multiple samples from a single distribution, it needs to be
        possible to analyze that distribution (and fit to it), so a conversion is needed.

        The experiment is going to be collapsed by this process because multiple measurements will be turned
        into a single measurement.

        The label for the scalar being collapsed should become the label of the bin centers, the counts are
        not related to the original labels.

        Parameters
        ----------
        dataset_indices : list of int
        bins : int or sequence of scalars or str
        histogram_range : (float, float)
        weights : array_like
        density : bool
        """
        if dataset_indices is None:
            dataset_indices = [0]

        histograms = []
        for dataset_index in dataset_indices:
            dependent_variables = [measurement_.datasets[dataset_index].dependent_variable_data for measurement_ in
                                   self.measurements]

            counts, bins = np.histogram(dependent_variables,
                                        bins=bins,
                                        range=histogram_range,
                                        weights=weights,
                                        density=density)

            centers = [bin_a + ((bin_b - bin_a) / 2) for i, (bin_a, bin_b) in enumerate(zip(bins[1:], bins[:-1]))]

            histogram_ = histogram.Histogram.create(x_values=centers,
                                                    y_values=counts,
                                                    x_name=f"Bin Center")

            histograms.append(measurement.Measurement(datasets=[histogram_]))

        self.repetitions = histograms

    @property
    def repetitions(self) -> List[measurement.Measurement]:
        """The replicate measurements of this experiment."""
        return self.measurements

    @repetitions.setter
    def repetitions(self, repetitions: List[measurement.Measurement]) -> None:
        self.measurements = repetitions
        if self.are_not_repetitions():
            raise ReplicateError("Cannot update experiment: measurements are not repetitions.")

    @property
    def conditions(self) -> Dict[str, object]:  # alias
        """The conditions under which this experiment was carried out."""
        return self.equal_shared_conditions

    @property
    def condition_names(self) -> Set[str]:
        """The names of conditions under which this experiment was carried out."""
        return self.equal_shared_condition_names

    def add_condition(self, name: str, value: object) -> None:
        """Add a condition to the repetitions of this experiment.

        This method is just an alias for add_equal_shared_condition.

        Parameters
        ----------
        name : str
            The name of the condition to add.
        value : object
            The value of the condition to add.
        """
        self.add_equal_shared_condition(name, value)

    def add_conditions(self, names: List[str], values: List[str]) -> None:
        """Add conditions to the repetitions of this experiment.

        This method is just an alias for add_equal_shared_conditions.

        Parameters
        ----------
        names : list of str
            The names of the conditions to add.
        values : list of object
            The values of the conditions to add.
        """
        self.add_equal_shared_conditions(names, values)

    def add_varying_shared_condition(self, name: str, values: list):
        """Add a shared, varying condition to this experiment."""
        raise MetadataEditingError("Cannot add shared, varying condition to experiment.")

    def add_varying_shared_conditions(self, name: str, values: list):
        """Add shared, varying conditions to this experiment."""
        raise MetadataEditingError("Cannot add shared, varying conditions to experiment.")

    def detail_to_condition(self, detail_name: str):
        """Convert a detail to a condition.

        This operation is only possible when the detail is already shared
        by all measurements and has an equal value in each.

        Parameters
        ----------
        detail_name : str
            The name of the detail to convert to a condition.
        """
        if detail_name in self.equal_shared_detail_names:
            self.detail_to_condition(detail_name)
        else:
            MetadataEditingError("Cannot change shared, varying detail to a shared, varying condition in experiment.")

    def flatten(self,
                force_tidiness: bool = False,
                include_unshared_conditions: bool = False,
                include_equal_conditions: bool = True,
                default_value: object = None,
                expand_replicates: bool = True) -> Tuple[List[str], List[List[object]]]:
        """Turn the experiment into a set of columns/rows.

        The tabular form of an experiment has a row for each repetition and a column for each
        independent variable of each dataset. There is nothing forcing

        By definition, all of the conditions in this class are present and
        equal so the set of metadata columns is fixed.
        The datasets are not guaranteed to share independent data values.

        Parameters
        ----------
        force_tidiness : bool
            Whether to make this experiment tidy before flattening.
        include_unshared_conditions : bool
            Whether to include conditions not defined for all measurements.
            Default is a tidy dataset which has specific values for all entries.
        include_equal_conditions : bool
            Whether to include conditions which are equal for all measurements.
            Default is a tidy dataset which does not include redundant columns.
        default_value : object
            The value to set for missing conditions.
            USed only if include_unshared_conditions is set true.
        expand_replicates : bool
            Whether to expand replicates or return the flat mean experiment.

        Returns
        -------
        list
            The experiment as a rectangular set of data.
        """
        if expand_replicates:

            return super().flatten(force_tidiness=force_tidiness,
                                   include_unshared_conditions=include_unshared_conditions,
                                   include_equal_conditions=include_equal_conditions,
                                   default_value=default_value)

        else:
            experiment = Experiment(measurements=[self.average_measurement])
            return experiment.flatten(force_tidiness=force_tidiness,
                                      include_unshared_conditions=include_unshared_conditions,
                                      include_equal_conditions=include_equal_conditions,
                                      default_value=default_value,
                                      expand_replicates=False)

    def flatten_datasets(self,
                         force_tidiness=True,
                         expand_replicates=True) -> Tuple[List[numpy.typing.NDArray],
                                                       List[List[object]]]:
        """Flatten the datasets of this experiment into headers and data.

        Parameters
        ----------
        force_tidiness : bool
            Whether to make the experiment set tidy before flattening.
        expand_replicates: bool
            Whether to include all replicates (the default) or the average measurement.
        """
        if expand_replicates:
            return super().flatten_datasets(force_tidiness=force_tidiness)
        else:
            average_measurement = self.average_measurement
            return average_measurement.flatten_datasets()

    # same as condition_label
    def create_description_from_conditions(self, keys=None, sep="_"):
        """Create a single-line description of this experiment.

        Parameters
        ----------
        keys : set
            A list of conditions to use in producing the description.
        sep : str
            A string to separate key-value pairs in the description.

        Returns
        -------
        str
            One line describing the experiment.
        """
        if keys is None:
            keys = self.conditions.keys()
        output = ""
        for key in keys:
            output += str(key) + "=" + str(self.conditions[key]) + sep
        return output[:-1]

    def visualize(self,
                  axes: matplotlib.axes.Axes or List[matplotlib.axes.Axes] = None,
                  expand_datasets: bool = True,
                  expand_replicates: bool = True,
                  include_text: bool = True,
                  figure_title: str = None,
                  total_figsize: Tuple[int] = None,
                  **plot_kwargs) -> Tuple[matplotlib.figure.Figure,
                                          Union[matplotlib.axes.Axes,
                                                List[matplotlib.axes.Axes]]]:
        """Create a visual representation of this experiment.

        Parameters
        ----------
        axes : matplotlib.axes.Axes or list of matplotlib.axes.Axes
            One or more axes on which to plot this experiment's datasets.
        expand_datasets : bool
            Whether to plot each measurement's datasets on their own axes object.
            Default is to do this.
        expand_replicates : bool
            Whether to visualize all repetitions or a statistical summary.
            By default only show the summarized information.
        include_text : bool
            Whether to display the str representation of the measurement set.
            Default is to include this information.
        figure_title : str
            A title for the figure.
        total_figsize : tuple of float
            The size (in inches) of the complete figure.
        plot_kwargs
            Keyword arguments for the matplotlib plot function.

        Returns
        -------
        fig : matplotlib.figure.Figure
            A matplotlib figure containing the axes.
        axes : matplotlib.axes.Axes or list of matplotlib.axes.Axes
            Axes containing the plotted experiment.
        """
        if "expand_measurements" in plot_kwargs.keys():
            print("Warning: Expansion of measurements is not supported for experiments.")
            plot_kwargs.pop("expand_measurements")

        if include_text:
            print(str(self))

        if not expand_datasets:
            if not self._has_collapsible_datasets():
                print("Warning: Datasets are not collapsible.")
                expand_datasets = True

        fig, axes = self._setup_fig_and_axes(axes=axes,
                                             expand_datasets=expand_datasets,
                                             expand_replicates=expand_replicates,
                                             total_figsize=total_figsize,
                                             title=figure_title)

        self.visualize_on_plots(fig=fig,
                                axes=axes,
                                expand_datasets=expand_datasets,
                                expand_replicates=expand_replicates,
                                **plot_kwargs)
        plt.tight_layout()

        return fig, axes

    def _setup_fig_and_axes(self,
                            axes: Union[matplotlib.axes.Axes, List[matplotlib.axes.Axes]] = None,
                            expand_datasets: bool = True,
                            expand_replicates: bool = True,
                            total_figsize: tuple = None,
                            title: str = None) -> Tuple[matplotlib.figure.Figure,
                                                        Union[matplotlib.axes.Axes, List[matplotlib.axes.Axes]]]:
        """Validate/prepare a figure and axes for plotting this experiment.

        Parameters
        ----------
        axes : matplotlib.axes.Axes or list of matplotlib.axes.Axes
            The axes on which to plot this experiment.
        expand_datasets : bool
            Whether to plot each measurement's datasets on their own axes object.
            Default is to do this, as multiple y-axes is a special case.
        expand_replicates : bool
            Whether to visualize all repetitions or a statistical summary.
            By default only show the summarized information.
        total_figsize : tuple(float)
            A tuple of 2 numbers setting the figure size.
        title : str
            A title for the visualization, overriding the metadata title.

        Returns
        -------
        fig : matplotlib.figure.Figure
            A matplotlib figure containing the axes.
        axes : matplotlib.axes.Axes
            Matplotlib axes containing the plotted experiment set.
        """
        if axes is None:  # create them
            total_figsize = self._validate_figsize(figsize=total_figsize,
                                                   expand_datasets=expand_datasets,
                                                   expand_replicates=expand_replicates)

            if expand_datasets:
                fig, axes = plt.subplots(1,
                                         self.num_datasets,
                                         figsize=total_figsize)
            else:
                fig, axes = plt.subplots(1, 1, figsize=total_figsize)

        else:
            if isinstance(axes, matplotlib.axes.Axes):
                fig = axes.get_figure()
            else:
                fig = axes.flat[0].get_figure()

        return fig, axes

    def visualize_on_plots(self,
                           fig: matplotlib.figure.Figure,
                           axes: Union[matplotlib.axes.Axes, List[matplotlib.axes.Axes]],
                           expand_datasets: bool = True,
                           expand_replicates: bool = True,
                           **plot_kwargs) -> None:
        """

        Parameters
        ----------
        fig : matplotlib.figure.Figure
        axes : matplotlib.axes.Axes or list of matplotlib.axes.Axes
        expand_datasets : bool
        expand_replicates : bool

        Returns
        -------

        """
        if expand_replicates:
            # plot all of the measurements on the axes
            for i, measurement_ in enumerate(self.measurements):
                measurement_.visualize(axes,
                                       expand_datasets=expand_datasets,
                                       include_text=False,
                                       **plot_kwargs)

        elif not expand_replicates:
            self.summary_measurement.visualize(axes=axes,
                                               expand_datasets=expand_datasets,
                                               include_text=False,
                                               **plot_kwargs)

    def _validate_figsize(self,
                          figsize: Tuple[int],
                          expand_datasets: bool,
                          expand_replicates: bool) -> Tuple[int]:
        """Assess and potentially create a size for the complete figure."""
        if figsize is None:
            figsize = self.compute_figure_size(expand_datasets=expand_datasets,
                                               expand_replicates=expand_replicates)

        return figsize

    def compute_figure_size(self,
                            expand_replicates: bool,
                            expand_datasets: bool) -> Tuple[int, int]:
        """Determine an overall size for the figure based on its components.

        Parameters
        ----------
        expand_replicates : boolean
            Whether to plot all replicates individually or use the mean and variation.
        expand_datasets : boolean
            Whether to plot all datasets on one plot or use separate.
        """

        height = max([dataset_type.DEFAULT_FIGURE_SIZE[1] for dataset_type in self.dataset_types])

        if expand_datasets:
            width = sum([dataset_type.DEFAULT_FIGURE_SIZE[0] for dataset_type in self.dataset_types])
        else:
            width = max([dataset_type.DEFAULT_FIGURE_SIZE[1] for dataset_type in self.dataset_types])

        return int(width), int(height)

    @property
    def summary_measurement(self):
        return self._compute_summary()

    def _compute_summary(self):
        return summary.SummaryMeasurement(self.average_measurement.datasets,
                                          self.variation_measurement.datasets,
                                          self.equal_shared_conditions,
                                          self.equal_shared_details)

    @property
    def average_measurement(self) -> measurement.Measurement:
        """The average measurement of this experiment.

        The average measurement of this experiment will be a single
        measurement with the same number of datasets as this consistent
        measurement set, where the independent variable values of each dataset
        must be chosen from those of all of the measurements, and the
        dependent variable values determined at those points.

        For this to be possible the experiment must not just be consistent, but
        must also be tidy (i.e. each dataset in a column must share the same
        independent variable values). Computing the average therefore means that
        a tidying scheme be in place for each dataset. Default tidying schemes
        should belong to dataset classes (as class methods which take a set of
        datasets of that class' type to generate their parameters). The act of
        tidying is the application of a pipeline to the experiment.
        """
        return self._compute_average_measurement()

    def _compute_average_measurement(self) -> measurement.Measurement:
        """Compute the average measurement of this experiment.

        If there is one measurement/replicate, the mean is just that
        measurement.
        If not, the metadata can be dealt with first as this is simplest. The conditions
        of the average measurement will be the shared and equal conditions of the set of
        replicates. All other metadata will be lost.
        The datasets are then per-dataset averages over all the replicates, which requires
        this experiment to be tidy (all datasets in a given column share the same independent
        variable values).
        """
        if not self.is_tidy:
            self.measurements = self.to_tidy_measurement_set().measurements

        conditions = self.equal_shared_conditions
        details = self.equal_shared_details

        if len(self.measurements) == 1:
            return self.measurements[0]
        else:
            average_datasets = []
            for dataset_index in range(self.num_datasets):

                dataset_type = self.dataset_types[dataset_index]

                datasets = [measurement_.datasets[dataset_index] for measurement_ in self.measurements]
                average_datasets.append(dataset_type.compute_average(datasets))

        return measurement.Measurement(datasets=average_datasets,
                                       conditions=conditions,
                                       details=details)

    @property
    def variation_measurement(self) -> measurement.Measurement:
        """The variation across this experiment."""
        return self._compute_variation_measurement()

    def _compute_variation_measurement(self) -> measurement.Measurement:
        """Compute the variation across this experiment."""
        if len(self.measurements) == 1:
            return self.measurements[0]
        else:
            average_datasets = []
            for dataset_index in range(self.num_datasets):
                datasets = [measurement_.datasets[dataset_index] for measurement_ in self.measurements]
                average_datasets.append(self.dataset_types[dataset_index].compute_variation(datasets))

        return measurement.Measurement(datasets=average_datasets,
                                       conditions=self.equal_shared_conditions,
                                       details=self.equal_shared_details)

    def detect_outliers(self):
        """Return a set of outlier measurements based on a derived property."""
        return NotImplemented

    def __getitem__(self, position: Union[int, slice]) -> Union[measurement.Measurement, "Experiment"]:
        if isinstance(position, slice):
            return Experiment(self.measurements[position])

        return self.measurements[position]

    def __setitem__(self, index: int, value: measurement.Measurement) -> None:
        if measurement.Measurement.are_repetitions([self.measurements[0], value]):
            self.measurements[index] = value
        else:
            raise ReplicateError("Cannot add non-replicate measurement to experiment.")

    def __str__(self):
        """Create human-readable representation of this experiment.

        Listing out all of the replicate measurements for the user is probably
        not useful, but a statistical approach to reporting may not be either
        for datasets with many points.
        """
        str_rep = f"Experiment ({self.num_measurements} repetitions)\n"
        underline = "-" * (len(str_rep) - 1)
        str_rep += f"{underline}\n\n"

        str_rep += "Conditions\n----------\n"
        for key, value in self.conditions.items():
            str_rep += key + " = " + str(value) + "\n"

        str_rep += "\n"

        str_rep += "Repetitions\n-----------\n"
        for i, replicate in enumerate(self.measurements):
            str_rep += "Replicate " + str(i) + ": " + \
                       replicate.one_line_dataset_string() + "\n"

        return str_rep + "\n"

    # @property
    # def one_line_description(self) -> str:
    #     """A single-line human-readable description of this experiment."""
    #     str_rep = ""
    #     for replicate in self.repetitions:
    #         str_rep += f"{replicate.one_line_str()}\n"
    #     return str_rep

    def __eq__(self, other):
        """Return True is self is equal to other.

        The conditions are guaranteed by the test in the initializer to be
        the same for all measurements, so two experiments are equal if their
        repetitions are the same.

        Parameters
        ----------
        other : Experiment
            The experiment to check for equality.

        Returns
        -------
        bool
            True iff this experiment is equal to the other experiment.
        """
        if self is other:
            return True

        if len(self.datasets) != len(other.datasets):
            return False

        for replicate in self.datasets:
            if replicate not in other.datasets:
                return False

        return True

    def __cmp__(self):
        """Comparison based on magnitudes is purposely not defined."""
        return NotImplemented

    def to_pandas(self, with_cralds_class: bool = True):
        """Converts Experiment into a Pandas DataFrame.

        Parameters
        ----------
        with_cralds_class : bool, optional
            If True, a column corresponding to the source cralds Dataset class is
            added to the DataFrame. This simplifies conversion back into a cralds
            data structure.

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

        return _to_pandas(self, with_cralds_class, concat=False)
