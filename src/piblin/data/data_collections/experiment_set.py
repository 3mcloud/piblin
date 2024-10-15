from typing import List, Set, Tuple, Union
import copy
import numpy as np
import numpy.typing
import matplotlib.axes
import matplotlib.figure
import matplotlib.cm as cm
import matplotlib.pyplot as plt

import piblin.data.datasets.abc.unambiguous_datasets.unambiguous_dataset
import piblin.data.datasets.abc.split_datasets.zero_dimensional_dataset as zero_dimensional_dataset
import piblin.data.datasets.abc.split_datasets.one_dimensional_dataset as one_dimensional_dataset
import piblin.data.datasets.abc.split_datasets.two_dimensional_dataset as two_dimensional_dataset

import piblin.data.datasets.abc.dataset_factory as dataset_factory
import piblin.data.data_collections.measurement_set as measurement_set_
import piblin.data.data_collections.consistent_measurement_set as consistent_measurement_set_
import piblin.data.data_collections.experiment as experiment_
import piblin.data.data_collections.measurement as measurement_
import piblin.data.data_collections.data_collection_factory as measurement_set_factory_
import piblin.data.datasets.abc.dataset
from piblin.data.data_collections.consistent_measurement_set import InconsistentMeasurementsError


class ExperimentSet(consistent_measurement_set_.ConsistentMeasurementSet):
    """A set of experiments and their related metadata.

    The experiment set wraps a measurement set and updates itself whenever
    that measurement set is edited. It exposes its experiments as a
    property which depend on the underlying measurement set metadata.
    This is the class with which users are most likely to interact directly.
    A choice has to be made between wrapping a MeasurementSet and inheriting
    from MeasurementSet, for now we have chosen the former pending more
    investigation. Basically, every time the underlying measurement set
    metadata is edited we need to call __update_experiments() to ensure that
    the experiments member is correct.

    Parameters
    ----------
    measurements : list of Measurement
        A set of measurements underlying a set of experiments.

    Methods
    -------
    The following are implemented to wrap the measurement_set data member and
    must call update_experiments.

    condition_to_detail
        Convert a condition to a detail.
    detail_to_condition
        Convert a detail to a condition.
    add_condition
        Add a condition to all measurements in the collection.
    add_detail
        Add a detail to all measurements in the collection.
    remove_condition
        Remove a condition from all measurements in the collection.
    remove_detail
        Remove a detail from all measurements in the collection.
    condition_to_dataset
        Convert a condition to a zero-dimensional dataset.
    dataset_to_condition
        Convert the dataset at the given index to a condition.
    """
    def __init__(self,
                 measurements: List[measurement_.Measurement] = None,
                 merge_redundant: bool = True):
        try:
            super().__init__(measurements=measurements,
                             merge_redundant=merge_redundant)
        except InconsistentMeasurementsError:
            raise InconsistentMeasurementsError("Measurements are not consistent." + self.consistency_str)

        self._experiments = None
        self._update_experiments()

        self._varying_conditions = self._determine_varying_conditions()

    def condition_to_dataset(self, condition_name: str) -> None:
        """Convert a condition to a zero-dimensional dataset.

        Parameters
        ----------
        condition_name : str
            The name of the condition to convert to a zero-dimensional dataset.

        Raises
        ------
        ValueError
            If the condition name is not a varying, shared condition of this experiment set.
        """
        if condition_name not in self.varying_shared_condition_names:
            raise ValueError(f"Condition with name {condition_name} not a shared varying condition.")

        for experiment in self.experiments:
            for replicate in experiment.repetitions:
                dataset = zero_dimensional_dataset.ZeroDimensionalDataset.create(
                     value=replicate.conditions[condition_name],
                     label=condition_name)
                replicate.datasets.append(dataset)

        self.remove_condition(condition_name)
        self._update_experiments()

    def dataset_to_condition(self, dataset_index: int) -> None:
        """Convert the dataset at the given index to a condition.

        Parameters
        ----------
        dataset_index : int
            The index of the dataset to convert to a condition.

        Raises
        ------
        ValueError
            If the dataset at the given index is not zero-dimensional.
        """
        if self.dataset_dimensionalities[dataset_index] != 0:
            raise ValueError("Only zero-dimensional datasets can be converted to conditions.")

        for measurement in self.measurements:
            dataset_name = f"{measurement.datasets[dataset_index].name}" \
                           f"({measurement.datasets[dataset_index].unit})"
            dataset_value = measurement.datasets[dataset_index].value

            measurement.update_condition(dataset_name, dataset_value)

        for measurement in self.measurements:
            measurement.remove_dataset_at_index(dataset_index)

        self._update_experiments()

    def promote_condition(self, condition_name: str) -> None:
        """Convert a condition to an independent variable.

        Parameters
        ----------
        condition_name : str
            The name of the condition to convert to an independent variable.
        """
        self.condition_to_independent_variable(condition_name)

    def combine_zerod_datasets(self, x_name: str, y_name: str, merge_datasets: bool = True) -> None:
        """Perform combination of 0D datasets on a per-experiment basis.

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
        for experiment in self.experiments:
            experiment.combine_zerod_datasets(x_name=x_name,
                                              y_name=y_name,
                                              merge_datasets=merge_datasets)

    def combine_multi_zerod_datasets(self,
                                     x_name: str,
                                     y_names: List[str] = None,
                                     merge_datasets: bool = True) -> None:
        """Perform combination of 0D datasets on a per-experiment basis.

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
        for experiment in self.experiments:
            experiment.combine_multi_zerod_datasets(x_name=x_name,
                                                    y_names=y_names,
                                                    merge_datasets=merge_datasets)

    def combine_oned_datasets(self, x_name, y_name):
        """Combine multiple one-dimensional datasets.

        This procedure is done at the measurement level so this method
        simply iterates over experiments.

        Parameters
        ----------
        x_name : str
            The name of the independent variable.
        y_name : str
            The name of the dependent variable.
        """
        for experiment in self.experiments:
            experiment.combine_oned_datasets(x_name=x_name, y_name=y_name)

    def condition_to_independent_variable(self, condition_name: str) -> None:
        """Convert a condition to an independent variable.

        Parameters
        ----------
        condition_name : str
            The name of the condition to convert to an independent variable.
        """
        indices_of_dataset_to_expand = list(range(self.num_datasets))

        self.condition_to_dataset(condition_name)
        index_of_dataset_to_convert_to_independent = self.num_datasets - 1  # added to the end

        new_measurements = []
        for experiment in self.experiments:

            new_independent_variable_data = []

            dataset_compiled_dependent_variables: List[np.typing.ndarray] = [[] for _ in indices_of_dataset_to_expand]
            for replicate in experiment:

                new_independent_variable_data.append(
                    replicate.datasets[index_of_dataset_to_convert_to_independent].value)

                for index_of_dataset_to_expand in indices_of_dataset_to_expand:
                    dataset_compiled_dependent_variables[index_of_dataset_to_expand].append(
                        np.squeeze(replicate.datasets[index_of_dataset_to_expand].dependent_variable_data))

            new_datasets = []
            for index_of_dataset_to_expand in indices_of_dataset_to_expand:

                all_independent_variable_data = [new_independent_variable_data]
                for ex in experiment[0].datasets[index_of_dataset_to_expand].independent_variable_data:
                    all_independent_variable_data.append(ex)

                new_datasets.append(dataset_factory.DatasetFactory.from_split_data(
                                    dependent_variable_data=dataset_compiled_dependent_variables[index_of_dataset_to_expand],
                                    dependent_variable_names=experiment[0].datasets[index_of_dataset_to_expand].dependent_variable_names,
                                    dependent_variable_units=experiment[0].datasets[index_of_dataset_to_expand].dependent_variable_units,
                                    independent_variable_data=all_independent_variable_data,
                                    independent_variable_names=[condition_name],
                                    independent_variable_units=None,
                                    source=None))

            new_measurements.append(measurement_.Measurement(datasets=new_datasets,
                                                             conditions=experiment.conditions))

        self.measurements = new_measurements
        self._update_experiments()

    def independent_variable_to_condition(self, independent_variable_name: str) -> measurement_set_.MeasurementSet:
        """Convert the independent variable with the specified name to a condition.

        The measurements of this experiment set have the same number and type of datasets with the
        same variable names and units, however not every dataset will have the independent variable
        of interest. This is by virtue of being a consistent measurement set.
        The process of converting the independent variable with the specified name to a
        condition therefore starts with splitting this experiment set in two experiment sets, one with
        and one without datasets that have the specified independent variable name. This process is actually
        a split of each measurement into two measurements, one with the correct datasets and one without.
        The latter can then be ignored safely until the manipulation of the former is complete.
        This implementation does not yet deal with re-incorporating the latter.

        The experiment set containing measurements that have datasets with the specified independent variable name is
        then to be changed. This set is still guaranteed consistent, but there are no restrictions on the
        values of the independent variable in each dataset. Per measurement of the experiment set, all
        datasets that have the same value of the independent variable with the specified name will share
        the same condition value once transformed, and thus a split can be performed on each measurement
        according to the values of the independent variable with the specified name. This split inserts
        the independent variable name and value into the conditions of each new measurement, and then the
        independent variable is to be removed from the datasets.
        Finally all of the measurements must be combined. There is no guarantee that these measurements
        remain consistent because the numbers and types of dataset in each measurement can differ.
        Ultimately this should return an object of the correct type so that users can write
        experiments = experiments.independent_variable_to_condition("independent_variable_name")
        i.e. it should not edit the experiments in place.

        Parameters
        ----------
        independent_variable_name : str
            The name of the independent variable to be converted to a condition.
        """
        experiments_with_independent_variable: ExperimentSet
        experiments_without_independent_variable: ExperimentSet

        experiments_with_independent_variable, \
            experiments_without_independent_variable = \
            self.split_by_dataset_independent_variable_name(independent_variable_name)  # defined in measurement set

        # TODO - must deal with the experiments without the independent variable, by combination

        if not experiments_with_independent_variable:
            raise ValueError(f"Cannot convert independent variable \"{independent_variable_name}\" to a condition."
                             "No measurements of this experiment set have datasets with the named variable.")
        # TODO - in this case, could just return self because technically we did the work

        # this part only works on measurements with datasets that do have the independent variable name
        new_measurements = []
        for measurement in experiments_with_independent_variable.measurements:

            measurements_with_unique_values = \
                measurement.split_by_dataset_independent_variable_value(independent_variable_name)

            for measurement_with_unique_values in measurements_with_unique_values:
                new_datasets = []
                for dataset in measurement_with_unique_values.datasets:
                    new_datasets.append(dataset.remove_independent_variable_by_name(independent_variable_name))

                new_measurement_with_unique_values = \
                    measurement_.Measurement(datasets=new_datasets,
                                             conditions=measurement_with_unique_values.conditions,
                                             details=measurement_with_unique_values.details)

                new_measurements.append(new_measurement_with_unique_values)

        new_measurement_set = measurement_set_factory_.DataCollectionFactory.from_measurements(new_measurements)

        if isinstance(new_measurement_set, ExperimentSet):
            self.measurements = new_measurements
            self._update_experiments()
        else:
            # TODO - return it anyway?
            raise ValueError("Converting independent variable to condition has removed consistency.")

    def independent_variables_to_conditions(self, independent_variable_names: List[str]):
        """Convert a list of named independent variables to conditions.

        This is an operation that actually happens on measurements, and the datasets within them, therefore it can
        be achieved by iteration over the measurements in this set.

        For this to result in changes to a measurement, at least one dataset of the measurement must have the
        specified independent variable names.
        This is a consistent measurement set, so all measurements will be affected in the same way.
        """
        experiments_with_independent_variables, \
            experiments_without_independent_variables = \
            self.split_by_dataset_independent_variable_names(independent_variable_names)

        if not experiments_with_independent_variables:
            raise ValueError(f"No measurements with independent variable {independent_variable_names}")

        # now we just have to work on those datasets that do have the right independent variable names
        # in this case z = f(x, y) whatever z is
        new_measurements = []
        for measurement in experiments_with_independent_variables.measurements:
            measurements_with_unique_values = \
                measurement.split_by_dataset_independent_variable_values(independent_variable_names)

            for measurement_with_unique_values in measurements_with_unique_values:

                new_datasets = []
                for dataset in measurement_with_unique_values.datasets:
                    new_datasets.append(dataset.remove_independent_variables_by_name(independent_variable_names))

                new_measurement_with_unique_values = \
                    measurement_.Measurement(datasets=new_datasets,
                                             conditions=measurement_with_unique_values.conditions,
                                             details=measurement_with_unique_values.details)

                new_measurements.append(new_measurement_with_unique_values)

        new_measurement_set = measurement_set_factory_.DataCollectionFactory.from_measurements(new_measurements)

        if isinstance(new_measurement_set, ExperimentSet):
            self.measurements = new_measurements
            self._update_experiments()
        else:
            # TODO - return it anyway?
            raise ValueError("Converting independent variable to condition has removed consistency.")

    def combine_with(self, experiment_set: "ExperimentSet", merge_redundant: bool):
        return self.combine(measurement_sets=[self, experiment_set],
                            merge_redundant=merge_redundant)

    def remove_measurements_with_condition_name(self, name: str) -> None:
        super().remove_measurements_with_condition_name(name)
        self._update_experiments()

    def remove_measurements_without_condition_name(self, name: str) -> None:
        super().remove_measurements_without_condition_name(name)
        self._update_experiments()

    def remove_measurements_with_condition(self, name: str, value: object) -> None:
        super().remove_measurements_with_condition(name, value)
        self._update_experiments()

    def remove_measurements_without_condition(self, name: str, value: object) -> None:
        super().remove_measurements_without_condition(name, value)
        self._update_experiments()

    def remove_measurements_by_condition_test(self, name, test) -> None:
        super().remove_measurements_by_condition_test(name, test)
        self._update_experiments()

    def remove_measurements_by_conditions_test(self, names, test) -> None:
        super().remove_measurements_by_conditions_test(names, test)
        self._update_experiments()

    def remove_replicate_from_experiment(self, experiment_index, replicate_index):
        del self.experiments[experiment_index].measurements[replicate_index]

    def condition_to_detail(self, condition_name: str) -> None:
        super().condition_to_detail(condition_name)
        self._update_experiments()

    def conditions_to_details(self, condition_names: List[str]) -> None:
        for name in condition_names:
            super().condition_to_detail(name)

        self._update_experiments()

    def detail_to_condition(self, detail_name: str) -> None:
        """Convert a detail to a condition and update experiments.
        Parameters
        ----------
        detail_name : str
            The name of the detail to convert to a condition.
        """
        super().detail_to_condition(detail_name)
        self._update_experiments()

    def details_to_conditions(self, detail_names: List[str]) -> None:
        """Convert a list of details to conditions and update experiments.
        Parameters
        ----------
        detail_names : list of str
            The names of the details to convert to conditions.
        """
        for name in detail_names:
            super().detail_to_condition(name)

        self._update_experiments()

    def add_equal_shared_condition(self,
                                   name: str,
                                   value: object = None) -> None:
        """Add a condition with an optional value to all experiments.
        Parameters
        ----------
        name : str
            The name of the condition.
        value : object
            The value to set for all experiments.
            Default is None as it is assumed values will be reset.
        """
        super().add_equal_shared_condition(name, value)
        self._update_experiments()

    def add_equal_shared_conditions(self,
                                    names: List[str],
                                    values: List[object]) -> None:
        """Add a new equal shared condition to this experiment set."""
        for name, value in zip(names, values):
            super().add_equal_shared_condition(name, value)
        self._update_experiments()

    def add_varying_shared_condition(self, name: str, values: List[object]) -> None:
        super().add_varying_shared_condition(name, values)
        self._update_experiments()

    def add_varying_shared_conditions(self,
                                      names: List[str],
                                      values: List[List[object]]) -> None:
        for name, value in zip(names, values):
            super().add_varying_shared_condition(name, value)
        self._update_experiments()

    def remove_condition(self, name: str) -> None:
        """Remove a condition from all experiments.
        Parameters
        ----------
        name : str
            The name of the condition to remove from all experiments.
        """
        super().remove_condition(name)
        self._update_experiments()

    def remove_conditions(self, names: List[str]) -> None:
        super().remove_conditions(names)
        self._update_experiments()

    @staticmethod
    def from_experiment_sets(experiment_sets: List["ExperimentSet"], merge_redundant: bool = True) -> "ExperimentSet":
        """Create an experiment set from a collection of experiment sets."""
        measurements = []
        for experiment_set in experiment_sets:
            measurements.extend(experiment_set.measurements)

        return ExperimentSet(measurements=measurements, merge_redundant=merge_redundant)

    @staticmethod
    def from_measurement_set(measurement_set: "measurement_set_.MeasurementSet",
                             merge_redundant: bool = True) -> Union["ExperimentSet", "measurement_set_.MeasurementSet"]:
        """Create an experiment set from a given measurement set.

        Parameters
        ----------
        measurement_set : MeasurementSet
            The set of measurements from which to create an experiment set.
        merge_redundant : bool
            Whether to merge measurements that are redundant based on conditions.

        Returns
        -------
        ExperimentSet
            The set of experiments created from the set of measurements.
        """
        try:
            return ExperimentSet(measurement_set.measurements, merge_redundant=merge_redundant)
        except InconsistentMeasurementsError:
            return measurement_set_.MeasurementSet(measurement_set.measurements,
                                                   merge_redundant=merge_redundant)

    def to_measurement_set(self,
                           expand_replicates: bool = True,
                           merge_redundant: bool = True) -> "measurement_set_.MeasurementSet":
        """Convert this set of experiments to a set of measurements.

        Parameters
        ----------
        expand_replicates : bool
            Whether to include all replicates or their means.
            Default is to include all data.
        merge_redundant : bool
            Whether to merge measurements that are redundant based on conditions.

        Returns
        -------
        MeasurementSet
            The set of measurements corresponding to this set of experiments.
        """
        if expand_replicates:
            return measurement_set_.MeasurementSet(self.measurements,
                                                   merge_redundant=merge_redundant)
        else:
            measurements = []
            for experiment in self.experiments:
                measurements.append(experiment.mean)

            return measurement_set_.MeasurementSet(measurements,
                                                   merge_redundant=merge_redundant)

    def flatten(self,
                force_tidiness: bool = False,
                include_unshared_conditions: bool = False,
                include_equal_conditions: bool = True,
                default_value: object = None,
                expand_replicates: bool = True) -> Tuple[List[str], List[List[object]]]:

        if expand_replicates:

            return super().flatten(force_tidiness=force_tidiness,
                                   include_unshared_conditions=include_unshared_conditions,
                                   include_equal_conditions=include_equal_conditions,
                                   default_value=default_value)

        else:
            average_measurements = [experiment.average_measurement for experiment in self.experiments]
            average_experiment_set = ExperimentSet(measurements=average_measurements)
            return average_experiment_set.flatten(force_tidiness=True,
                                                  include_unshared_conditions=include_unshared_conditions,
                                                  include_equal_conditions=include_equal_conditions,
                                                  default_value=default_value,
                                                  expand_replicates=True)

    def flatten_datasets(self,
                         force_tidiness=True,
                         expand_replicates=True) -> Tuple[List[numpy.typing.NDArray],
                                                          List[List[object]]]:
        """Flatten the datasets of this experiment set.

        Parameters
        ----------
        force_tidiness : bool
            Whether to force the experiment set to be tidy before flattening.
        expand_replicates : bool
            Whether to include all replicates or only the average measurement.
        """
        if expand_replicates:
            return super().flatten_datasets(force_tidiness=force_tidiness)
        else:
            average_measurements = [experiment.average_measurement for experiment in self.experiments]
            average_experiment_set = ExperimentSet(measurements=average_measurements)
            return average_experiment_set.flatten_datasets(force_tidiness=True)

    @property
    def experiments(self) -> list:
        """The experiments of this experiment set."""
        return self._experiments

    def remove_all_datasets_at_index(self, index: int) -> None:
        """Delete the dataset from the given index from all measurements.
        Parameters
        ----------
        index : int
            The index of the dataset to remove from all measurements.
        """
        for measurement in self.measurements:
            measurement.datasets = measurement.datasets[0:index] + measurement.datasets[index+1:]
        self._update_experiments()

    @property
    def varying_conditions(self) -> Set[str]:
        """The names of the conditions which vary across the experiments.
        Returns
        -------
        set of str
            The names of the conditions which vary across the experiments of this set.
        """
        return self._determine_varying_conditions()

    def _determine_varying_conditions(self) -> Set[str]:
        """Create a set of conditions which can be treated as variables.

        If a set of experiments has a condition in common, the variation of
        the measurement data (or any derived data) can be visualized as a
        function of that condition. Experimental conditions that can be treated
        as variables are those that appear in the conditions of every
        experiment in a given project, and have at least 2 values across
        the experiment set.

        Returns
        -------
        set of str
            The keys into the condition metadata defined for all experiments.
        """
        experiment_conditions = [experiment.conditions for experiment in self.experiments]

        keys: Set = set()
        for i, experiment in enumerate(self.experiments):
            for key, value in experiment_conditions[i].items():
                condition_in_all_experiments = True
                for j, other_experiment in enumerate(self.experiments):
                    if other_experiment is experiment:
                        continue
                    if key not in experiment_conditions[j]:
                        condition_in_all_experiments = False

                if condition_in_all_experiments is True:
                    keys.add(key)

        constant_keys = []
        for key in keys:
            constant_key = True
            first_value = self.experiments[0].conditions[key]
            for experiment in self.experiments:
                if experiment.conditions[key] != first_value:
                    constant_key = False
                    break

            if constant_key:
                constant_keys.append(key)

        for key in constant_keys:
            keys.remove(key)

        return keys

    def get_varying_condition_values(self, keys: List[str] = None):
        """The values of specified conditions which vary across the experiments of this set.

        Parameters
        ----------
        keys : list of str
            The names of the varying conditions to include.
            By default all are included.

        Returns
        -------
        conditions_as_variables : list of list
            The values of the specified varying conditions for each experiment.
        """
        if keys is None:
            keys = self.varying_conditions
        else:
            for key in keys:
                if key not in self.varying_conditions:
                    raise KeyError("Name of varying condition not recognised: ", key)

        new_experiment_set = copy.deepcopy(self)  # copy this experiment set
        new_experiment_set.condition_to_detail(keys[0])  # remove the key of interest

        key_values = [experiment.conditions[key] for experiment in self.experiments for key in keys]

        # this feels like it shouldn't be here
        conditions_as_variables = []
        key_index = 0
        for experiment in new_experiment_set:

            experiment_independent_variable_data = []
            for measurement_index in range(experiment.num_measurements):
                experiment_independent_variable_data.append(key_values[key_index])
                key_index += 1

            conditions_as_variables.append(experiment_independent_variable_data)

        return conditions_as_variables

    def get_varying_datasets(self,
                             condition_name: str,
                             dataset_indices: List[int] = None,
                             expand_replicates: bool = False):
        """Return the specified datasets across the experiments.

        Parameters
        ----------
        condition_name : str
            Name of the condition being promoted.
        dataset_indices : list of int
            Indices of datasets to include.
            By default all are used.
        expand_replicates : bool
            Whether to include all replicates separately.
            Default is to use the mean measurement.
        """
        new_experiment_set = copy.deepcopy(self)
        new_experiment_set.condition_to_detail(condition_name)

        if dataset_indices is None:
            dataset_indices = range(self.num_datasets)

        # first flatten out the data of interest
        data_to_convert = []
        for index in dataset_indices:
            experiment_data = []
            for experiment in self.experiments:
                experiment_data.append(experiment.mean.datasets[index].dependent_variable_data)

            data_to_convert.append(experiment_data)

        new_dependent_data = []
        start_index = 0
        for experiment in new_experiment_set:

            experiment_data = []
            for index in dataset_indices:
                dataset_data = data_to_convert[index][start_index: start_index + len(experiment)]
                experiment_data.append(dataset_data)

            start_index += len(experiment)

            new_dependent_data.append(experiment_data)

        return new_dependent_data

    def _update_experiments(self) -> None:
        """Update the set of experiments based on the condition metadata."""
        # which conditions are present in all measurements?
        shared_condition_names = \
            self.shared_condition_names

        # cache these for performance reasons
        measurement_conditions = [measurement.conditions for measurement in self.measurements]

        # what are the unique combinations of values for these conditions?
        unique_combinations = self._determine_unique_condition_combinations()

        # for each unique combination, must create an Experiment
        experiment_lists = []
        for unique_combination in unique_combinations:

            experimental_measurements = []

            # create a list of measurements with the given conditions
            for i, measurement in enumerate(self.measurements):

                shared_conditions = {}

                for condition_name in measurement.condition_names:
                    if condition_name in shared_condition_names:
                        shared_conditions[condition_name] = \
                            measurement_conditions[i][condition_name]

                if shared_conditions == unique_combination:
                    experimental_measurements.append(measurement)

            experiment_lists.append(experimental_measurements)

        experiments = []
        for experiment in experiment_lists:
            experiments.append(experiment_.Experiment(experiment, merge_redundant=False))

        self._experiments = experiments

        self.measurements = []
        for experiment in self.experiments:
            self.measurements.extend(experiment.measurements)

        self._varying_conditions = self._determine_varying_conditions()

    def sort_by_condition(self, condition_name: str) -> None:
        """Sort the list of experiments by the values of a specified condition."""
        def get_key(experiment):
            return experiment.conditions[condition_name]

        self.experiments.sort(key=get_key)

    def summary_table(self,
                      keys: Set[str] = None,
                      expand_replicates: bool = True) -> str:
        """Create a table giving details of all experimental conditions.

        Parameters
        ----------
        keys : set
            List of keys to include in table. Default (None) uses all keys.
        expand_replicates : bool
            Whether to include all replicates in the summary.

        Returns
        -------
        output : str
        """
        if keys is None:
            if self.num_experiments == 1:
                keys = self.all_condition_names
            else:
                keys = self._varying_conditions

        output = "index"

        if not expand_replicates:
            output += "\t" + "n_rep"

        for key in keys:
            output += "\t" + key
        output += "\t"

        for _ in self.experiments[0].datasets[0]:  # if there are a lot of datasets, there are lots of columns that need to be summarized
            output += "Data Types" + "\t"

        output += "\n"

        if self.num_experiments < 500:  # this will be a cutoff based on rows. If there are loads, need a more summarized version

            if expand_replicates:  # one repetition per line
                for i, experiment in enumerate(self.experiments):
                    for replicate in experiment.datasets:

                        output += str(i) + "\t"
                        for key in keys:
                            output += str(experiment.conditions[key]) + "\t"

                        for dataset in replicate:
                            output += dataset.one_line_description + "\t"
                        output += "\n"

            else:
                for i, experiment in enumerate(self.experiments):
                    output += str(i) + "\t" + str(len(experiment)) + "\t"
                    for key in keys:
                        output += str(experiment.conditions[key]) + "\t"

                    for dataset in experiment.datasets[0]:
                        output += dataset.one_line_description + "\t"
                    output += "\n"

        else:

            output += str(0) + "\t" + str(len(self.experiments[0])) + "\t"
            for key in keys:
                output += str(self.experiments[0].conditions[key]) + "\t"

            for dataset in self.experiments[0].datasets[0]:
                output += dataset.one_line_description + "\t"
            output += "\n"

        return output

    @property
    def num_experiments(self) -> int:
        return len(self.experiments)

    def produce_color_map(self):
        """Create a color map to differentiate between experiments."""
        return iter(cm.get_cmap("cool")(np.linspace(0, 1, len(self.experiments))))

    def condition_labels(self, include_name: bool = True) -> List[str]:
        return [experiment.condition_label(include_name) for experiment in self.experiments]

    def produce_legend_labels(self) -> List[str]:
        """Create labels to be inserted into the figure legend.

        Returns
        -------
        list of str
            A label for each experiment for inclusion in the legend.
        """
        legend_labels = []
        for experiment in self.experiments:

            legend_label = ""
            for condition_name in self.varying_conditions:
                legend_label += condition_name + "=" + str(experiment.conditions[condition_name]) + ", "

            legend_labels.append(legend_label[:-2])

        return legend_labels

    def visualize(self,
                  include_text: bool = True,
                  expand_datasets: bool = True,
                  expand_replicates: bool = False,
                  expand_experiments: bool = False,
                  figure_title: str = None,
                  total_figsize: Tuple[int] = None,
                  **plot_kwargs):
        """Visualize the project's experiments.

        Each experiment needs to be displayed along with its relevant
        metadata. By default, the human-readable string representation of
        the experiments is provided first, followed by dataset
        visualizations created using matplotlib.

        Parameters
        ----------
        include_text : bool
            Whether to display the str representation of the experiment.
            Default True, which is correct if this method is called directly.
        expand_datasets: bool
            Whether to place each dataset on its own axes.
            By default each dataset will have its own axes.
        expand_replicates : bool
            Whether to visualize all repetitions or a statistical summary.
            By default only show the summarized information.
        expand_experiments : bool
            Whether to place each experiment on its own axes.
            Default is to use a single plot.
        figure_title : str
            A title for the visualization, overriding the metadata title.
        total_figsize : tuple
            A tuple of 2 numbers setting the figure size.
        plot_kwargs : dict
            Keyword arguments for matplotlib.axes.Axes.plot function.

        Returns
        -------
        fig : matplotlib.figure.Figure
            A matplotlib figure containing the axes.
        axes : matplotlib.axes.Axes
            Matplotlib axes containing the plotted experiment set.
        """
        if include_text:
            print(str(self))

        for dataset_list in self.datasets:
            for dataset in dataset_list:
                if isinstance(dataset, two_dimensional_dataset.TwoDimensionalDataset):

                    expand_experiments = True
                    expand_replicates = True
                    expand_datasets = True

        legend_labels = self.produce_legend_labels()  # one label per-experiment

        colormap = None
        if expand_experiments:
            colormap = self.produce_color_map()

        fig, axes = self._setup_fig_and_axes(axes=None,
                                             expand_experiments=expand_experiments,
                                             expand_datasets=expand_datasets,
                                             total_figsize=total_figsize,
                                             figure_title=figure_title)

        total_index = 0
        for i, experiment in enumerate(self.experiments):

            if colormap:
                color = next(colormap)
            else:
                color = None

            if not expand_experiments:  # one row
                experiment_axes = axes
            elif expand_experiments:
                experiment_axes = axes[i]

            experiment.visualize(axes=experiment_axes,
                                 include_text=False,
                                 expand_datasets=expand_datasets,
                                 expand_replicates=expand_replicates,
                                 color=color,
                                 **plot_kwargs)

            # if not expand_experiments:
            #     if self.num_datasets == 1:
            #         axes.legend(legend_labels,
            #                     fontsize=13,
            #                     edgecolor="black",
            #                     bbox_to_anchor=(0.0, 1.02, 1.0, 0.102),
            #                     loc='lower left',
            #                     ncol=2,
            #                     borderaxespad=0.0)
            #     else:
            #         fig.legend(legend_labels,
            #                    fontsize=13,
            #                    edgecolor="black",
            #                    bbox_to_anchor=(0.0, 1.02, 1.0, 0.102),
            #                    loc='lower left',
            #                    ncol=2,
            #                    borderaxespad=0.0)

        return fig, axes

    def compute_figure_size(self,
                            expand_datasets: bool = False,
                            expand_experiments: bool = False) -> Tuple[int]:
        """Compute an appropriate size for the complete figure.

        If a single plot has been requested, then the figure will
        be the same size as a single measurement from the set, so
        the largest measurement will be used.
        If multiple plots have been requested, the size will be a
        sum over the sizes of the set of measurements.

        Parameters
        ----------
        expand_datasets : bool
            Whether to plot each dataset of each experiment on its own plot.
        expand_experiments : bool
            Whether to plot each experiment of this experiment set on its own plot.

        Returns
        -------
        tuple
            The size of the complete figure.
        """
        figsize = [0, 0]

        if not expand_experiments:

            for experiment in self.experiments:
                experiment_size = experiment.compute_figure_size(expand_replicates=True,
                                                                 expand_datasets=expand_datasets)
                if experiment_size[0] > figsize[0]:
                    figsize[0] = experiment_size[0]
                if experiment_size[1] > figsize[1]:
                    figsize[1] = experiment_size[1]

        else:  # is not on a single plot

            for experiment in self.experiments:
                experiment_size = experiment.compute_figure_size(expand_replicates=True,
                                                                 expand_datasets=expand_datasets)

                if experiment_size[0] > figsize[0]:
                    figsize[0] = experiment_size[0]

                figsize[1] += experiment_size[1]

        return tuple(figsize)

    def _validate_figsize(self,
                          figsize: Tuple[int],
                          expand_experiments: bool = False,
                          expand_datasets: bool = False) -> Tuple[int, int]:
        """Assess and potentially create a title for the complete figure."""
        if figsize is None:
            figsize = self.compute_figure_size(expand_experiments=expand_experiments,
                                               expand_datasets=expand_datasets)

        return figsize

    def _setup_fig_and_axes(self,
                            axes: matplotlib.axes.Axes = None,
                            expand_datasets: bool = False,
                            expand_experiments: bool = False,
                            total_figsize: Tuple[int] = None,
                            figure_title: str = None) -> Tuple[matplotlib.figure.Figure,
                                                               Union[matplotlib.axes.Axes,
                                                                     List[matplotlib.axes.Axes]]]:
        """Prepare a figure and axes for plotting this measurement set.

        Parameters
        ----------
        expand_datasets : bool
            Whether to plot all repetitions or a representation of each experiment.
        expand_experiments : bool
            Whether to plot all experiments on a single axes.
        axes : matplotlib.axes.Axes
            The matplotlib axes on which to plot the experiment set.
        total_figsize : tuple of int
            The size of the complete figure.
        figure_title : str
            The title of the complete figure.

        Returns
        -------
        fig : matplotlib.figure.Figure
        axes : matplotlib.axes.Axes or list of matplotlib.axes.Axes
        """
        if axes is not None:
            raise ValueError("An experiment set must create its own fig and axes.")

        total_figsize = self._validate_figsize(total_figsize,
                                               expand_datasets=expand_datasets,
                                               expand_experiments=expand_experiments)

        if not expand_experiments:

            if not expand_datasets:
                fig, axes = plt.subplots(1,
                                         1,
                                         figsize=total_figsize)
            else:
                fig, axes = plt.subplots(1,
                                         self.num_datasets,
                                         figsize=total_figsize)

        elif expand_experiments:

            if not expand_datasets:
                fig, axes = plt.subplots(self.num_experiments,
                                         1,
                                         figsize=total_figsize)
            elif expand_datasets:

                fig, axes = plt.subplots(self.num_experiments,
                                         self.num_datasets,
                                         figsize=total_figsize)

        fig.suptitle(self._validate_title(figure_title))
        fig.set_tight_layout({"rect": [0, 0.03, 1, 0.9]})
        return fig, axes

    def condition_to_data(self,
                          condition_name: str,
                          new_condition_name: str = None,
                          condition_unit: str = None) -> None:
        """Convert a condition to a scalar dataset.

        Parameters
        ----------
        condition_name : str
            The name of the condition to convert to a scalar dataset.
        new_condition_name : str
            A replacement for the original condition name.
        condition_unit : str
            A unit for the condition name.
        """
        if not new_condition_name:
            new_condition_name = condition_name

        for measurement in self.measurements:
            measurement.datasets.append(
                zero_dimensional_dataset.ZeroDimensionalDataset.create(label=new_condition_name,
                                                                       unit=condition_unit,
                                                                       value=measurement.conditions[condition_name]))

        self.remove_condition(condition_name)
        self._update_experiments()

    def compress_conditions(self,
                            condition_names: List[str],
                            separator: str = ",",
                            merge_redundant: bool = True) -> "ExperimentSet":
        """Convert the conditions of each measurement to a single condition.

        Parameters
        ----------
        condition_names : list of str
            The conditions to include.
        separator : str
            The separator to place between each condition.
        merge_redundant : bool
            Whether to merge measurements that are redundant based on conditions.
        """
        measurements = []
        for measurement in self.measurements:
            label = ""
            for condition_name in condition_names:
                if condition_name in self.varying_conditions:
                    label += condition_name + "=" + str(measurement.conditions[condition_name]) + separator + " "

            label = label[:-2]

            measurements.append(piblin.data.data_collections.measurement.Measurement(measurement.datasets, conditions={"label": label}))

        return ExperimentSet.from_measurement_set(measurement_set_.MeasurementSet(measurements),
                                                  merge_redundant=merge_redundant)

    def __len__(self):
        """The number of experiments in this set.

        Returns
        -------
        int
            The number of experiments in this set.
        """
        return len(self.experiments)

    def __getitem__(self, position: Union[int, slice]):
        """The experiment(s) at the specified position in the set.

        Parameters
        ----------
        position : int or slice
            The position(s) of the experiments to return.

        Returns
        -------
        Experiment or ExperimentSet
            If position is an int, returns the appropriate experiment.
            If position is a slice, returns an experiment set of appropriate experiments.
        """
        if isinstance(position, slice):
            measurements = []
            for experiment in self.experiments[position]:
                measurements.extend(experiment.to_consistent_measurement_set().measurements)
            return ExperimentSet(measurements)

        return self.experiments[position]

    def __delitem__(self, position):
        """Delete the experiment at the specified position.

        Parameters
        ----------
        position : int or slice
            The position at which to delete experiment(s).
        """
        del self.experiments[position]
        self._update_experiments()

    def __setitem__(self, position, value):
        """Change the experiment at the given index to value.

        This method will not necessarily behave intuitively.
        The set of experiments is computed from the set of measurements,
        and as such cannot be edited directly. The method is guaranteed to
        add the measurements of the experiments to the experiment set, but
        is then forced to update its experiments such that the requested
        position may not be respected.

        Parameters
        ----------
        position : int or slice
            The position at which to change the experiments.
        value : Experiment or list of Experiment
            The experiments to place at the given position.
        """
        measurements = [experiment.to_measurement_set().measurements for experiment in self.experiments[position]]
        for measurement in measurements:
            self.measurements.remove(measurement)

        if isinstance(position, slice):
            measurements = [experiment.to_measurement_set().measurements for experiment in value]
            self.measurements.append(measurements)
        else:
            self.measurements.append(value.to_measurement_set().measurements)

        self._update_experiments()

    def insert(self, index: int, value) -> None:
        """Insert the given experiment(s) at the given position.

        See the comment for __setitem__ for caveats to this method. It
        will not necessarily behave intuitively but is guaranteed to add
        the measurements of the experiments to the experiment set.
        """
        self.measurements.extend(value.to_measurement_set().measurements)
        self._update_experiments()

    def __str__(self):
        """Return human-readable representation of this project.

        The direct printing of all experiments is not particularly useful for
        the user so the information has to be reduced.
        """
        classname = "Experiment Set"
        str_rep = self.__class__.__name__ + "\n" + ((len(classname) - 1) * "-") + "\n"

        str_rep += "\nSummary\n-------\n"
        str_rep += f"Number of Experiments: {self.num_experiments}\n"
        if self.num_experiments < 500 and self.num_datasets < 20:

            str_rep += "\nVaried Conditions\n-----------------\n"
            for name in self.varying_conditions:
                str_rep += str(name) + ", "
            str_rep = str_rep[:-2] + "\n\n"

            str_rep += self.summary_table(expand_replicates=False) + "\n"
        else:  # too many experiments or datasets for the table to look nice

            previous_dataset_type = self.dataset_types[0]
            dataset_type_str = ""
            dataset_type_count = 0
            for dataset_type in self.dataset_types:
                if dataset_type != previous_dataset_type:
                    dataset_type_str += f"{dataset_type.__name__} ({dataset_type_count}), "
                    dataset_type_count = 0
                else:
                    dataset_type_count += 1

            dataset_type_str += f"{self.dataset_types[-1].__name__} ({dataset_type_count})"

            str_rep += "\nExperiment Dataset Types: " + dataset_type_str + "\n"

            # dataset_names = [dataset.one_line_description for dataset in self.measurements[0].datasets]
            # unique_dataset_names = set(dataset_names)
            # # str_rep += f"\nNumber of Unique Dataset Names: {len(unique_dataset_names)}\n\n"
            # for unique_dataset_name in unique_dataset_names:
            #     str_rep += f"{unique_dataset_name}:\t(count = {dataset_names.count(unique_dataset_name)})\n"

            if len(self.varying_shared_condition_names) > 0:
                str_rep += "\nVaried Conditions\n-----------------\n\n"

                for condition_name, condition_value in self.varying_shared_conditions.items():
                    unique_condition_values = set(sorted([experiment.conditions[condition_name]
                                                         for experiment in self.experiments]))

                    if all([type(condition_value) == str for condition_value in unique_condition_values]):
                        str_rep += f"{condition_name}: ["
                        for condition_value in unique_condition_values:
                            str_rep += f"{condition_value}, "
                        str_rep = str_rep[:-2] + "]\n"
                    else:
                        str_rep += f"{condition_name}:\trange from [{min(condition_value)}, {max(condition_value)}]\n"
            else:
                str_rep += "\nNo Varying Conditions\n"

        return str_rep

    def __repr__(self):
        """Return eval()-able representation of this project."""
        str_rep = self.__class__.__name__ + "("

        str_rep += "experiments=["
        for experiment in self.experiments:
            str_rep += repr(experiment) + ", "
        return str_rep[:-2] + ")"

    def __eq__(self, other: "ExperimentSet"):
        """Return True is self is equal to other."""
        if self is other:
            return True

        if len(self) != len(other):
            return False

        for experiment in self.experiments:
            if experiment not in other.experiments:
                return False

        return True

    def __cmp__(self):
        """Comparison based on magnitudes is purposely not defined."""
        return NotImplemented
