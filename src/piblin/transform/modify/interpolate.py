"""
In cases where a set of measurements (or experiments) is used to
determine the parameters of a transform, that transform can be
created by analysis of the experiment set and then applied to
the experiment set as if it were a one-in-one-out transform.
"""
from typing import List, Tuple, Union
import math
import copy
import numpy as np
import piblin.transform.abc.dataset_transform as dataset_transform
import piblin.data.datasets.abc.dataset as dataset
import piblin.data.data_collections.measurement as measurement
import piblin.data.data_collections.measurement_set as measurement_set
import piblin.data.data_collections.experiment as experiment
import piblin.data.data_collections.experiment_set as experiment_set
import piblin.data.datasets.abc.split_datasets.one_dimensional_dataset as one_dimensional_dataset


class Interpolate1D(dataset_transform.DatasetTransform):
    """Equally-spaced interpolation to be applied to an arbitrary dataset.

    The initializer method can take fixed values for the minimum value,
    maximum value and spacing. If these are provided, there should be no
    reason to recompute them from data collections and so this transform
    will use user-provided parameters in all cases.
    Any parameters provided to the initializer will be frozen.

    Parameters
    ----------
    min_value : float
        The minimum value of the new set of independent variables.
        Default of None results in computation from a data collection.
    max_value : float
        The maximum value of the new set of independent variables.
        Default of None results in computation from a data collection.
    spacing : float
        The difference between consecutive pairs of independent variables.
        Default of None results in computation from a data collection.
    snap_to_ints : bool
        Whether to force the min and max values to be integers.
        Default is to use input values directly.
    respect_bounds : bool
        Whether to force min and max integer values to be inside the original values.
        Default is to limit the values as described.

    Attributes
    ----------
    min_value -> float
        The minimum value of the new set of independent variables.
    max_value -> float
        The maximum value of the new set of independent variables.
    spacing -> float
        The difference between consecutive pairs of independent variables.
    snap_to_ints -> bool
        Whether to force the min and max values to be integers.
    respect_bounds -> bool
            Whether to force min and max integer values to be inside the original values.

    Methods
    -------
    from_dataset
    from_datasets
    from_measurement_set
    list_from_measurement_set
    from_experiment_set
    """

    def __init__(self,
                 min_value: float = None,
                 max_value: float = None,
                 spacing: float = None,
                 snap_to_ints: bool = False,
                 respect_bounds: bool = True):

        self.snap_to_ints = snap_to_ints
        self.respect_bounds = respect_bounds

        super().__init__(data_independent_parameters=[self.snap_to_ints,
                                                      self.respect_bounds])

        self.min_value = min_value
        self.max_value = max_value
        self.spacing = spacing

        if self.min_value and self.max_value:
            if self.min_value > self.max_value:
                print("Warning: switching min and max value parameters in interpolation.")
                self.min_value, self.max_value = self.max_value, self.min_value

        self.is_frozen = False

        if self.min_value is not None:
            self._min_value_is_frozen = True

        if self.max_value is not None:
            self._max_value_is_frozen = True

        if self.spacing is not None:
            self._spacing_is_frozen = True

        if min_value is not None and max_value is not None and spacing is not None:
            self._independent_variable_values = self.__compute_independent_variable_values()
        else:
            self._independent_variable_values = None

    @property
    def is_frozen(self) -> bool:
        """Determine whether this transform updates based on input data.

        Returns
        -------
        bool
            Whether this transform updates based on input data.
        """
        return self._min_value_is_frozen and self._max_value_is_frozen and self._spacing_is_frozen

    @is_frozen.setter
    def is_frozen(self, is_frozen: bool) -> None:
        if is_frozen:
            self._min_value_is_frozen = True
            self._max_value_is_frozen = True
            self._spacing_is_frozen = True
        else:
            self._min_value_is_frozen = False
            self._max_value_is_frozen = False
            self._spacing_is_frozen = False

    def freeze(self) -> None:
        """Prevent this transform from updating based on input data."""
        if None in {self.min_value, self.max_value, self.spacing}:
            raise ValueError("Cannot freeze transform with undefined parameter")
        else:
            self._min_value_is_frozen = True
            self._max_value_is_frozen = True
            self._spacing_is_frozen = True

    @property
    def min_value(self) -> Union[int, float, None]:
        """The minimum independent variable value."""
        return self._min_value

    @min_value.setter
    def min_value(self, min_value: Union[int, float, None]) -> None:
        """Set the minimum independent variable value."""
        if self.snap_to_ints and min_value is not None:
            self._min_value = self.__min_value_float_to_int(min_value,
                                                            self._respect_bounds)
        else:
            self._min_value = min_value

    @property
    def max_value(self) -> Union[int, float, None]:
        """The maximum independent variable value."""
        return self._max_value

    @max_value.setter
    def max_value(self, max_value: Union[int, float, None]) -> None:
        """Set the maximum independent variable value."""
        if self.snap_to_ints and max_value is not None:
            self._max_value = self.__max_value_float_to_int(max_value,
                                                            self._respect_bounds)
        else:
            self._max_value = max_value

    @property
    def spacing(self) -> Union[int, float, None]:
        """The spacing between independent variable values."""
        return self._spacing

    @spacing.setter
    def spacing(self, spacing: Union[int, float, None]) -> None:
        """Set the spacing between independent variable values."""
        self._spacing = spacing

    @property
    def snap_to_ints(self) -> bool:
        """Whether independent variable extreme values are rounded to integers."""
        return self._snap_to_ints

    @snap_to_ints.setter
    def snap_to_ints(self, snap_to_ints: bool) -> None:
        """Set whether independent variable extreme values are rounded to integers."""
        self._snap_to_ints = snap_to_ints

    @property
    def respect_bounds(self) -> bool:
        return self._respect_bounds

    @respect_bounds.setter
    def respect_bounds(self, respect_bounds: bool) -> None:
        self._respect_bounds = respect_bounds

    def _apply(self, target: one_dimensional_dataset.OneDimensionalDataset, **kwargs):
        """Apply this interpolation to a one-dimensional dataset.

        Parameters
        ----------
        target : cralds.data.datasets.OneDimensionalDataset
            The dataset to apply this interpolation to.

        Returns
        -------
        cralds.data.datasets.OneDimensionalDataset
            The transformed dataset after the interpolation.
        """
        target.update_data(independent_variable_data=self._independent_variable_values,
                           dependent_variable_data=np.interp(self._independent_variable_values,
                                                   target.independent_variable_data[0],
                                                   target.dependent_variable_data))

        return target

    def compute_data_dependent_parameters(self, parameter_source: Union[dataset.Dataset,
                                                                        measurement.Measurement,
                                                                        measurement_set.MeasurementSet,
                                                                        experiment.Experiment,
                                                                        experiment_set.ExperimentSet]) -> List[object]:
        """Compute parameters of this transform from the given data collection.

        Only those parameters which have not been set either in the initializer
        of this class or via its properties are to be updated in this method.
        The setter is not called as these parameters should still be updated in
        future calls to apply_to.

        Parameters
        ----------
        parameter_source : {Dataset, Measurement, MeasurementSet, ExperimentSet}
            The data collection from which to compute the parameters.
        """
        if isinstance(parameter_source, dataset.Dataset):
            datasets = [parameter_source]
        else:
            datasets = parameter_source.datasets  # is this always a flat list of datasets?
            # Measurement = yes
            # MeasurementSet = ?

        if not self._spacing_is_frozen:
            self._spacing = Interpolate1D.__compute_mean_spacing(datasets)

        if self.snap_to_ints:
            independent_variable_range = Interpolate1D.__compute_integer_range(datasets, self.respect_bounds)
        else:
            independent_variable_range = Interpolate1D.__compute_range(datasets)

        if not self._min_value_is_frozen:
            self._min_value = independent_variable_range[0]

        if not self._max_value_is_frozen:
            self._max_value = independent_variable_range[1]

        return [self.min_value, self.max_value, self.spacing]

    @property
    def data_dependent_parameters(self) -> List[object]:
        """Data-dependent parameters for this transform."""
        return [self.min_value, self.max_value, self.spacing]

    @data_dependent_parameters.setter
    def data_dependent_parameters(self, data_dependent_parameters: List[Union[int, float]]) -> None:
        """Set the data-dependent parameters for this transform."""
        self.min_value = data_dependent_parameters[0]
        self.max_value = data_dependent_parameters[1]
        self.spacing = data_dependent_parameters[2]
        self._independent_variable_values = self.__compute_independent_variable_values()

    @classmethod
    def from_dataset(cls,
                     dataset_: dataset.Dataset,
                     spacing: float = None,
                     snap_to_ints: bool = False,
                     respect_bounds: bool = False) -> "Interpolate1D":
        """Create an interpolate transform from a dataset.

        Parameters
        ----------
        dataset_ : Dataset
            The dataset from which to create the transform.
        spacing : float
            The spacing between independent variables for the interpolation.
        snap_to_ints : bool
            Whether to force the min and max values to be integers.
        respect_bounds : bool
            Whether to force min and max value to be inside the original values.

        Returns
        -------
        Interpolate1D
        """
        return cls.from_datasets([dataset_],
                                 spacing,
                                 snap_to_ints,
                                 respect_bounds)

    @classmethod
    def from_datasets(cls,
                      datasets: List[dataset.Dataset],
                      spacing: float = None,
                      snap_to_ints: bool = False,
                      respect_bounds: bool = False) -> "Interpolate1D":
        """Create an interpolate transform from a list of datasets.

        Parameters
        ----------
        datasets : list of Dataset
            The datasets from which to create the transform.
        spacing : float
            The spacing between independent variables for the interpolation.
        snap_to_ints : bool
            Whether to force the min and max values to be integers.
        respect_bounds : bool
            Whether to force min and max value to be inside the original values.

        Returns
        -------
        Interpolate1D
        """

        if spacing is None:
            spacing = Interpolate1D.__compute_mean_spacing(datasets)

        if snap_to_ints:
            x_range = Interpolate1D.__compute_integer_range(datasets, respect_bounds)
        else:
            x_range = Interpolate1D.__compute_range(datasets)

        return cls(x_range[0], x_range[1], spacing)

    @classmethod
    def from_measurement_set_datasets(cls,
                                      measurement_set_,
                                      dataset_index: int,
                                      spacing: float = None,
                                      snap_to_ints: bool = False,
                                      respect_bounds: bool = False) -> "Interpolate1D":
        """Create an interpolation spanning a measurement set.

        Parameters
        ----------
        measurement_set_ : ConsistentMeasurementSet
            The consistent measurement set to interpolate.
        dataset_index : int
            The index of the dataset to interpolate.
        spacing : float
            The spacing between interpolated data points.
        snap_to_ints : bool
        respect_bounds : bool

        Returns
        -------
        Interpolate1D
            The interpolation spanning the measurement set.
        """
        return Interpolate1D.from_datasets(measurement_set_.datasets_at_index(dataset_index),
                                           spacing,
                                           snap_to_ints,
                                           respect_bounds)

    @staticmethod
    def list_from_measurement_set(measurement_set,
                                  spacing: float = None,
                                  snap_to_ints: bool = False,
                                  respect_bounds: bool = False):

        interpolations = []
        for dataset_index in range(len(measurement_set[0].datasets)):
            interpolations.append(Interpolate1D.from_measurement_set_datasets(measurement_set=measurement_set,
                                                                              dataset_index=dataset_index,
                                                                              spacing=spacing,
                                                                              snap_to_ints=snap_to_ints,
                                                                              respect_bounds=respect_bounds))

        return interpolations

    @staticmethod
    def from_experiment_set_datasets(experiment_set,
                                     dataset_index: int,
                                     spacing: float = None,
                                     snap_to_ints: bool = False,
                                     respect_bounds: bool = False):
        """Create an interpolation spanning an experiment set.

        Parameters
        ----------
        experiment_set : ExperimentSet
            The experiment set to interpolate.
        dataset_index : int
            The index of the dataset to interpolate.
        spacing : float
            The spacing between interpolated data points.
        snap_to_ints : bool
        respect_bounds : bool

        Returns
        -------
        Interpolate1D
            The interpolation spanning the experiment set.
        """
        return Interpolate1D.from_measurement_set_datasets(experiment_set.measurement_set_,
                                                           dataset_index,
                                                           spacing,
                                                           snap_to_ints,
                                                           respect_bounds)

    @staticmethod
    def __compute_mean_spacing(datasets: List[one_dimensional_dataset.OneDimensionalDataset]) -> float:
        """Determine the mean spacing across all datasets.

        This is default behaviour for determining the spacing to use in
        generating a new set of independent variables for datasets. It
        determines the mean spacing between consecutive data points across
        all the datasets.

        Parameters
        ----------
        datasets : list of OneDimensionalDataset
            The datasets from which to determine the mean spacing.

        Returns
        -------
        float
            The average inter-point difference in independent variable values.
        """
        step = 0.0
        count = 0
        for dataset_ in datasets:
            for i in range(len(dataset_.independent_variable_data[0]) - 1):
                step += dataset_.independent_variable_data[0][i + 1] - dataset_.independent_variable_data[0][i]
                count += 1

        return abs(step) / count

    @staticmethod
    def __compute_integer_range(datasets: List[one_dimensional_dataset.OneDimensionalDataset],
                                respect_bounds=True):
        """Find the minimum and maximum x-value across a number of datasets.

        Parameters
        ----------
        datasets : list of Dataset
            The set of measurements to find extreme independent variable values for.
        respect_bounds : bool
            Whether to force all independent variable values to be in the x-range of the datasets.

        Returns
        -------
        x_range : tuple of float
            The minimum and maximum independent variable values across all spectra,
        """
        min_x = np.inf
        max_x = np.NINF

        for dataset_ in datasets:

            min_x_dataset, max_x_dataset = \
                Interpolate1D.__extreme_values_float_to_int(min_value=np.min(dataset_.independent_variable_data),
                                                            max_value=np.max(dataset_.independent_variable_data),
                                                            respect_bounds=respect_bounds)

            if min_x_dataset < min_x:
                min_x = min_x_dataset
            if max_x_dataset > max_x:
                max_x = max_x_dataset

        return min_x, max_x

    @staticmethod
    def __extreme_values_float_to_int(min_value, max_value, respect_bounds) -> Tuple[int, int]:
        """Convert input extreme values to integers.

        Parameters
        ----------
        min_value : float
            The minimum value of the new set of independent variables.
        max_value : float
            The maximum value of the new set of independent variables.
        respect_bounds : bool
            Whether to force min and max value to be inside the original values.

        Returns
        -------
        Tuple of int
            The integers to use as extreme values of independent variables.
        """
        return (Interpolate1D.__min_value_float_to_int(min_value, respect_bounds),
                Interpolate1D.__max_value_float_to_int(max_value, respect_bounds))

    @staticmethod
    def __min_value_float_to_int(min_value: float, respect_bounds: bool) -> int:
        """Convert a floating point minimum value to an integer.

        Parameters
        ----------
        min_value : float
            The value to convert from floating point to integer.
        respect_bounds : bool
            Whether to ensure the converted value is greater than the input.

        Returns
        -------
        min_value : int
            The input value converted to an integer.
        """
        if respect_bounds:
            min_value = math.ceil(min_value)
        else:
            min_value = math.floor(min_value)

        return min_value

    @staticmethod
    def __max_value_float_to_int(max_value: float, respect_bounds: bool) -> int:
        """Convert a floating point maximum value to an integer.

        Parameters
        ----------
        max_value : float
            The value to convert from floating point to integer.
        respect_bounds : bool
            Whether to ensure the converted value is less than the input.

        Returns
        -------
        max_value : int
            The input value converted to an integer.
        """
        if respect_bounds:
            max_value = math.floor(max_value)
        else:
            max_value = math.ceil(max_value)

        return max_value

    @staticmethod
    def __compute_range(datasets):
        """Find the minimum and maximum independent variable value across a set of datasets.

        Parameters
        ----------
        datasets : list of Dataset
            The datasets to determine the extreme independent variable values of.

        Returns
        -------
        tuple of float
            The extreme values across all datasets.
        """
        min_independent_variable = np.inf
        max_independent_variable = np.NINF

        for dataset_ in datasets:
            if np.min(dataset_.independent_variable_data) < min_independent_variable:
                min_independent_variable = np.min(dataset_.independent_variable_data)
            if np.max(dataset_.independent_variable_data) > max_independent_variable:
                max_independent_variable = np.max(dataset_.independent_variable_data)

        return min_independent_variable, max_independent_variable

    def __compute_independent_variable_values(self):
        """Compute the independent variable values from the extreme values and spacing.

        This transform has three parameters which are ultimately used to
        generate a single array of independent variable values.

        Returns
        -------
        numpy.ndarray
            The independent variable corresponding to the extreme values and spacing.
        """
        n_values = int(abs(self.max_value - self.min_value) / self.spacing) + 1
        return np.linspace(self.min_value, self.max_value, num=n_values)

    def visualize_transformed_data(self, dataset_, axes):
        """Visualize the result of the application of this transform.

        This method overrides the base class implementation in order to
        ensure it is plotted with the interpolated points shown separately.

        Parameters
        ----------
        dataset_ : Dataset
            The dataset to which to apply the transform.
        axes : matplotlib.axes.Axes
            The axes on which to plot the transformed dataset.
        """
        dataset_ = copy.deepcopy(dataset_)

        transformed_data = self.apply_to(dataset_,
                                         make_copy=True)

        transformed_data.visualize(axes=axes,
                                   include_text=False,
                                   markersize=3)

    def visualize_transform_parameters(self, axes):
        """Visualize the parameters of this transform.

        The result of the three parameters of this transform is a new
        set of independent variable values.

        Parameters
        ----------
        axes : matplotlib.axes.Axes
            The axes on which to visualize this transform's parameters.
        """
        for value in self._independent_variable_values:
            axes.axvline(value, color="gray", linewidth=0.2)

    def __str__(self):
        """Create a human-readable representation of this transform."""

        str_rep = "Interpolation Transform\nData-Dependent Parameters:\n"

        str_rep += f"\tThe minimum x-coordinate value is {self.min_value}"
        if self._min_value_is_frozen:
            str_rep += " (frozen)\n"
        else:
            str_rep += " (not frozen)\n"

        str_rep += f"\tThe maximum x-coordinate value is {self.max_value}"
        if self._max_value_is_frozen:
            str_rep += " (frozen)\n"
        else:
            str_rep += " (not frozen)\n"

        str_rep += f"\tThe x-coordinate spacing is {self.spacing}"
        if self._spacing_is_frozen:
            str_rep += " (frozen)\n"
        else:
            str_rep += " (not frozen)\n"

        if not None in {self.min_value, self.max_value, self.spacing}:
            str_rep += f"\tThe resulting x-values are: {self._independent_variable_values}\n"

        return str_rep

    def __repr__(self):
        """Create and eval-able representation of this transform."""

        str_rep = self.__class__.__name__ + "("

        if self.min_value is not None:
            str_rep += f"min_value={self.min_value}"

        if self.max_value is not None:
            str_rep += f", max_value={self.max_value}"

        if self.spacing is not None:
            str_rep += f", spacing={self.spacing}"

        if self.snap_to_ints:
            str_rep += f", snap_to_ints=True, respect_bounds={self.respect_bounds}"

        return str_rep + ")"
