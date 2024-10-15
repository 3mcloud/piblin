from typing import Union, List, Set, Tuple
import matplotlib.axes
import matplotlib.lines
import matplotlib.collections
import numpy as np
import numpy.typing
import piblin.data.datasets.roi as roi
import piblin.data.datasets.abc.split_datasets as split_datasets
import piblin.data.datasets.abc.unambiguous_datasets.unambiguous_dataset as unambiguous_dataset
import piblin.data.datasets.abc.split_datasets.zero_dimensional_dataset as zero_dimensional_dataset


class OneDimensionalDataset(split_datasets.SplitDataset):
    """A split dataset with a single independent variable.

    This dataset is stored split for efficiency, so that when
    only dependent or independent variables are needed they do not
    have to be computed or stored separately in memory. This split
    makes the mapping between dependent and independent variables
    implicit, i.e. via a relationship between indices.
    This dataset splits D and I assuming that their list indices
    map one-to-one

    The data will be converted to a numpy array, so can be any array_like
    object. All other numpy array() defaults are used.

    See:
    https://numpy.org/doc/stable/reference/generated/numpy.array.html#numpy.array

    This allows us to use the full numpy ndarray feature set for manipulating
    data, as well as the ndim and shape properties for checking data.

    https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html

    Parameters
    ----------
    dependent_variable_data : numpy.typing.ArrayLike
        The dependent variable values for each data point.
    dependent_variable_names : List of str
        The names of the dependent variables.
        Length must be equal to the number of dependent dimensions.
    dependent_variable_units : List of str
        The units of the dependent variables.
        Length must be equal to the number of dependent dimensions.
    independent_variable_data : list of numpy.typing.ArrayLike, optional
        The values of independent variables along the independent axis.
        This list must contain an array-like whose ndarray representation has ndim=1.
        If not provided, array indices are used instead.
    independent_variable_names : List of str
        The names of the independent variable.
        Length must be one.
    independent_variable_units : List of str
        The units of the independent variables.
        Length must be equal to the number of independent dimensions.
    source : str
        A human-readable representation of the source of this dataset.

    Attributes
    ----------
    dependent_variable_data -> np.ndarray
        The dependent variable values for each data point.
    number_of_dependent_dimensions -> int
        The dimensionality shared by dependent variables for all data points.
        This is the number of dimensions of the dependent variable data array.
    dependent_variable_names -> List[str]
        The names of the dependent variables.
        The length of this list is the number of dependent dimensions.
    dependent_variable_name -> str
        The name of the dependent variable of this dataset, if one-dimensional.
    dependent_variable_units -> List[str]
        The units of the dependent variables.
        The length of this list is the number of dependent dimensions.
    dependent_variable_unit -> str
        The unit of the dependent variable of this dataset, if one-dimensional.
    dependent_variable_axis_labels -> list of str
        The axis labels for the dependent variables of this dataset.
    independent_variable_data -> List of np.ndarray
        The independent variable values for each data point.
    number_of_independent_dimensions -> int
        The dimensionality shared by independent variables for all data points.
        This is the length of the independent variable list for split data.
    independent_variable_names -> List[str]
        The names of the independent variables.
        The length of this list is the number of independent dimensions.
    independent_variable_name -> str
        The name of the independent variable of this dataset, if one-dimensional.
    independent_variable_units -> List[str]
        The units of the independent variables.
        The length of this list is the number of independent dimensions.
    independent_variable_unit -> str
        The unit of the independent variable of this dataset, if one-dimensional.
    independent_variable_axis_labels -> list of str
        The axis labels for the independent variables of this dataset.
    number_of_points -> int
        The number of points in the dataset.
    source -> str
        A human-readable representation of the source of this dataset.
    one_line_description -> str
        A single-line human-readable description of this dataset.
    """
    DEFAULT_X_NAME: str = "x"
    """The default name for the independent variable."""
    DEFAULT_Y_NAME: str = "y"
    """The default name for the dependent variable."""

    def __init__(
            self,
            dependent_variable_data: np.typing.ArrayLike,
            dependent_variable_names: Union[str, List[str]] = None,
            dependent_variable_units: Union[str, List[str]] = None,
            independent_variable_data: List[np.typing.ArrayLike] = None,
            independent_variable_names: List[str] = None,
            independent_variable_units: List[str] = None,
            source: str = None
    ):

        if independent_variable_names is None:
            independent_variable_names = [self.DEFAULT_X_NAME]

        if dependent_variable_names is None:
            dependent_variable_names = [self.DEFAULT_Y_NAME]

        super().__init__(
            dependent_variable_data=dependent_variable_data,
            dependent_variable_names=dependent_variable_names,
            dependent_variable_units=dependent_variable_units,
            independent_variable_data=independent_variable_data,
            independent_variable_names=independent_variable_names,
            independent_variable_units=independent_variable_units,
            source=source
        )

        if self.number_of_independent_dimensions != 1:
            raise ValueError(
                f"Independent variable data must be 1-dimensional. "
                f"ndim={self.number_of_independent_dimensions}"
            )

        if self.dependent_variable_data.shape[0] != \
                len(self.independent_variable_data[0]):
            raise ValueError("Wrong length for independent variable data.")

    @classmethod
    def create(cls,
               y_values: numpy.typing.ArrayLike,
               x_values: numpy.typing.ArrayLike = None,
               x_name: str = None,
               x_unit: str = None,
               y_name: str = None,
               y_unit: str = None,
               source: str = None) -> "OneDimensionalDataset":
        """Construct an instance of the class from simple parameters.

        It is often simpler to consider a 1D dataset as a set of paired
        x and y values with a label for each. This creation method allows
        for objects to be instantiated from this perspective.

        Parameters
        ----------
        x_values : numpy.typing.ArrayLike
            Values of the independent variable at which the dataset is
            recorded.
        y_values : numpy.typing.ArrayLike
            Values of the dependent variables at the specified independent
            values.
        x_name : str, optional
            A description of the data in `x_values` (e.g. "Time (hrs)").
            By default, this is set to the generic "x".
        x_unit : str, optional
            The unit of the x values.
        y_name : str, optional
            A description of the data in `y_values` (e.g. "Count").
            By default, this is set to the generic "y".
        y_unit : str, optional
            The unit of the y-values.
        source : str
            A human-readable representation of the source of this dataset.
        """
        if x_values is None:
            independent_variable_data = None
        else:
            independent_variable_data = [x_values]

        if x_name is None:
            independent_variable_names = None
        else:
            independent_variable_names = [x_name]

        if y_name is None:
            dependent_variable_names = None
        else:
            dependent_variable_names = [y_name]

        if x_unit is None:
            independent_variable_units = None
        else:
            independent_variable_units = [x_unit]

        if y_unit is None:
            dependent_variable_units = None
        else:
            dependent_variable_units = [y_unit]

        return cls(dependent_variable_data=y_values,
                   dependent_variable_names=dependent_variable_names,
                   dependent_variable_units=dependent_variable_units,
                   independent_variable_data=independent_variable_data,
                   independent_variable_names=independent_variable_names,
                   independent_variable_units=independent_variable_units,
                   source=source)

    @classmethod
    def from_datasets(
            cls, datasets: List["OneDimensionalDataset"]
    ) -> "OneDimensionalDataset":
        """Create a dataset from a set of one-dimensional datasets.

        This method effectively merges multiple one-dimensional datasets
        into a single one-dimensional dataset. For this to be possible,
        all of those datasets must be of the same type and have the same
        a- and y-names and units.

        Parameters
        ----------
        datasets : List of Dataset
            The datasets to combine into a single dataset.
            The datasets must all be of the same type as a minimum
            requirement.
            The datasets must also be the same dimensionality and have the
            same name and unit
            for both dependent and independent variables.

        Returns
        -------
        OneDimensionalDataset
            The one-dimensional dataset resulting from the merging of the
            provided datasets.

        Raises
        ------
        ValueError
            If the provided one-dimensional datasets are not compatible.
        """
        target_type = type(datasets[0])
        if not all([isinstance(dataset, target_type) for dataset in datasets[1:]]):
            raise ValueError("Cannot combine datasets of different type.")

        target_independent_variable_names = datasets[0].independent_variable_names
        if not all([dataset.independent_variable_names == target_independent_variable_names for dataset in datasets[1:]]):
            raise ValueError("Cannot combine datasets with different independent variable names")

        target_independent_variable_units = datasets[0].independent_variable_units
        if not all([dataset.independent_variable_units == target_independent_variable_units for dataset in datasets[1:]]):
            raise ValueError("Cannot combine datasets with different independent variable units")

        target_dependent_variable_names = datasets[0].dependent_variable_names
        if not all([dataset.dependent_variable_names == target_dependent_variable_names for dataset in datasets[1:]]):
            raise ValueError("Cannot combine datasets with different dependent variable names")

        target_dependent_variable_units = datasets[0].dependent_variable_units
        if not all([dataset.dependent_variable_units == target_dependent_variable_units for dataset in datasets[1:]]):
            raise ValueError("Cannot combine datasets with different dependent variable units")

        independent_variable_data = np.concatenate([dataset.independent_variable_data[0] for dataset in datasets])
        dependent_variable_data = np.concatenate([dataset.dependent_variable_data for dataset in datasets])

        values, \
            unique_values_indices, \
            unique_value_counts = np.unique(independent_variable_data,
                                            return_index=True,
                                            return_counts=True)

        duplicate_independent_variable_values = values[unique_value_counts > 1]

        summarized_dependent_variable_data = []
        for duplicate_independent_variable_value in duplicate_independent_variable_values:
            indices_to_summarize = np.where(independent_variable_data == duplicate_independent_variable_value)[0]
            summarized_dependent_variable_data.append(np.mean(dependent_variable_data[indices_to_summarize]))

        indices_to_keep = [j for point_index, j in enumerate(unique_values_indices)
                           if unique_value_counts[point_index] == 1]

        independent_variable_data = independent_variable_data[indices_to_keep]
        dependent_variable_data = dependent_variable_data[indices_to_keep]

        independent_variable_data = np.concatenate([independent_variable_data,
                                                    np.array(duplicate_independent_variable_values)])

        dependent_variable_data = np.concatenate([dependent_variable_data,
                                                  np.array(summarized_dependent_variable_data)])

        args = independent_variable_data.argsort()
        independent_variable_data.sort()
        dependent_variable_data = dependent_variable_data[args]

        return target_type(
            dependent_variable_data=dependent_variable_data,
            dependent_variable_names=datasets[0].dependent_variable_names,
            dependent_variable_units=datasets[0].dependent_variable_units,
            independent_variable_data=[independent_variable_data],
            independent_variable_names=datasets[0].independent_variable_names,
            independent_variable_units=datasets[0].independent_variable_units
        )

    @staticmethod
    def from_zerod_datasets(
            x_dataset: zero_dimensional_dataset.ZeroDimensionalDataset,
            y_dataset: zero_dimensional_dataset.ZeroDimensionalDataset
    ) -> "OneDimensionalDataset":
        """Create a (single-point) one-dimensional dataset from two 0D datasets.

        This combination process necessarily creates a one-dimensional
        dataset with a single data point. In general, two zero-dimensional
        datasets can be combined in either of two ways, using each as either
        the dependent or independent variable. There are no restrictions on
        the variable names or units.

        Parameters
        ----------
        x_dataset : ZeroDimensionalDataset
            The 0D dataset from which to source the independent variable.
        y_dataset : ZeroDimensionalDataset
            The 0D dataset from which to source the dependent variable.

        Returns
        -------
        OneDimensionalDataset
            The single-point one-dimensional dataset made by combining the 0D datasets.
        """
        return OneDimensionalDataset.create(x_values=[x_dataset.value],
                                            x_name=x_dataset.name,
                                            x_unit=x_dataset.unit,
                                            y_values=[y_dataset.value],
                                            y_name=y_dataset.name,
                                            y_unit=y_dataset.unit)

    @staticmethod
    def from_multiple_zerod_datasets(x_datasets: List[zero_dimensional_dataset.ZeroDimensionalDataset],
                                     y_datasets: List[zero_dimensional_dataset.ZeroDimensionalDataset],
                                     merge_datasets: bool = True) -> Union["OneDimensionalDataset", List["OneDimensionalDataset"]]:
        """Create a one-dimensional dataset from two sets of 0D datasets.

        The datasets from which the x-values are drawn must have the same
        name and unit, and ditto for the y-value datasets.

        Parameters
        ----------
        x_datasets : List of ZeroDimensionalDataset
            The zero-dimensional datasets to use as x-values.
        y_datasets : List of ZeroDimensionalDataset
            The zero-dimensional datasets to use as y-values.
        merge_datasets : bool
            Whether to merge the resulting single-point datasets to
            a single multipoint dataset.

        Returns
        -------
        OneDimensionalDataset
            A one-dimensional dataset created from the provided 0D datasets.
        """
        x_names = [x_dataset.name for x_dataset in x_datasets]
        x_units = [x_dataset.unit for x_dataset in x_datasets]
        if not x_names[:-1] == x_names[1:] or not x_units[:-1] == x_units[1:]:
            raise ValueError(
                "Zero-dimensional datasets have inconsistent x-value names or units."
            )

        y_names = [y_dataset.name for y_dataset in y_datasets]
        y_units = [y_dataset.unit for y_dataset in y_datasets]
        if not y_names[:-1] == y_names[1:] or not y_units[:-1] == y_units[1:]:
            raise ValueError(
                "Zero-dimensional datasets have inconsistent y-value names."
            )

        single_point_datasets = [
            OneDimensionalDataset.from_zerod_datasets(x_dataset, y_dataset)
            for x_dataset, y_dataset in zip(x_datasets, y_datasets)
        ]

        if merge_datasets:
            return OneDimensionalDataset.from_datasets(single_point_datasets)
        else:
            return single_point_datasets

    @property
    def data(self) -> Set[Tuple[Tuple, Tuple]]:
        """The set of data points of this split dataset."""
        data = set()
        for x_value, y_value in zip(self.x_values, self.y_values):
            data.add(((y_value,), (x_value,)))

        return data

    @property
    def variable_names(self) -> Tuple[Tuple[str], Tuple[str]]:
        """The names of the dependent and independent variables."""
        return (self.y_name,), (self.x_name,)

    @property
    def variable_units(self) -> Tuple[Tuple[str], Tuple[str]]:
        """The units of the dependent and independent variables."""
        return (self.y_unit,), (self.x_unit,)

    # TODO - this can go to level up if data, variable_names, variable_units are abstract in SplitDataset
    def to_unambiguous_dataset(self) -> unambiguous_dataset.UnambiguousDataset:
        """Convert this one-dimensional split dataset to an unambiguous dataset."""
        return unambiguous_dataset.UnambiguousDataset(data=self.data,
                                                      variable_names=self.variable_names,
                                                      variable_units=self.variable_units,
                                                      source=self.source)

    def split_by_callable(self, classifier: callable) -> List["OneDimensionalDataset"]:
        """Split this dataset using a classifier for its points.

        Parameters
        ----------
        classifier : callable
            A callable that takes a pair of dependent and independent variables
            and returns a classification for the data point. A classification of
            None will result in the point being deleted.

        Returns
        -------
        datasets : List of OneDimensionalDataset
            The split datasets computed using the classifier.
        """
        classifications = [classifier(x_value, y_value) for x_value, y_value in
                           zip(self.x_values, self.y_values)]

        unique_classifications = set(classifications)
        if None in unique_classifications:
            unique_classifications.remove(None)

        datasets = []
        for unique_classification in unique_classifications:
            x_values = []
            y_values = []
            for x_value, y_value, classification in zip(self.x_values, self.y_values, classifications):
                if classification == unique_classification:
                    x_values.append(x_value)
                    y_values.append(y_value)

            datasets.append(OneDimensionalDataset.create(x_values=x_values,
                                                         y_values=y_values))

        return datasets

    def split_by_independent_variable_regions(self,
                                              split_regions: roi.CompoundRegion) -> List["OneDimensionalDataset"]:
        """Split this dataset into multiple datasets using the specified independent variable regions.

        Parameters
        ----------
        split_regions : piblin.data.CompoundRegion
            The compound region containing the regions to split the dataset into.
        """
        datasets = []
        for split_region in split_regions:

            x_values = []
            y_values = []
            for x_value, y_value in zip(self.x_values, self.y_values):
                if split_region.contains(x_value):
                    x_values.append(x_value)
                    y_values.append(y_value)

            datasets.append(OneDimensionalDataset.create(x_values=x_values,
                                                         x_unit=self.x_unit,
                                                         x_name=self.x_name,
                                                         y_values=y_values,
                                                         y_unit=self.y_unit,
                                                         y_name=self.y_name))

        return datasets

    def value_at(self,
                 independent_variable_value: Union[int, float, complex],
                 single_value: bool = True):
        """Determine the dependent variable value corresponding to a given independent variable value.

        Parameters
        ----------
        independent_variable_value : float
            The independent variable value at which to determine the
            dependent variable value.
        single_value : bool

        Returns
        -------
        float
            The dependent variable value (or an estimate) at the given
            independent variable value.

        Raises
        ------
        ValueError
            If the independent variable value recurs in the dataset.
            If the requested variable is outside the range covered by the
            dataset.

        Notes
        -----
        If the given independent variable value is already in the set of
        independent variable values at which this dataset is defined, the
        corresponding dependent variable value can be returned. If it is
        not, an estimate must be produced, and the linear interpolation
        approach is taken. This is not guaranteed to work for all datasets.
        """
        if independent_variable_value in self.independent_variable_data[0]:
            indices = np.where(self.independent_variable_data[0] == independent_variable_value)[0]
            if len(indices) == 1:
                return self.dependent_variable_data[indices[0]]
            else:
                dependent_values = [self.dependent_variable_data[i] for i in indices]
                if not single_value:
                    return dependent_values
                else:
                    return np.median(dependent_values)

        else:
            if self.independent_variable_data[0].min() <= \
                independent_variable_value <= \
                    self.independent_variable_data[0].max():

                return np.interp(independent_variable_value,
                                 self.independent_variable_data[0],
                                 self.dependent_variable_data)
            else:
                raise ValueError("Cannot estimate dependent variable values outside defined region.")

    @property
    def dependent_variable_data(self) -> np.ndarray:
        return self._dependent_variable_data

    @dependent_variable_data.setter
    def dependent_variable_data(self, dependent_variable_data) -> None:
        dependent_variable_data = np.array(dependent_variable_data)
        if dependent_variable_data.shape != self.dependent_variable_data.shape:
            raise ValueError(f"{dependent_variable_data.shape} != {self.dependent_variable_data.shape}")
        else:
            self._dependent_variable_data = dependent_variable_data

    @property
    def dependent_variable_name(self) -> str:
        return self.dependent_variable_names[0]

    @dependent_variable_name.setter
    def dependent_variable_name(self, dependent_variable_name: str) -> None:
        self._dependent_variable_names[0] = dependent_variable_name

    @property
    def dependent_variable_unit(self) -> str:
        return self.dependent_variable_units[0]

    @dependent_variable_unit.setter
    def dependent_variable_unit(self, dependent_variable_unit: str) -> None:
        self._dependent_variable_units[0] = dependent_variable_unit

    @property
    def independent_variable_data(self) -> List[np.ndarray]:
        return self._independent_variable_data

    @independent_variable_data.setter
    def independent_variable_data(self, independent_variable_data: np.typing.ArrayLike) -> None:
        independent_variable_data = np.array(independent_variable_data)
        if independent_variable_data.shape != self.independent_variable_data[0].shape:
            raise ValueError(f"New independent vars must be same shape. Old: {self.independent_variable_data[0].shape} New: {independent_variable_data.shape}")

        else:
            self._independent_variable_data = [independent_variable_data]

    def update_data(self,
                    dependent_variable_data: np.typing.ArrayLike,
                    independent_variable_data: np.typing.ArrayLike):
        """Update the x- and y- values simultaneously."""
        self._independent_variable_data[0] = np.array(independent_variable_data)
        self._dependent_variable_data = np.array(dependent_variable_data)

        self._validate_attributes()

    @property
    def independent_variable_name(self) -> str:
        return self.independent_variable_names[0]

    @independent_variable_name.setter
    def independent_variable_name(self, independent_variable_name):
        self.independent_variable_names = [independent_variable_name]

    @property
    def independent_variable_unit(self) -> str:
        return self.independent_variable_units[0]

    @independent_variable_unit.setter
    def independent_variable_unit(self, independent_variable_unit):
        self.independent_variable_units = [independent_variable_unit]

    @property
    def x_name(self) -> str:
        return self.independent_variable_name

    @x_name.setter
    def x_name(self, x_name: str):
        self.independent_variable_name = x_name

    @property
    def x_unit(self):
        return self.independent_variable_unit

    @x_unit.setter
    def x_unit(self, x_unit):
        self.independent_variable_unit = x_unit

    @property
    def x_label(self):
        return self.independent_variable_axis_labels[0]

    @property
    def y_name(self):
        return self.dependent_variable_name

    @y_name.setter
    def y_name(self, y_name: str):
        self.dependent_variable_name = y_name

    @property
    def y_unit(self):
        return self.dependent_variable_unit

    @y_unit.setter
    def y_unit(self, y_unit):
        self.dependent_variable_unit = y_unit

    @property
    def y_label(self):
        return self.dependent_variable_axis_labels[0]

    @property
    def x_values(self) -> np.ndarray:
        return self.independent_variable_data[0]

    @x_values.setter
    def x_values(self, x_values: np.ndarray):
        self.independent_variable_data = x_values

    @property
    def y_values(self):
        return self.dependent_variable_data

    @y_values.setter
    def y_values(self, y_values) -> None:
        self.dependent_variable_data = y_values

    def as_ndarray(self) -> np.ndarray:
        """Return this dataset as a numpy ndarray.

        Returns
        -------
        array_like
            The x- and y-values as a single array.
        """
        return np.vstack([self.x_values, self.y_values])

    def to_csv(self, filepath: str) -> None:
        """Save this dataset to a csv file.

        Parameters
        ----------
        filepath : str
            The path to which to save the dataset.
        """
        output_data = np.array([self.x_values, self.y_values])
        np.savetxt(filepath, output_data.T, delimiter=",")

    # def flatten_dependent_variables(self) -> np.ndarray:
    #     """Flatten the function values of this dataset into a 1-dimensional row.
    #
    #     Returns
    #     -------
    #     numpy.ndarray
    #         The y-values as a single row.
    #     """
    #     return self.y_values
    #
    # @classmethod
    # def unflatten_dependent_variables(cls, y_values: numpy.typing.ArrayLike) -> np.ndarray:
    #     """Convert a list of values to a 1D dataset's dependent variables.
    #
    #     This may appear useless but can be more complex in higher-dimensional subclasses.
    #     This method will need to know about x values for higher-D subclasses.
    #
    #     Parameters
    #     ----------
    #     y_values : array_like
    #         The values to be used as a 1D dataset's dependent variables.
    #
    #     Returns
    #     -------
    #     numpy.ndarray
    #         An array to be used as a 1D dataset's dependent variables.
    #     """
    #     return np.array(y_values)

    def _encode_column_labels(self) -> List[str]:
        """Compute labels for the y-values of the dataset from x-values.

        The 1D dataset stores a pairwise relationship between the
        independent variable x and a dependent variable y, i.e. for
        a set of values of x it tells us the corresponding value
        of y=f(x). Rather than storing the numerical values of x,
        we may want to store the y-values alongside string representations
        of the function, making it explicit rather than implicit.
        (x, y) becomes (f(x=...), y)

        Returns
        -------
        list of str
            The labels f(x-values) for each column of this dataset.
        """
        column_labels = []
        for x_value in self.x_values:
            label = f"{self.y_name}({self.y_unit})=f({self.x_name}({self.x_unit})={x_value})"
            column_labels.append(label)

        return column_labels
    #
    # @staticmethod
    # def decode_column_labels(column_labels: List[str]):
    #     """Convert a set of column labels into x, y labels and x values.
    #
    #     Parameters
    #     ----------
    #     column_labels : list of str
    #         The labels of the columns of the dataset.
    #
    #     Returns
    #     -------
    #     list of str
    #         The independent variable names of the dataset.
    #         This is the x-label as a length-1 list.
    #     y_label : str
    #         The dependent variable name of the dataset.
    #     x_values : numpy.ndarray
    #         The independent variable values of the dataset.
    #     """
    #     column_label_r = r"(\w+)\((\w+)=(.*)\)"
    #
    #     x_label = None
    #     y_label = None
    #
    #     x_values = []
    #     for column_label in column_labels:
    #         match = re.search(column_label_r, column_label)
    #         if match:
    #             x_label = match.groups()[1]
    #             y_label = match.groups()[0]
    #             x_values.append(float(match.groups()[2]))
    #
    #     return [x_label], y_label, np.array(x_values)

    @classmethod
    def unflatten_dependent_variables(cls, values: np.ndarray):
        pass

    @staticmethod
    def decode_column_labels(column_labels: List[str]):
        pass

    def __str__(self) -> str:
        """Create a human-readable representation of this 1D dataset."""
        str_rep = super().__str__() + "\n\n"
        str_rep += f"Values of {self.y_name} ({self.y_unit}) as a function of {self.x_name} ({self.x_unit})\n\n"
        longest_length = max(len(self.x_name), len(self.y_name))
        str_rep += f"{self.x_name:{longest_length}} = {self.x_values}\n"
        str_rep += f"{self.y_name:{longest_length}} = {self.y_values}"

        return str_rep

    @classmethod
    def _validate_axes_plotting_kwargs(cls, **axes_plotting_kwargs):
        """Validate the kwargs to be passed to the axes plotting method.

        This class uses the scatter method to plot its points. Setting the
        "c" keyword argument to None makes the colour of the point be
        determined by the "color" keyword argument which is set in the
        superclass to the default if not present in the keyword arguments
        passed to visualize.
        """
        if "marker" not in axes_plotting_kwargs:
            axes_plotting_kwargs["marker"] = None

        axes_plotting_kwargs = \
            super()._validate_axes_plotting_kwargs(**axes_plotting_kwargs)

        if "linestyle" not in axes_plotting_kwargs.keys():
            axes_plotting_kwargs["linestyle"] = cls.DEFAULT_LINE_STYLE

        return axes_plotting_kwargs

    def _plot_on_axes(self,
                      axes: matplotlib.axes.Axes,
                      **axes_plotting_kwargs) -> None:
        """Plot this 1D dataset on a given axes.

        Parameters
        ----------
        axes : matplotlib.axes.Axes
            The axes on which to plot the 1D dataset.
        """

        if "label" not in axes_plotting_kwargs.keys():
            axes_plotting_kwargs["label"] = f"{self.y_name} ({self.dependent_variable_unit})"

        args = [self.x_values,
                self.y_values]

        if len(self.dependent_variable_data) == 1:
            axes_plotting_kwargs["marker"] = self.DEFAULT_MARKER
            axes.scatter(x=self.x_values[0],
                         y=self.y_values[0],
                         cmap=None,
                         norm=None,
                         plotnonfinite=False,
                         data=None,
                         **axes_plotting_kwargs)
        else:
            _: List[matplotlib.lines.Line2D] = axes.plot(*args,
                                                         scalex=True,
                                                         scaley=True,
                                                         data=None,
                                                         **axes_plotting_kwargs)

    def _plot_on_axes_with_variation(self,
                                     axes: matplotlib.axes.Axes,
                                     variation,
                                     **axes_plotting_kwargs) -> None:
        """Visualize the given variation against this 1D dataset.

        The 1D dataset uses the fill_between method to visualize its variation.
        The original plot and variation plot share plot kwargs which is non-ideal.

        Parameters
        ----------
        axes : matplotlib.axes.Axes
            The axes on which to plot this scalar.
        variation : Dataset
            A dataset quantifying variation of this dataset.
        axes_plotting_kwargs : dict
            Keywords which will be passed to the scatter method.
            Empty by default.
        """
        self._plot_on_axes(axes=axes,
                           **axes_plotting_kwargs)

        if "marker" in axes_plotting_kwargs:
            del axes_plotting_kwargs["marker"]

        if "markersize" in axes_plotting_kwargs:
            del axes_plotting_kwargs["markersize"]

        axes_plotting_kwargs["alpha"] = axes_plotting_kwargs["alpha"] / 4

        _: matplotlib.collections.PolyCollection = \
            axes.fill_between(x=self.x_values,
                              y1=self.y_values + variation.dependent_variable_data,
                              y2=self.y_values - variation.dependent_variable_data,
                              where=None,
                              interpolate=False,
                              step=None,
                              data=None,
                              **axes_plotting_kwargs)

    def _label_axes(self, axes: matplotlib.axes.Axes) -> None:
        """Add labels the axes on which this 1D dataset will be plotted.

        Parameters
        ----------
        axes : matplotlib.axes.Axes
            The axes on which this dataset will be plotted.
        """
        axes.set_xlabel(self.x_label)
        axes.set_ylabel(self.y_label)

    @staticmethod
    def compute_average(
            datasets: List["OneDimensionalDataset"]
    ) -> "OneDimensionalDataset":
        """Compute the average of a set of one-dimensional datasets.

        Parameters
        ----------
        datasets : list of OneDimensionalDataset
            The datasets of which the average is to be computed.

        Returns
        -------
        dataset : OneDimensionalDataset
            The average dataset of the provided datasets.
        """
        if len(datasets) == 1:
            return datasets[0]
        else:

            representative_dataset = datasets[0]

            collected_data = [dataset.flatten_dependent_variables() for dataset in datasets]
            collected_data = np.array(collected_data).T

            dependent_variable_data = np.mean(collected_data, axis=1)

            return OneDimensionalDataset(
                dependent_variable_data=dependent_variable_data,
                dependent_variable_names=representative_dataset.dependent_variable_names,
                dependent_variable_units=representative_dataset.dependent_variable_units,
                independent_variable_data=representative_dataset.independent_variable_data,
                independent_variable_names=representative_dataset.independent_variable_names,
                independent_variable_units=representative_dataset.independent_variable_units,
                source="cralds averaging process"
            )

    @staticmethod
    def compute_variation(
            datasets: List["OneDimensionalDataset"]
    ) -> "OneDimensionalDataset":
        """Compute the variation across a set of one-dimensional datasets.

        Parameters
        ----------
        datasets : list of OneDimensionalDataset
            The datasets over which the variation is to be computed.

        Returns
        -------
        dataset : OneDimensionalDataset
            The variation over the provided datasets.
        """
        if len(datasets) == 1:
            return OneDimensionalDataset.create(
                datasets[0].x_values,
                np.zeros_like(datasets[0].y_values)
            )
        else:
            representative_dataset = datasets[0]

            collected_data = [dataset.flatten_dependent_variables()
                              for dataset in datasets]
            collected_data = np.array(collected_data).T

            dependent_variable_data = np.std(collected_data, axis=1)

            return OneDimensionalDataset(
                dependent_variable_data=dependent_variable_data,
                dependent_variable_names=representative_dataset.dependent_variable_names,
                dependent_variable_units=representative_dataset.dependent_variable_units,
                independent_variable_data=representative_dataset.independent_variable_data,
                independent_variable_names=representative_dataset.independent_variable_names,
                independent_variable_units=representative_dataset.independent_variable_units,
                source="cralds variation process")

    def one_line_str(self) -> str:
        """Create a single-line human-readable representation of this 1D dataset."""
        return self.__class__.__name__ + f": {self.y_name}=f({self.x_name})"

    # @property
    # def one_line_description(self) -> str:
    #     """Create a single-line human-readable representation of this 1D dataset."""
    #     return self.__class__.__name__ + f": {self.y_name}=f({self.x_name})"
