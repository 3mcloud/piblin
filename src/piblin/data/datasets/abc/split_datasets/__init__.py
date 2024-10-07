from typing import List, Set, Tuple, Union
import re
import copy

import matplotlib.axes
import numpy as np
import numpy.typing
import piblin.data.datasets.abc.dataset as dataset
# import piblin.data.datasets.abc.dataset_factory as dataset_factory


class SplitDataset(dataset.Dataset):
    """A dataset with gridded independent variable values.

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
        The values of independent variables along each axis.
        This list must contain array-likes whose ndarray representations have ndim=1.
        If not provided, array indices are used instead.
    independent_variable_names : List of str
        The names of the independent variables.
        Length must be equal to the number of independent dimensions.
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

    Methods
    -------
    compute_average
        Compute the average of a set of datasets.
    compute_variation
        Compute the variation across a set of datasets.
    to_pandas
        Convert this dataset into a pandas data frame.
    visualize
        Represent and present this dataset to facilitate understanding.
    create_color_map
        Create a color map for the requested number of datasets of this type.
    """
    COLUMN_LABEL_REGEX = r"(\w+)\((\w+)\)=(\d+)"
    """Regular expression for extracting dataset information from column labels of flat data."""

    def __init__(self,
                 dependent_variable_data: numpy.typing.ArrayLike,
                 dependent_variable_names: List[str] = None,
                 dependent_variable_units: List[str] = None,
                 independent_variable_data: List[numpy.typing.ArrayLike] = None,
                 independent_variable_names: List[str] = None,
                 independent_variable_units: List[str] = None,
                 source: str = None):

        super().__init__(source=source)

        self._dependent_variable_data = np.array(dependent_variable_data)

        if independent_variable_data is None:
            self._independent_variable_data = [np.array(range(number_of_points))
                                               for number_of_points in
                                               self.dependent_variable_data.shape]

        else:
            self._independent_variable_data = [np.array(independent_axis_values) for independent_axis_values in
                                               independent_variable_data]

            # TODO - should be in a validation method
            for independent_axis_values in self.independent_variable_data:
                if independent_axis_values.ndim != 1:
                    raise ValueError(f"Incorrect number of dimensions for independent axis: "
                                     f"{independent_axis_values.ndim} != 1, "
                                     f"shape={independent_axis_values.shape}")

            if len(self.independent_variable_data) != self.number_of_independent_dimensions:
                raise ValueError(f"Incorrect number of independent axes provided. "
                                 f"Expected: {self.number_of_dependent_dimensions}, "
                                 f"Received: {len(self.independent_variable_data)}")

            for i, dependent_variable_data_length in enumerate(self.dependent_variable_data.shape):
                if dependent_variable_data_length != len(self.independent_variable_data[i]):
                    raise ValueError(f"Independent variable lengths do not match shape of dependent variable data. "
                                     f"Expected: {self.dependent_variable_data.shape} "
                                     f"Received: {[len(ind) for ind in self.independent_variable_data]}")

        if dependent_variable_names is None:
            self._dependent_variable_names = \
                self._generate_default_variable_names(prefix="d", n=self.number_of_dependent_dimensions)
        else:
            self._dependent_variable_names = dependent_variable_names

        if dependent_variable_units is None:
            self._dependent_variable_units = \
                self._generate_default_variable_units(n=self.number_of_dependent_dimensions)
        else:
            self._dependent_variable_units = dependent_variable_units

        if independent_variable_names is None:
            self._independent_variable_names = \
                self._generate_default_variable_names(prefix="i", n=self.number_of_independent_dimensions)
        else:
            self._independent_variable_names = independent_variable_names

        if independent_variable_units is None:
            self._independent_variable_units = \
                self._generate_default_variable_units(n=self.number_of_independent_dimensions)
        else:
            self._independent_variable_units = independent_variable_units

        self._validate_attributes()

    @property
    def data(self) -> Set[Tuple[Tuple[Union[bool, int, float]],
                                Tuple[Union[bool, int, float]]]]:
        """Convert this split dataset to an unambiguous form."""
        pass

    @property
    def dependent_variable_data(self) -> np.ndarray:
        """The dependent variable data of this dataset."""
        return self._dependent_variable_data

    @dependent_variable_data.setter
    def dependent_variable_data(self, dependent_variable_data) -> None:
        """Set the dependent variable data of this dataset.

        The validity of the dataset depends on the shape of the dependent variable
        data (i.e its dimensionality and number of points) being preserved. For this
        reason new values must be validated before being set.
        """
        if dependent_variable_data.shape != self.dependent_variable_data.shape:
            print(f"New shape: {dependent_variable_data.shape}\nOriginal Shape: { self.dependent_variable_data.shape}")
            raise ValueError("Cannot replace dependent variable data with different shape")
        else:
            self._dependent_variable_data = dependent_variable_data

    @property
    def number_of_dependent_dimensions(self) -> int:
        """The dimensionality of the dependent variables of each data point of this dataset."""
        return 1  #self.dependent_variable_data.ndim

    @property
    def dependent_variable_range(self) -> Tuple[float, float]:
        """The ranges (min, max) of the dependent variable.

        Returns
        -------
        Tuple of float
            The minimum and maximum values of the dependent variable.
        """
        return np.min(self.dependent_variable_data), np.max(self.dependent_variable_data)

    @property
    def dependent_variable_names(self) -> List[str]:
        """The names of the dependent variables."""
        return self._dependent_variable_names

    @dependent_variable_names.setter
    def dependent_variable_names(self, dependent_variable_names: List[str]) -> None:
        if len(dependent_variable_names) != self.number_of_dependent_dimensions:
            raise ValueError(f"Incorrect number ({len(dependent_variable_names)}) of dependent variable names. "
                             f"Expected {self.number_of_dependent_dimensions}")
        else:
            self._dependent_variable_names = dependent_variable_names

    @property
    def dependent_variable_units(self) -> List[str]:
        """The units of the dependent variables."""
        return self._dependent_variable_units

    @dependent_variable_units.setter
    def dependent_variable_units(self, dependent_variable_units) -> None:
        if len(dependent_variable_units) != self.number_of_dependent_dimensions:
            raise ValueError(f"Incorrect number ({len(dependent_variable_units)}) of dependent variable units. "
                             f"Expected {self.number_of_dependent_dimensions}")
        else:
            self._dependent_variable_units = dependent_variable_units

    def change_dependent_variable_data_unit(self,
                                            factor: float,
                                            unit_name: str) -> None:
        """Change the units of the independent variable by a multiplicative factor.

        Parameters
        ----------
        factor : float
            The multiplicative factor.
        unit_name : str
            The new unit name.
        """
        self.dependent_variable_data *= factor
        self.dependent_variable_units = [unit_name]

    @property
    def independent_variable_data(self) -> List[np.ndarray]:
        """The values of independent variables along each axis."""
        return self._independent_variable_data

    @property
    def number_of_independent_dimensions(self) -> int:
        """The dimensionality of the independent variables of each data point of this dataset."""
        return len(self.independent_variable_data)

    @property
    def independent_variable_ranges(self) -> List[Tuple[float, float]]:
        """The ranges (min, max) of each independent variable.

        Returns
        -------
        List of Tuple of float
            The minimum and maximum values of each independent variable.
        """
        return [(np.min(independent_variables),
                 np.max(independent_variables)) for independent_variables in self.independent_variable_data]

    def step_size_along_independent_dimension(self, dimension: int) -> float:
        """Determine the pixel size along the given dimension.

        Parameters
        ----------
        dimension : int
            The dimension along which to determine the pixel size.

        Returns
        -------
        float
            The pixel size along the given dimension.
        """
        first_step_size = self.independent_variable_data[dimension][1] - self.independent_variable_data[dimension][0]
        for pair in ((self.independent_variable_data[dimension][i],
                      self.independent_variable_data[dimension][i + 1])
                     for i in range(0, len(self.independent_variable_data[dimension]) - 1)):

            step_size = pair[1] - pair[0]
            if not np.isclose(step_size, first_step_size):
                raise ValueError(f"2D dataset does not have constant x pixel size: {first_step_size} != {step_size}")

        return float(first_step_size)

    @property
    def size(self) -> List[float]:
        """The size of the dataset along each independent dimension"""
        return [variable_range[1] - variable_range[0]
                for variable_range in self.independent_variable_ranges]

    @property
    def center(self) -> List[float]:
        """The center of the dataset along each independent dimension"""
        return [np.min(independent_variables) +
                ((np.max(independent_variables) -
                  np.min(independent_variables)) / 2) for independent_variables in self.independent_variable_data]

    @property
    def independent_variable_names(self) -> List[str]:
        """The names of the independent variables."""
        return self._independent_variable_names

    @independent_variable_names.setter
    def independent_variable_names(self, independent_variable_names: List[str]) -> None:
        if len(independent_variable_names) != self.number_of_independent_dimensions:
            raise ValueError(f"Incorrect number ({len(independent_variable_names)}) of independent variable names. "
                             f"Expected {self.number_of_independent_dimensions}")
        else:
            self._independent_variable_names = independent_variable_names

    def remove_independent_variable_by_name(self, independent_variable_name: str) -> "SplitDataset":
        """Create a new dataset without the specified independent variable name.

        Parameters
        ----------
        independent_variable_name : str
            The name of the independent variable to remove.

        Returns
        -------
        SplitDataset
            A split dataset with the named independent variable removed.
        """
        independent_variable_index = self.independent_variable_index(independent_variable_name)
        return self.remove_independent_variable(independent_variable_index)

    def remove_independent_variables_by_name(self, independent_variable_names: List[str]):
        """Create a new dataset without the specified independent variable names.

        Parameters
        ----------
        independent_variable_names : List of str
            The names of the independent variable to remove.

        Returns
        -------
        SplitDataset
            A split dataset with the named independent variables removed.
        """
        independent_variable_indices = [self.independent_variable_index(name) for name in independent_variable_names]
        return self.remove_independent_variables(independent_variable_indices)

    def remove_independent_variable(self, independent_variable_index: int) -> "SplitDataset":
        """Create a new dataset without the specified independent variable.

        Parameters
        ----------
        independent_variable_index : int

        Returns
        -------
        SplitDataset
            A split dataset with one fewer dimensions than this dataset.
        """
        dataset = copy.deepcopy(self)

        independent_variable_data_to_keep = \
            dataset.independent_variable_data[:independent_variable_index] + \
            dataset.independent_variable_data[independent_variable_index + 1:]

        independent_variable_names_to_keep = \
            dataset.independent_variable_names[:independent_variable_index] + \
            dataset.independent_variable_names[independent_variable_index + 1:]

        independent_variable_units_to_keep = \
            dataset.independent_variable_units[:independent_variable_index] + \
            dataset.independent_variable_units[independent_variable_index + 1:]

        new_dependent_variable_data = np.squeeze(dataset.dependent_variable_data, -1)

        import \
            piblin.data.datasets.abc.dataset_factory as dataset_factory

        return dataset_factory.DatasetFactory.from_split_data(
            dependent_variable_data=new_dependent_variable_data,
            dependent_variable_names=dataset.dependent_variable_names,
            dependent_variable_units=dataset.dependent_variable_units,
            independent_variable_data=independent_variable_data_to_keep,
            independent_variable_names=independent_variable_names_to_keep,
            independent_variable_units=independent_variable_units_to_keep
        )

    def remove_independent_variables(self, independent_variable_indices: List[int]) -> "SplitDataset":
        """Create a new dataset without the specified independent variables.

        Parameters
        ----------
        independent_variable_indices : List of int

        Returns
        -------
        SplitDataset
            A split dataset with N fewer dimensions than this dataset, where N is the number of specified indices.
        """
        dataset = copy.deepcopy(self)

        independent_variable_data_to_keep = \
            [independent_variable_data
             for independent_variable_index, independent_variable_data in enumerate(dataset.independent_variable_data)
             if independent_variable_index not in independent_variable_indices]

        independent_variable_names_to_keep = \
            [independent_variable_name
             for independent_variable_index, independent_variable_name in enumerate(dataset.independent_variable_names)
             if independent_variable_index not in independent_variable_indices]

        independent_variable_units_to_keep = \
            [independent_variable_unit
             for independent_variable_index, independent_variable_unit in enumerate(dataset.independent_variable_units)
             if independent_variable_index not in independent_variable_indices]

        new_dependent_variable_data = np.squeeze(dataset.dependent_variable_data, -1)

        import \
            piblin.data.datasets.abc.dataset_factory as dataset_factory

        return dataset_factory.DatasetFactory.from_split_data(
            dependent_variable_data=new_dependent_variable_data,
            dependent_variable_names=dataset.dependent_variable_names,
            dependent_variable_units=dataset.dependent_variable_units,
            independent_variable_data=independent_variable_data_to_keep,
            independent_variable_names=independent_variable_names_to_keep,
            independent_variable_units=independent_variable_units_to_keep
        )

    def independent_variable_index(self, independent_variable_name: str) -> int:
        """Get the index of the independent variable with a specified name.

        Parameters
        ----------
        independent_variable_name : str
            The name of the independent variable to get the index of.

        Returns
        -------
        int
            The index of the independent variable.

        Raises
        ------
        ValueError
            If this dataset does not have an independent variable with the specified name.
        """
        try:
            return self.independent_variable_names.index(independent_variable_name)
        except ValueError:
            raise ValueError("No independent variable with name: {independent_variable_name}")

    @property
    def independent_variable_units(self) -> List[str]:
        """The units of the independent variables."""
        return self._independent_variable_units

    @independent_variable_units.setter
    def independent_variable_units(self, independent_variable_units) -> None:
        if len(independent_variable_units) != self.number_of_independent_dimensions:
            raise ValueError(f"Incorrect number ({len(independent_variable_units)}) of independent variable units. "
                             f"Expected {self.number_of_independent_dimensions}")
        else:
            self._independent_variable_units = independent_variable_units

    def change_independent_variable_data_units(
            self,
            factors: List[float],
            unit_names: List[str]) -> None:
        """Change the units of the independent variable by a factor.

        Parameters
        ----------
        factors : List of float
            The (per-independent variable) multiplicative factors.
        unit_names : List of str
            The (per-independent variable) new unit names.
        """
        for i, (factor, unit_name) in enumerate(zip(factors, unit_names)):
            self.independent_variable_data[i] = \
                self.independent_variable_data[i] * factor
            self.independent_variable_units[i] = unit_name

    def number_of_points(self) -> int:
        """The number of points in the dataset."""
        return self.dependent_variable_data.size

    @staticmethod
    def compute_average(datasets: List["SplitDataset"]) -> "SplitDataset":
        """Compute the average split dataset from a given set."""
        raise NotImplementedError(
            "Averaging of generic split datasets is not implemented."
        )

    @staticmethod
    def compute_variation(
            datasets: List["SplitDataset"]
    ) -> "SplitDataset":
        """Compute the variation across a set of split datasets."""
        raise NotImplementedError(
            "Variation in generic split datasets is not implemented."
        )

    def _plot_on_axes(self,
                      axes: matplotlib.axes.Axes,
                      **axes_plotting_kwargs) -> None:
        """Plot this dataset on a given matplotlib axes."""
        raise NotImplementedError(
            "Visualization for generic split dataset is not supported."
            "Consider using/creating a subclass of defined dimensionality."
        )

    def _plot_on_axes_with_variation(self,
                                     axes: matplotlib.axes.Axes,
                                     variation,
                                     **plot_kwargs) -> None:
        """Visualize the given variation in this dataset."""
        raise NotImplementedError(
            "Visualization for generic split dataset is not supported."
            "Consider using/creating a subclass of defined dimensionality."
        )

    def _label_axes(self, axes) -> None:
        """Add labels to the given axes."""
        raise NotImplementedError(
            "Visualization for generic split dataset is not supported."
            "Consider using/creating a subclass of defined dimensionality."
        )

    @property
    def one_line_description(self) -> str:
        """A single-line human-readable description of this split dataset.

        This property is to be used where a short description of the
        dataset is required. It must capture the properties of the dataset
        in the fewest possible characters. This string should not need to
        advertise that it's a dataset.
        """
        if self.number_of_dependent_dimensions == 1:
            str_rep = f"{self.dependent_variable_name}"
            if self.dependent_variable_unit is not None:
                str_rep += f"({self.dependent_variable_unit})"

            if self.number_of_independent_dimensions != 0:
                str_rep += "=f("
                for independent_variable_name, independent_variable_unit in zip(self.independent_variable_names,
                                                                                self.independent_variable_units):
                    str_rep += f"{independent_variable_name}"
                    if independent_variable_unit is not None:
                        str_rep += f"[{independent_variable_unit}]"

                    str_rep += ", "

                str_rep = str_rep[:-2] + ")"

                str_rep += f" ({self.number_of_points()} points)"
        else:
            raise NotImplementedError("Support for multiple dependent variables not complete.")

        return str_rep

    def _encode_column_labels(self) -> List[str]:
        """Create a label for each column of a flattened dataset.

        The column labels need to follow a strict format so that they
        can be decoded later if needed. There will be an entry in the
        label of a data point for each independent dimension of the
        dataset. For a zero-dimensional dataset, the label will just be
        the dataset's name and unit, e.g. temperature(K).
        For a one-dimensional dataset the label will contain a name,
        unit and value, e.g. A(nu)=0.1.
        For a 2D dataset, the label will contain two names, units and
        values, e.g. x(m)=0.5,y(um)=0.1.
        The implicit mapping between dependent and independent variables
        is required to make the connection between flattened dependent
        variables and the correct independent variables. ZeroD is easy.
        OneD the mapping is just by index D_i, I_i and there are no
        other sensible choices.
        In 2D we need to know the flattening order of the numpy method
        so that we can follow it.
        The elements of a are read using this index order.
        ‘C’ means to index the elements in row-major, C-style order,
         with the last axis index changing fastest,
         back to the first axis index changing slowest.

        Returns
        -------
        column_labels : List of str
            A label for each column of the flattened dataset.
        """
        label = ""
        for dependent_variable_axis_label in self.dependent_variable_axis_labels:
            label += dependent_variable_axis_label

        if self.number_of_independent_dimensions == 0:
            return [label]

        coordinates = np.meshgrid(*self.independent_variable_data)
        coordinates = np.array(coordinates).T.reshape(-1, self.number_of_independent_dimensions)

        labels = []
        for coordinate_vector in coordinates:
            point_label = ""
            for i, value in enumerate(coordinate_vector):
                point_label += f"{self.independent_variable_axis_labels[i]}={value},"
            point_label = point_label[:-1]

            labels.append(f"{label}=f({point_label})")

        return labels

    def flatten_dependent_variables(self) -> np.ndarray:
        """Create a flat array of the dependent variable values of this dataset.

        The array will be flattened in row-major, C-style order.

        data = np.array([[0, 1, 2],
                         [3, 4, 5]]) (shape=(2,3)) -> array([0, 1, 2, 3, 4, 5])

        data = np.array([[0, 1],
                         [2, 3],
                         [4, 5]]) (shape=(3,2)) -> array([0, 1, 2, 3, 4, 5])

        Shape information is lost on flattening so the shape must be recoverable
        from the column labels.

        Returns
        -------
        numpy.ndarray
            The dependent variable data as a 1D numpy array.
        """
        return self.dependent_variable_data.flatten(order="C")
    
    def to_xarray(self):
        """Converts Dataset to xarray.DataArray

        Returns
        -------
        xarray.DataArray
            Dataset converted to xarray DataArray object.
        """
        import xarray as xr
        tech = self.__class__.__module__ + "." + self.__class__.__name__
        xr_data = xr.DataArray(
            self.dependent_variable_data,
            name=self.dependent_variable_name,  # 1-D only?
            coords=self.independent_variable_data,
            dims=self.independent_variable_names)
        xr_data.attrs["cralds_cls"] = tech
        xr_data.attrs[
            "dependent_variable_units"] = self.dependent_variable_units
        xr_data.attrs[
            "independent_variable_units"] = self.independent_variable_units
        xr_data.attrs["source"] = self.source
        return xr_data

    @classmethod
    def from_xarray(cls, xr_data):
        """Creates new dataset from xarray.DataArray.

        Parameters
        ----------
        xr_data : xarray.DataArry
            DataArray to be converted to a cralds Dataset

        Returns
        -------
        Dataset
            Dataset created from DataArry object.
        """
        import xarray as xr
        dependent_variable_data = xr_data.values
        dependent_variable_names = [
            xr_data.attrs.get("dependent_variable_name", xr_data.name)
        ]
        independent_variable_data = [
            coord.values for coord in xr_data.coords.values()
        ]
        independent_variable_names = list(xr_data.coords.keys())
        dependent_variable_units = xr_data.attrs["dependent_variable_units"]
        independent_variable_units = xr_data.attrs[
            "independent_variable_units"]
        source = xr_data.attrs["source"]
        return cls(dependent_variable_data=dependent_variable_data,
                   dependent_variable_names=dependent_variable_names,
                   dependent_variable_units=dependent_variable_units,
                   independent_variable_data=independent_variable_data,
                   independent_variable_names=independent_variable_names,
                   independent_variable_units=independent_variable_units,
                   source=source)

    @classmethod
    def unflatten(cls,
                  column_labels: List[str],
                  values: np.ndarray) -> "SplitDataset":
        """Turn the provided column labels and values into a dataset.

        Parameters
        ----------
        column_labels : list of str
            Labels for the columns of the dataset.
            Expected to contain independent variables and all labels.
        values : numpy.ndarray
            Flattened dependent variable values of the dataset.

        Returns
        -------
        Dataset
            The dataset as defined by the column labels and values.
        """
        variable_split_regex = r"(.*\(.*\))=f\((.*)\)"
        dependent_variable_label_regex = r"(\w+)\((\w+)\)"
        column_label_pattern = re.compile(cls.COLUMN_LABEL_REGEX)

        all_coords = []
        for label in column_labels:

            variable_split_pattern = re.compile(variable_split_regex)
            match = variable_split_pattern.match(label)
            if match is not None:
                dep_label = match.group(1)
                label = match.group(2)
            else:
                raise ValueError(f"No match for split: {label}, {variable_split_regex}")

            dependent_variable_label_pattern = re.compile(dependent_variable_label_regex)
            match = dependent_variable_label_pattern.match(dep_label)
            if match is not None:
                dependent_variable_name = match.group(1)
                dependent_variable_unit = match.group(2)
                if dependent_variable_unit == "None":
                    dependent_variable_unit = None

            inds = label.split(",")

            names = []
            coords = []
            units = []

            for ind in inds:
                match_data = column_label_pattern.match(ind)
                if match_data is not None:
                    names.append(match_data.group(1))
                    unit = match_data.group(2)
                    if unit == "None":
                        unit = None

                    units.append(unit)
                    coords.append(int(match_data.group(3)))
                else:
                    ...
                    # raise ValueError("Column labels not in correct format.")

            all_coords.append(coords)

        all_coords = np.array(all_coords).T

        independent_variable_data = []
        for row in all_coords:
            independent_variable_data.append(list(dict.fromkeys(row, None).keys()))

        new_shape = [len(ind_data) for ind_data in independent_variable_data]
        if new_shape == []:
            dependent_variable_data = values
        else:
            dependent_variable_data = values.reshape(new_shape)

        return cls(dependent_variable_data=dependent_variable_data,
                   dependent_variable_names=[dependent_variable_name],
                   dependent_variable_units=[dependent_variable_unit],
                   independent_variable_data=independent_variable_data,
                   independent_variable_names=names,
                   independent_variable_units=units,
                   source=None)

    @classmethod
    def unflatten_dependent_variables(cls, values: np.ndarray):
        """Convert a flat thing into a dataset.

        This probably just calls the right numpy unflatten.

        Parameters
        ----------
        values : array_like
            The dependent variable value as a list

        Returns
        -------
        numpy.ndarray
            The dependent variable value as a length-1 list.
        """
        raise NotImplementedError("To be implemented.")

    @staticmethod
    def decode_column_labels(column_labels: List[str]):
        """Turn the provided column label into dataset information.

        For a scalar there is only one label, for the y-value, and
        no independent information so return None.

        Parameters
        ----------
        list of str

        Returns
        -------
        list of str
            The label for the independent variable.
        str
            The label for the dependent variable.
        array_like
            The values of the independent variables.
        """
        raise NotImplementedError("To be implemented.")

    def __str__(self) -> str:
        """Create a human-readable representation of this split dataset."""
        str_rep = f"{self.__class__.__name__}\n" + "-" * len(self.__class__.__name__)
        str_rep += "\n"
        if self.source is not None:
            str_rep += f"Dataset Source: {self.source}\n"

        str_rep += f"\nDependent Variable Data Properties: " \
                   f"ndim={self.dependent_variable_data.ndim}, " \
                   f"size={self.dependent_variable_data.size} ({self.dependent_variable_data.size} bytes), " \
                   f"shape={self.dependent_variable_data.shape}\n"

        str_rep += f"Dependent Variable Labels: "
        for label in self.dependent_variable_axis_labels:
            str_rep += f"{label}, "
        str_rep = str_rep[:-2]

        if self.number_of_independent_dimensions == 0:
            return str_rep
        else:
            str_rep += f"\n\nIndependent Variable Data Properties: " \
                       f"n={self.number_of_independent_dimensions}, lengths={[len(axis) for axis in self.independent_variable_data]}\n"

            str_rep += f"Independent Variable Labels: "
            for label in self.independent_variable_axis_labels:
                str_rep += f"{label}, "

            return str_rep[:-2]

    def __repr__(self) -> str:
        """Create an eval-able representation of this split dataset."""
        str_rep = f"{self.__class__.__name__}("
        str_rep += f"dependent_variable_data={self.dependent_variable_data}, "
        str_rep += f"dependent_variable_names={self.dependent_variable_names}, "
        str_rep += f"dependent_variable_units={self.dependent_variable_units}, "
        str_rep += f"independent_variable_data={self.independent_variable_data}, "
        str_rep += f"independent_variable_names={self.independent_variable_names}, "
        str_rep += f"independent_variable_units={self.independent_variable_units}, "
        str_rep += f"source={self.source}"

        return f"{str_rep})"

    def __eq__(self, other: "SplitDataset") -> bool:
        """Determine if this split dataset is equal to another split dataset."""
        if self is other:
            return True

        if not np.array_equal(self._dependent_variable_data,
                              other.dependent_variable_data):
            return False

        for this_axis, other_axis in zip(self.independent_variable_data,
                                         other.independent_variable_data):
            if not np.array_equal(this_axis,
                                  other_axis):
                return False

        if self.dependent_variable_names != other.dependent_variable_names:
            return False

        if self.dependent_variable_units != other.dependent_variable_units:
            return False

        if self.independent_variable_names != other.independent_variable_names:
            return False

        if self.independent_variable_units != other.independent_variable_units:
            return False

        return True

    def __cmp__(self) -> NotImplemented:
        """Ensure comparisons other than equality are not implemented."""
        return NotImplemented
