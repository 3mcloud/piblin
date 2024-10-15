from typing import Set, Tuple, List
import itertools
import random
import numpy as np
import matplotlib.axes
from piblin.data.datasets.abc.dataset import Dataset


class UnambiguousDataset(Dataset):
    """A dataset with paired sets of independent and dependent values.

    An unambiguous dataset has an explicit mapping between the dependent
    and independent variable values at each data point. Each data point contains
    two ordered sets, one identified with the independent variables and one with
    the dependent variables. There is an ambiguity which is resolved by defining
    whether the first or second element of the data point is the set of dependent
    variables. In this class the dependent variables are listed first, and the
    independent variables second.

    Parameters
    ----------
    data : set of tuple of tuple of object
        The data points of this dataset.
    variable_names : tuple of tuple of str
        The labels of this dataset.
    variable_units : tuple of tuple of str
        The units of this dataset.
    """
    DEPENDENT_INDEX = 0
    """The index of each data point corresponding to the dependent variable values."""
    INDEPENDENT_INDEX = 1
    """The index of each data point corresponding to the independent variable values."""

    def __init__(self,
                 data: Set[Tuple[Tuple, Tuple]],
                 variable_names: Tuple[Tuple[str], Tuple[str]] = None,
                 variable_units: Tuple[Tuple[str], Tuple[str]] = None,
                 source: str = None):

        super().__init__(source=source)

        self._data = data

        self._number_of_dependent_dimensions = len(random.choice(tuple(self._data))[self.DEPENDENT_INDEX])
        self._number_of_independent_dimensions = len(random.choice(tuple(self._data))[self.INDEPENDENT_INDEX])

        self.assert_consistency()

        if variable_names is None:
            self._variable_names = self.generate_labels()
        else:
            self._variable_names = variable_names

        if variable_units is None:
            self._variable_units = self.generate_units()
        else:
            self._variable_units = variable_units

        self.assert_consistency()

        self._number_of_dependent_dimensions = len(random.choice(tuple(self._data))[self.DEPENDENT_INDEX])
        self._number_of_independent_dimensions = len(random.choice(tuple(self._data))[self.INDEPENDENT_INDEX])

    @property
    def number_of_points(self):
        return len(self.data)

    @classmethod
    def unflatten(cls, column_labels, values):
        ...

    # TODO - abstract properties?
    @property
    def variable_names(self) -> Tuple[Tuple[str], Tuple[str]]:
        """The names of the dependent and independent variables."""
        return self._variable_names

    @property
    def variable_units(self) -> Tuple[Tuple[str], Tuple[str]]:
        """The units of the dependent and independent variables."""
        return self._variable_units

    def _encode_column_labels(self) -> List[str]:
        """Create labels for the dependent variable data of this dataset."""
        ...

    def flatten_dependent_variables(self):
        """Create a flat (1D) form of the dependent variable data of this dataset."""
        ...

    @staticmethod
    def compute_average(datasets: List["Dataset"]) -> "Dataset":
        pass

    @staticmethod
    def compute_variation(datasets: List["Dataset"]) -> "Dataset":
        pass

    def assert_consistency(self):
        """Ensure this dataset's points have the same dimensionality."""
        for point in self._data:
            if isinstance(point[self.DEPENDENT_INDEX], int):
                raise ValueError("Dependent variables must be a tuple")

            if isinstance(point[self.INDEPENDENT_INDEX], int):
                raise ValueError("Independent variables must be a tuple")

        if len(self._data) == 1:
            return

        ordered_data = list(self._data)
        number_of_independent_dimensions = len(ordered_data[0][self.INDEPENDENT_INDEX])
        for point in ordered_data[1:]:
            if len(point[self.INDEPENDENT_INDEX]) != number_of_independent_dimensions:
                raise ValueError(
                    f"Not all points have the same independent variable dimensionality.{len(point[self.INDEPENDENT_INDEX])} {number_of_independent_dimensions}")

        number_of_dependent_dimensions = len(ordered_data[0][self.DEPENDENT_INDEX])
        for point in ordered_data[1:]:
            if len(point[self.DEPENDENT_INDEX]) != number_of_dependent_dimensions:
                raise ValueError("Not all points have the same dependent variable dimensionality.")

        self.check_for_independent_duplicates()

    def check_for_independent_duplicates(self):
        """Ensure there are no duplicate points in the domain."""
        data_point_pairs = itertools.permutations(self.independent_variable_data, 2)
        for data_point_pair in data_point_pairs:
            if np.array_equal(data_point_pair[0], data_point_pair[1]):
                raise ValueError(
                    f"At least one pair of data points shares independent variable values: {data_point_pair}")

    def generate_labels(self) -> Tuple[List[str], List[str]]:
        """Generate labels for this dataset."""
        dependent_variable_labels = [f"d_{i}" for i in range(self.number_of_dependent_dimensions)]
        independent_variable_labels = [f"i_{i}" for i in range(self.number_of_independent_dimensions)]
        return dependent_variable_labels, independent_variable_labels

    def generate_units(self) -> Tuple[List[None], List[None]]:
        """Generate empty units for this dataset."""
        dependent_variable_units = [None] * self.number_of_dependent_dimensions
        independent_variable_units = [None] * self.number_of_independent_dimensions
        return dependent_variable_units, independent_variable_units

    @property
    def data(self) -> Set[Tuple[Tuple[object], Tuple[object]]]:
        """The data of this dataset."""
        return self._data

    def one_line_description(self) -> str:
        return self.__class__.__name__

    @property
    def dependent_variable_data(self) -> np.ndarray:
        return np.array([data_point[self.DEPENDENT_INDEX] for data_point in self._data])

    @property
    def dependent_variable_names(self) -> List[str]:
        return self._variable_names[self.DEPENDENT_INDEX]

    @property
    def dependent_variable_units(self) -> List[str]:
        return self._variable_units[self.DEPENDENT_INDEX]

    @property
    def number_of_dependent_dimensions(self):
        return self._number_of_dependent_dimensions

    @property
    def independent_variable_data(self) -> List[np.ndarray]:
        """Must have shape (number of points, number of independent dimensions)."""
        if self.number_of_independent_dimensions == 0:
            return []

        # TODO - this is wrong
        return [np.array([data_point[self.INDEPENDENT_INDEX] for data_point in self._data])]

    @property
    def independent_variable_names(self) -> List[str]:
        return self._variable_names[self.INDEPENDENT_INDEX]

    @property
    def independent_variable_units(self) -> List[str]:
        return self._variable_units[self.INDEPENDENT_INDEX]

    @property
    def number_of_independent_dimensions(self):
        return self._number_of_independent_dimensions

    def _plot_on_axes(self, axes: matplotlib.axes.Axes, **axes_plotting_kwargs) -> None:
        """Produce a scatter plot of this dataset."""
        raise NotImplementedError("Plotting of generic unambiguous datasets is not supported.")

    def _plot_on_axes_with_variation(self,
                                     axes: matplotlib.axes.Axes,
                                     variation,
                                     **plot_kwargs) -> None:
        """Produce a scatter plot of this dataset with an indication of variation."""
        raise NotImplementedError("Plotting of generic unambiguous datasets is not supported.")

    def _label_axes(self, axes: matplotlib.axes.Axes) -> None:
        """Add labels to the given axes."""
        raise NotImplementedError("Plotting of generic unambiguous datasets is not supported.")

    def __str__(self):
        """Return a human-readable representation of this dataset."""
        str_rep = f"{self.__class__.__name__}\n"
        str_rep += "-" * len(str_rep) + "\n"
        str_rep += f"Num. Independent Dimensions: {self.number_of_independent_dimensions}"
        str_rep += f"\tLabels: {self._variable_names[self.INDEPENDENT_INDEX]} "
        str_rep += f"\tUnits: {self._variable_units[self.INDEPENDENT_INDEX]}\n"
        str_rep += f"Num. Dependent Dimensions:   {self.number_of_dependent_dimensions}"
        str_rep += f"\tLabels: {self._variable_names[self.DEPENDENT_INDEX]} "
        str_rep += f"\tUnits: {self._variable_units[self.DEPENDENT_INDEX]}\n\n"
        str_rep += f"Data Points\n-----------\n{self._variable_names[self.INDEPENDENT_INDEX]}, {self._variable_names[self.DEPENDENT_INDEX]}\n"
        for point in self._data:
            str_rep += f"{point[self.INDEPENDENT_INDEX]}, {point[self.DEPENDENT_INDEX]}\n"

        return str_rep

    def __repr__(self) -> str:
        """Return an eval()-able representation of this dataset."""
        str_rep = f"{self.__class__.__name__}("
        return f"{str_rep})"
