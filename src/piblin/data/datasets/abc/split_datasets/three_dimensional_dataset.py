from typing import List
import numpy.typing
from piblin.data.datasets.abc.split_datasets import SplitDataset


class ThreeDimensionalDataset(SplitDataset):
    """A dataset with three independent dimensions on a grid."""
    DEFAULT_X_NAME: str = "x"
    """The default name for the first independent variable."""
    DEFAULT_Y_NAME: str = "y"
    """The default name for the second independent variable."""
    DEFAULT_Z_NAME: str = "z"
    """The default name for the third independent variable."""
    DEFAULT_Q_NAME: str = "q"
    """The default name for the dependent variable."""

    def __init__(self,
                 dependent_variable_data: numpy.typing.ArrayLike,
                 dependent_variable_names: List[str] = None,
                 dependent_variable_units: List[str] = None,
                 independent_variable_data: List[numpy.typing.ArrayLike] = None,
                 independent_variable_names: List[str] = None,
                 independent_variable_units: List[str] = None,
                 source: str = None):

        if independent_variable_names is None:
            independent_variable_names = [self.DEFAULT_X_NAME,
                                          self.DEFAULT_Y_NAME,
                                          self.DEFAULT_Z_NAME]

        if dependent_variable_names is None:
            dependent_variable_names = [self.DEFAULT_Q_NAME]

        super().__init__(
            dependent_variable_data=dependent_variable_data,
            dependent_variable_names=dependent_variable_names,
            dependent_variable_units=dependent_variable_units,
            independent_variable_data=independent_variable_data,
            independent_variable_names=independent_variable_names,
            independent_variable_units=independent_variable_units,
            source=source
        )

        if self.dependent_variable_data.ndim != 3:
            raise ValueError("Non-3D dataset!")

        if not self.dependent_variable_data.shape[0] == \
                len(self.independent_variable_data[0]):
            raise ValueError(f"{self.dependent_variable_data.shape[0]} != "
                             f"{len(self.independent_variable_data[0])}")

        if not self.dependent_variable_data.shape[1] == \
               len(self.independent_variable_data[1]):
            raise ValueError(f"{self.dependent_variable_data.shape[1]} != "
                             f"{len(self.independent_variable_data[1])}")

        if not self.dependent_variable_data.shape[2] == \
               len(self.independent_variable_data[2]):
            raise ValueError(f"{self.dependent_variable_data.shape[2]} != "
                             f"{len(self.independent_variable_data[2])}")

    @classmethod
    def create(cls,
               q_values: numpy.typing.ArrayLike,
               q_name: str = None,
               q_unit: str = None,
               x_values: numpy.typing.ArrayLike = None,
               x_name: str = None,
               x_unit: str = None,
               y_values: numpy.typing.ArrayLike = None,
               y_name: str = None,
               y_unit: str = None,
               z_values: numpy.typing.ArrayLike = None,
               z_name: str = None,
               z_unit: str = None,
               source: str = None):

        if x_values is None and y_values is None and z_values is None:
            independent_variable_data = None
        else:
            independent_variable_data = [x_values, y_values, z_values]

        if x_name is None and y_name is None and z_name is None:
            independent_variable_names = None
        else:
            independent_variable_names = [x_name, y_name, z_name]

        if x_unit is None and y_unit is None and z_unit is None:
            independent_variable_units = None
        else:
            independent_variable_units = [x_unit, y_unit, z_unit]

        if q_unit is None:
            dependent_variable_units = None
        else:
            dependent_variable_units = [q_unit]

        if q_name is None:
            dependent_variable_names = None
        else:
            dependent_variable_names = [q_name]

        return cls(dependent_variable_data=q_values,
                   dependent_variable_names=dependent_variable_names,
                   dependent_variable_units=dependent_variable_units,
                   independent_variable_data=independent_variable_data,
                   independent_variable_names=independent_variable_names,
                   independent_variable_units=independent_variable_units,
                   source=source)

    @property
    def x_values(self):
        return self.independent_variable_data[0]

    @x_values.setter
    def x_values(self, x_values):
        self.independent_variable_data[0] = x_values

    @property
    def y_values(self):
        return self.independent_variable_data[1]

    @y_values.setter
    def y_values(self, y_values):
        self.independent_variable_data[1] = y_values

    @property
    def z_values(self):
        return self.independent_variable_data[2]

    @z_values.setter
    def z_values(self, z_values):
        self.independent_variable_data[2] = z_values

    @property
    def x_name(self) -> str:
        return self.independent_variable_names[0]

    @x_name.setter
    def x_name(self, x_name: str) -> None:
        self.independent_variable_names[0] = x_name

    @property
    def x_unit(self) -> str:
        return self.independent_variable_units[0]

    @property
    def x_axis_label(self) -> str:
        return self.independent_variable_axis_labels[0]

    @property
    def y_name(self) -> str:
        return self.independent_variable_names[1]

    @y_name.setter
    def y_name(self, y_name: str) -> None:
        self.independent_variable_names[1] = y_name

    @property
    def y_unit(self) -> str:
        return self.independent_variable_units[1]

    @property
    def y_axis_label(self) -> str:
        return self.independent_variable_axis_labels[1]

    @property
    def z_name(self) -> str:
        return self.independent_variable_names[2]

    @z_name.setter
    def z_name(self, z_name: str) -> None:
        self.independent_variable_names[2] = z_name

    @property
    def z_unit(self) -> str:
        return self.independent_variable_units[2]

    @property
    def z_axis_label(self) -> str:
        return self.independent_variable_axis_labels[2]

    @property
    def x_range(self):
        return self.independent_variable_ranges[0]

    @property
    def y_range(self):
        return self.independent_variable_ranges[1]

    @property
    def z_range(self):
        return self.independent_variable_ranges[2]

    @property
    def x_size(self):
        return self.size[0]

    @property
    def y_size(self):
        return self.size[1]

    @property
    def z_size(self):
        return self.size[2]

    def voxel_size_along_dimension(self, dimension: int) -> float:
        return self.step_size_along_independent_dimension(dimension)

    @property
    def x_voxel_size(self) -> float:
        """The voxel size along the x-dimension."""
        return self.step_size_along_independent_dimension(dimension=0)

    @property
    def y_voxel_size(self):
        """The voxel size along the y-dimension."""
        return self.step_size_along_independent_dimension(dimension=1)

    @property
    def z_voxel_size(self):
        """The voxel size along the z-dimension."""
        return self.step_size_along_independent_dimension(dimension=2)

    @property
    def number_of_dependent_dimensions(self) -> int:
        return 1

    @property
    def number_of_independent_dimensions(self) -> int:
        return 3

    @property
    def one_line_description(self) -> str:
        """Create a single-line human-readable representation of this dataset."""
        return self.__class__.__name__ + \
            (f": {self.dependent_variable_name}=f("
             f"{self.x_name}, "
             f"{self.y_name},"
             f"{self.z_name})")

    def __str__(self):
        str_rep = super().__str__() + "\n\n"
        str_rep += (f"Values of {self.y_name} ({self.y_unit}) "
                    f"as a function of {self.x_name} ({self.x_unit})\n\n")
        longest_length = max(len(self.x_name), len(self.y_name))
        str_rep += f"{self.x_name:{longest_length}} = {self.x_values}\n"
        str_rep += f"{self.y_name:{longest_length}} = {self.y_values}\n"
        str_rep += f"{self.z_name:{longest_length}} = {self.z_values}\n\n"
        str_rep += f"{self.dependent_variable_name} =\n\n{self.dependent_variable_data}"

        return str_rep
