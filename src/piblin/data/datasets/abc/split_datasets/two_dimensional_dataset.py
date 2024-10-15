from typing import Union, List, Set, Tuple

import matplotlib.axes
import matplotlib.image
import matplotlib.cm
import matplotlib.colors
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import numpy.typing
from piblin.data.datasets.abc.split_datasets import SplitDataset
from scipy.interpolate import RegularGridInterpolator


class TwoDimensionalDataset(SplitDataset):
    """A dataset with two independent dimensions on a grid.

    Parameters
    ----------
    dependent_variable_data : numpy.typing.ArrayLike
        The values of the dependent variables for the data points.
        This must be an array-like whose ndarray representation has ndim=2.
        The cralds convention is that the first index runs over columns,
        and the second over rows.
    independent_variable_data : list of numpy.typing.ArrayLike, optional
        The values of independent variables along each axis.
        This list must contain array-likes whose ndarray representations
        have ndim=1. If not provided, array indices are used instead.
        The first array_like in the list contains x-values, i.e. the values
        corresponding to the columns of the dataset.
        The second contains y-values, i.e. the values corresponding to the
        rows of the dataset.

        y (rows)
        ^
        |
        |
        __________> x (columns)
    """
    DEFAULT_FIGURE_SIZE = (8, 8)
    """Default size of 2D dataset visualization (in inches)."""
    DEFAULT_COLOR_MAP: matplotlib.colors.Colormap = (
        matplotlib.cm.get_cmap("inferno"))
    """Default color map for dataset visualizations."""
    DEFAULT_X_NAME: str = "x"
    """The default name for the first independent variable."""
    DEFAULT_Y_NAME: str = "y"
    """The default name for the second independent variable."""
    DEFAULT_Z_NAME: str = "z"
    """The default name for the dependent variable."""

    def __init__(self,
                 dependent_variable_data: numpy.typing.ArrayLike,
                 dependent_variable_names: List[str] = None,
                 dependent_variable_units: List[str] = None,
                 independent_variable_data: List[np.ndarray] = None,
                 independent_variable_names: List[str] = None,
                 independent_variable_units: List[str] = None,
                 source: str = None):

        if independent_variable_names is None:
            independent_variable_names = [self.DEFAULT_X_NAME,
                                          self.DEFAULT_Y_NAME]

        if dependent_variable_names is None:
            dependent_variable_names = [self.DEFAULT_Z_NAME]

        super().__init__(
            dependent_variable_data=dependent_variable_data,
            dependent_variable_names=dependent_variable_names,
            dependent_variable_units=dependent_variable_units,
            independent_variable_data=independent_variable_data,
            independent_variable_names=independent_variable_names,
            independent_variable_units=independent_variable_units,
            source=source
        )

        if self.dependent_variable_data.ndim != 2:
            raise ValueError("Non-2D dataset!")

        if not self.dependent_variable_data.shape[0] == \
                len(self.independent_variable_data[0]):
            raise ValueError(f"{self.dependent_variable_data.shape[0]} != "
                             f"{len(self.independent_variable_data[0])}")

        if not self.dependent_variable_data.shape[1] == \
               len(self.independent_variable_data[1]):
            raise ValueError(f"{self.dependent_variable_data.shape[1]} != "
                             f"{len(self.independent_variable_data[1])}")

    @classmethod
    def create(cls,
               z_values: numpy.typing.ArrayLike,
               x_values: numpy.typing.ArrayLike = None,
               y_values: numpy.typing.ArrayLike = None,
               x_name: str = None,
               x_unit: str = None,
               y_name: str = None,
               y_unit: str = None,
               z_name: str = None,
               z_unit: str = None,
               source: str = None):

        if x_values is None and y_values is None:
            independent_variable_data = None
        else:
            independent_variable_data = [x_values, y_values]

        if x_name is None and y_name is None:
            independent_variable_names = None
        else:
            independent_variable_names = [x_name, y_name]

        if x_unit is None and y_unit is None:
            independent_variable_units = None
        else:
            independent_variable_units = [x_unit, y_unit]

        if z_unit is None:
            dependent_variable_units = None
        else:
            dependent_variable_units = [z_unit]

        if z_name is None:
            dependent_variable_names = None
        else:
            dependent_variable_names = [z_name]

        return cls(dependent_variable_data=z_values,
                   dependent_variable_names=dependent_variable_names,
                   dependent_variable_units=dependent_variable_units,
                   independent_variable_data=independent_variable_data,
                   independent_variable_names=independent_variable_names,
                   independent_variable_units=independent_variable_units,
                   source=source)

    def value_at(self,
                 independent_variable_value_x: Union[int, float, complex],
                 independent_variable_value_y: Union[int, float, complex],
                 single_value: bool = True):
        """Determine the z-value corresponding to a given x,y-value.

        Parameters
        ----------
        independent_variable_value_x, independent_variable_value_y : float
            The independent variable value at which to determine the
            dependent variable value.
        single_value : bool

        Returns
        -------
        float
            The dependent variable value (or an estimate) at the given
            independent variable values.

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
        if independent_variable_value_x in self.independent_variable_data[0] and \
                independent_variable_value_y in self.independent_variable_data[1]:

            indices_x, indices_y = np.where(self.independent_variable_data[0] == independent_variable_value_x)[0], \
                                   np.where(self.independent_variable_data[1] == independent_variable_value_y)[0]
            if len(indices_x) == len(indices_y) == 1:
                return [self.dependent_variable_data[indices_x[0], indices_y[0]]]
            else:
                dependent_values = [self.dependent_variable_data[i, j] for i, j in zip(indices_x, indices_y)]
                if not single_value:
                    return dependent_values
                else:
                    return [np.median(dependent_values)]

        else:
            interp = RegularGridInterpolator((self.independent_variable_data[0],
                                              self.independent_variable_data[1]),
                                             self.dependent_variable_data, method='slinear', bounds_error=True)
            return interp((independent_variable_value_x, independent_variable_value_y))

    @property
    def data(self) -> Set[Tuple[Tuple, Tuple]]:
        raise NotImplementedError("Conversion from 2D to unambiguous not implemented")

    @classmethod
    def unflatten_dependent_variables(cls, values: np.ndarray):
        pass

    @staticmethod
    def decode_column_labels(column_labels: List[str]):
        pass

    @staticmethod
    def compute_average(datasets: List["TwoDimensionalDataset"]) -> "TwoDimensionalDataset":
        pass

    @staticmethod
    def compute_variation(datasets: List["TwoDimensionalDataset"]) -> "TwoDimensionalDataset":
        pass

    @property
    def x_values(self):
        return self.independent_variable_data[0]

    @property
    def y_values(self):
        return self.independent_variable_data[1]

    @property
    def x_range(self):
        return self.independent_variable_ranges[0]

    @property
    def y_range(self):
        return self.independent_variable_ranges[1]

    @property
    def x_size(self):
        return self.size[0]

    @property
    def y_size(self):
        return self.size[1]

    @property
    def z_values(self):
        return self.dependent_variable_data

    @z_values.setter
    def z_values(self, z_values) -> None:
        self.dependent_variable_data = z_values

    @property
    def x_name(self) -> str:
        return self.independent_variable_names[0]

    @property
    def x_unit(self) -> str:
        return self.independent_variable_units[0]

    @property
    def x_axis_label(self):
        return self.independent_variable_axis_labels[0]

    @property
    def y_name(self) -> str:
        return self.independent_variable_names[1]

    @property
    def y_unit(self) -> str:
        return self.independent_variable_units[1]

    @property
    def y_axis_label(self):
        return self.independent_variable_axis_labels[1]

    @property
    def z_name(self) -> str:
        return self.dependent_variable_names[0]

    @z_name.setter
    def z_name(self, z_name: str) -> None:
        self.dependent_variable_names[0] = z_name

    @property
    def z_unit(self) -> str:
        return self.dependent_variable_units[0]

    @property
    def z_axis_label(self) -> str:
        return self.dependent_variable_axis_labels[0]

    @property
    def pixel_size(self) -> float:
        """The (shared) pixel size along the x- and y-dimensions."""
        if not np.isclose(self.x_pixel_size,
                          self.y_pixel_size,
                          atol=1e-05):
            raise ValueError(
                "x and y pixel sizes are not equal, "
                "no single pixel size is defined for this dataset."
            )
        else:
            return self.x_pixel_size

    @property
    def x_pixel_size(self) -> float:
        """The pixel size along the x-dimension."""
        return self.step_size_along_independent_dimension(dimension=0)

    @property
    def y_pixel_size(self):
        """The pixel size along the y-dimension."""
        return self.step_size_along_independent_dimension(dimension=1)

    def pixel_size_along_dimension(self, dimension: int) -> float:
        return self.step_size_along_independent_dimension(dimension)

    @property
    def number_of_dependent_dimensions(self) -> int:
        return 1

    @property
    def number_of_independent_dimensions(self) -> int:
        return 2

    @classmethod
    def _validate_axes_plotting_kwargs(cls, **axes_plotting_kwargs):
        """Validate the keyword arguments for the axes plotting method.

        This class uses the scatter method to plot its points. Setting the
        "c" keyword argument to None makes the colour of the point be
        determined by the "color" keyword argument which is set in the
        superclass to the default if not present in the keyword arguments
        passed to visualize.
        """
        axes_plotting_kwargs = (
            super()._validate_axes_plotting_kwargs(**axes_plotting_kwargs)
        )

        if "marker" in axes_plotting_kwargs:
            del axes_plotting_kwargs["marker"]

        if "cmap" not in axes_plotting_kwargs:
            axes_plotting_kwargs["cmap"] = cls.DEFAULT_COLOR_MAP

        if "color" in axes_plotting_kwargs.keys():
            del axes_plotting_kwargs["color"]

        return axes_plotting_kwargs

    def _plot_on_axes(self,
                      axes: matplotlib.axes.Axes,
                      **axes_plotting_kwargs) -> None:
        """Plot this 2D split dataset on a given matplotlib axes.

        The dependent variable data is stored in column, row order.
        The zeroth independent variable contains the x-values (which
        are the column coordinates).
        The first independent variable contains the y-values (which
        are the row coordinates).
        The imshow method expects the data in the array in row, column
        format so it hsa to be transposed. We also want the origin on the
        lower left for ease of reading which is not the imshow default.

        If this is a single-point dataset, it will be plotted using the
        scatter method as a single marker. If not, the imshow method is
        used to create the plot.

        Parameters
        ----------
        axes : matplotlib.axes.Axes
            The axes on which to plot this scalar.
        axes_plotting_kwargs : dict
            Keywords which will be passed to the scatter method.
            Empty by default.

        Notes
        -----
        This method is to be overridden in subclasses.
        """
        # if "color" in axes_plotting_kwargs.keys():
        #     del axes_plotting_kwargs["color"]

        if self.dependent_variable_data.size == 1:
            axes.scatter(x=self.independent_variable_data[0][0],
                         y=self.independent_variable_data[1][0],
                         c=self.dependent_variable_data[0][0])
            return

        else:

            # masked_array = np.ma.array(self.dependent_variable_data,
            #                            mask=np.isnan(self.dependent_variable_data))

            masked_array = np.ma.array(self.dependent_variable_data,
                                       mask=self.dependent_variable_data == 0)

            if "cmap" not in axes_plotting_kwargs.keys():
                print("missing CMAP")
                current_cmap = self.DEFAULT_COLOR_MAP.copy()
                current_cmap.set_bad(color='white')
                axes_plotting_kwargs["cmap"] = current_cmap

            vmin = None
            if "vmin" in axes_plotting_kwargs:
                vmin = axes_plotting_kwargs["vmin"]
                del axes_plotting_kwargs["vmin"]

            vmax = None
            if "vmax" in axes_plotting_kwargs:
                vmax = axes_plotting_kwargs["vmax"]
                del axes_plotting_kwargs["vmax"]

            image: matplotlib.image.AxesImage = \
                axes.imshow(
                    X=masked_array.T,
                    norm=None,
                    aspect='equal',
                    interpolation='nearest',
                    vmin=vmin,
                    vmax=vmax,
                    origin="lower",
                    # the extent is left, right, bottom, top
                    extent=(
                        # leftmost (lowest) x-value
                        float(self.independent_variable_data[0][0]),
                        # rightmost (highest) x-value
                        float(self.independent_variable_data[0][-1]),
                        # bottom (lowest) y-value
                        float(self.independent_variable_data[1][0]),
                        # top (highest) y-value
                        float(self.independent_variable_data[1][-1])
                    ),
                    filternorm=True,
                    filterrad=4.0,
                    resample=True,
                    url=None,
                    data=None,
                    **axes_plotting_kwargs)

        if self.dependent_variable_data.dtype != np.bool_:
            fig = axes.get_figure()

            divider = make_axes_locatable(axes)
            cax = divider.append_axes("right", size="2%", pad=0.08)

            colorbar = fig.colorbar(image, cax=cax)
            colorbar.set_label(self.dependent_variable_axis_labels[0])

    def _plot_on_axes_with_variation(self,
                                     axes: matplotlib.axes.Axes,
                                     variation,
                                     **plot_kwargs) -> None:
        raise NotImplementedError(
            "Variation of 2D dataset class is not visualized in cralds."
        )

    def _label_axes(self, axes: matplotlib.axes.Axes) -> None:
        """Add labels to a matplotlib axes."""
        axes.set_xlabel(self.independent_variable_axis_labels[0])
        axes.set_ylabel(self.independent_variable_axis_labels[1])

    @property
    def one_line_description(self) -> str:
        """Create a single-line human-readable representation of this dataset."""
        return self.__class__.__name__ + \
            f": {self.z_name}=f({self.x_name}, {self.y_name})"

    def __str__(self):
        str_rep = super().__str__() + "\n\n"
        str_rep += (f"Values of {self.y_name} ({self.y_unit}) "
                    f"as a function of {self.x_name} ({self.x_unit})\n\n")
        longest_length = max(len(self.x_name), len(self.y_name))
        str_rep += f"{self.x_name:{longest_length}} = {self.x_values}\n"
        str_rep += f"{self.y_name:{longest_length}} = {self.y_values}\n\n"
        str_rep += f"{self.z_name} =\n\n{self.z_values}"

        return str_rep
