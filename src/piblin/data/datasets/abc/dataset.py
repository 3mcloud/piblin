from typing import List, Set, Tuple, Iterator, Union
from abc import ABC, abstractmethod
import numpy as np
import matplotlib.figure
import matplotlib.axes
import matplotlib.colors
import matplotlib.cm
import matplotlib.pyplot as plt
import piblin.data


class VisualizationError(Exception):
    """Raised when an incorrect object is provided for plotting."""


class Dataset(ABC):
    """Collection of labelled independent/dependent variable pairs.

    This abstract base class defines the properties that a dataset must
    have. It does not provide a general initializer method; these are left
    to subclasses and therefore so is the validation of the inputs.

    Parameters
    ----------
    source : str
        A human-readable representation of the source of this dataset.

    Attributes
    ----------
    dependent_variable_data -> np.ndarray
        The dependent variable values for each data point.
    number_of_dependent_dimensions -> int
        The dimensionality of dependent variables for all data points.
        This is the number of dimensions of the dependent variable array.
    dependent_variable_names -> List[str]
        The names of the dependent variables.
        The length of this list is the number of dependent dimensions.
    dependent_variable_name -> str
        The name of the dependent variable of this dataset, if 1D.
    dependent_variable_units -> List[str]
        The units of the dependent variables.
        The length of this list is the number of dependent dimensions.
    dependent_variable_unit -> str
        The unit of the dependent variable of this dataset, if 1D.
    dependent_variable_axis_labels -> list of str
        The axis labels for the dependent variables of this dataset.
    independent_variable_data -> np.ndarray
        The independent variable values for each data point.
    number_of_independent_dimensions -> int
        The dimensionality of independent variables for all data points.
        This is the length of the independent variable list for split data.
    independent_variable_names -> List[str]
        The names of the independent variables.
        The length of this list is the number of independent dimensions.
    independent_variable_name -> str
        The name of the independent variable of this dataset, if 1D.
    independent_variable_units -> List[str]
        The units of the independent variables.
        The length of this list is the number of independent dimensions.
    independent_variable_unit -> str
        The unit of the independent variable of this dataset, if 1D.
    independent_variable_axis_labels -> list of str
        The axis labels for the independent variables of this dataset.
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
        Create color map for the requested number of datasets of this type.
    flatten
        Convert this dataset to a flat representation.
    unflatten
        Convert a flat representation to a dataset.
    """
    DEPENDENT_INDEX = 0
    """The index of  the dependent variable values."""
    INDEPENDENT_INDEX = 1
    """The index of the independent variable values."""

    DEFAULT_FIGURE_SIZE: Tuple[int, int] = (5, 5)
    """Default size of dataset visualization (in inches)."""
    DEFAULT_COLOR: str = "black"
    """Default color for dataset data plots."""
    DEFAULT_COLOR_MAP: matplotlib.colors.Colormap = \
        matplotlib.cm.get_cmap("RdBu")
    """Default color map for dataset visualizations."""
    DEFAULT_MARKER: str = "x"
    """Default marker for dataset visualizations."""
    DEFAULT_LINE_STYLE: str = "solid"
    """Default line style for data plots."""

    def __init__(self, source: str = None):
        self._source = source

    def _validate_attributes(self):
        """Check whether all attributes of this dataset are consistent."""
        if len(self.dependent_variable_names) != \
                self.number_of_dependent_dimensions:
            raise ValueError(
                f"Incorrect number of dependent names. "
                f"Got {len(self.dependent_variable_names)}, "
                f"expected {self.number_of_dependent_dimensions}"
            )

        if len(self.dependent_variable_units) != \
                self.number_of_dependent_dimensions:
            raise ValueError(
                f"Incorrect number of dependent units. "
                f"Got {len(self.dependent_variable_units)}, "
                f"expected {self.number_of_dependent_dimensions}"
            )

        if len(self.independent_variable_names) != \
                self.number_of_independent_dimensions:
            raise ValueError(
                f"Incorrect number of dependent names. "
                f"Got {len(self.independent_variable_names)}, "
                f"expected {self.number_of_independent_dimensions}"
            )

        if len(self.independent_variable_units) != \
                self.number_of_independent_dimensions:
            raise ValueError(
                f"Incorrect number of dependent units. "
                f"Got {len(self.independent_variable_units)}, "
                f"expected {self.number_of_independent_dimensions}"
            )

    def split(self) -> List["Dataset"]:
        """Alias for to_datasets method."""
        return self.to_datasets()

    def to_datasets(self) -> List["Dataset"]:
        """Split this dataset into multiple datasets."""
        pass

    @classmethod
    def combine(cls, datasets: List["Dataset"]) -> "Dataset":
        """Alias for from_datasets method."""
        return cls.from_datasets(datasets)

    @classmethod
    def from_datasets(cls, datasets: List["Dataset"]) -> "Dataset":
        """Combine multiple datasets into one.

        Datasets can be combined provided they have consistent independent
        and dependent dimensions.
        """
        if len(datasets) == 1:
            return datasets[0]

        for dataset in datasets[1:]:
            if dataset.number_of_independent_dimensions != \
                    datasets[0].number_of_independent_dimensions:
                raise ValueError("Cannot combine datasets with different "
                                 "independent dimensions.")

        for dataset in datasets[1:]:
            if dataset.number_of_dependent_dimensions != \
                    datasets[0].number_of_dependent_dimensions:
                raise ValueError("Cannot combine datasets with different "
                                 "dependent dimensions.")

        for dataset in datasets[1:]:
            if dataset.dependent_variable_names != \
                    datasets[0].dependent_variable_names:
                print("Warning: Dependent variable names are not "
                      "consistent across datasets. Using generic labels.")
                dependent_variable_names = None
                break

        for dataset in datasets[1:]:
            if dataset.dependent_variable_units != \
                    datasets[0].dependent_variable_units:
                print("Warning: Dependent variable units are not "
                      "consistent across datasets. Removing units.")
                dependent_variable_units = None
                break

        for dataset in datasets[1:]:
            if dataset.independent_variable_names != \
                    datasets[0].independent_variable_names:
                print("Warning: Independent variable names are not "
                      "consistent across datasets. Using generic labels.")
                independent_variable_names = None
                break

        for dataset in datasets[1:]:
            if dataset.independent_variable_units != \
                    datasets[0].independent_variable_units:
                print("Warning: Independent variable names are not "
                      "consistent across datasets. Removing units.")
                independent_variable_units = None
                break

        source = ""
        sources = [f"{dataset.source}, "
                   for dataset in datasets if dataset.source is not None]
        for line in sources:
            source += line

        if source != "":
            source = source[:-2]
        else:
            source = None

        target_type = type(datasets[0])
        if not all([isinstance(dataset, target_type)
                    for dataset in datasets]):
            print("Warning: Types are not consistent across datasets. "
                  "Using generic class.")
            target_type = Dataset

        # create and return a new dataset,
        # this is where a common creation interface is necessary
        # print(source,
        #       dependent_variable_names,
        #       dependent_variable_units,
        #       independent_variable_names,
        #       independent_variable_units,
        #       target_type)

        # get the new dependent and independent variable data
        cls.combine_dataset_variable_data(datasets)
        # implement on per-dataset class level?

        # return target_type(
        # dependent_variable_data=dependent_variable_data,
        # dependent_variable_names=dependent_variable_names,
        # dependent_variable_units=dependent_variable_units,
        # independent_variable_data=independent_variable_data,
        # independent_variable_names=independent_variable_names,
        # independent_variable_units=independent_variable_units,
        # source=source
        # )

    @property
    def source(self) -> str:
        """A human-readable representation of the source of this dataset.

        Returns
        -------
        str
            A human-readable representation of the source of this dataset.
        """
        return self._source

    @source.setter
    def source(self, source: str) -> None:
        self._source = source

    @property
    @abstractmethod
    def data(self) -> Set[Tuple[Tuple[Union[bool, int, float]],
                                Tuple[Union[bool, int, float]]]]:
        """The set of data points of this dataset."""
        ...

    @property
    @abstractmethod
    def dependent_variable_data(self) -> np.ndarray:
        """The dependent variable data of this dataset."""
        ...

    @property
    @abstractmethod
    def number_of_dependent_dimensions(self) -> int:
        """Dimensionality of the dependent variables of this dataset."""
        ...

    @property
    @abstractmethod
    def dependent_variable_names(self) -> List[str]:
        """The names of the dependent variables of this dataset."""
        ...

    @property
    def dependent_variable_name(self) -> str:
        """Name of the dependent variable of this dataset, if 1D."""
        if self.number_of_dependent_dimensions == 1:
            return self.dependent_variable_names[0]
        else:
            raise ValueError(
                f"Dataset has {self.number_of_dependent_dimensions} "
                f"dependent variables. "
                f"A single dependent variable name is not defined."
            )

    @property
    @abstractmethod
    def dependent_variable_units(self) -> List[str]:
        """The units of the dependent variables of this dataset."""
        ...

    @property
    def dependent_variable_unit(self) -> str:
        """The unit of the dependent variable of this dataset if 1D."""
        if self.number_of_dependent_dimensions == 1:
            return self.dependent_variable_units[0]
        else:
            raise ValueError(
                f"Dataset has {self.number_of_dependent_dimensions} "
                f"dependent variables.\n"
                f"A single dependent variable unit is not defined."
            )

    @property
    def dependent_variable_axis_labels(self) -> List[str]:
        """Axis labels for the dependent variables."""
        return self._generate_variable_axis_labels(
            self.dependent_variable_names,
            self.dependent_variable_units
        )

    @property
    @abstractmethod
    def independent_variable_data(self) -> np.ndarray:
        """The independent variable data of this dataset."""
        ...

    @property
    @abstractmethod
    def number_of_independent_dimensions(self) -> int:
        """Dimensionality of the independent variables of this dataset."""
        ...

    @property
    @abstractmethod
    def independent_variable_names(self) -> List[str]:
        """The names of the independent variables of this dataset."""
        ...

    @property
    def independent_variable_name(self) -> str:
        if self.number_of_independent_dimensions == 1:
            return self.independent_variable_names[0]
        else:
            raise ValueError(
                f"Dataset has {self.number_of_independent_dimensions} "
                f"independent variables.\n"
                f"A single independent variable name is not defined."
            )

    @property
    @abstractmethod
    def independent_variable_units(self) -> List[str]:
        """The units of the dependent variables of this dataset."""
        ...

    @property
    def independent_variable_unit(self) -> str:
        """The unit of the independent variable (if defined)."""
        if self.number_of_independent_dimensions == 1:
            return self.independent_variable_units[0]
        else:
            raise ValueError(
                f"Dataset has {self.number_of_independent_dimensions} "
                f"independent variables.\n"
                f"A single independent variable unit is not defined."
            )

    @property
    def independent_variable_axis_labels(self) -> List[str]:
        """Axis labels for the independent variables."""
        return self._generate_variable_axis_labels(
            self.independent_variable_names,
            self.independent_variable_units
        )

    @property
    @abstractmethod
    def number_of_points(self):
        ...

    @property
    def one_line_description(self) -> str:
        """A human-readable one-line string representation."""
        return self.__class__.__name__

    def flatten(self) -> Tuple[List[str], np.ndarray]:
        """Convert dependent variables to a single row with column headers.

        Returns
        -------
        List of str
            A row of labels, one for each column of the flattened dataset.
        numpy.ndarray
            A row of dependent variable data as a 1D numpy array.
        """
        return (self._encode_column_labels(),
                self.flatten_dependent_variables())

    @abstractmethod
    def _encode_column_labels(self) -> List[str]:
        """Create labels for the dependent variable data."""
        ...

    @abstractmethod
    def flatten_dependent_variables(self) -> np.ndarray:
        """Create a flat (1D) form of the dependent variable data."""
        ...

    @classmethod
    @abstractmethod
    def unflatten(cls, column_labels, values):
        ...

    # @abstractmethod
    # def unflatten_dependent_variables(
    # cls, values: np.ndarray
    # ) -> "Dataset":
    #     ...
    #
    # @abstractmethod
    # def decode_column_labels(self):
    #     ...

    @staticmethod
    def _generate_default_variable_units(n: int) -> List[None]:
        """Generate a set of default units for this dataset.

        Parameters
        ----------
        n : int
            The number of variables.

        Returns
        -------
        List of None
            An empty unit for each variable.
        """
        return [None] * n

    @staticmethod
    def _generate_default_variable_names(prefix: str, n: int) -> List[str]:
        """Generate a set of default names for this dataset.

        Parameters
        ----------
        prefix : str
            A common prefix for the variable names.
        n : int
            The number of variables.

        Returns
        -------
        list of str
            A name for each variable.
        """
        return [f"{prefix}_{i}" for i in range(n)]

    @staticmethod
    @abstractmethod
    def compute_average(datasets: List["Dataset"]) -> "Dataset":
        """Compute the average of a set of datasets."""
        ...

    @staticmethod
    @abstractmethod
    def compute_variation(datasets: List["Dataset"]) -> "Dataset":
        """Compute the variation across a set of datasets."""
        ...

    def to_pandas(self, with_cralds_class: bool = True):
        """Convert this dataset into a pandas data frame.

        Parameters
        ----------
        with_cralds_class : bool, optional
            If True, a column corresponding to the source cralds dataset
            class is added to the data frame. This simplifies conversion
            back into a cralds data structure.

        Returns
        -------
        converted_data : Union[pd.DataFrame, List[pd.DataFrame]]
            Data frame or list of data frames based on source data.

        Raises
        ------
        ImportError
            Raised if `pandas` is not installed in environment.
        """
        from piblin.dataio.pandas import _to_pandas

        return _to_pandas(cralds_object=self,
                          with_cralds_class=with_cralds_class,
                          concat=False)

    def visualize(
            self,
            axes: matplotlib.axes.Axes = None,
            variation: "Dataset" = None,
            include_text: bool = False,
            figure_title: str = None,
            axis_title: str = None,
            total_figsize: Tuple[int, int] = None,
            **axes_plotting_kwargs
    ) -> Tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]:
        """Represent and present this dataset to facilitate understanding.

        This method provides a general approach to visualizing datasets.
        It can be customized in subclasses by overriding the various
        methods that it calls. Note that no mechanism is provided for
        passing args or kwargs to the matplotlib figure and axes creation
        methods. It is preferred that the user manipulated the returned
        figure and axes in order to replace default behaviour.

        Parameters
        ----------
        axes : matplotlib.axes.Axes
            A matplotlib axes object on which to plot this dataset.
            The default of None results in creation of a figure and axes.
        include_text : bool
            Whether to display the str representation of the dataset.
            The default is to not include the text.
        variation : Dataset
            A dataset containing the variation of this dataset.
        figure_title : str
            A title for the figure.
            The default is to not title the figure.
            Note that this argument is only respected if no axes is passed.
        axis_title : str
            A title for the axes.
            The default is to not title the axes.
        total_figsize
            The size of the figure to create in inches.
            The default is to use the class constant DEFAULT_FIGURE_SIZE.
            Note that this argument is only respected if no axes is passed.

        Returns
        -------
        matplotlib.figure.Figure
            The figure created by this method.
        matplotlib.axes.Axes
            The set of axes of the figure.

        Raises
        ------
        NotImplementedError
            If called on a class which has not implemented plot_on_axes.

        Notes
        -----
        This method is implemented such that it can receive either no axes
        on which to plot the dataset, in which case it must create a figure
        and axes, or it can receive an existing axes, in which case it must
        just plot the dataset on that axes. In the former case,
        responsibility for the appearance of the figure lies with this
        method. In the latter case, the figure should not be altered by
        this method. Subclasses can in general defer to this method but set
        their own default color and figure size properties for basic
        customization, or implement axes styling by overriding
        style_axes(). The dataset has all responsibility for the axes
        object it is plotted on and so can manipulate it.
        """
        if axes is not None and \
                (not isinstance(axes, matplotlib.axes.Axes)):
            raise VisualizationError(
                f"Dataset can only be visualized on a single Axes. "
                f"{axes} was supplied."
            )

        if variation is not None and type(variation) != type(self):
            if isinstance(self, piblin.data.ZeroDimensionalDataset) \
                    and isinstance(variation, float):
                variation = piblin.data.ZeroDimensionalDataset.create(
                        value=variation
                )

            elif isinstance(self, piblin.data.OneDimensionalDataset) \
                    and isinstance(variation, np.ndarray):
                variation = piblin.data.OneDimensionalDataset.create(
                    x_values=self.x_values,
                    y_values=variation
                )
            else:
                raise ValueError(
                    f"Variation dataset is not compatible: "
                    f"{type(variation)} != {type(self)}"
                )

        if include_text:
            print(self)

        if axis_title is None:
            axis_title = self.one_line_description

        fig, axes = self.__setup_fig_and_axes(axes=axes,
                                              axis_title=axis_title,
                                              figure_size=total_figsize,
                                              figure_title=figure_title)

        axes_plotting_kwargs = \
            self._validate_axes_plotting_kwargs(**axes_plotting_kwargs)

        if variation is None:
            self._plot_on_axes(axes=axes,
                               **axes_plotting_kwargs)
        else:
            self._plot_on_axes_with_variation(axes=axes,
                                              variation=variation,
                                              **axes_plotting_kwargs)

        return fig, axes

    @classmethod
    def _validate_axes_plotting_kwargs(cls, **axes_plotting_kwargs):
        """Validate the kwargs to be passed to the axes plotting method."""
        if "color" not in axes_plotting_kwargs:
            axes_plotting_kwargs["color"] = cls.DEFAULT_COLOR

        if "marker" not in axes_plotting_kwargs:
            axes_plotting_kwargs["marker"] = cls.DEFAULT_MARKER

        if "alpha" not in axes_plotting_kwargs:
            axes_plotting_kwargs["alpha"] = 1.0

        return axes_plotting_kwargs

    def __setup_fig_and_axes(
            self,
            axes: matplotlib.axes.Axes = None,
            axis_title: str = None,
            figure_size: Tuple[int, int] = None,
            figure_title: str = None
    ) -> Tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]:
        """Set up a matplotlib figure and axes for plotting data.

        This method is generic for all classes inheriting from the
        dataset class. It provides a single figure and single axes object
        which can be used to visualize a single dataset by calling the
        appropriate method of the axes class with that dataset's data
        as the parameters.

        Parameters
        ----------
        axes : matplotlib.axes.Axes
            A matplotlib axes for plotting the dataset.
            Default is to create a new instance.
        figure_size : tuple
            The size (in inches) of the matplotlib figure.
            Default of none uses the class default.
        figure_title : str
            A title for the matplotlib figure.
            Default of none results in not figure title.

        Returns
        -------
        fig : matplotlib.figure.Figure
            A matplotlib figure holding the plot axes.
        axes : matplotlib.axes.Axes
            A matplotlib axes for plotting the dataset.
        """
        if axes is None:

            if figure_size is None:
                figure_size = self.DEFAULT_FIGURE_SIZE

            fig = plt.figure()
            fig.set_size_inches(figure_size)

            if figure_title is not None:
                fig.suptitle(figure_title)

            axes = fig.add_subplot(1, 1, 1)
            self._title_axes(axes, title=axis_title)

        else:
            if figure_size is not None:
                print("Warning: figure_size argument ignored "
                      "when axes is supplied.")

            if figure_title is not None:
                print("Warning: figure_title argument ignored "
                      "when axes is supplied.")

            if axis_title != self.one_line_description:
                print("Warning: axis_title argument ignored "
                      "when axes is supplied.")

            fig = axes.figure

        fig.set_tight_layout(tight=None)
        self._style_axes(axes)
        self._label_axes(axes)

        return fig, axes

    @abstractmethod
    def _plot_on_axes(self,
                      axes: matplotlib.axes.Axes,
                      **axes_plotting_kwargs) -> None:
        """Plot this dataset on a given matplotlib axes.

        Parameters
        ----------
        axes : matplotlib.axes.Axes
            The axes on which to plot this scalar.
        axes_plotting_kwargs : dict
            Keywords which will be passed to the scatter method.
            Empty by default.

        Raises
        ------
        NotImplementedError
            In all cases.

        Notes
        -----
        To be overridden in subclasses to enable visualization.
        """
        ...

    @abstractmethod
    def _plot_on_axes_with_variation(self,
                                     axes: matplotlib.axes.Axes,
                                     variation,
                                     **plot_kwargs) -> None:
        """Visualize the given variation in this dataset.

        Parameters
        ----------
        axes : matplotlib.axes.Axes
            The axes on which to plot this scalar.
        variation : Dataset
            A dataset quantifying variation of this dataset.
        plot_kwargs : dict
            Keywords which will be passed to the scatter method.
            Empty by default.

        Raises
        ------
        NotImplementedError
            In all cases.

        Notes
        -----
        To be overridden in subclasses to enable visualization.
        """
        ...

    @abstractmethod
    def _label_axes(self, axes: matplotlib.axes.Axes) -> None:
        """Add labels to the given axes.

        This method is abstract because it is not possible to make an
        assumption about which variable will be placed along each plot
        axis without specific knowledge of the dimensionality of the
        dataset, which is only present in some subclasses.

        Parameters
        ----------
        axes : matplotlib.axes.Axes
        """
        ...

    @staticmethod
    def _generate_variable_axis_labels(
            variable_names: List[str],
            variable_units: List[str]
    ) -> List[str]:
        """Generate a set of axis labels for either type of variable.

        Parameters
        ----------
        variable_names : List of str
            The names of the variables.
        variable_units : List of str
            The units of the variables.

        Returns
        -------
        axis_labels : List of str
            The labels for the axes.
        """
        axis_labels: List[str] = []
        for name, unit in zip(variable_names, variable_units):
            label = name
            label += f"({unit})" if unit is not None else "(None)"
            axis_labels.append(label)

        return axis_labels

    @staticmethod
    def _title_axes(axes: matplotlib.axes.Axes,
                    title: str) -> None:
        """Add a title to the given axes.

        This method can be generic for all datasets because of the 1:1
        mapping between dataset and matplotlib axes. A dataset will always
        be visualized on a single axes object and can be described with a
        single title.

        Parameters
        ----------
        axes : matplotlib.axes.Axes
            The axes to label.
        title : str
            The title to apply to the axes.
            By default, the class' one line description will be used.
        """
        axes.set_title(title)

    @staticmethod
    def _style_axes(axes: matplotlib.axes.Axes) -> None:
        """Apply the class style to a matplotlib axes.

        Parameters
        ----------
        axes : matplotlib.axes.Axes
            The Axes object to apply the class style to.
        """
        axes.spines['top'].set_visible(False)
        axes.spines['right'].set_visible(False)

        axes.spines['bottom'].set_color('black')
        axes.spines['left'].set_color('black')
        axes.tick_params(top=False)
        axes.tick_params(right=False)
        axes.tick_params(axis='x', colors='black')
        axes.tick_params(axis='y', colors='black')
        axes.yaxis.label.set_color('black')

    @classmethod
    def create_color_map(cls, n: int = 1) -> Iterator:
        """Create a color map for the requested number of datasets.

        Given the default color map for this class, generate an evenly
        spaced set of colors to be used to identify the n datasets to be
        plotted on a single matplotlib axes.

        Parameters
        ----------
        n : int
            The number of datasets for which to generate colors.

        Returns
        -------
        Iterator
            A color map for a set of n generic datasets.
        """
        return iter(cls.DEFAULT_COLOR_MAP(np.linspace(0, 1, n)))
