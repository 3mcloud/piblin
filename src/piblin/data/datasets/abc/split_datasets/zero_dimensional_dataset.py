from typing import List, Set, Tuple
import re
import numpy as np
import numpy.typing
import matplotlib.axes
import matplotlib.figure
import piblin.data.datasets.abc.split_datasets as split_datasets
import piblin.data.datasets.abc.unambiguous_datasets.unambiguous_dataset as unambiguous_dataset


class ZeroDimensionalDataset(split_datasets.SplitDataset):
    """A dataset with a single dependent variable value.

    Parameters
    ----------
    dependent_variable_data : np.typing.ArrayLike
    dependent_variable_names : List[str]
    dependent_variable_units : List[str]
    independent_variable_data : List[np.typing.ArrayLike]
    independent_variable_names : List[str]
    independent_variable_units : List[str]
    source : str

    Attributes
    ----------

    Methods
    -------
    create -> ZeroDimensionalDataset
    """

    VARIABLE_SPLIT_REGEX: str = r"(.*\(.*\))"
    DEFAULT_NAME: str = "variable"
    """The default name for the dependent variable."""

    def __init__(
            self,
            dependent_variable_data: numpy.typing.ArrayLike,
            dependent_variable_names: List[str] = None,
            dependent_variable_units: List[str] = None,
            independent_variable_data: List[numpy.typing.ArrayLike] = None,
            independent_variable_names: List[str] = None,
            independent_variable_units: List[str] = None,
            source: str = None
    ):

        if dependent_variable_names == [None]:
            dependent_variable_names = [self.DEFAULT_NAME]

        if independent_variable_names is not None:
            if independent_variable_names:
                raise ValueError(
                    "0D dataset cannot have independent labels."
                )
            else:
                independent_variable_names = None

        if independent_variable_units is not None:
            if independent_variable_units:
                raise ValueError(
                    "0D dataset cannot have independent units."
                )
            else:
                independent_variable_units = None

        if independent_variable_data is not None:
            if independent_variable_data:
                raise ValueError(
                    "0D dataset cannot have independent data."
                )
            else:
                independent_variable_data = None
        
        if not isinstance(dependent_variable_data, np.ndarray):
            dependent_variable_data = np.array(dependent_variable_data)

        dependent_variable_data = np.squeeze(dependent_variable_data)
        
        if len(dependent_variable_data.shape) != 0:
            raise ValueError(
                "`dependent_variable_data` must be a scalar value"
            )

        super().__init__(
            dependent_variable_data=dependent_variable_data,
            dependent_variable_names=dependent_variable_names,
            dependent_variable_units=dependent_variable_units,
            independent_variable_data=independent_variable_data,
            independent_variable_names=independent_variable_names,
            independent_variable_units=independent_variable_units,
            source=source
        )

    @classmethod
    def create(cls,
               value: np.typing.ArrayLike,
               label: str = None,
               unit: str = None,
               source: str = None) -> "ZeroDimensionalDataset":
        """Create a 0D dataset from a value and label.

        The scalar class helps resolve an amibiguity in the
        representation of data as a numpy array. This method
        takes either a raw python value or a 1-length array-like
        and returns a Scalar object with the appropriate internal
        representation of the dataset as a scalar.

        Parameters
        ----------
        value : variable
            The value of the scalar.
        label : str
            The label describing the scalar.
        unit : str
            The unit of this scalar.
        source : str

        Returns
        -------
        piblin.data.datasets.Scalar
            A scalar dataset with the given value and label.
        """
        if not np.isscalar(value):  # in case a list is passed
            try:
                value = value[0]
            except IndexError:
                value = value.item()

        if not isinstance(value, np.ndarray):
            value = np.array(value)

        value = np.squeeze(value)

        return cls(dependent_variable_data=value,
                   dependent_variable_names=[label],
                   dependent_variable_units=[unit],
                   independent_variable_data=None,
                   independent_variable_names=None,
                   independent_variable_units=None,
                   source=source)

    @property
    def data(self) -> Set[Tuple[Tuple, Tuple]]:
        """The set of data points of this split dataset."""
        return set(((self.value,), ()))

    @property
    def variable_names(self) -> Tuple[Tuple[str], Tuple[str]]:
        """The names of the dependent and independent variables."""
        return (self.name,), ()

    @property
    def variable_units(self) -> Tuple[Tuple[str], Tuple[str]]:
        """The units of the dependent and independent variables."""
        return (self.unit,), ()

    def to_unambiguous_dataset(self) -> \
            unambiguous_dataset.UnambiguousDataset:
        """Convert this to an unambiguous dataset."""
        return unambiguous_dataset.UnambiguousDataset(
            data=self.data,
            variable_names=self.variable_names,
            variable_units=self.variable_units,
            source=self.source
        )

    @property
    def one_line_description(self) -> str:
        """Return a single-line human-readable representation."""
        return f"{self.name} = {self.value} {self.unit}"

    def _encode_column_labels(self) -> List[str]:
        """A label for this zero-dimensional dataset."""
        return [f"{self.name}({self.unit})"]

    @property
    def value(self) -> float:
        """The value of this zero-dimensional dataset."""
        return float(self.dependent_variable_data)

    @value.setter
    def value(self, value: float) -> None:
        """Change the value of this zero-dimensional dataset."""
        self._dependent_variable_data = np.array(value)

    @property
    def label(self) -> str:
        """The label of this zero-dimensional dataset."""
        return self.dependent_variable_name

    @label.setter
    def label(self, label: str) -> None:
        """Change the label of this zero-dimensional dataset."""
        self.dependent_variable_names[0] = label

    @property
    def name(self) -> str:
        """The name of this zero-dimensional dataset."""
        return self.dependent_variable_name

    @name.setter
    def name(self, name: str) -> None:
        """Change the name of this zero-dimensional dataset."""
        self.dependent_variable_names[0] = name

    @property
    def unit(self) -> str:
        """The unit of the dependent variables."""
        return self.dependent_variable_units[0]

    @unit.setter
    def unit(self, unit: str) -> None:
        self.dependent_variable_units[0] = unit

    @property
    def independent_variable_data(self) -> List[np.ndarray]:
        """The values of independent variables along each axis.

        By definition the zero-dimensional dataset has no independent
        variables, so this list is empty.
        """
        return []

    @property
    def independent_variable_names(self) -> List[str]:
        """The names of the independent variables.

        By definition the zero-dimensional dataset has no independent
        variables, so this list is empty.
        """
        return []

    @property
    def number_of_independent_dimensions(self) -> int:
        """The dimensionality of the independent variables.

        By definition the zero-dimensional dataset has no independent
        variables, so this value is always zero.
        """
        return 0

    @staticmethod
    def compute_average(
            datasets: List["ZeroDimensionalDataset"]
    ) -> "ZeroDimensionalDataset":
        """Compute the average of a set of scalars.

        Parameters
        ----------
        datasets : list of Scalar
            Teh scalars of which the average is to be computed.

        Returns
        -------
        dataset : Scalar
            the average scalar of the provided scalars.
        """
        if len(datasets) == 1:
            return datasets[0]

        label = datasets[0].label
        unit = datasets[0].unit
        for dataset in datasets[1:]:
            if dataset.label != label:
                labels = [dataset.label for dataset in datasets]
                raise ValueError(
                    f"Inconsistent labels for 0D datasets: {labels}"
                )
            if dataset.unit != unit:
                units = [dataset.unit for dataset in datasets]
                raise ValueError(
                    f"Inconsistent units for 0D datasets: {units}"
                )

        return ZeroDimensionalDataset.create(
            value=np.mean(np.array([dataset.value for dataset in datasets])),
            label=label,
            unit=unit)

    @staticmethod
    def compute_variation(
            datasets: List["ZeroDimensionalDataset"]
    ) -> "ZeroDimensionalDataset":
        """Compute the variation across a set of scalars.

        Parameters
        ----------
        datasets : list of Scalar
            Teh scalars over which the variation is to be computed.

        Returns
        -------
        dataset : Scalar
            The variation over the provided scalars.
        """
        if len(datasets) == 1:
            return datasets[0]

        label = datasets[0].label
        unit = datasets[0].unit
        for dataset in datasets[1:]:
            if dataset.label != label:
                labels = [dataset.label for dataset in datasets]
                raise ValueError(
                    f"Inconsistent labels for 0D datasets: {labels}"
                )
            if dataset.unit != unit:
                units = [dataset.unit for dataset in datasets]
                raise ValueError(
                    f"Inconsistent units for 0D datasets: {units}"
                )

        return ZeroDimensionalDataset.create(
            value=np.std(np.array([dataset.value
                                   for dataset in datasets])),
            label=label,
            unit=unit
        )
    
    @classmethod
    def unflatten(cls,
                  column_labels: List[str],
                  values: np.ndarray) -> "ZeroDimensionalDataset":
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
        variable_split_regex = r"(.*\(.*\))"
        dependent_variable_label_regex = r"(\w+)\((\w+)\)"
        column_label_pattern = re.compile(cls.COLUMN_LABEL_REGEX)

        # Check that value is scalar
        if not isinstance(values, np.ndarray):
            values = np.array(values)

        values = np.squeeze(values)
        
        if len(values.shape) != 0:
            raise ValueError("`values` must be a scalar value")

        all_coords = []
        for label in column_labels:

            variable_split_pattern = re.compile(variable_split_regex)
            match = variable_split_pattern.match(label)
            if match is not None:
                dep_label = match.group(1)
            else:
                raise ValueError(
                    f"No match for split: {label}, {variable_split_regex}"
                )

            dependent_variable_label_pattern = \
                re.compile(dependent_variable_label_regex)

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
        pass

    @staticmethod
    def decode_column_labels(column_labels: List[str]):
        pass

    @property
    def dependent_variable_data(self) -> np.ndarray:
        """The dependent variable values for each data point."""
        return self._dependent_variable_data

    @dependent_variable_data.setter
    def dependent_variable_data(self, dependent_variable_data: np.ndarray) -> None:
        self._dependent_variable_data = dependent_variable_data

    @property
    def number_of_dependent_dimensions(self):
        return 1

    @property
    def dependent_variable_names(self):
        """The names of the dependent variables."""
        return self._dependent_variable_names

    @property
    def dependent_variable_units(self):
        """The units of the dependent variables."""
        return self._dependent_variable_units

    def _plot_on_axes(self,
                      axes: matplotlib.axes.Axes,
                      **axes_plotting_kwargs) -> None:
        """Plot this scalar on a given axes with the specified kwargs.

        This method will place a single point on a scatter plot for the
        value of the dataset. When plotting a single dataset this is not
        very informative, but is the best approach when placing many
        datasets on the same plot.

        Parameters
        ----------
        axes : matplotlib.axes.Axes
            The axes on which to plot this scalar.
        axes_plotting_kwargs : dict
            Keywords which will be passed to the scatter method.
            Empty by default.
        """
        axes.scatter(x=0,
                     y=self.dependent_variable_data,
                     cmap=None,
                     norm=None,
                     vmin=None,
                     vmax=None,
                     # alpha=None,
                     plotnonfinite=False,
                     data=None,
                     **axes_plotting_kwargs)

    def _plot_on_axes_with_variation(self,
                                     axes: matplotlib.axes.Axes,
                                     variation: "ZeroDimensionalDataset",
                                     **plot_kwargs) -> None:
        """Visualize the given variation in this dataset.

        This method will draw the dataset along with an error bar showing
        the variation in the value.

        Parameters
        ----------
        axes : matplotlib.axes.Axes
            The axes on which to plot the variation.
        variation
            The variation in the dataset.
        plot_kwargs : dict
            Keywords which will be passed to the scatter method.
            Empty by default.
        """
        axes.errorbar(x=[self.label],
                      y=self.value,
                      yerr=self.value,  # TODO - check this
                      **plot_kwargs)

    def _label_axes(self, axes: matplotlib.axes.Axes) -> None:
        """Add labels to the given axes."""
        axes.set_ylabel(self.dependent_variable_axis_labels[0])

    @classmethod
    def _validate_axes_plotting_kwargs(cls, **axes_plotting_kwargs):
        """Validate the kwargs to be passed to the axes plotting method.

        This class uses the scatter method to plot its points. Setting the
        "c" keyword argument to None makes the colour of the point be
        determined by the "color" keyword argument which is set in the
        superclass to the default if not present in the keyword arguments
        passed to visualize.
        """
        axes_plotting_kwargs = \
            super()._validate_axes_plotting_kwargs(**axes_plotting_kwargs)

        if "c" in axes_plotting_kwargs:
            del axes_plotting_kwargs["c"]

        return axes_plotting_kwargs

    @staticmethod
    def _style_axes(axes: matplotlib.axes.Axes) -> None:
        """Apply the class style to a matplotlib axes.

        When plotting zero-dimensional datasets on an axis, there is no
        information to be used along the x-axis. The value is plotted on
        the y-axis, and the name and unit are used to label the y-axis.
        For this reason, the x-axis is removed entirely from the plot.

        Parameters
        ----------
        axes : matplotlib.axes.Axes
            The Axes object to apply the class style to.
        """
        super(ZeroDimensionalDataset,
              ZeroDimensionalDataset)._style_axes(axes=axes)

        axes.xaxis.set_visible(False)
        axes.spines['bottom'].set_visible(False)
