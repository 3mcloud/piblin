"""
This module is for fixtures that can be used in any sub-package of the data
package, including datasets and organize.
"""
from typing import Tuple
import pytest
import matplotlib.pyplot as plt
import matplotlib.axes
import piblin.data.datasets.abc.split_datasets.zero_dimensional_dataset as zero_dimensional_dataset
import piblin.data.datasets.abc.split_datasets.one_dimensional_dataset as one_dimensional_dataset
import piblin.data.datasets.abc.split_datasets.two_dimensional_dataset as two_dimensional_dataset


@pytest.fixture()
def empty_axes() -> matplotlib.axes.Axes:
    """An empty matplotlib axes to plot on."""
    return plt.subplot()


@pytest.fixture()
def nondefault_boolean_variable_name() -> str:
    """A name for a boolean variable."""
    return "Boolean"


@pytest.fixture()
def nondefault_integer_variable_name() -> str:
    """A name for an integer variable."""
    return "Integer"


@pytest.fixture()
def nondefault_float_variable_name() -> str:
    """A name for a float variable."""
    return "Float"


@pytest.fixture()
def nondefault_xvalue_name():
    return "i0"


@pytest.fixture()
def nondefault_yvalue_name():
    return "d0"


@pytest.fixture()
def nondefault_unit():
    return "units"


@pytest.fixture()
def nondefault_figure_title() -> str:
    """A title for a figure."""
    return "Figure Title"


@pytest.fixture()
def nondefault_axes_title() -> str:
    """A title for an axes."""
    return "Axes Title"


@pytest.fixture()
def nondefault_unit() -> str:
    """A unit for a variable."""
    return "Units"


@pytest.fixture()
def specific_figsize() -> Tuple[int, int]:
    """A figure size for a matplotlib figure."""
    return 10, 10


@pytest.fixture()
def zerod_dataset_false_default_name_and_unit() -> \
        zero_dimensional_dataset.ZeroDimensionalDataset:
    """Create a 0D dataset with value False."""
    return zero_dimensional_dataset.ZeroDimensionalDataset.create(
        value=False
    )


@pytest.fixture()
def zerod_dataset_false_nondefault_name_default_unit(
        nondefault_boolean_variable_name
) -> zero_dimensional_dataset.ZeroDimensionalDataset:
    """Create a 0D dataset with value False and non-default name."""
    return (
        zero_dimensional_dataset.ZeroDimensionalDataset.create(
            value=False,
            label=nondefault_boolean_variable_name,
            unit=None)
    )


@pytest.fixture()
def zerod_dataset_true_default_name_and_unit() -> \
        zero_dimensional_dataset.ZeroDimensionalDataset:
    """Create a 0D dataset with value True."""
    return zero_dimensional_dataset.ZeroDimensionalDataset.create(
        value=True
    )


@pytest.fixture()
def zerod_dataset_true_nondefault_name_default_unit(
        nondefault_boolean_variable_name
) -> zero_dimensional_dataset.ZeroDimensionalDataset:
    """Create a 0D dataset with value False and non-default name."""
    return zero_dimensional_dataset.ZeroDimensionalDataset.create(
        value=True,
        label=nondefault_boolean_variable_name,
        unit=None
    )


@pytest.fixture()
def zero_integer_dataset(
        nondefault_integer_variable_name,
        nondefault_unit
) -> zero_dimensional_dataset.ZeroDimensionalDataset:
    """"""
    return zero_dimensional_dataset.ZeroDimensionalDataset.create(
        value=0,
        label=nondefault_integer_variable_name,
        unit=nondefault_unit
    )


@pytest.fixture()
def zero_float_dataset(
        nondefault_float_variable_name,
        nondefault_unit
) -> zero_dimensional_dataset.ZeroDimensionalDataset:
    """"""
    return zero_dimensional_dataset.ZeroDimensionalDataset.create(
        value=0.0,
        label=nondefault_float_variable_name,
        unit=nondefault_unit
    )


@pytest.fixture()
def oned_dataset_single_point() -> \
        one_dimensional_dataset.OneDimensionalDataset:
    """"""
    return one_dimensional_dataset.OneDimensionalDataset.create(
        x_values=[0.0],
        y_values=[0.0]
    )


@pytest.fixture()
def oned_dataset_multiple_points() -> \
        one_dimensional_dataset.OneDimensionalDataset:
    """"""
    return one_dimensional_dataset.OneDimensionalDataset.create(
        x_values=[0.0, 1.0],
        y_values=[0.0, 0.0]
    )


@pytest.fixture()
def oned_dataset_multiple_points_and_values() -> \
        one_dimensional_dataset.OneDimensionalDataset:
    """"""
    return one_dimensional_dataset.OneDimensionalDataset.create(
        x_values=[0.0, 1.0],
        y_values=[0.0, 1.0]
    )


@pytest.fixture()
def oned_dataset_repeated_values() -> \
        one_dimensional_dataset.OneDimensionalDataset:
    """"""
    return one_dimensional_dataset.OneDimensionalDataset.create(
        x_values=[0.0, 0.0],
        y_values=[0.0, 1.0],
        x_name="x",
        y_name="y"
    )


@pytest.fixture()
def oned_dataset_single_point_labelled(
        nondefault_xvalue_name,
        nondefault_yvalue_name,
        nondefault_unit
) -> one_dimensional_dataset.OneDimensionalDataset:
    """"""
    return one_dimensional_dataset.OneDimensionalDataset.create(
        x_values=[0.0],
        y_values=[0.0],
        x_name=nondefault_xvalue_name,
        x_unit=nondefault_unit,
        y_name=nondefault_yvalue_name,
        y_unit=nondefault_unit
    )


@pytest.fixture()
def twod_dataset_four_points() -> \
        two_dimensional_dataset.TwoDimensionalDataset:
    """"""
    return two_dimensional_dataset.TwoDimensionalDataset.create(
        x_values=[0, 1],
        y_values=[0, 1],
        z_values=[[1, 1], [0, 0]]
    )
