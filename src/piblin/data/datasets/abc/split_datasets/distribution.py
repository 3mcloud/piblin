from typing import List
import numpy as np
import matplotlib.axes
import piblin.data.datasets.abc.split_datasets.one_dimensional_dataset as one_dimensional_dataset
from piblin.data.datasets.abc.split_datasets.two_dimensional_dataset import TwoDimensionalDataset
from piblin.data.datasets.abc.split_datasets.one_dimensional_dataset import OneDimensionalDataset


class Distribution(one_dimensional_dataset.OneDimensionalDataset):
    """A set of datasets to be treated as a distribution.

    This class uses the numpy histogram method to compute the histogram
    of the datasets that it stores.

    Parameters
    ----------
    datasets : List of Dataset
    """
    DEFAULT_X_NAME: str = "Bin Centers"
    """The default name for the independent variable."""
    DEFAULT_Y_NAME: str = "Counts"
    """The default name for the dependent variable."""

    def __init__(self,
                 datasets: List = None,
                 name: str = "",
                 bins: int = 10,
                 density=None,
                 source=None):

        if datasets is None:
            self._datasets = []
        else:
            self._datasets = datasets

        independent_variable_data, dependent_variable_data = self.__compute_variables(bins, density)

        super().__init__(dependent_variable_data=dependent_variable_data,
                         dependent_variable_names=["Counts"],
                         dependent_variable_units=["None"],
                         independent_variable_data=[independent_variable_data],
                         independent_variable_names=[name],
                         independent_variable_units=[self._determine_unit(self._datasets)],
                         source=source)

    def __compute_variables(self, bins, density):
        if all(isinstance(x, OneDimensionalDataset) for x in self._datasets):
            dataset_values = [dataset.value for dataset in self._datasets]
        elif all(isinstance(x, TwoDimensionalDataset) for x in self._datasets):
            dataset_values = [dataset.dependent_variable_data for dataset in self._datasets]
        else:
            dataset_types = set([type(item) for item in self._datasets])
            raise AttributeError(f"is not defined for {dataset_types}")
        hist, bin_edges = np.histogram(a=dataset_values, bins=bins, density=density)
        bin_centers = np.linspace(np.min(bin_edges), np.max(bin_edges), len(hist))

        return bin_centers, hist

    @staticmethod
    def _determine_unit(datasets):
        """Determine the unit of this distribution's independent variables."""
        units = []
        for dataset in datasets:
            if type(dataset) == OneDimensionalDataset:
                units.append(dataset.unit)
            elif type(dataset) == TwoDimensionalDataset:
                units.append(dataset.dependent_variable_unit)
        if len(set(units)) == 1:
            unit = units[0]
        else:
            raise ValueError("Cannot create a distribution from datasets with different units.")
        return unit

    @property
    def datasets(self):
        return self._datasets

    @property
    def bins(self) -> np.ndarray:
        return self._independent_variable_data[0]

    @property
    def counts(self) -> np.ndarray:
        return self._dependent_variable_data

    def _plot_on_axes(self,
                      axes: matplotlib.axes.Axes,
                      **axes_plotting_kwargs):
        """Plot this histogram on the given matplotlib axes.

        Parameters
        ----------
        axes : matplotlib.axes.Axes
            The axes on which to plot this histogram.
        """
        if 'marker' in axes_plotting_kwargs.keys():
            del axes_plotting_kwargs['marker']
        args = []
        axes.bar(self.bins,
                 self.counts,
                 label=f"{self.y_name} ({self.dependent_variable_unit})",
                 *args,
                 **axes_plotting_kwargs)
