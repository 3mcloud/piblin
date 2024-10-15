import matplotlib.axes
import numpy as np
from piblin.data.datasets.abc.split_datasets.one_dimensional_dataset import OneDimensionalDataset


# TODO - get rid of this in favor of the distribution class
class Histogram(OneDimensionalDataset):
    """A histogram.

    Attributes
    ----------
    bins -> np.ndarray
        The centers of the bins of this histogram.
    counts -> np.ndarray
        The counts of the bins of this histogram.
    """
    DEFAULT_X_NAME: str = "Bin Centers"
    """The default name for the independent variable."""
    DEFAULT_Y_NAME: str = "Counts"
    """The default name for the dependent variable."""

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
        args = []

        axes.bar(self.bins,
                 self.counts,
                 label=f"{self.y_name} ({self.dependent_variable_unit})",
                 *args,
                 **axes_plotting_kwargs)
