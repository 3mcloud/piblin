import matplotlib.axes
import piblin.data.datasets.abc.unambiguous_datasets.unambiguous_dataset as unambiguous_dataset


class OneDimensionalUnambiguousDataset(unambiguous_dataset.UnambiguousDataset):
    def _plot_on_axes_with_variation(self, axes: matplotlib.axes.Axes, variation, **plot_kwargs) -> None:
        pass

    def _label_axes(self, axes: matplotlib.axes.Axes) -> None:
        pass

    def _plot_on_axes(self, axes: matplotlib.axes.Axes, **axes_plotting_kwargs) -> None:
        pass
