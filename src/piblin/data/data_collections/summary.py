from typing import List, Dict, Union, Tuple
import matplotlib.axes
import matplotlib.figure
import numpy as np
import piblin.data.datasets.abc.dataset as dataset_
import piblin.data.data_collections.measurement as measurement_


class SummaryMeasurement(measurement_.Measurement):
    """A single measurement summarizing a set of measurements.

    Parameters
    ----------
    average_datasets : list of Dataset
        Datasets representative of the datasets of the summarized set of measurements.
    variation_datasets : list of Dataset
        Datasets representative of the variation among the datasets of the summarized set of measurements.
    conditions : dict
        The conditions shared by the summarized set of measurements.
    details : dict
        The details shared by the summarized set of measurements.
    """
    def __init__(self,
                 average_datasets: List[dataset_.Dataset] = None,
                 variation_datasets: List[dataset_.Dataset] = None,
                 conditions: Dict[str, object] = None,
                 details: Dict[str, object] = None):

        super().__init__(datasets=average_datasets,
                         conditions=conditions,
                         details=details)

        self._variation_datasets = variation_datasets

    @property
    def variation_datasets(self):
        return self._variation_datasets

    @variation_datasets.setter
    def variation_datasets(self, variation_datasets):
        self._variation_datasets = variation_datasets

    @property
    def average_datasets(self):
        return self._datasets

    @average_datasets.setter
    def average_datasets(self, average_datasets):
        self._datasets = average_datasets

    def visualize(self,
                  axes: Union[matplotlib.axes.Axes, List[matplotlib.axes.Axes]] = None,
                  expand_datasets: bool = True,
                  include_text: bool = True,
                  figure_title: str = None,
                  total_figsize: Tuple[int] = None,
                  dataset_colors: list = None,
                  **axes_plotting_kwargs) -> Tuple[matplotlib.figure.Figure,
                                                   Union[matplotlib.axes.Axes,
                                                List[matplotlib.axes.Axes]]]:
        """Visualize this measurement's datasets.

        Create a textual representation of the conditions and details of this
        measurement, along with a visual representation of of its dataset.

        Parameters
        ----------
        axes : matplotlib.axes.Axes or list of matplotlib.axes.Axes
            One or more matplotlib axes on which to plot this measurement's datasets.
            The default of None results in creation of a figure and axes.
        expand_datasets : bool
            Whether to plot each dataset on its own axes object.
            Default is to do this, as multiple y-axes is a special case.
        include_text : bool
            Whether to print this measurement's text representation.
        figure_title : str
            A title for the plot, overriding the metadata title.
        total_figsize : tuple
            A tuple of 2 numbers setting the total figure size.
        dataset_colors : list of Color
            List of colors for the plotted datasets.

        Returns
        -------
        fig : matplotlib.figure.Figure
            A matplotlib figure visualizing this measurement.
        axes : matplotlib.axes.Axes or list of matplotlib.axes.Axes
            The axes of the matplotlib figure.
        """
        if not expand_datasets:
            if not self.has_collapsible_datasets():
                expand_datasets = True
                print(f"Warning: Datasets are not collapsible.: {self.dataset_types}")

        fig: matplotlib.figure.Figure
        axes: Union[matplotlib.axes.Axes, List[matplotlib.axes.Axes]]

        fig, axes = self._setup_fig_and_axes(axes=axes,
                                             expand_datasets=expand_datasets,
                                             total_figsize=total_figsize,
                                             figure_title=figure_title)

        if expand_datasets:  # plot each dataset on its own axes object

            # deal with the edge case of a single, expanded dataset
            if isinstance(axes, matplotlib.axes.Axes):
                axes = [axes]

            i = 0
            for j, (mean_dataset, variation_dataset, axis) in enumerate(zip(self.datasets,
                                                                                 self.variation_datasets,
                                                                                 axes)):

                if "color" not in axes_plotting_kwargs.keys():
                    if dataset_colors is not None:
                        axes_plotting_kwargs["color"] = dataset_colors[i]
                        i += 1

                mean_dataset.visualize(axes=axis,
                                       variation=variation_dataset,
                                       **axes_plotting_kwargs)

        elif not expand_datasets:  # plot each dataset on a single axes object

            assert isinstance(axes, matplotlib.axes.Axes)
            twin_axes = []
            for i in range(self.num_datasets - 1):
                t = axes.twinx()
                twin_axes.append(t)

            if "color" not in axes_plotting_kwargs.keys():
                if dataset_colors is not None:
                    axes_plotting_kwargs["color"] = dataset_colors[0]

            # the first dataset goes on the normal axes
            self.datasets[0].visualize(axes=axes,
                                       include_text=False,
                                       **axes_plotting_kwargs)
            i = 1
            for mean_dataset in self.datasets[1:]:

                if "color" not in axes_plotting_kwargs.keys():
                    if dataset_colors is not None:
                        axes_plotting_kwargs["color"] = dataset_colors[i]
                        i += 1

                mean_dataset.visualize(axes=twin_axes[i - 1],
                                       include_text=False,
                                       **axes_plotting_kwargs)

                shift = 1 + ((i - 1) * 0.25)
                twin_axes[i - 1].spines.right.set_position(("axes", shift))
                twin_axes[i - 1].spines['right'].set_visible(True)
                twin_axes[i - 1].spines['right'].set_color("k")

                min_val = np.amin(mean_dataset.dependent_variable_data)
                max_val = np.amax(mean_dataset.dependent_variable_data)

                twin_axes[i - 1].set_ylim(min_val - min_val * 0.1,
                                          max_val + max_val * 0.1)

                i += 1

            if axes.get_lines():
                axes_legend_handles = [axes.get_lines()[0]]
                for twin_axis in twin_axes:
                    axes_legend_handles.append(twin_axis.get_lines()[0])
                axes.legend(handles=axes_legend_handles)

        return fig, axes
