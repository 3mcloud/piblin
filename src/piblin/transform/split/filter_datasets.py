from collections.abc import Callable
from typing import Union

import piblin.data.datasets.abc.dataset as dataset_
import piblin.transform.abc.measurement_transform as measurement_transform_
from piblin.data.datasets.abc import dataset as dataset
from piblin.data.data_collections import measurement as measurement, measurement_set as measurement_set, \
    experiment as experiment, experiment_set as experiment_set


class FilterDatasets(measurement_transform_.MeasurementTransform):
    """Remove datasets from a measurement using the filter builtin.

    """

    def _apply(self, target: measurement.Measurement, **kwargs) -> measurement.Measurement:
        """Apply the filter builtin to a target measurement.



        Parameters
        ----------
        target : Measurement
            The measurement whose datasets are to be filtered using the predicate function.

        Returns
        -------
        target : Measurement
            The measurement after filtering with the predicate function.
        """
        target.datasets = list(filter(self.function, target.datasets))
        return target

    def __init__(self, function: Callable[[dataset_.Dataset], bool]):
        super().__init__(data_independent_parameters=[function])

        self._function = function

    @property
    def function(self):
        return self._function
