from abc import ABC
from typing import Union
import piblin.transform.abc.transform as transform
import piblin.data.datasets.abc.dataset as dataset
import piblin.data.data_collections.measurement as measurement
import piblin.data.data_collections.measurement_set as measurement_set
import piblin.data.data_collections.experiment as experiment
import piblin.data.data_collections.experiment_set as experiment_set


class MeasurementSetTransform(transform.Transform, ABC):
    """A transform that applies to a measurement set."""
    def _apply_to_dataset(
            self,
            target: dataset.Dataset,
            **kwargs
    ) -> Union[None,
               dataset.Dataset,
               measurement.Measurement,
               measurement_set.MeasurementSet,
               experiment.Experiment,
               experiment_set.ExperimentSet,
               object]:
        """Apply this transform to the provided target dataset.

        In order to apply this measurement set transform to a single
        dataset, the dataset must be wrapped in a measurement set.
        """
        measurement_set_ = measurement_set.MeasurementSet(
            measurements=[measurement.Measurement(datasets=[target])]
        )
        return self._apply(measurement_set_, **kwargs)

    def _apply_to_measurement(
            self,
            target: measurement.Measurement,
            **kwargs
    ) -> Union[None,
               dataset.Dataset,
               measurement.Measurement,
               measurement_set.MeasurementSet,
               experiment.Experiment,
               experiment_set.ExperimentSet,
               object]:
        """Apply this transform to the provided target measurement.

        In order to apply this measurement set transform to a single
        measurement, the measurement must be wrapped in a measurement set.
        """
        measurement_set_ = measurement_set.MeasurementSet(
            measurements=[target]
        )
        return self._apply(measurement_set_, **kwargs)

    def _apply_to_measurement_set(
            self,
            target: measurement_set.MeasurementSet,
            **kwargs
    ) -> Union[None,
               dataset.Dataset,
               measurement.Measurement,
               measurement_set.MeasurementSet,
               experiment.Experiment,
               experiment_set.ExperimentSet,
               object]:
        """Apply this transform to the provided target measurement set."""
        result = self._apply(target, **kwargs)
        if result:
            target = result

        return target

    def _apply_to_experiment(
            self,
            target: experiment.Experiment,
            **kwargs
    ) -> Union[None,
               dataset.Dataset,
               measurement.Measurement,
               measurement_set.MeasurementSet,
               experiment.Experiment,
               experiment_set.ExperimentSet,
               object]:
        """Apply this transform to the provided target experiment."""
        return self._apply(target, **kwargs)

    def _apply_to_experiment_set(
            self,
            target: experiment_set.ExperimentSet,
            **kwargs
    ) -> Union[None,
               dataset.Dataset,
               measurement.Measurement,
               measurement_set.MeasurementSet,
               experiment.Experiment,
               experiment_set.ExperimentSet,
               object]:
        """Apply this transform to the provided target experiment set."""
        return self._apply_to_measurement_set(target,
                                              **kwargs)
