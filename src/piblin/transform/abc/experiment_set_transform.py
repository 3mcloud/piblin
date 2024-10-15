from abc import ABC
import piblin.data.datasets.abc.dataset as dataset
import piblin.data.data_collections.measurement as measurement
import piblin.data.data_collections.measurement_set as measurement_set
import piblin.data.data_collections.experiment as experiment
import piblin.data.data_collections.experiment_set as experiment_set
import piblin.transform.abc.transform as transform


class ExperimentSetTransform(transform.Transform, ABC):
    """A transform that applies on an experiment set basis."""
    def _apply_to_dataset(self, target: dataset.Dataset, **kwargs):
        """Apply this transform to the provided target dataset."""
        measurement_set_ = measurement_set.MeasurementSet(measurements=[measurement.Measurement(datasets=[target])])
        raise ValueError(f"Cannot apply to {type(target)}")

    def _apply_to_measurement(self, target: measurement.Measurement, **kwargs):
        """Apply this transform to the provided target measurement."""
        measurement_set_ = measurement_set.MeasurementSet(measurements=[target])
        raise ValueError(f"Cannot apply to {type(target)}")

    def _apply_to_measurement_set(self, target: measurement_set.MeasurementSet, **kwargs):
        """Apply this transform to the provided target measurement set."""
        raise ValueError(f"Cannot apply to {type(target)}")

    def _apply_to_experiment(self, target: experiment.Experiment, **kwargs):
        """Apply this transform to the provided target experiment."""
        raise ValueError(f"Cannot apply to {type(target)}")

    def _apply_to_experiment_set(self, target: experiment_set.ExperimentSet, **kwargs):
        """Apply this transform to the provided target experiment set."""
        return self._apply(target, **kwargs)
