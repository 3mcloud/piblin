import copy
from typing import Union, List
from abc import ABC
import piblin.transform.abc.transform as transform
import piblin.data.datasets.abc.dataset as dataset
import piblin.data.data_collections.measurement as measurement
import piblin.data.data_collections.measurement_set as measurement_set
import piblin.data.data_collections.experiment as experiment
import piblin.data.data_collections.experiment_set as experiment_set


class MeasurementTransform(transform.Transform, ABC):
    """A transform that applies on a measurement basis."""
    def _apply(self,
               target: measurement.Measurement,
               **kwargs) -> Union[None,
                                  dataset.Dataset,
                                  List[dataset.Dataset],
                                  measurement.Measurement,
                                  measurement_set.MeasurementSet,
                                  experiment.Experiment,
                                  experiment_set.ExperimentSet,
                                  object]:
        """Apply this transform to a target measurement."""
        ...

    def _apply_to_dataset(self,
                          target: dataset.Dataset,
                          **kwargs) -> Union[None,
                                             dataset.Dataset,
                                             measurement.Measurement,
                                             measurement_set.MeasurementSet,
                                             experiment.Experiment,
                                             experiment_set.ExperimentSet,
                                             object]:
        """Apply this transform to the provided target dataset.

        In order to apply this measurement transform to a dataset, the
        dataset must be wrapped with an otherwise empty (meaning no details
        or conditions) measurement prior to the calling of _apply.
        """
        measurement_ = measurement.Measurement(datasets=[target])
        return self._apply(measurement_, **kwargs)

    def _apply_to_measurement(self,
                              target: measurement.Measurement,
                              **kwargs) -> Union[None,
                                                 dataset.Dataset,
                                                 measurement.Measurement,
                                                 measurement_set.MeasurementSet,
                                                 experiment.Experiment,
                                                 experiment_set.ExperimentSet,
                                                 object]:
        """Apply this transform to the provided target measurement."""
        # TODO - this insists that you return the transformed target, i.e. no in-place
        return self._apply(target, **kwargs)

    def _apply_to_measurement_set(self,
                                  target: measurement_set.MeasurementSet,
                                  **kwargs) -> Union[None,
                                                     dataset.Dataset,
                                                     measurement.Measurement,
                                                     measurement_set.MeasurementSet,
                                                     experiment.Experiment,
                                                     experiment_set.ExperimentSet,
                                                     object]:
        """Apply this transform to the provided target measurement set."""
        # TODO - the iteration is good, get the rest working
        measurements = []
        for measurement_ in target:
            return_value = self._apply_to_measurement(copy.deepcopy(measurement_), **kwargs)
            if return_value is not None:
                measurements.append(return_value)
            else:
                measurements.append(measurement_)

        target.measurements = measurements

        return target

    def _apply_to_experiment(self,
                             target: experiment.Experiment,
                             **kwargs) -> Union[None,
                                                dataset.Dataset,
                                                measurement.Measurement,
                                                measurement_set.MeasurementSet,
                                                experiment.Experiment,
                                                experiment_set.ExperimentSet,
                                                object]:
        """Apply this transform to the provided target experiment."""
        for replicate in target:
            self._apply_to_measurement(replicate, **kwargs)

        return target

    def _apply_to_experiment_set(self,
                                 target: experiment_set.ExperimentSet,
                                 **kwargs) -> Union[None,
                                                    dataset.Dataset,
                                                    measurement.Measurement,
                                                    measurement_set.MeasurementSet,
                                                    experiment.Experiment,
                                                    experiment_set.ExperimentSet,
                                                    object]:
        """Apply this transform to the provided target experiment set."""
        for experiment_ in target:
            self._apply_to_experiment(experiment_, **kwargs)

        return target
