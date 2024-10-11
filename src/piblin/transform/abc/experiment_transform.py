from typing import Union
from abc import ABC
import piblin.transform.abc.transform as transform
import piblin.data.datasets.abc.dataset as dataset
import piblin.data.data_collections.measurement as measurement
import piblin.data.data_collections.measurement_set as measurement_set
import piblin.data.data_collections.experiment as experiment
import piblin.data.data_collections.experiment_set as experiment_set


class ExperimentTransform(transform.Transform, ABC):
    """A transform that applies on a measurement basis."""

    def _apply_to_dataset(self,
                          target: dataset.Dataset,
                          **kwargs) -> Union[None,
                                             dataset.Dataset,
                                             measurement.Measurement,
                                             measurement_set.MeasurementSet,
                                             experiment.Experiment,
                                             experiment_set.ExperimentSet,
                                             object]:
        """"""
        experiment_ = experiment.Experiment(
            measurements=[measurement.Measurement(datasets=[target])]
        )
        return self._apply(experiment_, **kwargs)

    def _apply_to_measurement(self,
                              target: measurement.Measurement,
                              **kwargs) -> Union[None,
                                                 dataset.Dataset,
                                                 measurement.Measurement,
                                                 measurement_set.MeasurementSet,
                                                 experiment.Experiment,
                                                 experiment_set.ExperimentSet,
                                                 object]:
        """"""
        experiment_ = experiment.Experiment(measurements=[target])
        return self._apply(experiment_, **kwargs)

    def _apply_to_measurement_set(self,
                                  target: measurement_set.MeasurementSet,
                                  **kwargs) -> Union[None,
                                                     dataset.Dataset,
                                                     measurement.Measurement,
                                                     measurement_set.MeasurementSet,
                                                     experiment.Experiment,
                                                     experiment_set.ExperimentSet,
                                                     object]:
        """"""
        return self._apply_to_experiment_set(
            experiment_set.ExperimentSet.from_measurement_set(target)
        )

    def _apply_to_experiment(self,
                             target: experiment.Experiment,
                             **kwargs) -> Union[None,
                                                dataset.Dataset,
                                                measurement.Measurement,
                                                measurement_set.MeasurementSet,
                                                experiment.Experiment,
                                                experiment_set.ExperimentSet,
                                                object]:
        """"""
        return self._apply(target, **kwargs)

    def _apply_to_experiment_set(self,
                                 target: experiment_set.ExperimentSet,
                                 **kwargs) -> Union[None,
                                                    dataset.Dataset,
                                                    measurement.Measurement,
                                                    measurement_set.MeasurementSet,
                                                    experiment.Experiment,
                                                    experiment_set.ExperimentSet,
                                                    object]:
        """"""
        experiments = []
        for experiment_ in target:
            experiments.append(self._apply_to_experiment(experiment_))

        target.measurements = [measurement_
                               for experiment_ in experiments
                               for measurement_ in experiment_.measurements
                               ]

        target._update_experiments()
        return target
