from typing import Union, Tuple, List
from abc import ABC, abstractmethod
import matplotlib.axes
import matplotlib.figure
import matplotlib.pyplot as plt

import piblin.data.datasets.abc.dataset
import piblin.transform.abc.transform as transform
import piblin.data.datasets.abc.dataset as dataset
import piblin.data.data_collections.measurement as measurement
import piblin.data.data_collections.measurement_set as measurement_set
import piblin.data.data_collections.experiment as experiment
import piblin.data.data_collections.experiment_set as experiment_set
import piblin.data.data_collections.data_collection_factory as data_collection_factory


class DatasetTransform(transform.Transform, ABC):
    """A transform that applies on a per-dataset basis.

    The simplest data collection to which a dataset transform may be
    applied is a dataset. Applying a dataset transform to more complex
    data collections is achieved by iteration, i.e. for a measurement, the
    application is done over each dataset. In a measurement set, iteration
    is over measurements and then datasets. A given dataset transform
    can return an instance of any other data collection from its _apply
    method, although a single dataset or list thereof is most common. The
    option to return a list of datasets greatly simplifies thinking about
    transforms that split a dataset into constituent parts, justifying its
    inclusion as a possible return type.

    Attributes
    ----------
    data_independent_parameters -> List of float or List of int
        The data-independent parameters of this transform.
    data_dependent_parameters -> List of object
        The data-dependent parameters of this transform.
    is_frozen -> bool
        Whether this transform's data-dependent parameters are to be
        recomputed or not.

    Methods
    -------
    apply_to(target: Dataset)
        Apply this transform to the given target dataset.
    freeze
        Prevent re-computation of this transform's data-dependent parameters.
    unfreeze
        Permit re-computation of this transform's data-dependent parameters.
    """

    APPLY_TYPE = piblin.data.datasets.abc.dataset.Dataset
    """The type to which this transform's _apply method may be applied."""

    @abstractmethod
    def _apply(self,
               target: dataset.Dataset,
               **kwargs) -> Union[None,
                                  dataset.Dataset,
                                  List[dataset.Dataset],
                                  measurement.Measurement,
                                  measurement_set.MeasurementSet,
                                  experiment.Experiment,
                                  experiment_set.ExperimentSet,
                                  object]:
        """Apply this transform to a target dataset.

        Any object can be returned by this transform. If the target is
        modified in-place, nothing (ie. None) should be returned. If a
        new object is created by the transform, it should be returned and
        will be dealt with based on data collection type if it is a cralds
        object. If not a cralds data collection object, the object should
        be returned and cannot be passed to another step in a transform.
        However, if the transform is not part of (or the final step in) a
        pipeline, any object can be returned.

        This method is intentionally defined with minimal restrictions as
        it will be overridden in every user-defined dataset transform.

        Returns
        -------
        None or Dataset or List[Dataset] or Measurement
            Returns None if the target dataset was altered in-place.
            # TODO - support other return types
        """
        ...

    def _apply_to_dataset(self,
                          target: dataset.Dataset,
                          **kwargs) \
            -> Union[None,
                     dataset.Dataset,
                     measurement.Measurement,
                     measurement_set.MeasurementSet,
                     experiment.Experiment,
                     experiment_set.ExperimentSet,
                     object]:
        """Apply this per-dataset transform to the provided target dataset.

        Parameters
        ----------
        target : dataset.Dataset
            The dataset to which to apply this transform.

        Returns
        -------
        None or Dataset or List[Dataset] or Measurement
            Returns None if the target was altered in-place.
        # TODO - support other return types
        """
        result = self._apply(target, **kwargs)
        return result


    def _apply_to_measurement(self,
                              target: measurement.Measurement,
                              **kwargs) \
            -> Union[None, measurement.Measurement]:
        """Apply this transform to the provided target measurement.

        This transform is applied on a per-dataset basis, so to apply it
        to a measurement, it is iteratively applied to every dataset in
        that measurement. This is implicitly the same as splitting the
        measurement into a set of one-dataset measurements and applying
        the transform to each. The result is a list of data collections
        (one per target dataset) returned by the transform. These are
        combined into a single data collection to be returned as the
        result of applying the dataset transform to the target measurement.

        Parameters
        ----------
        target : measurement.Measurement
            The measurement to which to apply this transform.

        Returns
        -------
        measurement.Measurement
            The data collection produced by transforming the target.
        # TODO - support other return types
        """
        results: List = []
        for i, target_dataset in enumerate(target.datasets):

            return_value = self._apply_to_dataset(target=target_dataset,
                                                  **kwargs)
            if return_value is None:
                results.append(target_dataset)
            elif isinstance(return_value, List):  # list of datasets or measurements
                results.extend(return_value)
            else:
                results.append(return_value)

        if all(isinstance(result, dataset.Dataset) for result in results):
            target.datasets = results
        elif all(isinstance(result, measurement.Measurement) for result in results):
            return measurement_set.MeasurementSet(measurements=results,
                                                  merge_redundant=False)
        else:
            raise TypeError(
                f"Result of transform not supported: "
                f"DatasetTransform cannot return {results}"
            )

        return target

    def _apply_to_measurement_set(self,
                                  target: measurement_set.MeasurementSet,
                                  **kwargs) \
            -> Union[None, measurement_set.MeasurementSet]:
        """Apply this transform to the provided target measurement set.

        This transform is applied on a per-dataset basis, so to apply it to
        a measurement set, it is iteratively applied to every
        measurement in that measurement set (and then to every dataset in
        that measurement). This is implicitly the same as splitting the
        measurement set into measurements, and each measurement into a set
        of one-dataset measurements, then applying the transform to each.
        The result is a list of data collections (one per target
        measurement) returned by the transform. These are combined into
        a single data collection to be returned as the result of applying
        the dataset transform to the target measurement set.

        Parameters
        ----------
        target : measurement_set.MeasurementSet
            The measurement set to which to apply this transform.

        Returns
        -------
        dataset.Dataset or
        measurement.Measurement or
        measurement_set.MeasurementSet or
        experiment.Experiment or
        experiment_set.ExperimentSet
            The data collection produced by transforming the target.
        """
        results = []
        for measurement_ in target.measurements:
            return_value = self._apply_to_measurement(
                target=measurement_,
                **kwargs
            )
            if isinstance(return_value, measurement.Measurement):
                results.append(return_value)
            elif return_value is None:
                results.append(measurement_)

        if len(results) == 0:
            target.measurements = results
            return target

    def _apply_to_experiment(self,
                             target: experiment.Experiment,
                             **kwargs) \
            -> Union[dataset.Dataset,
                     measurement.Measurement,
                     measurement_set.MeasurementSet,
                     experiment.Experiment,
                     experiment_set.ExperimentSet]:
        """Apply this transform to the provided target experiment.

        An experiment is a tidy, consistent measurement set where all
        measurements have equal values for shared condition metadata
        keys. A dataset transform therefore cannot tell the difference
        between a measurement set and an experiment because metadata
        does not reach the _apply method. This method defers to the
        _apply_to_measurement_set method.

        Parameters
        ----------
        target : experiment.Experiment
            The experiment to which to apply this transform.
        """
        return self._apply_to_measurement_set(target)

    def _apply_to_experiment_set(self,
                                 target: experiment_set.ExperimentSet,
                                 **kwargs) -> Union[dataset.Dataset,
                                                    measurement.Measurement,
                                                    measurement_set.MeasurementSet,
                                                    experiment.Experiment,
                                                    experiment_set.ExperimentSet]:
        """Apply this transform to the provided target experiment set.

        See _apply_to_experiment for reasoning behind deferring to
        _apply_to_measurement_set.

        Parameters
        ----------
        target : experiment_set.ExperimentSet
            The experiment set to which to apply this transform.
        """
        return self._apply_to_measurement_set(target, **kwargs)

    def visualize(self,  # analog of apply_to
                  target: Union[dataset.Dataset,
                                measurement.Measurement,
                                measurement_set.MeasurementSet,
                                experiment.Experiment,
                                experiment_set.ExperimentSet],
                  make_copy: bool = True,
                  total_figsize: Tuple[int, int] = None,
                  **kwargs) -> Tuple[matplotlib.figure.Figure,
                                     matplotlib.axes.Axes]:
        """Represent and present the effect of this dataset transform to facilitate understanding.

        Can we just by default visualize pre- and post- application?
        Will get a fig and axes for pre- and post- that would need to be combined.
        This approach stops you plotting parameters.

        Can the class have visualize_dataset, visualize_measurement, ... methods?
        This would give you enough granularity to create pre- and post- with parameters drawn for every dataset.
        Only works for a DatasetTransform, others will need more consideration.

        If the target is a dataset, no iteration will happen.
        If the target is a measurement, the transform will iterate over its datasets.
        If the

        Parameters
        ----------
        target
        make_copy : bool
        total_figsize : tuple of int

        Returns
        -------
        matplotlib.figure.Figure
        matplotlib.axes.Axes
        """
        if isinstance(target, dataset.Dataset):
            return self._visualize_dataset(target, total_figsize, **kwargs)
        elif isinstance(target, experiment_set.ExperimentSet):
            for experiment_ in target:
                for replicate_measurement in experiment_:
                    for dataset_ in replicate_measurement.datasets:
                        f, a = self._visualize_dataset(dataset_, total_figsize, **kwargs)
        else:
            raise ValueError(f"{self.__class__.__name__} cannot visualize self on object of type {type(target)}.")

    def _visualize_dataset(self,
                           target: dataset.Dataset,
                           total_figsize: Tuple[int],
                           **kwargs) -> Tuple[matplotlib.figure.Figure,
                                              matplotlib.axes.Axes]:
        """Visualize the action of this transform on a dataset.

        Parameters
        ----------
        target : dataset.Dataset
            The dataset to visualize the action of this transform upon.

        Returns
        -------
        matplotlib.figure.Figure
            A figure visualizing the action of this transform on the given dataset.
        matplotlib.axes.Axes
            the axes of the figure.
        """
        f, a = plt.subplots(1, 2)

        if total_figsize is None:
            total_figsize = list(target.DEFAULT_FIGURE_SIZE)
            total_figsize[0] *= 2
        f.set_size_inches(total_figsize)

        target.visualize(axes=a[0], **kwargs)
        result = self.apply_to(target)
        result.visualize(axes=a[1], **kwargs)

        f.set_tight_layout(tight=None)

        return f, a

    def _visualize_measurement(self, target: measurement.Measurement, **kwargs) -> Tuple[matplotlib.figure.Figure,
                                                                                         matplotlib.axes.Axes]:
        ...
