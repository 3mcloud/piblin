import copy
from typing import Union, List, Tuple
from abc import ABC, abstractmethod
import matplotlib.axes
import matplotlib.figure
import matplotlib.pyplot as plt

import piblin.transform.pipeline as pipeline
import piblin.data.datasets.abc.dataset as dataset
import piblin.data.data_collections.measurement as measurement
import piblin.data.data_collections.measurement_set as measurement_set
import piblin.data.data_collections.experiment as experiment
import piblin.data.data_collections.experiment_set as experiment_set
import piblin.data.data_collections.data_collection_factory as data_collection_factory


class Transform(ABC):
    """A transform which manipulates data.

    The heart of the cralds transform is its _apply method, which takes a
    target data collection, manipulates it, and returns a different data
    collection. This can involve modification in place or the creation of
    a new data collection on the basis of the input. This method is to be
    overridden by all subclasses. The user uses a transform by invoking
    its apply_to method, which provides the specific logic to apply the
    transform to the data collection it receives. After application, the
    results are combined into the appropriate output data collection and
    returned.

    It is not likely that a user will need to directly subclass this ABC.
    In most cases one of the other ABCs in the package that directly
    subclasses this ABC will be more immediately useful as the _apply_to_*
    methods will not need to be overridden. The data-collection specific
    subclasses DatasetTransform, MeasurementTransform and
    MeasurementSetTransform (and its further subclasses) should be chosen
    between when implementing a transform on the basis of what type of
    data collection the transform's _apply method is applied to.

    A cralds transform has two types of parameter that control its
    behaviour. One type is data-independent, that is, the set of
    parameters that are not computed from a data collection. These are not
    affected by cralds data collections passed into the _apply method, and
    are passed into the transform when created as a list of objects.
    Exposing these to the user of the class for getting/setting by a
    specific name is achieved by explicitly defining properties for a
    subclass of Transform.

    The other type are parameters with explicit dependence on a data
    collection. By default, a transform has no dataset-dependent
    parameters and this property is set to an empty list. A transform can
    be frozen to prevent re-computation of data-dependent parameters when
    applied to new data and also unfrozen, either by setting the value of
    is_frozen directly or by using the "freeze" or "unfreeze" convenience
    methods.

    The apply_to method has an analogous method, visualize, which takes a
    target data collection and visualizes the application of the transform.
    This is key to understanding how a transform is working (or more
    commonly, not working). For a pipeline (a series of transforms to be
    applied in order) to be able to visualize itself, the transformed data
    also need to be returned so that it can be passed into the next
    transform.

    Attributes
    ----------
    data_independent_parameters -> List of float or List of int
        The data-independent parameters of this transform.
    data_dependent_parameters -> List of object
        The data-dependent parameters of this transform.
    is_frozen -> bool
        Whether the data-dependent parameters are to be recomputed or not.

    Methods
    -------
    apply_to(target: Dataset)
        Apply this transform to the given target dataset.
    freeze
        Stop re-computation of this transform's data-dependent parameters.
    unfreeze
        Start re-computation of this transform's data-dependent parameters.
    visualize
        Represent and present the effect of this transform.
    """
    APPLY_TYPE = object
    """The type to which this transform's _apply method may be applied."""

    def __init__(self,
                 data_independent_parameters: List[object] = None,
                 *args,
                 **kwargs):

        if data_independent_parameters is None:
            self._data_independent_parameters = []
        else:
            self._data_independent_parameters = data_independent_parameters

        self._data_dependent_parameters = []
        self._is_frozen = False

    @property
    def data_independent_parameters(self) -> List[object]:
        """Externally-specified parameters for this transform."""
        return self._data_independent_parameters

    @data_independent_parameters.setter
    def data_independent_parameters(
            self,
            data_independent_parameters: List[object]
    ) -> None:
        """Set the externally-specified parameters for this transform."""
        self._data_independent_parameters = data_independent_parameters

    @property
    def data_dependent_parameters(self) -> List[object]:
        """Data-dependent parameters for this transform."""
        return self._data_dependent_parameters

    @data_dependent_parameters.setter
    def data_dependent_parameters(
            self,
            data_dependent_parameters: List[object]
    ) -> None:
        """Set the data-dependent parameters for this transform."""
        self._data_dependent_parameters = data_dependent_parameters

    def compute_data_dependent_parameters(
            self,
            parameter_source: Union[dataset.Dataset,
                                    measurement.Measurement,
                                    measurement_set.MeasurementSet,
                                    experiment.Experiment,
                                    experiment_set.ExperimentSet]
    ) -> List[object]:
        """Compute the data-dependent parameters of this transform.

        Parameters
        ----------
        parameter_source : DataCollection
            The type of this variable will depend on how this transform's
            data-dependent parameters are to be computed.
        """
        return []

    @property
    def is_frozen(self) -> bool:
        """Whether to recompute data-dependent parameters or not."""
        return self._is_frozen

    @is_frozen.setter
    def is_frozen(self, is_frozen: bool) -> None:
        """Set whether to recompute data-dependent parameters or not"""
        self._is_frozen = is_frozen

    def freeze(self) -> None:
        """Prevent re-computation of the data-dependent parameters."""
        self.is_frozen = True

    def unfreeze(self) -> None:
        """Permit re-computation of the data-dependent parameters."""
        self.is_frozen = False

    def apply_to(self, target: Union[dataset.Dataset,
                                     measurement.Measurement,
                                     measurement_set.MeasurementSet,
                                     experiment.Experiment,
                                     experiment_set.ExperimentSet],
                 make_copy: bool = True,
                 **kwargs) -> Union[dataset.Dataset,
                                    measurement.Measurement,
                                    measurement_set.MeasurementSet,
                                    experiment.Experiment,
                                    experiment_set.ExperimentSet]:
        """Apply this transform to a target.

        This is the entry and exit point for an individual transform.
        The return type is any data collection class. There are a set of
        subclasses of this ABC that make the type of the target explicit.
        These subclasses provide implementations of the _apply_to_dataset,
        _apply_to_measurement, _apply_to_measurement_set,
        _apply_to_experiment and _apply_to_experiment_set methods.

        Parameters
        ----------
        target : dataset.Dataset or measurement.Measurement or measurement_set.MeasurementSet or experiment.Experiment
            The target to which to apply this transform.
        make_copy : bool
            Whether to make a copy of the target before applying the transform.
            Default behaviour is to make a copy.

        Returns
        -------
        Dataset or List of Dataset or Measurement or MeasurementSet or Experiment or ExperimentSet
        """
        # TODO - this cannot always be here, it must be called according to
        # the needs of the transform in terms of what type it uses to
        # update its parameters. For example, on receiving an experiment
        # set, a transform may need to update parameters based on each
        # experiment and then apply to all datasets in that experiment.
        if not self.is_frozen:
            # TODO - figure out why this method can't just update self
            self.data_dependent_parameters = \
                self.compute_data_dependent_parameters(
                    parameter_source=target
                )
        # one workaround to this is to choose the Transform subclass based
        # on the most complex of either the type of data collection that
        # the transform is applied to, or the type of data collection from
        # which the data-dependent parameters are to be computed. If the
        # latter is more complex, the parameters can be updated e.g. from
        # an experiment, then the _apply method can iterate over that
        # experiment applying to each dataset. This does burden the user
        # with having to write the iteration which is not in the spirit of
        # the package.
        # If the former is more complex, the iteration still has to be
        # coded in with loop-based updates of the parameters. The former
        # seems much more common (maybe exclusively the case).

        if make_copy:
            target = copy.deepcopy(target)

        if isinstance(target, dataset.Dataset):
            return_value = self._apply_to_dataset(target,
                                                  **kwargs)

            # TODO - apply_to_dataset should be dealing with this - _apply_to_dataset cannot return a list
            if isinstance(return_value, List):
                target = measurement.Measurement(datasets=return_value)
            elif return_value is not None:
                target = return_value

        elif isinstance(target, measurement.Measurement):
            result = self._apply_to_measurement(target,
                                                **kwargs)
            if result:
                target = result

        elif isinstance(target, experiment.Experiment):
            result = self._apply_to_experiment(target,
                                               **kwargs)
            if result:
                target = result

        elif isinstance(target, experiment_set.ExperimentSet):
            result = self._apply_to_experiment_set(target,
                                                   **kwargs)
            if result:
                target = result

        elif isinstance(target, measurement_set.MeasurementSet):
            result = self._apply_to_measurement_set(target,
                                                    **kwargs)
            if result:
                target = result

        else:
            target_type = type(target)
            raise TypeError(f"Provided target cannot be transformed: "
                            f"{target_type}")

        if isinstance(target, experiment_set.ExperimentSet):
            target._update_experiments()

        return target

    @abstractmethod
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
        """Apply this transform to a target dataset."""
        ...

    @abstractmethod
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
        """Apply this transform to a target measurement."""
        ...

    @abstractmethod
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
        """Apply this transform to a target measurement set."""
        ...

    @abstractmethod
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
        """Apply this transform to a target experiment."""
        ...

    @abstractmethod
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
        """Apply this transform to a target experiment set."""
        ...

    @abstractmethod
    def _apply(self,
               target: Union[dataset.Dataset,
                             measurement.Measurement,
                             measurement_set.MeasurementSet,
                             experiment.Experiment,
                             experiment_set.ExperimentSet],
               **kwargs) -> Union[None,
                                  dataset.Dataset,
                                  List[dataset.Dataset],
                                  measurement.Measurement,
                                  measurement_set.MeasurementSet,
                                  experiment.Experiment,
                                  experiment_set.ExperimentSet,
                                  object]:
        """Apply this transform to a target data collection.

        This method will use any externally-specified or data-dependent
        parameters. The data-dependent parameters may have to be recomputed
        from the target. This method must return the transformed data
        collection, whether it is modified in place or a new object is
        created. For final transforms, the return value need not be a
        cralds data collection.

        Parameters
        ----------
        target : Dataset or Measurement or MeasurementSet or Experiment or ExperimentSet
            The type of this variable will be determined by the basis on which this transform is to be applied.
        """
        ...

    @classmethod
    def _is_applicable_to(cls, target: Union[dataset.Dataset,
                                             measurement.Measurement,
                                             measurement_set.MeasurementSet,
                                             experiment.Experiment,
                                             experiment_set.ExperimentSet]) -> bool:
        """Determine whether this transform's _apply method is applicable to the target.

        Parameters
        ----------
        target
            The target to which to apply this transform.
        """
        if not isinstance(target, cls.APPLY_TYPE):
            return False
        else:
            return True

    def __call__(self, target: Union[dataset.Dataset,
                                     measurement.Measurement,
                                     measurement_set.MeasurementSet,
                                     experiment.Experiment, experiment_set.ExperimentSet],
                 make_copy: bool = True,
                 **kwargs):
        """See documentation of the apply_to method."""
        return self.apply_to(target=target,
                             make_copy=make_copy,
                             **kwargs)

    def __add__(self, other: "Transform") -> pipeline.Pipeline:
        """Add this transform to another to create a pipeline."""
        return pipeline.Pipeline([self, other])

    def __radd__(self, other: "Transform") -> pipeline.Pipeline:
        """Right-add this transform to another to create a pipeline."""
        return pipeline.Pipeline([other, self])

    def __str__(self) -> str:
        """Create a human-readable representation of this transform."""
        return f"{self.__class__.__name__} Transform"

    def __repr__(self) -> str:
        """Create an eval()-able representation of this transform."""
        str_rep = (f"{self.__class__.__name__}"
                   f"(data_independent_parameters=[")
        for parameter in self.data_independent_parameters:
            str_rep += repr(parameter) + ", "
        return str_rep[:-2] + "])"

    def visualize(self,
                  target: Union[dataset.Dataset,
                                measurement.Measurement,
                                measurement_set.MeasurementSet,
                                experiment.Experiment,
                                experiment_set.ExperimentSet],
                  make_copy: bool = True,
                  **kwargs
                  ) -> Tuple[
        Union[dataset.Dataset,
              measurement.Measurement,
              measurement_set.MeasurementSet,
              experiment.Experiment,
              experiment_set.ExperimentSet],
        Tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]
    ]:
        """Represent and present the effect of this transform to facilitate understanding.

        Visualizing the effect of a transform on a target data collection requires applying the transform. The
        implementation of visualize for any transform will therefore have to call apply_to on the target it receives.
        This method should not reimplement any part of the _apply method, which means intermediate results need to be
        stored when the transform is applied.

        Parameters
        ----------
        target : Dataset or Measurement or MeasurementSet or Experiment or ExperimentSet
        make_copy : bool
            Whether to make a copy of the target before manipulating it.
        """
        transformed_target = self.apply_to(target=target,
                                           make_copy=make_copy,
                                           **kwargs)

        fig, axes = plt.subplots()

        return transformed_target, (fig, axes)

    # def _combine_results(self,
    #                      transformed_target: Union[dataset.Dataset,
    #                                   measurement.Measurement,
    #                                   measurement_set.MeasurementSet,
    #                                   experiment.Experiment,
    #                                   experiment_set.ExperimentSet],
    #                      ) -> Union[dataset.Dataset,
    #                                measurement.Measurement,
    #                                measurement_set.MeasurementSet,
    #                                experiment.Experiment,
    #                                experiment_set.ExperimentSet]:
    #     """Combine the results of applying the transform to the target.
    #
    #     The result of the application of a transform to a target data
    #     collection can be any data collection. When the type of target
    #     data collection is more structured than the type to which the
    #     _apply function applies, iterative application of the transform
    #     is required and the output is a list of data collections that must
    #     be combined back to a single data collection to be returned.
    #     It is possible to turn all individual returns into measurements
    #     and return the most structured data collection that can be
    #     created from the measurements. For datasets, though, the situation
    #     is slightly more complex.
    #
    #     Parameters
    #     ----------
    #     transformed_target
    #         The data collection to which this transform has been applied.
    #     transformed_target : List
    #         The result of applying the transform iteratively to the target.
    #
    #     Returns
    #     -------
    #
    #     """
    #     return transformed_target
    #     # TODO - this still has to deal with modifying in place versus creating new objects to return
    #     # this method is called at the end of apply_to which just takes
    #     # target and make_copy. if we asked apply_to to make a copy by
    #     # setting make_copy=True, it will deepcopy the entire target before
    #     # doing anything. Therefore whether or not we asked it to make a
    #     # copy, we are passing around a data collection that can be freely
    #     # modified in-place, which will ultimately be returned to caller
    #
    #     # the open question for me is related to being annoyed at having to
    #     # remember to return the modified target from the _apply method.
    #     # Surely if the thing being passed in can be modified in-place then
    #     # there is no need to return it.
    #
    #     # here we could have received any data collection in return
    #
    #     # if results is a list of datasets, the return from this function can be a measurement with the target metadata
    #     # and the results as its datasets.
    #
    #     # if results contains measurements, their metadata will determine what can be returned.
    #     # if they all have the same metadata, that can be combined with the target metadata and a single measurement
    #     # can be returned. Maybe they are all single-dataset replicates of an experiment though? Need to think of
    #     # some examples to understand this. This is a more general question about combining measurements.
    #     # If a set of measurements have the same condition keys and values they are replicates, but instead could be
    #     # combined along their datasets since the conditions are the same for all. In general I guess this means any
    #     # experiment is also a single measurement with all the datasets of all the replicates.
    #
    #     # if they don't all have the same metadata, we have distinct measurements that will share the target metadata
    #     # but differ by their specific metadata. We essentially have our pick of measurement set class to return so
    #     # should return the most structured one possible.
    #     # if it is consistent and tidy is there any reason not to return an experiment set?
    #
    #     # target type | transformed target type
    #     # _____________________________________
    #     # Dataset -> Dataset, just return the Dataset
    #     # Dataset -> List[Dataset], needs to be wrapped in a Measurement or multiple Measurements
    #     # Dataset -> Measurement, just return the Measurement?
    #     # Dataset -> MeasurementSet, just return the MeasurementSet?
    #     target_type = type(transformed_target)
    #     transformed_target_type = type(transformed_target)
    #
    #     if isinstance(transformed_target, list) and all(isinstance(item, dataset.Dataset) for item in transformed_target):
    #         # whatever went in, a list of datasets came back out
    #         if isinstance(transformed_target, dataset.Dataset):
    #             return measurement.Measurement(datasets=transformed_target)
    #         elif isinstance(transformed_target, measurement.Measurement):
    #             datasets = transformed_target.datasets
    #             datasets.extend(transformed_target)
    #             return measurement.Measurement(datasets=datasets)
    #
    #     if all([isinstance(result,
    #                        dataset.Dataset)
    #             for result in transformed_target]):
    #         # all results of applying the dataset transform to this data collection's datasets are datasets
    #         return measurement.Measurement(datasets=transformed_target,
    #                                        conditions=transformed_target.conditions,
    #                                        details=transformed_target.details)
    #
    #     elif all(
    #             [isinstance(result, measurement.Measurement) for result
    #              in transformed_target]):
    #         # all results of applying the dataset transform to this measurement's datasets are measurements
    #
    #         # do all the measurements have the same metadata?
    #         equal_conditions = True
    #         for measurement_ in transformed_target[1:]:
    #             if measurement_.conditions != transformed_target[0].conditions:
    #                 equal_conditions = False
    #
    #         if equal_conditions:
    #             # if so, just flatten them into a new list of datasets for the target measurement
    #             datasets = [dataset_ for measurement_ in transformed_target for
    #                         dataset_ in measurement_.datasets]
    #             return measurement.Measurement(datasets=datasets,
    #                                            conditions=transformed_target.conditions,
    #                                            details=transformed_target.details)
    #         else:
    #             # if not (i.e. the transform added varying condition metadata) distinct measurements must be
    #             # returned, which implies the return type is a measurement set or one of its subclasses
    #             # but how to pick the subclass? why not return the most structured measurement set possible?
    #             # this seems like the right idea in general - turn everything into measurements and feed the factory
    #             return data_collection_factory.DataCollectionFactory.from_measurements(
    #                 transformed_target, merge_redundant=False)
    #
    #     elif all(isinstance(result, measurement_set.MeasurementSet) for
    #              result in transformed_target):
    #
    #         measurements = []
    #         for measurement_set_ in transformed_target:
    #             for measurement_ in measurement_set_.measurements:
    #                 measurements.append(measurement_)
    #
    #         return data_collection_factory.DataCollectionFactory.from_measurements(
    #             measurements, merge_redundant=False)
    #
    #     else:
    #         unsupported_types = [type(result) for result in transformed_target]
    #         raise NotImplementedError(
    #             f"{self.__class__.__name__} cannot deal with return type "
    #             f"from _apply_to_dataset: {unsupported_types}")