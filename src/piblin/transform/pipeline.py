import matplotlib.figure
from typing import Union, List
import collections.abc
import copy
import piblin.data.datasets.abc.dataset as dataset
import piblin.data.data_collections.measurement as measurement
import piblin.data.data_collections.measurement_set as measurement_set
import piblin.data.data_collections.experiment as experiment
import piblin.data.data_collections.experiment_set as experiment_set


class Pipeline(collections.abc.MutableSequence):
    """A series of consecutively applied transforms.

    A pipeline is a list of transforms to be applied linearly,
    i.e. one after the other. Data flows through the pipeline
    to produce transformed data. When applied to a particular
    data collection, the object can either be modified in place
    or a copy can be made and returned instead.
    The pipeline is explicitly a mutable sequence of transforms.

    Parameters
    ----------
    transforms : list of Transform
        The transforms to be applied, in application order.

    Methods
    -------
    from_single_transform
    apply_to
    visualize

    Attributes
    ----------
    transforms -> List of Transform

    Notes
    -----
    Pipeline does not implement the count, index, pop, reverse or sort
    methods typical of mutable sequences as they are not likely to be
    relevant to transforming datasets. The addition operators are
    supported, but multiplication is not as it is unlikely that the
    same single transform needs to be applied many times in succession.
    """
    def __init__(self, transforms: List = None):
        super().__init__()
        if transforms is None:
            self.__transforms = []
        else:
            self.__transforms = transforms

    @classmethod
    def from_single_transform(cls, transform) -> "Pipeline":
        return cls([transform])

    @property
    def transforms(self):
        return self.__transforms

    def apply_to(self,
                 target: Union[dataset.Dataset,
                               measurement.Measurement,
                               measurement_set.MeasurementSet,
                               experiment.Experiment,
                               experiment_set.ExperimentSet],
                 make_copy: bool = True,
                 **kwargs):
        """Apply this pipeline to the given target object.

        Parameters
        ----------
        target : {ExperimentSet, MeasurementSet, Experiment, Measurement, Dataset}
            The target to which to apply this pipeline.
        make_copy : bool
            Whether to make a copy of the target before applying pipeline.
            The default behavior is to make a copy, not affecting the original target.

        Returns
        -------
        target
            The data collection after the application of the transform.
            If make_copy, this will be a different instance of type(target) .
            If not make_copy, this will be the original target instance, but transformed.

        """
        if make_copy:
            target = copy.deepcopy(target)

        for transform in self.__transforms:
            target = transform.apply_to(target,
                                        make_copy=False,
                                        **kwargs)

        return target

    def __call__(self,
                 target: Union[dataset.Dataset,
                               measurement.Measurement,
                               measurement_set.MeasurementSet,
                               experiment.Experiment,
                               experiment_set.ExperimentSet],
                 make_copy: bool = True,
                 **kwargs):
        return self.apply_to(target, make_copy, **kwargs)

    def __len__(self) -> int:
        return len(self.__transforms)

    def __getitem__(self, position):
        if isinstance(position, slice):
            return Pipeline(self.__transforms[position])
        return self.__transforms[position]

    def __delitem__(self, index) -> None:
        del self.__transforms[index]

    def __setitem__(self, index, value) -> None:
        self.__transforms[index] = value

    def insert(self, index: int, value) -> None:
        self.__transforms.insert(index, value)

    def __add__(self, other):
        self.append(other)
        return Pipeline(self.transforms)

    def __radd__(self, other):
        self.append(other)
        return Pipeline(self.transforms)

    def __iadd__(self, other):
        self.append(other)
        return Pipeline(self.transforms)

    def __cmp__(self):
        """Ensure comparisons other than equality are not implemented."""
        return NotImplemented

    def __eq__(self, other):
        """Define equality between two pipeline."""
        if self is other:
            return True

        if self.transforms != other.transforms:
            return False

        return True

    def __repr__(self):
        """Create an eval()-able representation of this pipeline."""
        str_rep = self.__class__.__name__ + "(old_transforms=["
        for transform in self.transforms:
            str_rep += repr(transform)
        return str_rep + "])"

    def visualize(self,
                  target: Union[dataset.Dataset,
                                measurement.Measurement,
                                measurement_set.MeasurementSet,
                                experiment.Experiment,
                                experiment_set.ExperimentSet],
                  make_copy: bool = True,
                  **kwargs):
        """Represent and present the effect of this pipeline to facilitate understanding.

        Applying a pipeline involves iterative application of a list of old_transforms. Each of these old_transforms has a
        visualize method which takes a target, applies the transform and returns a visualization. This approach does not
        lend itself to iterative application as the transformed data is not returned.
        I can think of two approaches: 1) return the transformed data, figure and axes from the visualize method, or
        2) return the transformed data and store the figure and axes in the transform. Either way the pipeline visualize
        method has to deal with assembling these into a single visualization of the pipeline.

        The below implementation is extremely slow because it applies every transform twice and has to copy the target
        each time to avoid problems.
        There is a problem with kwargs where the transform kwargs get passed into matplotlib and it freaks out.
        """
        if make_copy:
            target = copy.deepcopy(target)

        figures: List[matplotlib.figure.Figure] = []
        axes = []
        for transform in self.__transforms:
            f, a = transform.visualize(copy.deepcopy(target), **kwargs)
            figures.append(f)
            axes.append(a)

            target = transform.apply_to(target,
                                        make_copy=False,
                                        **kwargs)

        return target, figures, axes
