from collections.abc import Callable
from piblin.data.data_collections import measurement as measurement_
from piblin.data.data_collections import measurement_set as measurement_set_
from piblin.transform.abc.measurement_set_transform import MeasurementSetTransform


class FilterMeasurements(MeasurementSetTransform):
    """Remove measurements from a measurement set using the filter builtin.

    This class wraps python's filter functionality, which uses a predicate to remove
    all entries from an iterable that are not true for that predicate. This is an
    effective splitting into true and false sets. In practice, the iterable is a
    measurement set, and the predicate function must take a measurement.

    Parameters
    ----------
    function : Callable
        The function to be used in the filter.
    """
    def __init__(self, function: Callable[[measurement_.Measurement], bool]):
        super().__init__(data_independent_parameters=[function])

        self._function = function

    @property
    def function(self):
        return self._function

    def _apply(self,
               target: measurement_set_.MeasurementSet,
               **kwargs) -> measurement_set_.MeasurementSet:
        """Apply the filter builtin to a target measurement set.


        Parameters
        ----------
        target : MeasurementSet
            The measurements to be filtered using the predicate function.

        Returns
        -------
        target : MeasurementSet
            The measurements after filtering with the predicate function.
        """
        target.measurements = list(filter(self.function, target.measurements))

        return target
