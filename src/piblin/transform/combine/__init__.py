from typing import Set
import piblin.data.data_collections.measurement as measurement_


class CombineMeasurements(object):

    def _apply(self, measurements: Set[measurement_.Measurement]):
        """Combine measurements into a data collection.

        Given a set of measurements to be combined into the appropriate
        data collection.

        Parameters
        ----------
        measurements : set of Measurement
            The measurements to be combined into a data collection.
        """
        ...
