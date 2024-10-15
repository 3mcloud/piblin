from abc import ABC
from typing import Callable

from piblin.transform.abc import transform as transform


class LambdaTransform(transform.Transform, ABC):
    """Apply a custom transform function to data collection.

    The provided function will be passed a cralds data collection to
    transform, and must return a valid cralds data collection. This class
    defines the behaviour of transform application, but is abstract as a
    concrete subclass must be used to ensure correct application to any
    data collection.

    Parameters
    ----------
    transform_function : Callable
        The custom transform function
    """
    def __init__(self,
                 transform_function: Callable,
                 *args,
                 **kwargs
                 ):

        self._transform_function = transform_function

    def _apply(self, dataset_, **kwargs):
        """Applies custom function to dataset_.

        Parameters
        ----------
        dataset_ : piblin.data.Dataset

        Returns
        -------
        piblin.data.Dataset or list of piblin.data.Dataset
            The modified collection after the transform.
        """
        return self._transform_function(dataset_, **kwargs)
