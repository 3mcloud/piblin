from typing import Callable

from piblin.transform.abc import \
    dataset_transform as dataset_transform
from piblin.transform.abc.lambda_transform.lambda_transform import LambdaTransform


class DatasetLambdaTransform(
    LambdaTransform,
    dataset_transform.DatasetTransform
):
    """Apply a custom transform function to a dataset."""
    def __init__(self,
                 transform_function: Callable,
                 data_independent_parameters=None,
                 *args,
                 **kwargs):

        LambdaTransform.__init__(self,
                                 transform_function=transform_function)

        dataset_transform.DatasetTransform.__init__(
            self,
            data_independent_parameters=data_independent_parameters,
            *args,
            **kwargs
        )
