from typing import Callable

from piblin.transform.abc import \
    measurement_set_transform as measurement_set_transform
from piblin.transform.abc.lambda_transform.lambda_transform import LambdaTransform


class MeasurementSetLambdaTransform(
    LambdaTransform,
    measurement_set_transform.MeasurementSetTransform
):
    """Apply a custom transform function to a measurement set."""
    def __init__(self,
                 transform_function: Callable,
                 data_independent_parameters=None,
                 *args,
                 **kwargs):
        LambdaTransform.__init__(self,
                                 transform_function=transform_function)

        measurement_set_transform.MeasurementSetTransform.__init__(
            self,
            data_independent_parameters=data_independent_parameters,
            *args,
            **kwargs
        )
