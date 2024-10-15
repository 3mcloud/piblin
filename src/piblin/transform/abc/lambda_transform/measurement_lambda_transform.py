from typing import Callable

from piblin.transform.abc import \
    measurement_transform as measurement_transform
from piblin.transform.abc.lambda_transform.lambda_transform import LambdaTransform


class MeasurementLambdaTransform(
    LambdaTransform,
    measurement_transform.MeasurementTransform
):
    """Apply a custom transform function to a measurement."""
    def __init__(self,
                 transform_function: Callable,
                 data_independent_parameters=None,
                 *args,
                 **kwargs):

        LambdaTransform.__init__(self,
                                 transform_function=transform_function)

        measurement_transform.MeasurementTransform.__init__(
            self,
            data_independent_parameters=data_independent_parameters,
            *args,
            **kwargs
        )
