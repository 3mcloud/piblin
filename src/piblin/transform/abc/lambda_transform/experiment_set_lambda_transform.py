from typing import Callable

from piblin.transform.abc import experiment_set_transform as experiment_set_transform
from piblin.transform.abc.lambda_transform.lambda_transform import LambdaTransform


class ExperimentSetLambdaTransform(
    LambdaTransform,
    experiment_set_transform.ExperimentSetTransform
):
    """Apply a custom transform function to an experiment set."""
    def __init__(self,
                 transform_function: Callable,
                 data_independent_parameters=None,
                 *args,
                 **kwargs):
        LambdaTransform.__init__(self,
                                 transform_function=transform_function)

        experiment_set_transform.ExperimentSetTransform.__init__(
            self,
            data_independent_parameters=data_independent_parameters,
            *args,
            **kwargs
        )
