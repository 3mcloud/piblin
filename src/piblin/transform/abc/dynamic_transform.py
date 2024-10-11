from importlib import import_module
from piblin.transform.abc.dataset_transform import DatasetTransform


class DynamicTransform(DatasetTransform):
    """A dynamically created transform."""
    _PACKAGE = None

    def __new__(cls, transform: str, *args, **kwargs):
        transform_class = super().__new__(cls)
        if not hasattr(cls, "_transform_operation"):
            if cls._PACKAGE is None:
                raise ValueError("Class does not implement _PACKAGE")

            splits = transform.split(".")
            transform_name = splits[-1]
            tail = ".".join(splits[:-1])

            module_name = f"{cls._PACKAGE}.{tail}"
            client = import_module(name=module_name)

            transform = getattr(client, transform_name)
            setattr(transform_class, "_transform_operation", transform)
        return transform_class

    def __init__(self, transform: str, *args, **kwargs):
        if not hasattr(self, "_transform_operation"):
            self._transform_operation = None
        super().__init__(*args, **kwargs)
        self._transform = transform
        self.transform_kwargs = kwargs

    @property
    def transform(self) -> str:
        """Full module path of transform function.

        Returns
        -------
        str
            Full moduole path of transform.
        """
        return self._PACKAGE + "." + self._transform

    @property
    def transform_operation(self):
        """The transform operation of this dynamically-created transform."""
        return self._transform_operation

    def _apply(self, dataset_, **kwargs):  # only valid for functions
        transform_operation_output = self.transform_operation(dataset_.dependent_variable_data,
                                                              **self.transform_kwargs)

        return dataset_.__class__(dependent_variable_data=transform_operation_output,
                                  dependent_variable_names=dataset_.dependent_variable_names,
                                  dependent_variable_units=dataset_.dependent_variable_units,
                                  independent_variable_data=dataset_.independent_variable_data,
                                  independent_variable_names=dataset_.independent_variable_names,
                                  independent_variable_units=dataset_.independent_variable_units)
