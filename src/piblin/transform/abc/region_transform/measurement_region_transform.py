from abc import ABC
import piblin.transform.abc.region_transform.region_transform as region_transform
import piblin.transform.abc.measurement_transform as measurement_transform


class MeasurementRegionTransform(
    region_transform.RegionTransform,
    measurement_transform.MeasurementTransform,
    ABC
):
    ...
