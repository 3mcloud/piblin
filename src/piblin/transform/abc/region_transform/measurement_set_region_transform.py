from abc import ABC
import piblin.transform.abc.region_transform.region_transform as region_transform
import piblin.transform.abc.measurement_set_transform as measurement_set_transform


class MeasurementSetRegionTransform(
    region_transform.RegionTransform,
    measurement_set_transform.MeasurementSetTransform,
    ABC
):
    ...
