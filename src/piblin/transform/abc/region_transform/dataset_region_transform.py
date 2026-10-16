from abc import ABC

import piblin.transform.abc.region_transform.region_transform as region_transform
from piblin.transform.abc import dataset_transform as dataset_transform


class DatasetRegionTransform(
    region_transform.RegionTransform, dataset_transform.DatasetTransform, ABC
): ...
