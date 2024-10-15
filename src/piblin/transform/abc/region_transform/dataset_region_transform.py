from abc import ABC
from piblin.transform.abc import dataset_transform as dataset_transform
import piblin.transform.abc.region_transform.region_transform as region_transform


class DatasetRegionTransform(region_transform.RegionTransform,
                             dataset_transform.DatasetTransform,
                             ABC):
    ...
