from typing import List
from piblin.transform.abc.region_transform.dataset_region_transform import DatasetRegionTransform
import piblin.data.datasets.abc.split_datasets as split_dataset


class SplitByIndependentVariableRanges(DatasetRegionTransform):

    def _apply(
            self, target: split_dataset.SplitDataset, **kwargs
    ) -> List[split_dataset.SplitDataset]:
        """Split a dataset by independent variable regions.

        The compound region property of this class defines the datasets
        that will be returned following a split. Each linear region of the
        compound region is N-dimensional, and its dimensionality must match
        that of the dataset to which it is to be applied. A dataset will
        be returned (potentially empty) for each linear region of the
        compound region.

        Parameters
        ----------
        target
            A single split dataset of any dimensionality.

        Returns
        -------

        """
        return target.split_by_independent_variable_regions(
            self.compound_region
        )


class Truncate(SplitByIndependentVariableRanges):
    """Alias for SplitByIndependentVariableRanges."""
    ...
