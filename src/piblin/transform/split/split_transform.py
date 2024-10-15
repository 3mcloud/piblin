from typing import List
import piblin.transform.abc.dataset_transform as dataset_transform
import piblin.transform.abc.measurement_transform as measurement_transform
from piblin.data.datasets.abc import dataset as dataset
import piblin.data.data_collections.measurement as measurement


class DatasetSplitTransform(dataset_transform.DatasetTransform):
    """A transform that converts a dataset into a list of datasets."""
    def _apply(self,
               target: dataset.Dataset,
               **kwargs) -> List[dataset.Dataset]:
        """

        Parameters
        ----------
        target : dataset.Dataset
            The dataset on which to perform the split.

        Returns
        -------
        List of dataset.Dataset
            The list of datasets created by the split.
        """
        pass

    def partitioner(self):
        ...

    def _apply_to_dataset(self,
                          target: dataset.Dataset,
                          **kwargs) -> List[dataset.Dataset]:
        """Apply this per-dataset transform to the provided target dataset.

        Parameters
        ----------
        target : dataset.Dataset
            The dataset to which to apply this transform.

        Returns
        -------
        List of dataset.Dataset
            The list of datasets created by the split.
        """
        return self._apply(target, **kwargs)


class MeasurementSplitTransform(measurement_transform.MeasurementTransform):

    def _apply(self,
               target: measurement.Measurement,
               **kwargs) -> List[measurement.Measurement]:
        pass
