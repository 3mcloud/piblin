from typing import Set
from abc import ABC, abstractmethod
from piblin.data.datasets.abc.dataset import Dataset


class FileWriter(ABC):
    """Abstract class for file writers."""

    @property
    @abstractmethod
    def supported_extensions(self) -> Set[str]:
        pass

    @abstractmethod
    def write_data_collection(self, data_collection: Dataset, location: str):
        """Convert the contents of a data collection into the output file format."""
        ...
