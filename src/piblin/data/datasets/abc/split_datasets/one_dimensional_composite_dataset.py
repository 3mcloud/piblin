from typing import List, Union
import numpy as np
import piblin.data.datasets.abc.split_datasets.one_dimensional_dataset as one_dimensional_dataset
import piblin.data.datasets.roi as roi


class OneDimensionalCompositeDataset(one_dimensional_dataset.OneDimensionalDataset):
    """A one-dimensional split dataset composed of multiple switchable data arrays.

    Parameters
    ----------
    data_arrays: List of np.ndarray
        The multiple one-dimensional arrays of data.
    data_array_names: List of str
        The name of each one-dimensional array of data.
    data_array_units: List of str
        The unit of each one-dimensional array of data.
    default_independent_name: str, optional
        The name of the variable to use for the independent coordinate.
        If not provided, the
    default_dependent_name: str
        The name of the variable to use for the dependent coordinate.
    source: str
        THe source of this one-dimensional composite dataset.

    Attributes
    ----------
    data_arrays -> List of np.ndarray
        The multiple on-dimensional arrays of data.
    number_of_data_arrays -> int
        The number of one-dimensional arrays of data.
    data_array_names -> List of str
        The name of each one-dimensional array of data.
    data_array_units -> List of str
        The unit of each one-dimensional array of data.
    """

    @property
    def data_arrays(self) -> List[np.ndarray]:
        return self._data_arrays

    @data_arrays.setter
    def data_arrays(self, data_arrays: List[np.ndarray]):
        self._data_arrays = data_arrays

    @property
    def data_array_names(self):
        return self._data_array_names

    @property
    def data_array_units(self):
        return self._data_array_units

    @property
    def number_of_data_arrays(self) -> int:
        """The number of one-dimensional arrays of data."""
        return len(self._data_arrays)

    @property
    def dependent_variable_data(self) -> np.ndarray:
        return self.data_arrays[self._dependent_data_index]

    @dependent_variable_data.setter
    def dependent_variable_data(self, dependent_variable_data: np.ndarray):
        self.data_arrays[self._dependent_data_index] = dependent_variable_data

    @property
    def dependent_variable_names(self) -> List[str]:
        return [self._data_array_names[self._dependent_data_index]]

    @dependent_variable_names.setter
    def dependent_variable_names(self, dependent_variable_names: List[str]):
        self._data_array_names[self._dependent_data_index] = dependent_variable_names[0]

    @property
    def dependent_variable_units(self) -> List[str]:
        return [self._data_array_units[self._dependent_data_index]]

    @dependent_variable_units.setter
    def dependent_variable_units(self, dependent_variable_units: List[str]):
        self._data_array_units[self._dependent_data_index] = dependent_variable_units[0]

    @property
    def independent_variable_data(self) -> List[np.ndarray]:
        return [self._data_arrays[self._independent_data_index]]

    @independent_variable_data.setter
    def independent_variable_data(self, independent_variable_data: np.ndarray):
        self._data_arrays[self._independent_data_index] = independent_variable_data

    @property
    def independent_variable_names(self) -> List[str]:
        return [self._data_array_names[self._independent_data_index]]

    @independent_variable_names.setter
    def independent_variable_names(self, independent_variable_names: List[str]):
        self._data_array_names[self._independent_data_index] = independent_variable_names[0]

    @property
    def independent_variable_units(self) -> List[str]:
        return [self._data_array_units[self._independent_data_index]]

    @independent_variable_units.setter
    def independent_variable_units(self, independent_variable_units: List[str]):
        self._data_array_units[self._independent_data_index] = independent_variable_units[0]

    def __init__(self,
                 data_arrays: List[np.array],
                 data_array_names: List[str],
                 data_array_units: List[str],
                 default_independent_name: str = None,
                 default_dependent_name: str = None,
                 source=None):

        self._data_arrays = [np.array(data_array)
                             for data_array in data_arrays]
        self._data_array_names = data_array_names
        self._data_array_units = data_array_units

        if default_independent_name is None:
            self._independent_data_index = 0
        else:
            self._independent_data_index = data_array_names.index(default_independent_name)

        if default_dependent_name is None:
            self._dependent_data_index = 1
        else:
            self._dependent_data_index = data_array_names.index(default_dependent_name)

        super().__init__(
            dependent_variable_data=self.dependent_variable_data,
            dependent_variable_names=self.dependent_variable_names,
            dependent_variable_units=self.dependent_variable_units,
            independent_variable_data=[self.independent_variable_data],
            independent_variable_names=self.independent_variable_names,
            independent_variable_units=self.independent_variable_units,
            source=source
        )

    def switch_independent_coordinate(self, coordinate_name: str) -> None:
        """Change the independent coordinate to the named variable.

        Parameters
        ----------
        coordinate_name : str
            The name of the variable to switch the independent coordinates to.

        Raises
        ------
        ValueError
            When the dataset has no variable array with the provided coordinate name.
        """
        if coordinate_name not in self._data_array_names:
            error_str = ", ".join(self._data_array_names)
            raise ValueError(f"Dataset has no variable array with the name: {coordinate_name}\nNames:{error_str}")
        else:
            self._independent_data_index = \
                self._data_array_names.index(coordinate_name)

    def switch_dependent_coordinate(self, coordinate_name: str) -> None:
        """Change the dependent coordinate to the named variable.

        Parameters
        ----------
        coordinate_name : str
            The name of the variable to switch the dependent coordinates to.

        Raises
        ------
        ValueError
            When the dataset has no variable array with the provided coordinate name.
        """
        if coordinate_name not in self._data_array_names:
            error_str = ", ".join(self._data_array_names)
            raise ValueError(
                f"Dataset has no variable array with the name: {coordinate_name}\nNames:{error_str}"
            )
        else:
            self._dependent_data_index = \
                self._data_array_names.index(coordinate_name)

    def switch_coordinates(self, independent_name: str, dependent_name: str) -> None:
        """Change the independent and dependent coordinates to the named variables.

        Parameters
        ----------
        independent_name : str
            The name of the variable to switch the independent coordinates to.
        dependent_name : str
            The name of the variable to switch the dependent coordinate to.
        """
        if independent_name not in self._data_array_names:
            error_str = ", ".join(self._data_array_names)
            raise ValueError(f"Dataset has no variable array with the name: {independent_name}\nNames:{error_str}")

        if dependent_name not in self._data_array_names:
            error_str = ", ".join(self._data_array_names)
            raise ValueError(f"Dataset has no variable array with the name: {dependent_name}\nNames:{error_str}")

        self._independent_data_index = self._data_array_names.index(independent_name)
        self._dependent_data_index = self._data_array_names.index(dependent_name)

    def to_onedimensional_dataset(
            self,
            coord_names: Union[str, List[str]] = None
    ) -> Union[one_dimensional_dataset.OneDimensionalDataset,
               List[one_dimensional_dataset.OneDimensionalDataset]]:
        """Convert this composite dataset back to a non-composite dataset.

        Returns
        -------
        one_dimensional_dataset.OneDimensionalDataset
            A non-composite dataset with data from the current active coordinates.
        """
        if not coord_names:
            return one_dimensional_dataset.OneDimensionalDataset(
                independent_variable_data=self.independent_variable_data,
                independent_variable_names=self.independent_variable_names,
                independent_variable_units=self.independent_variable_units,
                dependent_variable_data=self.dependent_variable_data,
                dependent_variable_names=self.dependent_variable_names,
                dependent_variable_units=self.dependent_variable_units,
                source=self.source
            )
        elif isinstance(coord_names, str):
            coord_names = [coord_names]

        datasets = []
        for coord_name in coord_names:
            self.switch_independent_coordinate(coord_name)
            dataset = one_dimensional_dataset.OneDimensionalDataset(
                independent_variable_data=self.independent_variable_data,
                independent_variable_names=self.independent_variable_names,
                independent_variable_units=self.independent_variable_units,
                dependent_variable_data=self.dependent_variable_data,
                dependent_variable_names=self.dependent_variable_names,
                dependent_variable_units=self.dependent_variable_units,
                source=self.source
            )
            datasets.append(dataset)

        return datasets

    def to_xarray(self):
        """Convert this composite dataset to an xarray dataset."""
        raise NotImplementedError

    @classmethod
    def from_xarray(cls, xr_data):
        """Convert the provided xarray dataset to a composite dataset."""
        raise NotImplementedError

    @staticmethod
    def from_datasets(
            datasets:List["OneDimensionalCompositeDataset"]
    ) -> "OneDimensionalCompositeDataset":
        """
        Create a new OneDimensionalCompositeDataset from a list of OneDimensionalCompositeDataset objects.

        Args:
            datasets (List[OneDimensionalCompositeDataset]): A list of OneDimensionalCompositeDataset objects.

        Returns:
            OneDimensionalCompositeDataset: A new OneDimensionalCompositeDataset object created from the provided datasets.
        """
        data_arrays = []
        data_array_names = []
        data_array_units = []
        for dataset in datasets:
            data_arrays.extend(dataset.data_arrays)
            data_array_names.extend(dataset._data_array_names)
            data_array_units.extend(dataset._data_array_units)
        default_independent_name = None
        default_dependent_name = None
        source = ""

        return OneDimensionalCompositeDataset(
            data_arrays,
            data_array_names,
            data_array_units,
            default_independent_name,
            default_dependent_name,
            source
        )

    def split_by_independent_variable_regions(self,
                                              split_regions: roi.CompoundRegion) -> List["OneDimensionalDataset"]:
        """Split this dataset into multiple datasets using the specified independent variable regions.

        Parameters
        ----------
        split_regions : piblin.data.CompoundRegion
            The compound region containing the regions to split the dataset into.
        """
        datasets = []

        for region in split_regions:

            if self.x_values.max() < region.min or self.x_values.min() > region.max:
                print(f"Warning: split by [{region.min}, {region.max}], ({self.independent_variable_axis_labels[0]} range = [{self.x_values.min()}, {self.x_values.max()}]) will result in empty array")

            region_data_arrays = [[] for array in self.data_arrays]

            for i, independent_value in enumerate(self.x_values):

                if region.contains(independent_value):
                    for j in range(len(self.data_arrays)):
                        region_data_arrays[j].append(
                            self.data_arrays[j][i])

            datasets.append(self.__class__(
                data_arrays=region_data_arrays,
                data_array_names=self.data_array_names,
                data_array_units=self.data_array_units,
                default_independent_name=self.independent_variable_name,
                default_dependent_name=self.dependent_variable_name,
                source=self.source
            ))

        return datasets
