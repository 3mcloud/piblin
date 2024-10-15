# from cralds.data.datasets.base_classes.dataset_factory import DatasetFactory
# from cralds.data.datasets.base_classes.dataset import Dataset
# from cralds.data.datasets.base_classes.split_datasets.zero_dimensional_dataset import ZeroDimensionalDataset
# from cralds.data.datasets.base_classes.split_datasets.one_dimensional_dataset import OneDimensionalDataset
#
#
# def test_create_scalar():
#     assert type(DatasetFactory.from_split_data(0.0)) is ZeroDimensionalDataset
#     assert type(DatasetFactory.from_split_data([0.0])) is ZeroDimensionalDataset
#
#
# def test_create_oned():
#     assert type(DatasetFactory.from_split_data(0.0, 0.0)) is OneDimensionalDataset
#     assert type(DatasetFactory.from_split_data([0.0], 0.0)) is OneDimensionalDataset
#     assert type(DatasetFactory.from_split_data(0.0, [0.0])) is OneDimensionalDataset
#     assert type(DatasetFactory.from_split_data([0.0], [0.0])) is OneDimensionalDataset
#
#     assert type(DatasetFactory.from_split_data([0.0, 1.0], [0.0, 0.0])) is OneDimensionalDataset
#
#
# def test_generic():
#     assert type(DatasetFactory.from_split_data([[0.0, 0.0], [0.0, 0.0]], [[0.0, 1.0], [0.0, 1.0]])) is Dataset
