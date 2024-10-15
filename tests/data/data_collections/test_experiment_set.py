import pytest
from piblin.data.data_collections.measurement import Measurement
from piblin.data.data_collections.experiment_set import ExperimentSet
from piblin.data.datasets.abc.split_datasets.two_dimensional_dataset import TwoDimensionalDataset
from piblin.data.datasets.abc.split_datasets.zero_dimensional_dataset import ZeroDimensionalDataset
from piblin.data.datasets.abc.split_datasets.one_dimensional_dataset import OneDimensionalDataset


@pytest.fixture()
def experiment_set_one_scalar():
    dataset = ZeroDimensionalDataset.create(value=0)
    measurement = Measurement(datasets=[dataset])
    return ExperimentSet(measurements=[measurement])


@pytest.fixture()
def experiment_set_two_scalars():
    dataset_a = ZeroDimensionalDataset.create(value=0)
    dataset_b = ZeroDimensionalDataset.create(value=0)
    measurement = Measurement(datasets=[dataset_a,
                                        dataset_b])
    return ExperimentSet(measurements=[measurement])


@pytest.fixture()
def experiment_set_one_oned_dataset():
    dataset = OneDimensionalDataset.create(x_values=[0],
                                           y_values=[0])

    measurement = Measurement(datasets=[dataset])
    return ExperimentSet(measurements=[measurement])


@pytest.fixture()
def experiment_set_two_oned_dataset():

    dataset_a = OneDimensionalDataset.create(x_values=[0],
                                             y_values=[0])

    dataset_b = OneDimensionalDataset.create(x_values=[0],
                                             y_values=[0])

    measurement = Measurement(datasets=[dataset_a, dataset_b])
    return ExperimentSet(measurements=[measurement])


@pytest.fixture()
def experiment_set_one_twod_dataset():
    dataset = TwoDimensionalDataset.create(x_values=[0],
                                           y_values=[0],
                                           z_values=[[1]])

    measurement = Measurement(datasets=[dataset])
    return ExperimentSet(measurements=[measurement])


@pytest.fixture()
def experiment_set_one_twod_dataset_two_values():
    dataset = TwoDimensionalDataset.create(x_values=[0, 1],
                                           y_values=[0, 1],
                                           z_values=[[1, 1], [1, 1]])

    measurement = Measurement(datasets=[dataset])
    return ExperimentSet(measurements=[measurement])


def test_for_independent_variables(experiment_set_one_scalar,
                                   experiment_set_two_scalars):

    with pytest.raises(ValueError):
        experiment_set_one_scalar.independent_variable_to_condition("")

    with pytest.raises(ValueError):
        experiment_set_two_scalars.independent_variable_to_condition("")


def test_for_correct_independent_variable(experiment_set_one_oned_dataset):
    with pytest.raises(ValueError):
        experiment_set_one_oned_dataset.independent_variable_to_condition("")

    experiment_set_one_oned_dataset.independent_variable_to_condition("x")

    assert "x" in experiment_set_one_oned_dataset.conditions
    assert experiment_set_one_oned_dataset.num_datasets == 1


def test_for_correct_independent_variable_two(experiment_set_two_oned_dataset):
    with pytest.raises(ValueError):
        experiment_set_two_oned_dataset.independent_variable_to_condition("")

    experiment_set_two_oned_dataset.independent_variable_to_condition("x")

    assert "x" in experiment_set_two_oned_dataset.conditions
    assert experiment_set_two_oned_dataset.num_datasets == 2


def test_for_correct_independent_variable_twod(experiment_set_one_twod_dataset):
    with pytest.raises(ValueError):
        experiment_set_one_twod_dataset.independent_variable_to_condition("")

    experiment_set_one_twod_dataset.independent_variable_to_condition("x")

    assert "x" in experiment_set_one_twod_dataset.conditions
    assert experiment_set_one_twod_dataset.num_datasets == 1


def test_error_on_multipoint_dataset(experiment_set_one_twod_dataset_two_values):
    with pytest.raises(ValueError):
        experiment_set_one_twod_dataset_two_values.independent_variable_to_condition("")

#
# # import pytest
# # from cralds.data import Measurement, MeasurementSet, ExperimentSet
# # # from cralds.data import DatasetFactory
# #
# #
# # @pytest.fixture()
# # def condition_name():
# #     return "condition"
# #
# #
# # @pytest.fixture()
# # def condition_value():
# #     return 0.0
# #
# #
# # @pytest.fixture()
# # def dependent_variable_name():
# #     return "dependent_variable"
# #
# #
# # @pytest.fixture()
# # def dependent_variable_value():
# #     return 1.0
# #
# #
# # @pytest.fixture()
# # def measurement_single_scalar(condition_name,
# #                               condition_value,
# #                               dependent_variable_name,
# #                               dependent_variable_value):
# #
# #     scalar = DatasetFactory.create(dependent_variable_value,
# #                                    dependent_variable_name=dependent_variable_name)
# #
# #     return Measurement(datasets=[scalar], conditions={condition_name: condition_value})
# #
# #
# # @pytest.fixture()
# # def experiment_set_single_scalar_one_replicate(condition_name,
# #                                                dependent_variable_name,
# #                                                dependent_variable_value):
# #
# #     scalar_a = DatasetFactory.create(dependent_variable_value,
# #                                      dependent_variable_name=dependent_variable_name)
# #
# #     scalar_b = DatasetFactory.create(dependent_variable_value,
# #                                      dependent_variable_name=dependent_variable_name)
# #
# #     measurement_a = Measurement(datasets=[scalar_a], conditions={condition_name: 0.0})
# #
# #     measurement_b = Measurement(datasets=[scalar_b], conditions={condition_name: 1.0})
# #
# #     return ExperimentSet.from_measurement_set(MeasurementSet(measurements=[measurement_a,
# #                                                                            measurement_b]))
# #
# #
# # @pytest.fixture()
# # def experiment_single_scalar_two_repetitions(condition_name,
# #                                              condition_value,
# #                                              dependent_variable_name,
# #                                              dependent_variable_value):
# #
# #     scalar = DatasetFactory.create(dependent_variable_value,
# #                                    dependent_variable_name=dependent_variable_name)
# #
# #     measurement_a = Measurement(datasets=[scalar], conditions={condition_name: condition_value})
# #     measurement_b = Measurement(datasets=[scalar], conditions={condition_name: condition_value})
# #
# #     experiment_set = ExperimentSet.from_measurement_set(MeasurementSet(measurements=[measurement_a,
# #                                                                                      measurement_b]))
# #
# #     return experiment_set[0]
# #
# #
# # @pytest.fixture()
# # def experiment_single_one_d_dataset_two_repetitions(condition_name,
# #                                                     condition_value,
# #                                                     dependent_variable_name,
# #                                                     dependent_variable_value):
# #
# #     one_d_dataset = DatasetFactory.create(dependent_variable_data=dependent_variable_value,
# #                                           independent_variable_data=condition_value,
# #                                           independent_variable_names=[condition_name],
# #                                           dependent_variable_name=dependent_variable_name)
# #
# #     measurement_a = Measurement(datasets=[one_d_dataset],
# #                                 conditions={condition_name: condition_value})
# #
# #     measurement_b = Measurement(datasets=[one_d_dataset],
# #                                 conditions={condition_name: condition_value})
# #
# #     experiment_set = ExperimentSet.from_measurement_set(MeasurementSet(measurements=[measurement_a,
# #                                                                                      measurement_b]))
# #
# #     return experiment_set[0]
# #
# #
# # @pytest.fixture()
# # def measurement_two_scalars(condition_name,
# #                             condition_value,
# #                             dependent_variable_name,
# #                             dependent_variable_value):
# #
# #     scalar_a = DatasetFactory.create(dependent_variable_value,
# #                                      dependent_variable_name=dependent_variable_name)
# #
# #     scalar_b = DatasetFactory.create(dependent_variable_value,
# #                                      dependent_variable_name=dependent_variable_name)
# #
# #     return Measurement(datasets=[scalar_a, scalar_b],
# #                        conditions={condition_name: condition_value})
# #
# #
# # @pytest.fixture()
# # def measurement_one_d_and_scalar(condition_name,
# #                                  condition_value,
# #                                  dependent_variable_name,
# #                                  dependent_variable_value):
# #
# #     scalar = DatasetFactory.create(dependent_variable_value,
# #                                    dependent_variable_name=dependent_variable_name)
# #
# #     one_d_dataset = DatasetFactory.create(dependent_variable_data=dependent_variable_value,
# #                                           independent_variable_data=condition_value,
# #                                           independent_variable_names=[condition_name],
# #                                           dependent_variable_name=dependent_variable_name)
# #
# #     return Measurement(datasets=[one_d_dataset, scalar],
# #                        conditions={})
# #
# #
# # @pytest.fixture()
# # def measurement_single_one_d_dataset(condition_name,
# #                                      condition_value,
# #                                      dependent_variable_name,
# #                                      dependent_variable_value):
# #
# #     one_d_dataset = DatasetFactory.create(dependent_variable_data=dependent_variable_value,
# #                                           independent_variable_data=condition_value,
# #                                           independent_variable_names=[condition_name],
# #                                           dependent_variable_name=dependent_variable_name)
# #
# #     return Measurement(datasets=[one_d_dataset],
# #                        conditions={})
# #
# #
# # @pytest.fixture()
# # def measurement_two_one_d_datasets(condition_name,
# #                                    condition_value,
# #                                    dependent_variable_name,
# #                                    dependent_variable_value):
# #
# #     one_d_dataset_a = DatasetFactory.create(dependent_variable_data=dependent_variable_value,
# #                                             independent_variable_data=condition_value,
# #                                             independent_variable_names=[condition_name],
# #                                             dependent_variable_name=dependent_variable_name)
# #
# #     one_d_dataset_b = DatasetFactory.create(dependent_variable_data=dependent_variable_value,
# #                                             independent_variable_data=condition_value,
# #                                             independent_variable_names=[condition_name],
# #                                             dependent_variable_name=dependent_variable_name)
# #
# #     return Measurement(datasets=[one_d_dataset_a,
# #                                  one_d_dataset_b],
# #                        conditions={})
# #
# #
# # # def test_measurement_single_scalar(measurement_single_scalar,
# # #                                    condition_name,
# # #                                    measurement_single_one_d_dataset):
# # #     measurement_single_scalar.promote_condition(condition_name)
# # #     assert measurement_single_scalar == measurement_single_one_d_dataset
# # #
# # #
# # # def test_measurement_single_scalar_index(measurement_single_scalar,
# # #                                          condition_name,
# # #                                          measurement_single_one_d_dataset):
# # #
# # #     measurement_single_scalar.promote_condition(condition_name, dataset_indices=[0])
# # #     assert measurement_single_scalar == measurement_single_one_d_dataset
# # #
# # #
# # # def test_measurement_two_scalars_index_zero(measurement_two_scalars,
# # #                                             condition_name,
# # #                                             measurement_one_d_and_scalar):
# # #
# # #     measurement_two_scalars.promote_condition(condition_name, dataset_indices=[0])
# # #     assert measurement_two_scalars == measurement_one_d_and_scalar
# # #
# # #
# # # # def test_measurement_two_scalars_index_one(measurement_two_scalars,
# # # #                                            condition_name,
# # # #                                            measurement_one_d_and_scalar):
# # # #
# # # #     measurement_two_scalars.promote_condition(condition_name, dataset_indices=[1])
# # # #     assert measurement_two_scalars == measurement_one_d_and_scalar
# # #
# # #
# # # # def test_measurement_two_scalars(measurement_two_scalars,
# # # #                                  condition_name,
# # # #                                  measurement_two_one_d_datasets):
# # # #
# # # #     measurement_two_scalars.promote_condition(condition_name)
# # # #     assert measurement_two_scalars == measurement_two_one_d_datasets
# # #
# # #
# # # def test_experiment(experiment_single_scalar_two_repetitions,
# # #                     condition_name,
# # #                     experiment_single_one_d_dataset_two_repetitions):
# # #
# # #     experiment_single_scalar_two_repetitions.promote_condition(condition_name)
# # #     assert experiment_single_scalar_two_repetitions == experiment_single_one_d_dataset_two_repetitions
# # #
# # #
# # # # def test_print_experiment_set_single_scalar_one_replicate(experiment_set_single_scalar_one_replicate,
# # # #                                                           condition_name):
# # # #
# # # #     experiment_set_single_scalar_one_replicate.promote_condition(condition_name)
# # # #     print(experiment_set_single_scalar_one_replicate)
