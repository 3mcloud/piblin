import pytest
from piblin.data.data_collections.measurement_set import MeasurementSet
# from cralds.data import Scalar
from piblin.data.datasets.abc.split_datasets.zero_dimensional_dataset import ZeroDimensionalDataset


def test_from_single_measurement(measurement_empty,
                                 measurement_zerod_dataset_true_default_name_and_unit_no_metadata,
                                 measurement_single_oned_dataset_single_point,
                                 measurement_two_differing_zerod_datasets,
                                 measurement_single_true_condition):
    """Ensure that the from_single_measurement method returns the same result as passing a length-1 list to the init."""

    assert MeasurementSet.from_single_measurement(measurement_empty) == \
           MeasurementSet([measurement_empty])

    assert MeasurementSet.from_single_measurement(measurement_zerod_dataset_true_default_name_and_unit_no_metadata) == \
           MeasurementSet([measurement_zerod_dataset_true_default_name_and_unit_no_metadata])

    assert MeasurementSet.from_single_measurement(measurement_single_oned_dataset_single_point) == \
           MeasurementSet([measurement_single_oned_dataset_single_point])

    assert MeasurementSet.from_single_measurement(measurement_two_differing_zerod_datasets) == \
           MeasurementSet([measurement_two_differing_zerod_datasets])

    assert MeasurementSet.from_single_measurement(measurement_single_true_condition) == \
           MeasurementSet([measurement_single_true_condition])


def test_from_measurement_sets():
    ...


def test_default_measurement(measurement_empty):
    """A measurement with no conditions, details or datasets."""
    measurement_set = MeasurementSet.from_single_measurement(measurement_empty)
    assert isinstance(measurement_set, MeasurementSet)
    assert measurement_empty.num_datasets == 0
    assert measurement_empty.dataset_types == []
    assert measurement_empty.dataset_lengths == []


def test_measurement_single_true_scalar_dataset(measurement_zerod_dataset_true_default_name_and_unit_no_metadata):
    measurement_set = MeasurementSet.from_single_measurement(measurement_zerod_dataset_true_default_name_and_unit_no_metadata)

    assert measurement_set.measurements == [measurement_zerod_dataset_true_default_name_and_unit_no_metadata]
    assert measurement_set.num_measurements == 1
    # assert measurement_set.datasets == [[measurement_single_true_scalar_dataset]]
    assert measurement_set.num_datasets == [1]
    assert measurement_set.dataset_types == [[ZeroDimensionalDataset]]
    assert measurement_set.dataset_lengths == [[1]]
    # assert measurement_set.dataset_independent_variable_data == []
    assert measurement_set.is_consistent
    assert measurement_set.is_tidy
    assert measurement_set.all_condition_names == set()
    assert measurement_set.shared_condition_names == set()
    assert measurement_set.unshared_condition_names == set()
    assert measurement_set.equal_shared_condition_names == set()
    assert measurement_set.varying_shared_condition_names == set()
    assert measurement_set.equal_shared_conditions == {}
    assert measurement_set.varying_shared_conditions == {}
    assert measurement_set.has_varying_shared_conditions is False
    assert measurement_set.all_detail_names == set()
    assert measurement_set.shared_detail_names == set()
    assert measurement_set.unshared_detail_names == set()
    assert measurement_set.equal_shared_detail_names == set()
    assert measurement_set.varying_shared_detail_names == set()
    assert measurement_set.equal_shared_details == {}
    assert measurement_set.are_repetitions()


def test_default_measurement_set_properties(default_measurement_set):
    assert default_measurement_set.measurements == []
    assert default_measurement_set.num_measurements == 0
    assert default_measurement_set.datasets == []
    assert default_measurement_set.num_datasets == []
    assert default_measurement_set.dataset_types == []
    assert default_measurement_set.dataset_lengths == []
    assert default_measurement_set.dataset_independent_variable_data == []
    assert default_measurement_set.is_consistent
    assert default_measurement_set.is_tidy
    assert default_measurement_set.all_condition_names == set()
    assert default_measurement_set.shared_condition_names == set()
    assert default_measurement_set.unshared_condition_names == set()
    assert default_measurement_set.equal_shared_condition_names == set()
    assert default_measurement_set.varying_shared_condition_names == set()
    assert default_measurement_set.equal_shared_conditions == {}
    assert default_measurement_set.varying_shared_conditions == {}
    assert default_measurement_set.has_varying_shared_conditions is False
    assert default_measurement_set.all_detail_names == set()
    assert default_measurement_set.shared_detail_names == set()
    assert default_measurement_set.unshared_detail_names == set()
    assert default_measurement_set.equal_shared_detail_names == set()
    assert default_measurement_set.varying_shared_detail_names == set()
    assert default_measurement_set.equal_shared_details == {}
    assert default_measurement_set.are_repetitions()


def test_measurement_set_single_measurement_single_true_condition(
        measurement_set_single_measurement_single_true_condition):

    assert measurement_set_single_measurement_single_true_condition.num_measurements == 1
    assert measurement_set_single_measurement_single_true_condition.datasets == [[]]
    assert measurement_set_single_measurement_single_true_condition.num_datasets == [0]
    assert measurement_set_single_measurement_single_true_condition.dataset_types == [[]]
    assert measurement_set_single_measurement_single_true_condition.dataset_lengths == [[]]
    assert measurement_set_single_measurement_single_true_condition.dataset_independent_variable_data == [[]]
    assert measurement_set_single_measurement_single_true_condition.is_consistent
    assert measurement_set_single_measurement_single_true_condition.is_tidy
    assert measurement_set_single_measurement_single_true_condition.all_condition_names == {"condition"}
    assert measurement_set_single_measurement_single_true_condition.shared_condition_names == {"condition"}
    assert measurement_set_single_measurement_single_true_condition.unshared_condition_names == set()
    assert measurement_set_single_measurement_single_true_condition.equal_shared_condition_names == {"condition"}
    assert measurement_set_single_measurement_single_true_condition.varying_shared_condition_names == set()
    assert measurement_set_single_measurement_single_true_condition.equal_shared_conditions == {"condition": True}
    assert measurement_set_single_measurement_single_true_condition.varying_shared_conditions == {}
    assert measurement_set_single_measurement_single_true_condition.has_varying_shared_conditions is False
    assert measurement_set_single_measurement_single_true_condition.all_detail_names == set()
    assert measurement_set_single_measurement_single_true_condition.shared_detail_names == set()
    assert measurement_set_single_measurement_single_true_condition.unshared_detail_names == set()
    assert measurement_set_single_measurement_single_true_condition.equal_shared_detail_names == set()
    assert measurement_set_single_measurement_single_true_condition.varying_shared_detail_names == set()
    assert measurement_set_single_measurement_single_true_condition.equal_shared_details == {}
    assert measurement_set_single_measurement_single_true_condition.are_repetitions()


def test_measurement_set_single_measurement_two_true_conditions(
        measurement_set_single_measurement_two_true_conditions):
    assert measurement_set_single_measurement_two_true_conditions.num_measurements == 1
    assert measurement_set_single_measurement_two_true_conditions.datasets == [[]]
    assert measurement_set_single_measurement_two_true_conditions.num_datasets == [0]
    assert measurement_set_single_measurement_two_true_conditions.dataset_types == [[]]
    assert measurement_set_single_measurement_two_true_conditions.dataset_lengths == [[]]
    assert measurement_set_single_measurement_two_true_conditions.dataset_independent_variable_data == [[]]
    assert measurement_set_single_measurement_two_true_conditions.is_consistent
    assert measurement_set_single_measurement_two_true_conditions.is_tidy
    assert measurement_set_single_measurement_two_true_conditions.all_condition_names == {"condition",
                                                                                          "other_condition"}
    assert measurement_set_single_measurement_two_true_conditions.shared_condition_names == {"condition",
                                                                                             "other_condition"}
    assert measurement_set_single_measurement_two_true_conditions.unshared_condition_names == set()
    assert measurement_set_single_measurement_two_true_conditions.equal_shared_condition_names == {"condition",
                                                                                             "other_condition"}
    assert measurement_set_single_measurement_two_true_conditions.varying_shared_condition_names == set()
    assert measurement_set_single_measurement_two_true_conditions.equal_shared_conditions == {"condition": True,
                                                                                              "other_condition": True}
    assert measurement_set_single_measurement_two_true_conditions.varying_shared_conditions == {}
    assert measurement_set_single_measurement_two_true_conditions.has_varying_shared_conditions is False
    assert measurement_set_single_measurement_two_true_conditions.all_detail_names == set()
    assert measurement_set_single_measurement_two_true_conditions.shared_detail_names == set()
    assert measurement_set_single_measurement_two_true_conditions.unshared_detail_names == set()
    assert measurement_set_single_measurement_two_true_conditions.equal_shared_detail_names == set()
    assert measurement_set_single_measurement_two_true_conditions.varying_shared_detail_names == set()
    assert measurement_set_single_measurement_two_true_conditions.equal_shared_details == {}
    assert measurement_set_single_measurement_two_true_conditions.are_repetitions()

@pytest.mark.skip
@pytest.mark.mpl_image_compare(hash_library="../../matplotlib_test_hashes.json")
def test_visualize_two_false_measurements_true_condition_false_other_condition(measurement_set_two_false_measurements_true_condition_false_other_condition):

    figure, axes = measurement_set_two_false_measurements_true_condition_false_other_condition.visualize()

    return figure

@pytest.mark.skip
@pytest.mark.mpl_image_compare(hash_library="../../matplotlib_test_hashes.json")
def test_visualize_two_measurements_different_data_same_condition(measurement_set_two_measurements_different_data_same_condition):

    figure, axes = measurement_set_two_measurements_different_data_same_condition.visualize()

    return figure
