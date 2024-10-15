import pytest
from piblin.data.data_collections.consistent_measurement_set import ConsistentMeasurementSet, InconsistentMeasurementsError
from piblin.data.datasets.abc.split_datasets.zero_dimensional_dataset import ZeroDimensionalDataset
from piblin.data.data_collections.measurement import Measurement
from piblin.data.data_collections.measurement_set import MeasurementSet


def test_forced_consistency():

    dataset_a = ZeroDimensionalDataset.create(value=1.0, label="name_a")
    dataset_b = ZeroDimensionalDataset.create(value=0.5, label="name_b")

    measurement_a = Measurement(datasets=[dataset_a], conditions={"sample": "A"})
    measurement_b = Measurement(datasets=[dataset_b], conditions={"sample": "A"})

    cms_a = ConsistentMeasurementSet(measurements=[measurement_a])
    cms_b = ConsistentMeasurementSet(measurements=[measurement_b])

    cms_ab = ConsistentMeasurementSet.combine([cms_a, cms_b])
    assert cms_ab.num_measurements == 1
    assert cms_ab.num_datasets == 2


def test_from_default_measurement_set(default_measurement_set):
    """Ensure that the from_measurement_set method is present."""
    assert ConsistentMeasurementSet.from_measurement_set(default_measurement_set).num_measurements == 0
    assert ConsistentMeasurementSet.from_measurement_set(default_measurement_set).num_datasets == 0


def test_from_measurement_set(zerod_dataset_true_default_name_and_unit, zerod_dataset_false_default_name_and_unit):
    """"""
    measurement_true = Measurement.from_single_dataset(zerod_dataset_true_default_name_and_unit)
    measurement_false = Measurement.from_single_dataset(zerod_dataset_false_default_name_and_unit)
    scalar_measurement_set = MeasurementSet([measurement_true,
                                             measurement_false])

    consistent_set = ConsistentMeasurementSet.from_measurement_set(scalar_measurement_set)
    assert consistent_set.num_measurements == 1
    assert consistent_set.num_datasets == 2


def test_delete_dataset_of_measurement_at_index(measurement_zerod_dataset_true_default_name_and_unit_no_metadata,
                                                measurement_zerod_dataset_false_default_name_and_unit):

    measurement_set = ConsistentMeasurementSet([measurement_zerod_dataset_true_default_name_and_unit_no_metadata])

    assert measurement_set.num_measurements == 1
    assert measurement_set.num_datasets == 1

    measurement_set.remove_dataset_of_measurement_at_index(measurement_index=0,
                                                           dataset_index=0)
    assert measurement_set.num_measurements == 1
    assert measurement_set.num_datasets == 0


@pytest.mark.xfail
def test_delete_dataset_not_implemented(measurement_zerod_dataset_true_default_name_and_unit_no_metadata,
                                        measurement_zerod_dataset_false_default_name_and_unit):

    measurement_set = ConsistentMeasurementSet([measurement_zerod_dataset_true_default_name_and_unit_no_metadata,
                                                measurement_zerod_dataset_false_default_name_and_unit])

    with pytest.raises(NotImplementedError):
        measurement_set.remove_dataset_of_measurement_at_index(measurement_index=0,
                                                               dataset_index=0)


def test_default_measurement(measurement_empty):
    """A measurement with no conditions, details or datasets."""
    measurement_set = ConsistentMeasurementSet.from_single_measurement(measurement_empty)
    assert isinstance(measurement_set, ConsistentMeasurementSet)
    assert measurement_empty.num_datasets == 0
    assert measurement_empty.dataset_types == []
    assert measurement_empty.dataset_lengths == []


def test_measurement_single_true_scalar_dataset(measurement_zerod_dataset_true_default_name_and_unit_no_metadata):
    measurement_set = ConsistentMeasurementSet.from_single_measurement(measurement_zerod_dataset_true_default_name_and_unit_no_metadata)
    assert measurement_set.num_datasets == 1
    assert measurement_set.dataset_types == [ZeroDimensionalDataset]
    assert measurement_set.dataset_lengths == [[1]]


def test_default_consistent_measurement_set_properties(default_consistent_measurement_set):
    assert default_consistent_measurement_set.measurements == []
    assert default_consistent_measurement_set.num_measurements == 0
    assert default_consistent_measurement_set.datasets == []
    assert default_consistent_measurement_set.num_datasets == 0
    assert default_consistent_measurement_set.dataset_types == []
    assert default_consistent_measurement_set.dataset_lengths == []
    assert default_consistent_measurement_set.dataset_independent_variable_data == []
    assert default_consistent_measurement_set.is_consistent
    assert default_consistent_measurement_set.is_tidy
    assert default_consistent_measurement_set.all_condition_names == set()
    assert default_consistent_measurement_set.shared_condition_names == set()
    assert default_consistent_measurement_set.unshared_condition_names == set()
    assert default_consistent_measurement_set.equal_shared_condition_names == set()
    assert default_consistent_measurement_set.varying_shared_condition_names == set()
    assert default_consistent_measurement_set.equal_shared_conditions == {}
    assert default_consistent_measurement_set.varying_shared_conditions == {}
    assert default_consistent_measurement_set.has_varying_shared_conditions is False
    assert default_consistent_measurement_set.all_detail_names == set()
    assert default_consistent_measurement_set.shared_detail_names == set()
    assert default_consistent_measurement_set.unshared_detail_names == set()
    assert default_consistent_measurement_set.equal_shared_detail_names == set()
    assert default_consistent_measurement_set.varying_shared_detail_names == set()
    assert default_consistent_measurement_set.equal_shared_details == {}
    assert default_consistent_measurement_set.are_repetitions()


# @pytest.mark.mpl_image_compare(hash_library="../../matplotlib_test_hashes.json")
# def test_visualize(measurement_single_true_scalar_dataset):
#     consistent_measurement_set = ConsistentMeasurementSet([measurement_single_true_scalar_dataset])
#     fig, ax = consistent_measurement_set.visualize()
#     return fig
