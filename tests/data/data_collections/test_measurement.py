"""Tests for the `cralds.data.measurements.Measurement` class.

This module contains unit tests which are intended to thoroughly exercise the
public API of the Measurement class. The str method is not tested by choice.

Public Measurement Properties
-----------------------------
datasets : List[cralds.data.datasets.Dataset]
conditions : Dict[str, object]
details : Dict[str, object]

Functionality to be tested is:
initialization
property access
functionality of dunder methods:
    class repr can be eval'ed
    class str behaves as expected
    class eq behaves as expected
    class cmp is not implemented
checking for replicate measurements
visualization produces consistent output
"""
import pytest
from piblin.data.datasets.abc.dataset import Dataset  # not unused - do not delete
from numpy import array, array_equal  # not unused - do not delete
from piblin.data.data_collections.tabular_measurement_set import TabularMeasurementSet
from piblin.data.datasets.abc.split_datasets.zero_dimensional_dataset import ZeroDimensionalDataset
from piblin.data.datasets.abc.split_datasets.one_dimensional_dataset import OneDimensionalDataset
# from cralds.data import DatasetFactory
from piblin.data.data_collections.measurement import Measurement
from piblin.data.data_collections.measurement import ExistingMetadataError
from piblin.data.data_collections.measurement import MissingMetadataError
from piblin.data.data_collections.measurement import DuplicateMetadataError
from piblin.data.data_collections.measurement import DifferingConditionError
from piblin.data.data_collections.measurement import IncompatibleListError


def test_initialization_default(measurement_empty):
    """Ensure the default measurement has the correct properties."""
    assert measurement_empty.datasets == []
    assert measurement_empty.num_datasets == 0
    assert measurement_empty.dataset_types == []
    assert measurement_empty.dataset_lengths == []
    assert measurement_empty.dataset_independent_variable_data == []
    assert measurement_empty.conditions == {}
    assert measurement_empty.condition_names == set()
    assert measurement_empty.details == {}
    assert measurement_empty.detail_names == set()


def test_initialization_single_true_scalar_dataset(measurement_zerod_dataset_true_default_name_and_unit_no_metadata,
                                                   zerod_dataset_true_default_name_and_unit):
    """Ensure that a measurement with a single scalar has the correct properties."""
    assert measurement_zerod_dataset_true_default_name_and_unit_no_metadata.datasets == [zerod_dataset_true_default_name_and_unit]
    assert measurement_zerod_dataset_true_default_name_and_unit_no_metadata.num_datasets == 1
    assert measurement_zerod_dataset_true_default_name_and_unit_no_metadata.dataset_types == [ZeroDimensionalDataset]
    assert measurement_zerod_dataset_true_default_name_and_unit_no_metadata.dataset_lengths == [1]
    assert array_equal(measurement_zerod_dataset_true_default_name_and_unit_no_metadata.dataset_independent_variable_data, [array([])])
    assert measurement_zerod_dataset_true_default_name_and_unit_no_metadata.conditions == {}
    assert measurement_zerod_dataset_true_default_name_and_unit_no_metadata.condition_names == set()
    assert measurement_zerod_dataset_true_default_name_and_unit_no_metadata.details == {}
    assert measurement_zerod_dataset_true_default_name_and_unit_no_metadata.detail_names == set()


def test_initialization_single_true_oned_dataset(measurement_single_oned_dataset_single_point,
                                                 oned_dataset_single_point):
    """Ensure that a measurement with a single one-d dataset has the correct properties."""
    assert measurement_single_oned_dataset_single_point.datasets == [oned_dataset_single_point]
    assert measurement_single_oned_dataset_single_point.num_datasets == 1
    assert measurement_single_oned_dataset_single_point.dataset_types == [OneDimensionalDataset]
    assert measurement_single_oned_dataset_single_point.dataset_lengths == [1]
    assert measurement_single_oned_dataset_single_point.conditions == {}
    assert measurement_single_oned_dataset_single_point.condition_names == set()
    assert measurement_single_oned_dataset_single_point.details == {}
    assert measurement_single_oned_dataset_single_point.detail_names == set()


def test_initialization_two_datasets(measurement_two_differing_zerod_datasets,
                                     zerod_dataset_true_default_name_and_unit,
                                     zerod_dataset_false_default_name_and_unit):
    """Ensure that a measurement with two scalars has the correct properties."""
    assert measurement_two_differing_zerod_datasets.datasets == [zerod_dataset_true_default_name_and_unit, zerod_dataset_false_default_name_and_unit]
    assert measurement_two_differing_zerod_datasets.num_datasets == 2
    assert measurement_two_differing_zerod_datasets.dataset_types == [ZeroDimensionalDataset, ZeroDimensionalDataset]
    assert measurement_two_differing_zerod_datasets.dataset_lengths == [1, 1]
    assert measurement_two_differing_zerod_datasets.conditions == {}
    assert measurement_two_differing_zerod_datasets.condition_names == set()
    assert measurement_two_differing_zerod_datasets.details == {}
    assert measurement_two_differing_zerod_datasets.detail_names == set()


def test_initialization_single_condition(measurement_single_true_condition, true_condition):
    """Ensure that a measurement with a single condition has the correct properties."""
    assert measurement_single_true_condition.datasets == []
    assert measurement_single_true_condition.num_datasets == 0
    assert measurement_single_true_condition.dataset_types == []
    assert measurement_single_true_condition.dataset_lengths == []
    assert measurement_single_true_condition.conditions == true_condition
    assert measurement_single_true_condition.condition_names == {"condition"}
    assert measurement_single_true_condition.details == {}
    assert measurement_single_true_condition.detail_names == set()


def test_initialization_single_detail(measurement_single_true_detail, true_detail):
    """Ensure that a measurement with a single detail has the correct properties."""
    assert measurement_single_true_detail.datasets == []
    assert measurement_single_true_detail.num_datasets == 0
    assert measurement_single_true_detail.dataset_types == []
    assert measurement_single_true_detail.dataset_lengths == []
    assert measurement_single_true_detail.conditions == {}
    assert measurement_single_true_detail.condition_names == set()
    assert measurement_single_true_detail.details == true_detail
    assert measurement_single_true_detail.detail_names == {"detail"}


def test_initialization(measurement_single_dataset_condition_detail, zerod_dataset_true_default_name_and_unit, true_condition, true_detail):
    """Ensure that a measurement with a single scalar, condition and detail has the correct properties."""
    assert measurement_single_dataset_condition_detail.datasets == [zerod_dataset_true_default_name_and_unit]
    assert measurement_single_dataset_condition_detail.num_datasets == 1
    assert measurement_single_dataset_condition_detail.dataset_types == [ZeroDimensionalDataset]
    assert measurement_single_dataset_condition_detail.dataset_lengths == [1]
    assert measurement_single_dataset_condition_detail.conditions == true_condition
    assert measurement_single_dataset_condition_detail.condition_names == {"condition"}
    assert measurement_single_dataset_condition_detail.details == true_detail
    assert measurement_single_dataset_condition_detail.detail_names == {"detail"}


def test_from_single_dataset(measurement_zerod_dataset_true_default_name_and_unit_no_metadata, zerod_dataset_true_default_name_and_unit):
    """Test the creational method which takes a single dataset argument."""
    assert measurement_zerod_dataset_true_default_name_and_unit_no_metadata == Measurement.from_single_dataset(zerod_dataset_true_default_name_and_unit)


def test_from_single_dataset_no_arg(measurement_empty):
    """Test the creational method which takes a single dataset argument with no dataset."""
    assert Measurement.from_single_dataset() == measurement_empty


def test_datasets_setter(measurement_empty, zerod_dataset_true_default_name_and_unit):
    measurement_empty.datasets = [zerod_dataset_true_default_name_and_unit]
    assert measurement_empty.datasets == [zerod_dataset_true_default_name_and_unit]


def test_equality(zerod_dataset_true_default_name_and_unit, zerod_dataset_false_default_name_and_unit,
                  true_condition, false_condition,
                  true_detail, false_detail):
    """Ensure equality between measurements is implemented correctly."""

    # a measurement instance is equal to itself
    measurement = Measurement([zerod_dataset_true_default_name_and_unit])
    assert measurement == measurement
    # measurements with different data are not equal
    assert Measurement([zerod_dataset_true_default_name_and_unit]) != Measurement([zerod_dataset_false_default_name_and_unit])
    assert Measurement([zerod_dataset_true_default_name_and_unit]) != Measurement([zerod_dataset_true_default_name_and_unit, zerod_dataset_false_default_name_and_unit])
    # measurements with the same data under different conditions with no
    # details are not equal
    assert Measurement([zerod_dataset_true_default_name_and_unit], true_condition) != \
           Measurement([zerod_dataset_true_default_name_and_unit], false_condition)
    # measurements with the same data and no conditions with different details
    # are not equal
    assert Measurement([zerod_dataset_true_default_name_and_unit], details=true_detail) != \
           Measurement([zerod_dataset_true_default_name_and_unit], details=false_detail)
    # measurements with the same data under the same conditions with different
    # details are not equal
    assert Measurement([zerod_dataset_true_default_name_and_unit],
                       conditions=true_condition,
                       details=true_detail) != \
           Measurement([zerod_dataset_true_default_name_and_unit],
                       conditions=true_condition,
                       details=false_detail)
    # measurements with the same data under different conditions with the same
    # details are not equal
    assert Measurement([zerod_dataset_true_default_name_and_unit],
                       conditions=true_condition,
                       details=true_detail) != \
           Measurement([zerod_dataset_true_default_name_and_unit],
                       conditions=false_condition,
                       details=false_detail)
    # measurements with the same data under the same conditions with the same
    # details are equal.
    assert Measurement([zerod_dataset_true_default_name_and_unit], true_condition, true_detail) == \
           Measurement([zerod_dataset_true_default_name_and_unit], true_condition, true_detail)


def test_comparison_not_implemented(measurement_empty):
    """Check that the comparison dunder is specifically not implemented."""
    assert measurement_empty.__cmp__() is NotImplemented


def test_comparisons(measurement_zerod_dataset_true_default_name_and_unit_no_metadata,
                     measurement_zerod_dataset_false_default_name_and_unit):
    """Ensure comparison operators other than eq, neq are not implemented."""
    with pytest.raises(TypeError):
        _ = measurement_zerod_dataset_true_default_name_and_unit_no_metadata < measurement_zerod_dataset_false_default_name_and_unit

    with pytest.raises(TypeError):
        _ = measurement_zerod_dataset_true_default_name_and_unit_no_metadata > measurement_zerod_dataset_false_default_name_and_unit

    with pytest.raises(TypeError):
        _ = measurement_zerod_dataset_true_default_name_and_unit_no_metadata <= measurement_zerod_dataset_false_default_name_and_unit

    with pytest.raises(TypeError):
        _ = measurement_zerod_dataset_true_default_name_and_unit_no_metadata >= measurement_zerod_dataset_false_default_name_and_unit


def test_replicate(zerod_dataset_true_default_name_and_unit, zerod_dataset_false_default_name_and_unit, oned_dataset_single_point,
                   true_condition, false_condition,
                   true_detail, false_detail):
    """Ensure replicate relationship is implemented correctly."""

    # two measurements with data only are replicates
    assert Measurement([zerod_dataset_true_default_name_and_unit]).is_replicate_of(
        Measurement([zerod_dataset_false_default_name_and_unit]))
    # same data, same conditions are replicates
    assert Measurement([zerod_dataset_true_default_name_and_unit], true_condition).is_replicate_of(
        Measurement([zerod_dataset_true_default_name_and_unit], true_condition))
    # same data, different conditions are not replicates
    assert Measurement([zerod_dataset_true_default_name_and_unit], true_condition).is_not_replicate_of(
        Measurement([zerod_dataset_true_default_name_and_unit], false_condition))
    # different data, same conditions are replicates
    assert Measurement([zerod_dataset_true_default_name_and_unit], true_condition).is_replicate_of(
        Measurement([zerod_dataset_false_default_name_and_unit], true_condition))
    # different data, different conditions are not replicates
    assert Measurement([zerod_dataset_true_default_name_and_unit], true_condition).is_not_replicate_of(
        Measurement([zerod_dataset_false_default_name_and_unit], false_condition))
    # different data and details, same conditions are replicates
    assert Measurement([zerod_dataset_true_default_name_and_unit], true_condition, true_detail).\
        is_replicate_of(
        Measurement([zerod_dataset_false_default_name_and_unit], true_condition, false_detail))
    # same data and details, different conditions are not replicates
    assert Measurement([zerod_dataset_true_default_name_and_unit], true_condition, true_detail).\
        is_not_replicate_of(
           Measurement([zerod_dataset_true_default_name_and_unit], false_condition, false_detail))
    # measurements with different number of datasets are not replicates
    assert not Measurement([zerod_dataset_true_default_name_and_unit, zerod_dataset_false_default_name_and_unit]).\
        is_replicate_of(
           Measurement([zerod_dataset_true_default_name_and_unit]))
    # measurements with datasets of different types are not replicates
    assert not Measurement([zerod_dataset_true_default_name_and_unit]).\
        is_replicate_of(
           Measurement([oned_dataset_single_point]))


def test_equal_shared_conditions():
    """Ensure equal shared conditions relationship is implemented correctly."""
    assert Measurement(conditions={"condition": True}).\
        _has_equal_shared_conditions_to(
           Measurement(conditions={"condition": True}))

    assert not Measurement(conditions={"condition": True}).\
        _has_equal_shared_conditions_to(
           Measurement(conditions={"condition": False}))

    assert not Measurement(conditions={"condition": True}).\
        _has_equal_shared_conditions_to(
           Measurement(conditions={"condition": False}))


def test_replicate_sets(zerod_dataset_true_default_name_and_unit, true_condition):
    """Ensure 3 measurements with same conditions are replicates."""

    measurements = [Measurement([zerod_dataset_true_default_name_and_unit], true_condition),
                    Measurement([zerod_dataset_true_default_name_and_unit], true_condition),
                    Measurement([zerod_dataset_true_default_name_and_unit], true_condition)]
    assert Measurement.are_repetitions(measurements)


def test_non_replicate_sets(zerod_dataset_true_default_name_and_unit, true_condition, false_condition):
    """Ensure 3 measurements with differing conditions are not replicates."""
    measurements = [Measurement([zerod_dataset_true_default_name_and_unit], true_condition),
                    Measurement([zerod_dataset_true_default_name_and_unit], true_condition),
                    Measurement([zerod_dataset_true_default_name_and_unit], false_condition)]
    assert Measurement.are_not_repetitions(measurements)


def test_repr_no_metadata(measurement_single_dataset_condition_detail):
    """Ensure that a measurement with a single dataset, condition and detail's rept can be eval'd."""
    assert eval(repr(measurement_single_dataset_condition_detail)) == \
           measurement_single_dataset_condition_detail


def test_repr_single_metadata(measurement_single_dataset_condition_detail):
    """"""
    assert eval(repr(measurement_single_dataset_condition_detail)) == \
           measurement_single_dataset_condition_detail


def test_repr_multiple_metadata(zerod_dataset_false_default_name_and_unit,
                                true_condition,
                                other_true_condition,
                                true_detail,
                                other_true_detail):
    """Test that the repr can be eval'ed for a measurement with multiple metadata."""
    conditions = true_condition
    conditions.update(other_true_condition)

    details = true_detail
    details.update(other_true_detail)

    measurement = Measurement(datasets=[zerod_dataset_false_default_name_and_unit],
                              conditions=conditions,
                              details=details)
    assert eval(repr(measurement)) == measurement


def test_default_one_line_str(measurement_empty):
    assert measurement_empty.one_line_str() == "conditions={}, details={}, datasets=[]"


def test_one_line_str_single_scalar(measurement_zerod_dataset_true_default_name_and_unit_no_metadata):
    assert measurement_zerod_dataset_true_default_name_and_unit_no_metadata.one_line_str() == "conditions={}, details={}, datasets=[0: variable = 1.0 None]"


def test_one_line_str_single_true_condition(measurement_single_true_condition):
    assert measurement_single_true_condition.one_line_str() == "conditions={condition:True}, details={}, datasets=[]"


def test_one_line_str_single_true_detail(measurement_single_true_detail):
    assert measurement_single_true_detail.one_line_str() == "conditions={}, details={detail:True}, datasets=[]"


def test_one_line_str_single_dataset_condition_detail(measurement_single_dataset_condition_detail):
    assert measurement_single_dataset_condition_detail.one_line_str() == \
           "conditions={condition:True}, details={detail:True}, datasets=[0: variable = 1.0 None]"


def test_str_default(measurement_empty):
    assert str(measurement_empty).__contains__("Measurement")


def test_str(measurement_single_dataset_condition_detail):
    assert str(measurement_single_dataset_condition_detail).__contains__("Measurement")


def test_is_replicate():
    """Test that the replicate relationship holds for measurements with no metadata."""
    measurement_i = Measurement(datasets=None,
                                conditions={"condition": True},
                                details=None)

    measurement_j = Measurement(datasets=None,
                                conditions={"condition": True},
                                details=None)

    assert measurement_i.is_replicate_of(measurement_j)


def test_compute_matplotlib_figure_size(measurement_empty,
                                        measurement_zerod_dataset_true_default_name_and_unit_no_metadata):

    assert measurement_empty.compute_matplotlib_figure_size() == (0, 0)
    assert measurement_zerod_dataset_true_default_name_and_unit_no_metadata.compute_matplotlib_figure_size() == (5, 5)


def test_condition_to_detail(measurement_single_true_condition):
    """Test the conversion of a condition to a detail for a measurement with a single condition."""
    measurement_single_true_condition.condition_to_detail("condition")

    assert measurement_single_true_condition.conditions == {}
    assert measurement_single_true_condition.details == {"condition": True}


def test_conditions_to_details(measurement_two_true_conditions):
    measurement_two_true_conditions.conditions_to_details(["condition", "other_condition"])
    assert measurement_two_true_conditions.details == {"condition": True,
                                                       "other_condition": True}


def test_detail_to_condition(measurement_single_true_detail):
    """Test the conversion of a detail to a condition for a measurement with a single detail."""
    measurement_single_true_detail.detail_to_condition("detail")

    assert measurement_single_true_detail.details == {}
    assert measurement_single_true_detail.conditions == {"detail": True}


def test_details_to_condition(measurement_two_true_details):
    measurement_two_true_details.details_to_conditions(["detail", "other_detail"])

    assert measurement_two_true_details.conditions == {"detail": True,
                                                       "other_detail": True}


def test_add_condition(measurement_empty):
    """Test the ability to add a condition to a measurement."""
    measurement_empty.add_condition("a", False)
    assert measurement_empty.conditions["a"] is False


def test_add_conditions(measurement_empty):
    measurement_empty.add_conditions(["condition", "other_condition"], [True, True])
    assert measurement_empty.conditions == {"condition": True,
                                              "other_condition": True}


def test_add_details(measurement_empty):
    measurement_empty.add_details(["detail", "other_detail"], [True, True])
    assert measurement_empty.details == {"detail": True,
                                           "other_detail": True}


def test_add_details_incompatible_list(measurement_empty):
    with pytest.raises(IncompatibleListError):
        measurement_empty.add_details(["detail"], [True, True])


def test_add_conditions_incompatible_list(measurement_empty):
    """Ensure that adding multiple conditions raises the appropriate error if lists are incompatible."""
    with pytest.raises(IncompatibleListError):
        measurement_empty.add_conditions(["a"], [True, False])


def test_update_conditions_incompatible_list(measurement_empty):
    with pytest.raises(IncompatibleListError):
        measurement_empty.update_conditions(["condition"], [False, False])


def test_update_conditions(measurement_empty):
    measurement_empty.update_conditions(["condition", "other_condition"], [True, True])
    assert measurement_empty.conditions == {"condition": True,
                                              "other_condition": True}


def test_update_details_incompatible_list(measurement_empty):
    with pytest.raises(IncompatibleListError):
        measurement_empty.update_details(["detail"], [False, False])


def test_update_details(measurement_empty):
    measurement_empty.update_details(["detail", "other_detail"], [True, True])
    assert measurement_empty.details == {"detail": True,
                                           "other_detail": True}


def test_add_detail(measurement_empty):
    """Test the ability to add a detail to a measurement."""
    measurement_empty.add_detail("a", False)
    assert measurement_empty.details["a"] is False


def test_remove_condition(measurement_single_true_condition):
    """Test the ability to remove a condition from a measurement."""
    measurement_single_true_condition.remove_condition("condition")
    assert measurement_single_true_condition.conditions == {}


def test_remove_conditions(measurement_two_true_conditions):
    measurement_two_true_conditions.remove_conditions(["condition", "other_condition"])
    assert measurement_two_true_conditions.conditions == {}


def test_remove_detail(measurement_single_true_detail):
    """Test the ability to remove a detail from a measurement."""
    measurement_single_true_detail.remove_detail("detail")
    assert measurement_single_true_detail.details == {}


def test_remove_details(measurement_two_true_details):
    measurement_two_true_details.remove_details(["detail", "other_detail"])
    assert measurement_two_true_details.details == {}


def test_rename_metadata(measurement_single_true_condition):
    measurement_single_true_condition.rename_metadata("condition", "new_name")
    assert measurement_single_true_condition.has_condition_name("new_name")


def test_duplicate_metadata_name_error():
    """Ensure a measurement cannot be created with detail and condition metadata with equal names."""
    with pytest.raises(DuplicateMetadataError):
        Measurement(None,
                    details={"a": False},
                    conditions={"a": True})


def test_add_condition_existing_condition_error():
    """Ensure a condition cannot be updated with add_condition."""
    measurement = Measurement(conditions={"a": False})
    with pytest.raises(ExistingMetadataError):
        measurement.add_condition("a", True)


def test_add_condition_existing_detail_error():
    """Ensure an existing detail prevents adding a condition with the same name."""
    measurement = Measurement(details={"a": False})
    with pytest.raises(ExistingMetadataError):
        measurement.add_condition("a", False)


def test_add_detail_existing_condition_error():
    """Ensure an existing condition prevents adding a detail with the same name."""
    measurement = Measurement(conditions={"a": False})
    with pytest.raises(ExistingMetadataError):
        measurement.add_detail("a", False)


def test_add_detail_existing_detail_error():
    """Ensure a detail cannot be updated with add_detail."""
    measurement = Measurement(details={"a": False})
    with pytest.raises(ExistingMetadataError):
        measurement.add_detail("a", True)


def test_conditions_are_copies(measurement_empty):
    """Ensure the conditions property provides a copy of the class attribute."""
    measurement_empty.conditions["b"] = True
    assert "b" not in measurement_empty.conditions.keys()


def test_details_are_copies(measurement_empty):
    """Ensure the details property provides a copy of the class attribute."""
    measurement_empty.details["b"] = True
    assert "b" not in measurement_empty.details.keys()


def test_update_condition(measurement_single_true_condition):
    """Test the ability to update a condition of a measurement."""
    measurement_single_true_condition.update_condition("condition", False)
    assert measurement_single_true_condition.conditions["condition"] is False


def test_update_condition_non_existing_name(measurement_empty):
    """Ensure that updating a non-existing condition adds that condition."""
    measurement_empty.update_condition("condition", True)
    assert measurement_empty.conditions["condition"] is True


def test_update_condition_name_must_exist(measurement_empty):
    """Ensure that the name_must_exist flag works as expected."""
    with pytest.raises(MissingMetadataError):
        measurement_empty.update_condition("condition", True, name_must_exist=True)


def test_update_condition_existing_detail(measurement_single_true_detail):
    """Ensure that updating a condition with the same name as an existing detail is an error."""
    with pytest.raises(ExistingMetadataError):
        measurement_single_true_detail.update_condition("detail", True)


def test_update_detail(measurement_single_true_detail):
    """Test the ability to update a condition of a measurement."""
    measurement_single_true_detail.update_detail("detail", False)
    assert measurement_single_true_detail.details["detail"] is False


def test_update_detail_non_existing_name(measurement_empty):
    """Ensure that updating a non-existing condition adds that condition."""
    measurement_empty.update_detail("detail", True)
    assert measurement_empty.details["detail"] is True


def test_update_detail_name_must_exist(measurement_empty):
    """Ensure that the name_must_exist flag works as expected."""
    with pytest.raises(MissingMetadataError):
        measurement_empty.update_detail("detail", True, name_must_exist=True)


def test_update_detail_existing_condition(measurement_single_true_condition):
    """Ensure that updating a condition with the same name as an existing detail is an error."""
    with pytest.raises(ExistingMetadataError):
        measurement_single_true_condition.update_detail("condition", True)


def test_has_condition_name(measurement_single_true_condition):
    assert measurement_single_true_condition.has_condition_name("condition")
    assert not measurement_single_true_condition.has_condition_name("other_condition")


def test_has_condition_names(measurement_two_true_conditions):
    assert measurement_two_true_conditions.has_condition_names(["condition", "other_condition"])
    assert not measurement_two_true_conditions.has_condition_names(["condition", "missing_condition"])


def test_has_detail_name(measurement_single_true_detail):
    assert measurement_single_true_detail.has_detail_name("detail")
    assert not measurement_single_true_detail.has_detail_name("missing_detail")


def test_has_detail_names(measurement_two_true_details):
    assert measurement_two_true_details.has_detail_names(["detail", "other_detail"])
    assert not measurement_two_true_details.has_detail_names(["missing_detail"])


# def test_consolidation_two_measurements_equal_conditions():
#     """Test the from_measurements creational method."""
#     measurement_a = Measurement([DatasetFactory.create(False)])
#     measurement_b = Measurement([DatasetFactory.create(False)])
#
#     measurement_ab = Measurement([DatasetFactory.create(False),
#                                   DatasetFactory.create(False)])
#
#     assert Measurement.from_measurements([measurement_a,
#                                           measurement_b]) == measurement_ab
#
#
# def test_consolidation_three_measurements_equal_conditions():
#     """Test the from_measurements creational method."""
#     measurement_a = Measurement([DatasetFactory.create(False)])
#     measurement_b = Measurement([DatasetFactory.create(False)])
#     measurement_c = Measurement([DatasetFactory.create(False)])
#
#     measurement_abc = Measurement([DatasetFactory.create(False),
#                                    DatasetFactory.create(False),
#                                    DatasetFactory.create(False)])
#
#     assert Measurement.from_measurements([measurement_a,
#                                           measurement_b,
#                                           measurement_c]) == measurement_abc
#
#
# @pytest.mark.xfail
# def test_consolidation_different_conditions():
#     """Test the from_measurements creational method."""
#     measurement_a = Measurement([DatasetFactory.create(False)], conditions={"a": True})
#     measurement_b = Measurement([DatasetFactory.create(False)])
#
#     measurement_ab = Measurement([DatasetFactory.create(False),
#                                   DatasetFactory.create(False)],
#                                  conditions={"a": True})
#
#     with pytest.raises(ValueError):
#         Measurement.from_measurements([measurement_a, measurement_b])
#
#     assert measurement_ab == Measurement.from_measurements([measurement_a, measurement_b],
#                                                            allow_missing_conditions=True,
#                                                            keep_partial_conditions=True)
#
#     assert measurement_ab == Measurement.from_measurements([measurement_b, measurement_a],
#                                                            allow_missing_conditions=True,
#                                                            keep_partial_conditions=True)
#
#
# def test_consolidation_multiple_conditions():
#     """Test the from_measurements creational method."""
#     measurement_a = Measurement([DatasetFactory.create(False)], conditions={"a": True,
#                                                                             "b": True})
#     measurement_b = Measurement([DatasetFactory.create(False)], conditions={"a": True})
#
#     measurement_ab = Measurement([DatasetFactory.create(False),
#                                   DatasetFactory.create(False)],
#                                  conditions={"a": True, "b": True})
#
#     assert measurement_ab == Measurement.from_measurements([measurement_a, measurement_b],
#                                                            allow_missing_conditions=True,
#                                                            keep_partial_conditions=True)
#
#     assert measurement_ab == Measurement.from_measurements([measurement_b, measurement_a],
#                                                            allow_missing_conditions=True,
#                                                            keep_partial_conditions=True)
#
#
# def test_consolidation_multiple_conditions_no_partials():
#     """Test the from_measurements creational method."""
#     measurement_a = Measurement([DatasetFactory.create(False)], conditions={"a": True,
#                                                                             "b": True})
#     measurement_b = Measurement([DatasetFactory.create(False)], conditions={"a": True})
#
#     measurement_ab = Measurement([DatasetFactory.create(False),
#                                   DatasetFactory.create(False)],
#                                  conditions={"a": True})
#
#     assert measurement_ab == Measurement.from_measurements([measurement_a, measurement_b],
#                                                            allow_missing_conditions=True,
#                                                            keep_partial_conditions=False)
#
#     assert measurement_ab == Measurement.from_measurements([measurement_b, measurement_a],
#                                                            allow_missing_conditions=True,
#                                                            keep_partial_conditions=False)


def test_from_tabular():
    tabular_measurement_set = TabularMeasurementSet(data=[[0]],
                                                    n_metadata_columns=0,
                                                    column_headers=["scalar(None)=f()"],
                                                    dataset_types=[ZeroDimensionalDataset],
                                                    dataset_end_indices=[1])

    measurement = Measurement(datasets=[ZeroDimensionalDataset.create(value=[0], label="scalar")])
    other_measurement = Measurement.from_tabular_measurement_set(tabular_measurement_set)
    assert measurement == other_measurement


def test_flatten_metadata_default(measurement_empty):
    """

    Uses flatten_metadata and flatten_datasets to convert a measurement
    into a single row of a tabular dataset.
    flattening of metadata does not rely on underlying functionality,
    flattening of data does rely on the datasets to flatten themselves.
    A measurement will only ever produce a single row.

    Returns
    -------

    """
    names, metadata = measurement_empty.flatten_metadata()
    assert not names
    assert not metadata


def test_flatten_metadata(measurement_zerod_dataset_true_default_name_and_unit_no_metadata):
    names, metadata = measurement_zerod_dataset_true_default_name_and_unit_no_metadata.flatten_metadata()
    assert not names
    assert not metadata


def test_flatten_metadata_single_condition(measurement_single_true_condition):
    names, metadata = measurement_single_true_condition.flatten_metadata()
    assert names == ["condition"]
    assert metadata == [True]

    assert measurement_single_true_condition.flatten_metadata() == \
           measurement_single_true_condition.flatten_metadata(condition_names=["condition"])

    names, metadata = measurement_single_true_condition.flatten_metadata(condition_names=["other_condition"])
    assert names == ["other_condition"]
    assert metadata == [None]

    names, metadata = measurement_single_true_condition.flatten_metadata(condition_names=["other_condition"],
                                                                         default_value=True)
    assert names == ["other_condition"]
    assert metadata == [True]


def test_flatten_metadata_single_detail(measurement_single_true_detail):
    assert measurement_single_true_detail.flatten_metadata() == ([], [])
    names, metadata = measurement_single_true_detail.flatten_metadata(detail_names=["detail"])
    assert names == ["detail"]
    assert metadata == [True]

    names, metadata = measurement_single_true_detail.flatten_metadata(condition_names=["other_condition"],
                                                                      detail_names=["detail"])
    assert names == ["other_condition", "detail"]
    assert metadata == [None, True]


def test_flatten_metadata_condition_and_detail(measurement_one_condition_one_detail):
    names, metadata = measurement_one_condition_one_detail.flatten_metadata()
    assert names == ["condition"]
    assert metadata == [True]

    names, metadata = measurement_one_condition_one_detail.flatten_metadata(detail_names=["detail"])
    assert names == ["condition", "detail"]
    assert metadata == [True, True]


def test_flatten_metadata_two_conditions(measurement_two_true_conditions):
    names, metadata = measurement_two_true_conditions.flatten_metadata()
    assert names == ["condition", "other_condition"] or names == ["other_condition", "condition"]
    assert metadata == [True, True]

    names, metadata = measurement_two_true_conditions.flatten_metadata(condition_names=["condition"])
    assert names == ["condition"]
    assert metadata == [True]

    names, metadata = measurement_two_true_conditions.flatten_metadata(condition_names=["other_condition"])
    assert names == ["other_condition"]
    assert metadata == [True]

    names, metadata = measurement_two_true_conditions.flatten_metadata(detail_names=["other_detail"])
    assert names == ["condition", "other_condition", "other_detail"] or \
           names == ["other_condition", "condition", "other_detail"]
    assert metadata == [True, True, None]

    names, metadata = measurement_two_true_conditions.flatten_metadata(detail_names=["other_detail"],
                                                                       default_value=False)
    assert names == ["condition", "other_condition", "other_detail"] or \
           names == ["other_condition", "condition", "other_detail"]
    assert metadata == [True, True, False]


def test_flatten_datasets(measurement_empty):
    assert measurement_empty.flatten_datasets() == ([], [])


def test_flatten_scalar(measurement_zerod_dataset_true_default_name_and_unit_no_metadata):
    assert measurement_zerod_dataset_true_default_name_and_unit_no_metadata.flatten_datasets() == (["variable(None)"], [True])


# def test_visualize_measurement_two_differing_scalar_datasets(measurement_two_differing_scalar_datasets, tmp_path):
#     measurement_two_differing_scalar_datasets.visualize(expand_datasets=True)
#     plt.savefig(tmp_path / "measurement_two_differing_scalar_datasets_collapsed.png")
#
#
# def test_visualize_collapse_datasets(tmp_path):
#
#     dataset_a = OneDimensionalDataset.create(y_values=[1.0, 0.9, 1.1],
#                                              x_values=[0.0, 0.5, 1.0],
#                                              y_name="response",
#                                              y_unit="degrees",
#                                              x_name="time",
#                                              x_unit="seconds")
#
#     dataset_b = OneDimensionalDataset.create(y_values=[0.5, 0.4, 0.6],
#                                              x_values=[0.0, 0.5, 1.0],
#                                              y_name="response",
#                                              y_unit="degrees",
#                                              x_name="time",
#                                              x_unit="minutes")
#
#     measurement = Measurement([dataset_a, dataset_b])
#     measurement.visualize(include_text=False, expand_datasets=True)
#     plt.savefig(tmp_path / "m-collapse.png")


# def test_visualize_measurement_two_oned_datasets():
#
#     dataset_a = DatasetFactory.create(dependent_variable_data=[1.0, 0.9, 1.1],
#                                       independent_variable_data=[0.0, 0.5, 1.0],
#                                       dependent_variable_name="y",
#                                       independent_variable_names=["x"])
#
#     dataset_b = DatasetFactory.create(dependent_variable_data=[0.5, 0.4, 0.6],
#                                       independent_variable_data=[0.0, 0.5, 1.0],
#                                       dependent_variable_name="y",
#                                       independent_variable_names=["x"])
#
#     measurement = Measurement(datasets=[dataset_a, dataset_b])
#     measurement.visualize(expand_datasets=True)
#     plt.savefig(r"C:\Users\A835CZZ\AppData\Roaming\JetBrains\PyCharmCE2021.1\scratches\m-collapse.png")

from piblin.data.data_collections.measurement import Measurement
from piblin.data.datasets.abc.split_datasets.one_dimensional_dataset import OneDimensionalDataset
from piblin.data.datasets.abc.split_datasets.two_dimensional_dataset import TwoDimensionalDataset

@pytest.fixture()
def measurement_single_oned():
    dataset = OneDimensionalDataset.create(x_values=[0.0], x_name="x",
                                           y_values=[1.0])
    return Measurement(datasets=[dataset])


@pytest.fixture()
def measurement_single_twod():
    dataset = TwoDimensionalDataset.create(x_values=[0.0], x_name="x",
                                           y_values=[0.0], y_name="y",
                                           z_values=[[1.0]])
    return Measurement(datasets=[dataset])


@pytest.fixture()
def measurement_two_oned_same_name():
    dataset_a = OneDimensionalDataset.create(x_values=[0.0], x_name="x",
                                             y_values=[1.0])

    dataset_b = OneDimensionalDataset.create(x_values=[0.0], x_name="x",
                                             y_values=[1.0])

    return Measurement(datasets=[dataset_a, dataset_b])


@pytest.fixture()
def measurement_two_oned_different_names():
    dataset_a = OneDimensionalDataset.create(x_values=[0.0], x_name="x",
                                             y_values=[1.0])

    dataset_b = OneDimensionalDataset.create(x_values=[0.0], x_name="not_x",
                                             y_values=[1.0])

    return Measurement(datasets=[dataset_a, dataset_b])


@pytest.fixture()
def measurement_two_twod_same_name():
    dataset_a = TwoDimensionalDataset.create(x_values=[0.0], x_name="x",
                                             y_values=[0.0], y_name="y",
                                             z_values=[[1.0]])

    dataset_b = TwoDimensionalDataset.create(x_values=[0.0], x_name="x",
                                             y_values=[0.0], y_name="y",
                                             z_values=[[1.0]])

    return Measurement(datasets=[dataset_a, dataset_b])


@pytest.fixture()
def measurement_two_twod_different_names():
    dataset_a = TwoDimensionalDataset.create(x_values=[0.0], x_name="x",
                                             y_values=[0.0], y_name="y",
                                             z_values=[[1.0]])

    dataset_b = TwoDimensionalDataset.create(x_values=[0.0], x_name="not_x",
                                             y_values=[0.0], y_name="y",
                                             z_values=[[1.0]])

    return Measurement(datasets=[dataset_a, dataset_b])


def test_split_same_name(measurement_single_oned,
                         measurement_single_twod,
                         measurement_two_oned_same_name,
                         measurement_two_twod_same_name):

    with_name, without_name = \
        measurement_single_oned.split_by_dataset_independent_variable_name("x")

    assert not without_name.datasets
    assert with_name == measurement_single_oned

    with_name, without_name = \
        measurement_single_twod.split_by_dataset_independent_variable_name("x")

    assert not without_name.datasets
    assert with_name == measurement_single_twod

    with_name, without_name = \
        measurement_two_oned_same_name.split_by_dataset_independent_variable_name("x")

    assert not without_name.datasets
    assert with_name == measurement_two_oned_same_name

    with_name, without_name = \
        measurement_two_twod_same_name.split_by_dataset_independent_variable_name("x")

    assert not without_name.datasets
    assert with_name == measurement_two_twod_same_name
    assert with_name.num_datasets == 2


def test_split_different_names(measurement_two_oned_different_names,
                               measurement_two_twod_different_names):

    with_name, without_name = \
        measurement_two_oned_different_names.split_by_dataset_independent_variable_name("x")

    assert with_name.datasets
    assert without_name.datasets

    assert with_name.num_datasets == 1
    assert without_name.num_datasets == 1

    with_name, without_name = \
        measurement_two_twod_different_names.split_by_dataset_independent_variable_name("x")

    assert with_name.datasets
    assert without_name.datasets

    assert with_name.num_datasets == 1
    assert without_name.num_datasets == 1


@pytest.fixture()
def measurement_to_split():

    dataset_a = OneDimensionalDataset.create(x_values=[0.0], x_name="x", y_values=[1.0])
    dataset_b = OneDimensionalDataset.create(x_values=[0.0], x_name="x", y_values=[1.0])
    dataset_c = OneDimensionalDataset.create(x_values=[1.0], x_name="x", y_values=[1.0])

    return Measurement(datasets=[dataset_a, dataset_b, dataset_c])


def test_split_by_independent_variable_value(measurement_to_split):
    measurements = measurement_to_split.split_by_dataset_independent_variable_value("x")

    assert len(measurements) == 2
    assert {measurements[0].num_datasets, measurements[1].num_datasets} == {2, 1}
    assert {measurements[0].conditions["x"],
            measurements[1].conditions["x"]} == {0.0, 1.0}


def test_split_by_ind_var_two_oned(measurement_two_oned_same_name):
    measurements = measurement_two_oned_same_name.split_by_dataset_independent_variable_value("x")

    assert len(measurements) == 1
    assert measurements[0].num_datasets == 2
    assert measurements[0].conditions["x"] == 0.0


@pytest.fixture()
def twod_measurement_to_split():
    dataset_a = TwoDimensionalDataset.create(x_values=[0.0],
                                             x_name="x",
                                             y_values=[1.0],
                                             y_name="y",
                                             z_values=[[1.0]])

    dataset_b = TwoDimensionalDataset.create(x_values=[0.0],
                                             x_name="x",
                                             y_values=[1.0],
                                             y_name="y",
                                             z_values=[[1.0]])

    dataset_c = TwoDimensionalDataset.create(x_values=[1.0],
                                             x_name="x",
                                             y_values=[1.0],
                                             y_name="y",
                                             z_values=[[1.0]])

    return Measurement(datasets=[dataset_a, dataset_b, dataset_c])


def test_split_by_ind_var_two_twod(twod_measurement_to_split):

    measurements = twod_measurement_to_split.split_by_dataset_independent_variable_value("x")

    assert len(measurements) == 2
    assert {measurements[0].num_datasets, measurements[1].num_datasets} == {2, 1}
    assert {measurements[0].conditions["x"],
            measurements[1].conditions["x"]} == {0.0, 1.0}

@pytest.mark.skip
@pytest.mark.mpl_image_compare(hash_library="../../matplotlib_test_hashes.json")
def test_visualize_measurement_empty(measurement_empty):
    figure, axes = measurement_empty.visualize()
    return figure

@pytest.mark.skip
@pytest.mark.mpl_image_compare(hash_library="../../matplotlib_test_hashes.json")
def test_visualize_measurement_single_true_dataset_collapsed(
        measurement_zerod_dataset_true_default_name_and_unit_no_metadata, datadir):
    figure, axes = measurement_zerod_dataset_true_default_name_and_unit_no_metadata.visualize(expand_datasets=False)
    return figure

@pytest.mark.skip
@pytest.mark.mpl_image_compare(hash_library="../../matplotlib_test_hashes.json")
def test_visualize_measurement_single_true_dataset_expanded(
        measurement_zerod_dataset_true_default_name_and_unit_no_metadata, datadir):
    figure, axes = measurement_zerod_dataset_true_default_name_and_unit_no_metadata.visualize(expand_datasets=True)
    return figure

@pytest.mark.skip
@pytest.mark.mpl_image_compare(hash_library="../../matplotlib_test_hashes.json")
def test_visualize_measurement_single_one_dimensional_dataset_single_point(measurement_single_oned_dataset_single_point):
    figure, axes = measurement_single_oned_dataset_single_point.visualize(include_text=False)
    return figure

@pytest.mark.skip
@pytest.mark.mpl_image_compare(hash_library="../../matplotlib_test_hashes.json")
def test_visualize_measurement_two_differing_scalar_datasets(measurement_two_differing_zerod_datasets, datadir):
    figure, axes = measurement_two_differing_zerod_datasets.visualize()
    return figure

@pytest.mark.skip
@pytest.mark.mpl_image_compare(hash_library="../../matplotlib_test_hashes.json")
def test_visualize_true_measurement_with_true_condition(true_measurement_with_true_condition):
    figure, axes = true_measurement_with_true_condition.visualize()
    return figure

@pytest.mark.skip
@pytest.mark.mpl_image_compare(hash_library="../../matplotlib_test_hashes.json")
def test_visualize_true_measurement_with_true_condition(true_measurement_with_true_condition):
    figure, axes = true_measurement_with_true_condition.visualize(include_text=False)
    return figure
