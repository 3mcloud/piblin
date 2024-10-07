"""
Provide fixtures for testing the measurement-experiment-project hierarchy.

The fixtures defined in this test are designed to incrementally build up an
appropriate set of ExperimentSet instances while exposing Dataset, conditions and
details dicts, measurements and experiments for testing. The fixtures herein
are tested in the test modules `test_measurement`, `test_experiment` and
`test_experiment_set`.
"""
from typing import Dict
import pytest
from piblin.data.data_collections.measurement import Measurement
from piblin.data.data_collections.measurement_set import MeasurementSet
from piblin.data.data_collections.consistent_measurement_set import ConsistentMeasurementSet
from piblin.data.data_collections.experiment import Experiment
from piblin.data.data_collections.experiment_set import ExperimentSet


# Metadata instances for use in building Measurement instances


@pytest.fixture()
def true_condition() -> Dict[str, bool]:
    """Return a generic experimental condition which is true."""
    return {"condition": True}


@pytest.fixture()
def other_true_condition() -> Dict[str, bool]:
    """Return a generic experimental condition which is true."""
    return {"other_condition": True}


@pytest.fixture()
def false_condition() -> Dict[str, bool]:
    """Return a generic experimental condition which is false."""
    return {"condition": False}


@pytest.fixture()
def true_detail() -> Dict[str, bool]:
    """Return a generic experimental detail which is true."""
    return {"detail": True}


@pytest.fixture()
def other_true_detail() -> Dict[str, bool]:
    """Return a generic experimental detail which is true."""
    return {"other_detail": True}


@pytest.fixture()
def false_detail() -> Dict[str, bool]:
    """Return a generic experimental detail which is false"""
    return {"detail": False}


# Individual Measurement Instances


@pytest.fixture()
def measurement_empty():
    """A measurement with no datasets, conditions or details."""
    return Measurement()


@pytest.fixture()
def measurement_zerod_dataset_true_default_name_and_unit_no_metadata(zerod_dataset_true_default_name_and_unit):
    """Measurement containing a true-valued 0d dataset with no conditions or details."""
    return Measurement(datasets=[zerod_dataset_true_default_name_and_unit])


@pytest.fixture()
def measurement_zerod_dataset_false_default_name_and_unit(zerod_dataset_false_default_name_and_unit):
    """Measurement containing a false-valued 0d dataset with no conditions or details."""
    return Measurement(datasets=[zerod_dataset_false_default_name_and_unit])


@pytest.fixture()
def measurement_single_oned_dataset_single_point(oned_dataset_single_point):
    return Measurement(datasets=[oned_dataset_single_point])


@pytest.fixture()
def measurement_two_differing_zerod_datasets(zerod_dataset_true_default_name_and_unit, zerod_dataset_false_default_name_and_unit):
    return Measurement(datasets=[zerod_dataset_true_default_name_and_unit, zerod_dataset_false_default_name_and_unit])


@pytest.fixture()
def measurement_single_true_condition(true_condition):
    return Measurement(conditions=true_condition)


@pytest.fixture()
def measurement_two_true_conditions():
    return Measurement(conditions={"condition": True,
                                   "other_condition": True})


@pytest.fixture()
def measurement_two_true_details():
    return Measurement(details={"detail": True,
                                "other_detail": True})


@pytest.fixture()
def measurement_one_condition_one_detail(true_condition, true_detail):
    return Measurement(conditions=true_condition, details=true_detail)


@pytest.fixture()
def measurement_single_true_detail(true_detail):
    return Measurement(details=true_detail)


@pytest.fixture()
def true_measurement_with_true_condition(zerod_dataset_true_default_name_and_unit):
    """True-value measurement with a single true condition and no details."""
    return Measurement(datasets=[zerod_dataset_true_default_name_and_unit], conditions={"condition": True})


@pytest.fixture()
def true_measurement_with_true_detail(zerod_dataset_true_default_name_and_unit):
    """True-value measurement with no conditions and a single true detail."""
    return Measurement(datasets=[zerod_dataset_true_default_name_and_unit],
                       details={"detail": True})


@pytest.fixture()
def measurement_single_dataset_condition_detail(zerod_dataset_true_default_name_and_unit, true_condition, true_detail):
    return Measurement(datasets=[zerod_dataset_true_default_name_and_unit],
                       conditions=true_condition,
                       details=true_detail)


@pytest.fixture()
def false_measurement_with_true_condition(zerod_dataset_false_default_name_and_unit):
    return Measurement(datasets=[zerod_dataset_false_default_name_and_unit], conditions={"condition": True})


@pytest.fixture()
def true_measurement_with_false_condition(zerod_dataset_true_default_name_and_unit):
    return Measurement(datasets=[zerod_dataset_true_default_name_and_unit], conditions={"condition": False})


@pytest.fixture()
def false_measurement_with_false_condition(zerod_dataset_false_default_name_and_unit):
    return Measurement(datasets=[zerod_dataset_false_default_name_and_unit], conditions={"condition": False})


@pytest.fixture()
def true_measurement_with_other_false_condition(zerod_dataset_true_default_name_and_unit):
    return Measurement(datasets=[zerod_dataset_true_default_name_and_unit], conditions={"other_condition": False})


@pytest.fixture()
def false_measurement_with_other_false_condition(zerod_dataset_false_default_name_and_unit):
    return Measurement(datasets=[zerod_dataset_false_default_name_and_unit], conditions={"other_condition": False})


@pytest.fixture()
def true_measurement_with_true_condition_and_other_false_condition(zerod_dataset_true_default_name_and_unit):
    return Measurement(datasets=[zerod_dataset_true_default_name_and_unit], conditions={"condition": True,
                                                                    "other_condition": False})


@pytest.fixture()
def false_measurement_with_true_condition_and_other_false_condition(zerod_dataset_false_default_name_and_unit):
    return Measurement(datasets=[zerod_dataset_false_default_name_and_unit], conditions={"condition": True,
                                                                     "other_condition": False})


# Lists of Measurements


@pytest.fixture()
def two_false_measurements_true_condition_false_other_condition(
        false_measurement_with_true_condition_and_other_false_condition):
    return [false_measurement_with_true_condition_and_other_false_condition,
            false_measurement_with_true_condition_and_other_false_condition]


@pytest.fixture()
def two_measurements_different_data_same_condition(true_measurement_with_true_condition,
                                                   false_measurement_with_true_condition):
    return [true_measurement_with_true_condition,
            false_measurement_with_true_condition]


@pytest.fixture()
def two_measurements_different_data_same_false_condition(true_measurement_with_false_condition,
                                                         false_measurement_with_false_condition):
    return [true_measurement_with_false_condition,
            false_measurement_with_false_condition]


@pytest.fixture()
def two_measurements_different_data_same_other_false_condition(true_measurement_with_other_false_condition,
                                                               false_measurement_with_other_false_condition):
    return [true_measurement_with_other_false_condition,
            false_measurement_with_other_false_condition]


# Measurement Sets

@pytest.fixture()
def default_measurement_set():
    return MeasurementSet()


@pytest.fixture()
def measurement_set_single_measurement_single_true_condition(measurement_single_true_condition):
    return MeasurementSet([measurement_single_true_condition])


@pytest.fixture()
def measurement_set_single_measurement_two_true_conditions(measurement_two_true_conditions):
    return MeasurementSet([measurement_two_true_conditions])


@pytest.fixture()
def measurement_set_two_false_measurements_true_condition_false_other_condition(
        two_false_measurements_true_condition_false_other_condition):
    return MeasurementSet(two_false_measurements_true_condition_false_other_condition)


@pytest.fixture()
def measurement_set_two_measurements_different_data_same_condition(two_measurements_different_data_same_condition):
    return MeasurementSet(two_measurements_different_data_same_condition)

# Consistent Measurement Sets


@pytest.fixture()
def default_consistent_measurement_set():
    return ConsistentMeasurementSet()


# Experiments


@pytest.fixture()
def experiment_two_measurements_true_condition(two_measurements_different_data_same_condition):
    return Experiment(two_measurements_different_data_same_condition)


@pytest.fixture()
def experiment_two_measurements_false_condition(two_measurements_different_data_same_false_condition):
    return Experiment(two_measurements_different_data_same_false_condition)


@pytest.fixture()
def experiment_c(two_false_measurements_true_condition_false_other_condition):
    """An experiment with two repetitions under multiple conditions."""
    return Experiment(two_false_measurements_true_condition_false_other_condition)


# Lists of Experiments


@pytest.fixture()
def experiments_ab(experiment_two_measurements_true_condition,
                   experiment_two_measurements_false_condition):
    """A list of experiments with the same conditions."""
    return [experiment_two_measurements_true_condition, experiment_two_measurements_false_condition]


@pytest.fixture()
def experiments_ac(experiment_two_measurements_true_condition, experiment_c):
    """A list of experiments with different numbers of conditions."""
    return [experiment_two_measurements_true_condition, experiment_c]


@pytest.fixture()
def experiments_abc(experiment_two_measurements_true_condition,
                    experiment_two_measurements_false_condition, experiment_c):
    """A list of three experiments """
    return [experiment_two_measurements_true_condition, experiment_two_measurements_false_condition, experiment_c]


# Experiment Sets


@pytest.fixture()
def experiment_set_ab(experiments_ab):
    """An experiment set of two experiments.

    An experiment set which contains two experiments sharing the same set of conditions
    with different values. Each experiment contains two repetitions with
    different data.
    """
    return ExperimentSet(experiments_ab)


@pytest.fixture()
def experiment_set_ac(experiments_ac):
    """An experiment set of two experiments.

    An experiment set which contains two experiments sharing a single condition and
    differing by a single condition. Each experiment contains two repetitions.
    """
    return ExperimentSet(experiments_ac)


@pytest.fixture()
def experiment_set_abc(experiments_abc):
    """An experiment set of three experiments sharing a single condition."""
    return ExperimentSet(experiments_abc)
