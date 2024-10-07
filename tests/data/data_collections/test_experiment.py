"""Tests for the `cralds.data.measurements.Experiment` class.

This module contains unit tests which are intended to thoroughly exercise the
public API of the Experiment class.

An experiment has a list of neasurements (which by definition must be
repetitions, i.e. they share the same set of measurement conditions. An
experiment can therefore be examined for variation in its dataset when the set
of conditions is held constant.
"""

import pytest
from numpy import array  # not unused - do not delete
from piblin.data.data_collections.measurement import Measurement
from piblin.data.data_collections.measurement_set import MeasurementSet
from piblin.data.data_collections.experiment import Experiment
from piblin.data.datasets.abc.split_datasets.zero_dimensional_dataset import ZeroDimensionalDataset


def test_getters(true_measurement_with_true_condition,
                 false_measurement_with_true_condition,
                 two_measurements_different_data_same_condition):
    """Ensure that public members are accessible."""
    assert Experiment([true_measurement_with_true_condition])[0] == true_measurement_with_true_condition
    # experiment = Experiment(two_measurements_different_data_same_condition)
    # assert experiment[0] == true_measurement_with_true_condition
    # assert experiment[1] == false_measurement_with_true_condition


@pytest.mark.xfail()
def test_conditions(two_measurements_different_data_same_condition,
                    experiment_two_measurements_true_condition):
    assert experiment_two_measurements_true_condition.conditions == two_measurements_different_data_same_condition[0].conditions


def test_repr(experiment_two_measurements_true_condition):
    assert eval(repr(experiment_two_measurements_true_condition)) == \
           experiment_two_measurements_true_condition


def test_comparison_not_implemented(experiment_two_measurements_true_condition):
    """Check that the comparison dunder is specifically not implemented."""
    assert experiment_two_measurements_true_condition.__cmp__() is \
           NotImplemented


@pytest.mark.xfail()
def test_visualize():

    temperature_dataset_a = ZeroDimensionalDataset.create(295.0, "temperature")
    humidity_dataset_a = ZeroDimensionalDataset.create(36.0, "humidity")

    temperature_dataset_b = ZeroDimensionalDataset.create(301.0, "temperature")
    humidity_dataset_b = ZeroDimensionalDataset.create(34.0, "humidity")

    measurement_a = Measurement([temperature_dataset_a, humidity_dataset_a])
    measurement_b = Measurement([temperature_dataset_b, humidity_dataset_b])

    experiment = Experiment.from_measurement_set(MeasurementSet([measurement_a, measurement_b]))

    fig, axes = experiment.visualize(include_text=False,
                                     expand_experiments=True,
                                     expand_replicates=False)
    return fig
