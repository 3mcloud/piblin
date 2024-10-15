from typing import List, Union
import piblin.data.datasets.abc.dataset as dataset_
import piblin.data.data_collections.measurement as measurement_
import piblin.data.data_collections.measurement_set as measurement_set_
import piblin.data.data_collections.consistent_measurement_set as consistent_measurement_set_
import piblin.data.data_collections.tidy_measurement_set as tidy_measurement_set_
import piblin.data.data_collections.experiment_set as experiment_set_


class DataCollectionFactory(object):
    """A class to create instances of measurement set subclasses."""
    @staticmethod
    def from_measurements(measurements: List[measurement_.Measurement],
                          to_experiments: bool = True,
                          minimize_hierarchy: bool = False,
                          merge_redundant: bool = True) -> Union[measurement_set_.MeasurementSet,
                                                                 consistent_measurement_set_.ConsistentMeasurementSet,
                                                                 tidy_measurement_set_.TidyMeasurementSet,
                                                                 "experiment_set_.ExperimentSet"]:
        """Create an appropriate measurement set subclass from given measurements.

        Given a set of measurements, creates an instance of the least structured data collection class (the measurement
        set), then based on the properties of that measurement set (tidiness and consistency) creates the most
        structured data collection class possible. By default, a measurement set that is tidy or consistent is converted
        to the corresponding experiment set unless the caller specifies that conversion to experiments should not be
        carried out. In the latter case a tidy or consistent measurement set will be returned in lieu of an experiment
        set.

        Parameters
        ----------
        measurements : List of Measurement
            The measurements to convert to the appropriate measurement set subclass.
        to_experiments : bool
            Whether to convert to an experiment set if measurements are consistent.
        minimize_hierarchy : bool
            Whether to remove as much hierarchy as possible from the data collection.
        merge_redundant : bool
            Whether to merge measurements that are redundant based on conditions.

        Returns
        -------
        MeasurementSet or ConsistentMeasurementSet or TidyMeasurementSet or ExperimentSet
            The appropriate measurement set subclass.
        """
        measurement_set = measurement_set_.MeasurementSet(measurements,
                                                          merge_redundant=merge_redundant)

        if measurement_set.is_tidy and measurement_set.is_consistent:
            if not to_experiments:
                measurement_set = tidy_measurement_set_.TidyMeasurementSet(measurements=measurements,
                                                                           merge_redundant=merge_redundant)
            else:
                measurement_set = experiment_set_.ExperimentSet(measurements=measurements,
                                                                merge_redundant=merge_redundant)

        elif measurement_set.is_consistent and not measurement_set.is_tidy:
            if not to_experiments:
                measurement_set = consistent_measurement_set_.ConsistentMeasurementSet(measurements=measurements,
                                                                                       merge_redundant=merge_redundant)
            else:
                measurement_set = experiment_set_.ExperimentSet(measurements=measurements,
                                                                merge_redundant=merge_redundant)

        else:
            if to_experiments:
                print(f"Warning: Measurements are not consistent so could not be converted to experiments. "
                      f"A measurement set has been created.")
                print(measurement_set.consistency_str)
        if minimize_hierarchy:
            return DataCollectionFactory.__minimize_hierarchy(measurement_set)
        else:
            return measurement_set

    @staticmethod
    def __minimize_hierarchy(data_collection: Union[measurement_set_.MeasurementSet,
                                                    consistent_measurement_set_.ConsistentMeasurementSet,
                                                    tidy_measurement_set_.TidyMeasurementSet,
                                                    "experiment_set_.ExperimentSet"]) -> \
            Union[dataset_.Dataset,
                  measurement_.Measurement,
                  measurement_set_.MeasurementSet,
                  consistent_measurement_set_.ConsistentMeasurementSet,
                  tidy_measurement_set_.TidyMeasurementSet,
                  "experiment_set_.ExperimentSet"]:

        if len(data_collection) == 1:
            experiment = data_collection[0]
            if len(experiment) == 1:
                repetitions = experiment.datasets
                if len(repetitions) == 1:  # one repetition
                    if len(repetitions[0]) == 1:
                        return repetitions[0][0]
                    else:
                        return experiment  # should be a measurement
                else:
                    return experiment
            else:
                return experiment
        else:
            return data_collection
