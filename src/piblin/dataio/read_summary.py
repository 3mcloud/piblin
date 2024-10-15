"""
This module collects functionality related to reading summary files, which are single files connecting raw data
files to metadata. This is a safer alternative to the practice of putting metadata in filenames, which will still be
supported but has proven unscalable as expected.

It is important to note that this functionality only connects files containing experimental data to the relevant
condition and detail metadata. It is not intended as a storage format for sets of experiments - the replicate
relationships will be re-inferred when the summary file is read, and any transformations are lost when saving a
summary file. In general, this approach is more uesful for initially inputting metadata from an organized file
to be added to the datasets in question. Saving of experiment sets is currently done by pickling, with planned
support for an experiment set and pipeline file format in future.

"""
import os.path
import re
import csv
from piblin.dataio import read_file
import piblin.data.data_collections.experiment_set as experiment_set


# TODO - this should be a summary reader instead of its own isolated module.
def write_summary_csv(filepath: str) -> None:
    """Write a summary csv file.

    In cases where partial data has been accrued, write a .csv file for a
    scientist to use to add more files and metadata.

    Parameters
    ----------
    filepath : str
        The path to the file to be created.
    """
    raise NotImplementedError("Saving of summary csv files is not implemented")


def write_summary_json(filepath: str) -> None:
    """Write a summary json file.

    In cases where data files and metadata need to be stored together
    for future use, the summary should be written in json format.

    Parameters
    ----------
    filepath : str
        The path to the file to be created.
    """
    raise NotImplementedError("Saving of summary json files is not implemented")


def read_summary_csv(csv_filepath: str,
                     num_filenames: int = 1,
                     num_details: int = 0):
    """Read an experiment set from a summary .csv.

    Each line of the .csv must become a measurement, so must be parsed to filename(s), conditions and details.

    Parameters
    ----------
    csv_filepath : str
        The path to the csv file to be read.
    num_filenames : int
        The number of filenames for each measurement (row).
        The default is a single filename per measurement in the summary file.
    num_details : int
        The number of details for each measurement (row).
        The default is to place no detail metadata in the summary file.
    """
    with open(csv_filepath, newline='') as csv_file:

        reader = csv.reader(csv_file, delimiter=',')

        metadata_names = next(reader)[num_filenames:]
        num_conditions = len(metadata_names) - num_details

        condition_names = metadata_names[0:num_conditions]
        detail_names = metadata_names[num_conditions:]

        summary_dir, _ = os.path.split(csv_filepath)

        all_measurements = []
        for row in reader:

            datafile_paths = []
            for datafile_name in row[0:num_filenames]:

                file_dir, _ = os.path.split(datafile_name)
                if file_dir == '':
                    datafile_path = os.path.join(summary_dir, datafile_name)
                else:
                    datafile_path = datafile_name

                if not os.path.exists(datafile_path):

                    dir, name = os.path.split(datafile_path)
                    name = sanitize_filename_like_databricks(name)
                    datafile_path = os.path.join(dir, name)

                    if not os.path.exists(datafile_path):
                        raise IOError("File path in summary .csv does not exist.", datafile_path)

                datafile_paths.append(datafile_path)

            for datafile_path in datafile_paths:
                file_measurements = read_file(datafile_path).measurements

            metadata_values = row[num_filenames:]
            condition_values = metadata_values[0:len(metadata_names) - num_details]
            detail_values = metadata_values[num_conditions:]

            conditions = {name: value for name, value in zip(condition_names, condition_values)}
            details = {name: value for name, value in zip(detail_names, detail_values)}

            for name, value in conditions.items():
                file_measurements.add_condition(name, value)

            for name, value in details.items():
                file_measurements.add_detail(name, value)

            all_measurements.append(file_measurements)

        return experiment_set.ExperimentSet.from_measurement_sets(all_measurements)


def read_summary_json():
    """Read an experiment set from a summary .json."""
    return NotImplemented


def sanitize_filename_like_databricks(filename):
    """Process a filename in the same manner as the databricks uploader GUI.

    When files are moved to databricks via its uploader GUI, the files are renamed
    by replacing all non-alphanumerics with underscores. If a summary file is used,
    the filenames in the summary file will be incorrect and the reader will fail.
    By replicating the databricks renaming scheme, it is possible to alter the
    filenames from the summary file so that they will match the databricks equivalent.
    This is a best guess at the databricks renaming scheme based on experience.

    Parameters
    ----------
    filename : str
        The name of the file to sanitize like databricks.

    Returns
    -------
    str
        The sanitized filename.
    """
    return re.sub('[^0-9a-zA-Z]+', '_', os.path.splitext(filename)[0]) + os.path.splitext(filename)[1]
