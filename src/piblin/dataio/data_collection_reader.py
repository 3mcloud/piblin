"""Reading of measurements (data and metadata) from files.

This module provides the ability to read a set of measurement objects from a
set of one or more files located on disk. Several optional arguments are
included for filtering files with different extensions and reading metadata.

Classes
-------
MeasurementSetReader - Read measurements from files and directories thereof.
"""
import json
import csv
import pathlib
from pathlib import Path
from typing import List, Dict, Union
from piblin.dataio.fileio import list_files_in_directory
from piblin.dataio.fileio.autodetect import potential_file_readers_by_extension
from piblin.dataio.fileio.read import FileParsingException
import piblin.dataio.metadata_converter as filename_metadata
import piblin.data.data_collections.measurement_set as measurement_set_
import piblin.data.data_collections.consistent_measurement_set as consistent_measurement_set_
import piblin.data.data_collections.tidy_measurement_set as tidy_measurement_set_
import piblin.data.data_collections.tabular_measurement_set as tb
import piblin.data.data_collections.experiment_set as experiment_set_
import piblin.data.data_collections.data_collection_factory as measurement_set_factory


class UnsupportedFileSubtypeError(Exception):
    """Raised when the subtype of a specific file extension cannot be read."""


class DataCollectionReader(object):
    """Read measurements from files and directories thereof.

    Parameters
    ----------
    metadata_converter : MetadataConverter or Callable
        A converter between metadata dicts and strings.
        A default instance is created if none is provided via an argument.

    Raises
    ------
    TypeError
        If provided `metadata_converter` is not an instance of `MetadataConverter`
        class or a callable function.
    """
    __DIRECTORY_METADATA_FILENAME = "shared_metadata.json"

    def __init__(self, metadata_converter: filename_metadata.MetadataConverter = None):

        if metadata_converter is None:
            self.__metadata_converter = filename_metadata.MetadataConverter()
        elif isinstance(metadata_converter, filename_metadata.MetadataConverter) or callable(metadata_converter):
            self.__metadata_converter = metadata_converter
        else:
            raise TypeError("`metadata_converter` must be an instance of `MetadataConverter` or a callable function")

    def from_filepath(self,
                      filepath: str,
                      parse_filename_metadata: bool = False,
                      to_experiments: bool = True,
                      minimize_hierarchy: bool = False,
                      merge_redundant: bool = True,
                      **read_kwargs) -> \
            Union[measurement_set_.MeasurementSet,
                  consistent_measurement_set_.ConsistentMeasurementSet,
                  tidy_measurement_set_.TidyMeasurementSet,
                  experiment_set_.ExperimentSet]:
        """Read measurements from a file at a given filepath.

        Reading a file requires automatic determination of the particular
        reader which can parse the dataset and metadata from it. This function
        checks the package for applicable readers and tries them out.
        This function also handles checking for external metadata, either supplied
        in the filename or in a supporting per-file auxiliary .json file. The latter
        of these takes precedence when duplicate keys are present.

        Parameters
        ----------
        filepath : str
            The path to the (non-json) file to be read.
        parse_filename_metadata : bool
            Whether to attempt to parse metadata from the filename.
        to_experiments : bool, default is True
            Whether to attempt to return a set of experiments.
        minimize_hierarchy : bool
            Whether to remove as much hierarchy as possible from the data collection.
        merge_redundant : bool
            Whether to merge measurements that are redundant based on conditions.

        Returns
        -------
        MeasurementSet
            The measurements from the file.

        Raises
        ------
        UnsupportedFileSubtypeError
            When no potential file reader is able to read the given file.
        """
        def __parse_filename_metadata(parse_filename_metadata_: bool) -> Dict:
            """Extract external metadata from the filename.

            Parameters
            ----------
            parse_filename_metadata_ : bool
                Whether to parse filename metadata or not.

            Returns
            -------
            dict
                Any metadata extracted from the filename.
            """
            if parse_filename_metadata_:
                if isinstance(self.__metadata_converter, filename_metadata.MetadataConverter):
                    return self.__metadata_converter.string_to_dict(filename)
                else:
                    # Assume it is a callable function
                    return self.__metadata_converter(str(filepath.resolve()))
            else:
                return {}

        def __parse_metadata_file(filepath_: pathlib.Path) -> Dict:
            """Extract external metadata from the supporting metadata file.

            Parameters
            ----------
            filepath_ : pathlib.Path
                The path to the supporting metadata file.

            Returns
            -------
            dict
                Any metadata present in the supporting metadata file.

            """
            metadata_filepath = filepath_.parent.joinpath(filepath_.stem + ".json")
            if metadata_filepath.is_file():
                with open(metadata_filepath) as json_file:
                    try:
                        return json.load(json_file)
                    except json.decoder.JSONDecodeError:
                        raise ValueError(f"Decode error for .json file: {metadata_filepath}")
            else:
                return {}

        filepath = Path(filepath)
        filename = filepath.stem
        extension = filepath.suffix

        external_conditions = __parse_filename_metadata(parse_filename_metadata)
        external_conditions.update(__parse_metadata_file(filepath))

        potential_file_readers = potential_file_readers_by_extension(extension[1:])
        if len(potential_file_readers) == 0:
            raise UnsupportedFileSubtypeError(f"No cralds file reader can parse files with the extension {extension}")

        measurements = measurement_set_.MeasurementSet(merge_redundant=merge_redundant)
        error_messages = []
        for file_reader in potential_file_readers:
            try:
                measurements = \
                    file_reader().data_from_filepath(str(filepath), **read_kwargs)

                for measurement in measurements.measurements:
                    for filename_key, filename_value in external_conditions.items():
                        measurement.add_condition(filename_key, filename_value)

                break

            except FileParsingException as file_parsing_exception:

                error_messages.append(f"{file_reader.__module__}.{file_reader.__name__}: "
                                      f"{repr(file_parsing_exception)}")
                if file_reader is potential_file_readers[-1]:
                    message = f"No appropriate file reader was found for {filepath} " \
                                    f"with extension {extension[1:]}\n"
                    message += "File Parsing Exceptions:\n"
                    for error_message in error_messages:
                        message += f"{error_message}\n"

                    raise UnsupportedFileSubtypeError(message)

        return measurement_set_factory.DataCollectionFactory.from_measurements(measurements.measurements,
                                                                               to_experiments=to_experiments,
                                                                               minimize_hierarchy=minimize_hierarchy,
                                                                               merge_redundant=merge_redundant)

    @staticmethod
    def parse_directory_metadata(directory: str):
        """Extract any external metadata from the directory .json file.

        Parameters
        ----------
        directory : str
            The directory to load external metadata from.

        Returns
        -------
        dict
            A dictionary containing directory metadata.
        """
        directory_metadata_filepath = Path(directory).joinpath(DataCollectionReader.__DIRECTORY_METADATA_FILENAME)
        if directory_metadata_filepath.is_file():
            with open(directory_metadata_filepath) as json_file:
                try:
                    return json.load(json_file)
                except json.decoder.JSONDecodeError:
                    raise ValueError(f"Decode error for directory .json file: {directory_metadata_filepath}")
        else:
            return {}

    def from_directory(self,
                       directory: str,
                       recursive: bool = False,
                       parse_filename_metadata: bool = False,
                       to_experiments: bool = True,
                       minimize_hierarchy=False,
                       merge_redundant: bool = True,
                       **read_kwargs) -> \
            Union[measurement_set_.MeasurementSet,
                  consistent_measurement_set_.ConsistentMeasurementSet,
                  tidy_measurement_set_.TidyMeasurementSet,
                  experiment_set_.ExperimentSet]:
        """Read a set of measurements from all files in a given directory.

        This method uses the from_filepath method of this class.

        Parameters
        ----------
        directory : str
            The path to the directory containing files to be read.
        recursive : bool
            Whether to include the contents of subdirectories.
        parse_filename_metadata : bool
            Whether to attempt to parse metadata from the filename.
        to_experiments : bool, default is True
            Whether to attempt to return a set of experiments.
        minimize_hierarchy : bool
            Whether to remove as much hierarchy as possible from the data collection.
        merge_redundant : bool
            Whether to merge measurements that are redundant based on conditions.

        Returns
        -------
        MeasurementSet
            The set of measurements contained in the files in the directory.
        """
        directory_metadata = self.parse_directory_metadata(directory)

        measurements = []
        for filepath in list_files_in_directory(directory, recursive=recursive):
            if Path(filepath).suffix != ".json":
                file_measurements = self.from_filepath(filepath,
                                                       parse_filename_metadata,
                                                       **read_kwargs)

                measurements.extend(file_measurements.measurements)

        for measurement in measurements:
            for key, value in directory_metadata.items():
                measurement.add_condition(key, value)

        return measurement_set_factory.DataCollectionFactory.from_measurements(measurements,
                                                                               to_experiments=to_experiments,
                                                                               minimize_hierarchy=minimize_hierarchy,
                                                                               merge_redundant=merge_redundant)

    def from_directory_with_extension(self,
                                      directory: str,
                                      extension: str,
                                      recursive: bool = False,
                                      case_sensitive: bool = False,
                                      parse_filename_metadata: bool = False,
                                      to_experiments: bool = True,
                                      minimize_hierarchy: bool = False,
                                      merge_redundant: bool = True,
                                      **read_kwargs) -> \
            Union[measurement_set_.MeasurementSet,
                  consistent_measurement_set_.ConsistentMeasurementSet,
                  tidy_measurement_set_.TidyMeasurementSet,
                  experiment_set_.ExperimentSet]:
        """Read all files in a given directory with a given extension.

        Parameters
        ----------
        directory : str
            The path to the directory containing files to be read.
        extension : str
            The extension of files to be read.
        recursive : bool
            Whether to find files in subdirectories of the specified directory.
        case_sensitive : bool
            Whether to treat provided extension as case-sensitive.
        parse_filename_metadata : bool
            Whether to attempt to parse metadata from the filename.
        to_experiments : bool, default is True
            Whether to attempt to return a set of experiments.
        minimize_hierarchy : bool
            Whether to remove as much hierarchy as possible from the data collection.
        merge_redundant : bool
            Whether to merge measurements that are redundant based on conditions.

        Returns
        -------
        MeasurementSet
            The measurements from the files in the directory.
        """
        directory_metadata = self.parse_directory_metadata(directory)

        measurements = []
        for filepath in list_files_in_directory(directory,
                                                extension,
                                                case_sensitive,
                                                recursive):

            if Path(filepath).suffix != ".json":

                file_measurements = self.from_filepath(filepath,
                                                       parse_filename_metadata,
                                                       **read_kwargs)

                measurements.extend(file_measurements.measurements)

        for measurement in measurements:
            for key, value in directory_metadata.items():
                measurement.add_condition(key, value)

        return measurement_set_factory.DataCollectionFactory.from_measurements(measurements,
                                                                               to_experiments=to_experiments,
                                                                               minimize_hierarchy=minimize_hierarchy,
                                                                               merge_redundant=merge_redundant)

    def from_directories(self,
                         directories: List[str],
                         extension: str,
                         recursive: bool = False,
                         case_sensitive: bool = False,
                         parse_filename_metadata: bool = False,
                         to_experiments: bool = True,
                         minimize_hierarchy: bool = False,
                         merge_redundant: bool = True,
                         **read_kwargs) -> \
            Union[measurement_set_.MeasurementSet,
                  consistent_measurement_set_.ConsistentMeasurementSet,
                  tidy_measurement_set_.TidyMeasurementSet,
                  experiment_set_.ExperimentSet]:
        """Read all files in given directories with a given extension.

        Parameters
        ----------
        directories : list of str
            The paths to the directories containing files to be read.
        extension : str
            The extension of files to be read.
        recursive : bool
            Whether to find files in subdirectories of the specified directories.
        case_sensitive : bool
            Whether to treat provided extension as case-sensitive.
        parse_filename_metadata : bool
            Whether to attempt to parse metadata from the filename.
        to_experiments : bool, default is True
            Whether to attempt to return a set of experiments.
        minimize_hierarchy : bool
            Whether to remove as much hierarchy as possible from the data collection.
        merge_redundant : bool
            Whether to merge measurements that are redundant based on conditions.

        Returns
        -------
        MeasurementSet
            The measurements from the files in the directories.
        """
        measurements = measurement_set_.MeasurementSet(merge_redundant=merge_redundant)
        for directory in directories:
            directory_measurements = self.from_directory_with_extension(directory,
                                                                        extension,
                                                                        recursive,
                                                                        case_sensitive,
                                                                        parse_filename_metadata,
                                                                        **read_kwargs)
            measurements.extend(directory_measurements.measurements)

        return measurement_set_factory.DataCollectionFactory.from_measurements(measurements.measurements,
                                                                               to_experiments=to_experiments,
                                                                               minimize_hierarchy=minimize_hierarchy,
                                                                               merge_redundant=merge_redundant)

    def from_file_list(self,
                       filelist: List[str],
                       parse_filename_metadata: bool = False,
                       to_experiments: bool = True,
                       minimize_hierarchy: bool = False,
                       merge_redundant: bool = True,
                       **read_kwargs) -> \
            Union[measurement_set_.MeasurementSet,
                  consistent_measurement_set_.ConsistentMeasurementSet,
                  tidy_measurement_set_.TidyMeasurementSet,
                  experiment_set_.ExperimentSet]:
        """Read all measurements in a list of files.

        Parameters
        ----------
        filelist : list
            The paths to the individual files to be read.
        parse_filename_metadata : bool
            Whether to attempt to parse metadata from the filename.
        to_experiments : bool, default is True
            Whether to attempt to return a set of experiments.
        minimize_hierarchy : bool
            Whether to remove as much hierarchy as possible from the data collection.
        merge_redundant : bool
            Whether to merge measurements that are redundant based on conditions.

        Returns
        -------
        MeasurementSet
            The measurements from the files in the directory.
        """
        measurements = []
        for filepath in filelist:
            if Path(filepath).suffix != ".json":

                file_measurements = self.from_filepath(
                    filepath=filepath,
                    parse_filename_metadata=parse_filename_metadata,
                    to_experiments=to_experiments,
                    minimize_hierarchy=minimize_hierarchy,
                    merge_redundant=merge_redundant,
                    **read_kwargs)

                measurements.extend(file_measurements.measurements)

        return measurement_set_factory.DataCollectionFactory\
            .from_measurements(measurements=measurements,
                               to_experiments=to_experiments,
                               minimize_hierarchy=minimize_hierarchy,
                               merge_redundant=merge_redundant)

    @staticmethod
    def from_tabular_data_filepath(filepath: str,
                                   n_metadata_columns: int = 0,
                                   dataset_types: List = None,
                                   dataset_end_indices: List[int] = None,
                                   to_experiments: bool = True,
                                   minimize_hierarchy: bool = False,
                                   merge_redundant: bool = True,
                                   **read_kwargs) -> \
            Union[measurement_set_.MeasurementSet,
                  consistent_measurement_set_.ConsistentMeasurementSet,
                  tidy_measurement_set_.TidyMeasurementSet,
                  experiment_set_.ExperimentSet]:
        """Read a spreadsheet-like data table as a measurement set.

        Each column has a header and values of that property for each
        measurement (i.e. row). These data may be missing in which case we
        simply don't define their values in the metadata dictionaries of each
        measurement.
        There is a distinction between inputs and outputs that has to be made
        when reading this file.

        Parameters
        ----------
        filepath : str
            The path to the tabular data file.
        n_metadata_columns : int
            The number of initial columns of the file to treat as metadata.
        dataset_types : list of class
            The type of each dataset in the tabular data.
        dataset_end_indices : list of int
            The length of each dataset in the tabular data.
        to_experiments : bool, default is True
            Whether to attempt to return a set of experiments.
        minimize_hierarchy : bool
            Whether to remove as much hierarchy as possible from the data collection.
        merge_redundant : bool
            Whether to merge measurements that are redundant based on conditions.

        Returns
        -------
        MeasurementSet
            The set of measurements contained in the tabular data file.
        """
        data = []
        with open(filepath) as csvfile:
            csv_reader = csv.reader(csvfile)
            for row in csv_reader:
                data.append(row)

        tabular_dataset = tb.TabularMeasurementSet(data,
                                                   n_metadata_columns=n_metadata_columns,
                                                   column_headers=None,
                                                   dataset_types=dataset_types,
                                                   dataset_end_indices=dataset_end_indices)

        return measurement_set_factory.DataCollectionFactory.from_measurements(
            tabular_dataset.to_measurement_set().measurements,
            to_experiments=to_experiments,
            minimize_hierarchy=minimize_hierarchy,
            merge_redundant=merge_redundant)
