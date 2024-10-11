"""Modules for creating hierarchically structured data collections from files.

A set of modules for organizing experimental measurement data from files into meaningful hierarchies based on
metadata. The dataio subpackage provides a set of functions for reading data from files and directories with control
over the metadata parsing options and the type of hierarchy that is returned.

Modules
-------
data_collection_reader - Reading of data collections (data and metadata) from files and directories.
metadata_converter - Conversion between string and dictionary metadata forms.
pandas - Convert cralds data collections to pandas dataframes.
read_summary - Reading of data collections from summary .csv or .json files.

Functions
---------
read_file - Create a data collection from a single file.
read_files - Create a data collection from a list of files.
read_directory - Create a data collection from files in a given directory.
read_directories - Create a data collection from files in a given set of directories.
read_tabular_data - Create a data collection from tabular data in the file at the given path.
"""
from typing import List, Union
from .fileio.read.file_reader import FileReader
from .fileio.read.generic.generic_file_reader import GenericFileReader
from .fileio.read.multifile.multifile_reader import MultifileReader
from .fileio.read.summary.summary_file_reader import SummaryFileReader
from .fileio.write.file_writer import FileWriter

import piblin.dataio.data_collection_reader as data_collection_reader
import piblin.data.datasets.abc.dataset as dataset_
import piblin.data.data_collections.measurement as measurement_
import piblin.data.data_collections.measurement_set as measurement_set
import piblin.data.data_collections.consistent_measurement_set as consistent_measurement_set_
import piblin.data.data_collections.tidy_measurement_set as tidy_measurement_set_
import piblin.data.data_collections.experiment_set as experiment_set_


def read_file(filepath: str,
              parse_filename_metadata: bool = False,
              to_experiments: bool = True,
              minimize_hierarchy: bool = False,
              merge_redundant: bool = True,
              **read_kwargs) -> \
        Union[dataset_.Dataset,
              measurement_.Measurement,
              measurement_set.MeasurementSet,
              consistent_measurement_set_.ConsistentMeasurementSet,
              tidy_measurement_set_.TidyMeasurementSet,
              experiment_set_.ExperimentSet]:
    """Create a data collection from a single file.

    Parameters
    ----------
    filepath : str
        The path to the file containing measurements.
    parse_filename_metadata : bool
        Flag for whether to attempt to parse metadata from the filename.
        The default is False as this method for metadata input is purely a convenience.
    to_experiments : bool, default is True
        Whether to attempt to return a set of experiments.
    minimize_hierarchy : bool
        Whether to remove as much hierarchy as possible from the data collection.
        If a single experiment, measurement or dataset is read then they will be returned in their bare form if flatten
        is True.
        Default is to always return the most structured data collection.
    merge_redundant : bool
        Whether to merge measurements that are redundant based on conditions.

    Returns
    -------
    ExperimentSet
        The set of experiments contained in the file.
    """
    data_collection = \
        data_collection_reader.DataCollectionReader().from_filepath(filepath=filepath,
                                                                    parse_filename_metadata=parse_filename_metadata,
                                                                    to_experiments=to_experiments,
                                                                    minimize_hierarchy=minimize_hierarchy,
                                                                    merge_redundant=merge_redundant,
                                                                    **read_kwargs)

    return data_collection


def read_files(file_paths: List[str],
               parse_filename_metadata: bool = False,
               to_experiments: bool = True,
               minimize_hierarchy: bool = False,
               merge_redundant: bool = True,
               **read_kwargs) -> \
        Union[dataset_.Dataset,
              measurement_.Measurement,
              measurement_set.MeasurementSet,
              consistent_measurement_set_.ConsistentMeasurementSet,
              tidy_measurement_set_.TidyMeasurementSet,
              experiment_set_.ExperimentSet]:
    """Create a data collection from a list of files.

    Parameters
    ----------
    file_paths : list of str
        The locations of each file of interest.
    parse_filename_metadata : bool
        Flag for whether to attempt to parse metadata from the filename.
        The default is False as this method for metadata input is purely a convenience.
    to_experiments : bool, default is True
        Whether to attempt to return a set of experiments.
    minimize_hierarchy : bool
        Whether to remove as much hierarchy as possible from the data collection.
    merge_redundant : bool
        Whether to merge measurements that are redundant based on conditions.

    Returns
    -------
    ExperimentSet
        A single experiment set containing all measurements found in the
        listed files.
    """
    return data_collection_reader.DataCollectionReader().from_file_list(filelist=file_paths,
                                                                        parse_filename_metadata=parse_filename_metadata,
                                                                        to_experiments=to_experiments,
                                                                        minimize_hierarchy=minimize_hierarchy,
                                                                        merge_redundant=merge_redundant,
                                                                        **read_kwargs)


def read_directory(directory: str,
                   extension: str = None,
                   recursive: bool = False,
                   case_sensitive: bool = False,
                   parse_filename_metadata: bool = False,
                   to_experiments: bool = True,
                   minimize_hierarchy: bool = False,
                   merge_redundant: bool = True,
                   **read_kwargs) -> \
        Union[dataset_.Dataset,
              measurement_.Measurement,
              measurement_set.MeasurementSet,
              consistent_measurement_set_.ConsistentMeasurementSet,
              tidy_measurement_set_.TidyMeasurementSet,
              experiment_set_.ExperimentSet]:
    """Create a data collection from files in a given directory.

    Parameters
    ----------
    directory : str
        The location of the files of interest.
    extension : str
        The extension of the files of interest.
    recursive : bool
        Whether to search the directory for data recursively or not.
    case_sensitive : bool
        Whether to treat the extension as case-sensitive or not.
        The default is to ignore case.
    parse_filename_metadata : bool
        Flag for whether to attempt to parse metadata from the filename.
        The default is False as this method for metadata input is purely a convenience.
    to_experiments : bool, default is True
        Whether to attempt to return a set of experiments.
    minimize_hierarchy : bool
        Whether to remove as much hierarchy as possible from the data collection.
    merge_redundant : bool
        Whether to merge measurements that are redundant based on conditions.

    Returns
    -------
    ExperimentSet
        A single experiment set containing all measurements found in the
        specified location.
    """
    if extension is not None:
        return data_collection_reader.DataCollectionReader().from_directory_with_extension(
            directory=directory,
            extension=extension,
            recursive=recursive,
            case_sensitive=case_sensitive,
            parse_filename_metadata=parse_filename_metadata,
            to_experiments=to_experiments,
            minimize_hierarchy=minimize_hierarchy,
            merge_redundant=merge_redundant,
            **read_kwargs
        )
    else:
        return data_collection_reader.DataCollectionReader().from_directory(
            directory=directory,
            recursive=recursive,
            parse_filename_metadata=parse_filename_metadata,
            to_experiments=to_experiments,
            minimize_hierarchy=minimize_hierarchy,
            merge_redundant=merge_redundant,
            **read_kwargs
        )


def read_directories(directories: List[str],
                     extension: str = None,
                     recursive: bool = False,
                     case_sensitive: bool = False,
                     parse_filename_metadata: bool = False,
                     to_experiments: bool = True,
                     minimize_hierarchy: bool = False,
                     merge_redundant: bool = True,
                     **read_kwargs) -> \
        Union[dataset_.Dataset,
              measurement_.Measurement,
              measurement_set.MeasurementSet,
              consistent_measurement_set_.ConsistentMeasurementSet,
              tidy_measurement_set_.TidyMeasurementSet,
              experiment_set_.ExperimentSet]:
    """Create a data collection from files in a given set of directories.

    Parameters
    ----------
    directories : list of str
        The location of the files of interest.
    extension : str
        The extension of the files of interest.
    recursive : bool
        Whether to search the directory for data recursively or not.
    case_sensitive : bool
        Whether to treat the extension as case-sensitive or not.
        The default is to ignore case.
    parse_filename_metadata : bool
        Flag for whether to attempt to parse metadata from the filename.
        The default is False as this method for metadata input is purely a convenience.
    to_experiments : bool, default is True
        Whether to attempt to return a set of experiments.
    minimize_hierarchy : bool
        Whether to remove as much hierarchy as possible from the data collection.
    merge_redundant : bool
        Whether to merge measurements that are redundant based on conditions.

    Returns
    -------
    ExperimentSet
        A single experiment set containing all measurements found in the
        specified locations.
    """
    return data_collection_reader.DataCollectionReader().from_directories(
        directories=directories,
        extension=extension,
        recursive=recursive,
        case_sensitive=case_sensitive,
        parse_filename_metadata=parse_filename_metadata,
        to_experiments=to_experiments,
        minimize_hierarchy=minimize_hierarchy,
        merge_redundant=merge_redundant,
        **read_kwargs
    )


def read_tabular_data(filepath: str,
                      n_metadata_columns: int = 0,
                      dataset_types: List = None,
                      dataset_end_indices: List[int] = None,
                      to_experiments: bool = True,
                      minimize_hierarchy: bool = False,
                      merge_redundant: bool = True,
                      **read_kwargs) -> \
        Union[dataset_.Dataset,
              measurement_.Measurement,
              measurement_set.MeasurementSet,
              consistent_measurement_set_.ConsistentMeasurementSet,
              tidy_measurement_set_.TidyMeasurementSet,
              experiment_set_.ExperimentSet]:
    """Create a data collection from tabular data in the file at the given path.

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
    ExperimentSet
        The set of experiments contained in the tabular data file.
    """
    return data_collection_reader.DataCollectionReader().from_tabular_data_filepath(
        filepath=filepath,
        n_metadata_columns=n_metadata_columns,
        dataset_types=dataset_types,
        dataset_end_indices=dataset_end_indices,
        to_experiments=to_experiments,
        minimize_hierarchy=minimize_hierarchy,
        merge_redundant=merge_redundant,
        **read_kwargs
    )
