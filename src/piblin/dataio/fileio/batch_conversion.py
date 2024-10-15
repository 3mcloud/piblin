"""
Module providing functions for using cralds to perform batch conversion of files. This makes the file reader classes
of cralds available to users who wish to make use of other analysis tools which require a standard file format.
There should be a function here corresponding to each of the functions in the dataio public interface except for those
dealing with tabular data.

convert_file - Read a single file and write its contents to an output file.
convert_files - Read a set of files and write their contents to output files.
convert_directory - Read a directory and write its contents to output files.
convert_directories - Read directories and write their contents to output files.

All of these functions work by first calling the appropriate dataio function and then write the resulting experiment set
out with each dataset in an individual .csv file.
"""
from typing import List
import os
import copy
from glob import glob
import piblin.dataio as dataio


def convert_file(filepath: str,
                 output_directory: str = None,
                 allow_overwrite: bool = False,
                 flatten_to_single_file: bool = False,
                 create_metadata_filenames: bool = True,
                 suppress_condition_names: bool = False,
                 suppress_replicate_index: bool = True,
                 parse_filename_metadata: bool = False,
                 details_to_include: List[str] = None,
                 **read_kwargs) -> None:
    """Read a single file and write its contents to .csv files.

    Parameters
    ----------
    filepath : str
        The path to the file to be converted.
    output_directory : str
        The path to the directory in which to place converted data.
        By default, this is the path to the directory containing the input file.
    allow_overwrite : bool
        Whether to allow output files to overwrite existing files.
        Default is false for user safety.
    flatten_to_single_file : bool
        Whether to place the files in a single output file.
        Default is to create a file per dataset.
    create_metadata_filenames : bool
        Whether to create output filenames based on condition metadata.
        Default is to create rather than re-use the original source filename.
    suppress_condition_names : bool
        Whether to suppress condition names when creating output filenames.
        Default is to suppress names for readability.
    suppress_replicate_index : bool
        Whether to suppress replicate index when creating output filenames.
        Default is to suppress.
    parse_filename_metadata : bool
        Flag for whether to attempt to parse metadata from the
        filename.
        The default is False as this method for metadata input is purely a
        convenient stopgap.
    details_to_include : list of str
        Names of details to include in created metadata filenames.
    """
    if details_to_include is None:
        details_to_include = []

    experiment_set = dataio.read_file(filepath,
                                      attempt_flatten=False,
                                      parse_filename_metadata=parse_filename_metadata,
                                      **read_kwargs)

    if output_directory is None:
        output_directory = os.path.dirname(os.path.abspath(filepath))

    __convert_experiment_set(experiment_set,
                             output_directory,
                             allow_overwrite,
                             flatten_to_single_file,
                             create_metadata_filenames,
                             suppress_condition_names,
                             suppress_replicate_index,
                             details_to_include)


def convert_files(file_paths: List[str],
                  output_directory: str = None,
                  allow_overwrite: bool = False,
                  flatten_to_single_file: bool = False,
                  create_metadata_filenames: bool = True,
                  suppress_condition_names: bool = False,
                  suppress_replicate_index: bool = True,
                  parse_filename_metadata: bool = False,
                  details_to_include: List[str] = None,
                  **read_kwargs) -> None:
    """Read a set of files and write their contents to .csv files.

    Parameters
    ----------
    file_paths : list of str
        The locations of each file of interest.
    output_directory : str
            The path to the directory in which to place converted data.
            Default is to use the common prefix of all supplied files.
    allow_overwrite : bool
        Whether to allow output files to overwrite existing files.
        Default is false for user safety.
    flatten_to_single_file : bool
        Whether to place the files in a single output file.
        Default is to create a file per dataset.
    create_metadata_filenames : bool
        Whether to create output filenames based on condition metadata.
        Default is to create rather than re-use the original source filename.
    suppress_condition_names : bool
        Whether to suppress condition names when creating output filenames.
        Default is to suppress names for readability.
    suppress_replicate_index : bool
        Whether to suppress replicate index when creating output filenames.
        Default is to suppress.
    parse_filename_metadata : bool
        Flag for whether to attempt to parse metadata from the
        filename.
        The default is False as this method for metadata input is purely a
        convenient stopgap.
    details_to_include : list of str
        Names of details to include in created metadata filenames.
    """
    experiment_set = dataio.read_files(file_paths,
                                       parse_filename_metadata=parse_filename_metadata,
                                       **read_kwargs)
    if output_directory is None:
        output_directory = os.path.dirname(os.path.commonprefix(file_paths))

    __convert_experiment_set(experiment_set,
                             output_directory,
                             allow_overwrite,
                             flatten_to_single_file,
                             create_metadata_filenames,
                             suppress_condition_names,
                             suppress_replicate_index,
                             details_to_include)


def convert_directory(directory: str,
                      output_directory: bool = None,
                      allow_overwrite: bool = False,
                      flatten_to_single_file: bool = False,
                      create_metadata_filenames: bool = True,
                      suppress_condition_names: bool = False,
                      suppress_replicate_index: bool = True,
                      extension: str = None,
                      recursive: bool = False,
                      case_sensitive: bool = False,
                      parse_filename_metadata: bool = False,
                      details_to_include: List[str] = None,
                      **read_kwargs) -> None:
    """Read a directory and write its contents to .csv files.

    Parameters
    ----------
    directory : str
        The location of the files of interest.
    output_directory : str
        The path to the directory in which to place converted data.
        Default is to use the same directory as the input.
    allow_overwrite : bool
        Whether to allow output files to overwrite existing files.
        Default is false for user safety.
    flatten_to_single_file : bool
        Whether to place the files in a single output file.
        Default is to create a file per dataset.
    create_metadata_filenames : bool
        Whether to create output filenames based on condition metadata.
        Default is to create rather than re-use the original source filename.
    suppress_condition_names : bool
        Whether to suppress condition names when creating output filenames.
        Default is to suppress names for readability.
    suppress_replicate_index : bool
        Whether to suppress replicate index when creating output filenames.
        Default is to suppress.
    extension : str
        The extension of the files of interest.
    recursive : bool
        Whether to search the directory for data recursively or not.
    case_sensitive : bool
        Whether to treat the extension as case-sensitive or not.
        The default is to ignore case.
    parse_filename_metadata : bool
        Flag for whether to attempt to parse metadata from the
        filename.
        The default is False as this method for metadata input is purely a
        convenient stopgap.
    details_to_include : list of str
        Names of details to include in created metadata filenames.
    """
    experiment_set = dataio.read_directory(directory,
                                           extension=extension,
                                           recursive=recursive,
                                           case_sensitive=case_sensitive,
                                           parse_filename_metadata=parse_filename_metadata,
                                           **read_kwargs)

    if output_directory is None:
        output_directory = directory

    __convert_experiment_set(experiment_set,
                             output_directory,
                             allow_overwrite,
                             flatten_to_single_file,
                             create_metadata_filenames,
                             suppress_condition_names,
                             suppress_replicate_index,
                             details_to_include)


def convert_directories(directories: List[str],
                        output_directory: str = None,
                        allow_overwrite: bool = False,
                        flatten_to_single_file: bool = False,
                        create_metadata_filenames: bool = True,
                        suppress_condition_names: bool = False,
                        suppress_replicate_index: bool = True,
                        extension: str = None,
                        recursive: bool = False,
                        case_sensitive: bool = False,
                        parse_filename_metadata: bool = False,
                        details_to_include: List[str] = None,
                        **read_kwargs) -> None:
    """Read directories and write their contents to .csv files.

    Parameters
    ----------
    directories : list of str
        The location of the files of interest.
    output_directory : str
        The path to the directory in which to place converted data.
        Default is to use the common prefix of all supplied directories.
    allow_overwrite : bool
        Whether to allow output files to overwrite existing files.
        Default is false for user safety.
    flatten_to_single_file : bool
        Whether to place the files in a single output file.
        Default is to create a file per dataset.
    create_metadata_filenames : bool
        Whether to create output filenames based on condition metadata.
        Default is to create rather than re-use the original source filename.
    suppress_condition_names : bool
        Whether to suppress condition names when creating output filenames.
        Default is to suppress names for readability.
    suppress_replicate_index : bool
        Whether to suppress replicate index when creating output filenames.
        Default is to suppress.
    extension : str
        The extension of the files of interest.
    recursive : bool
        Whether to search the directory for data recursively or not.
    case_sensitive : bool
        Whether to treat the extension as case-sensitive or not.
        The default is to ignore case.
    parse_filename_metadata : bool
        Flag for whether to attempt to parse metadata from the
        filename.
        The default is False as this method for metadata input is purely a
        convenient stopgap.
    details_to_include : list of str
        Names of details to include in created metadata filenames.
    """
    experiment_set = dataio.read_directories(directories,
                                             extension=extension,
                                             recursive=recursive,
                                             case_sensitive=case_sensitive,
                                             parse_filename_metadata=parse_filename_metadata,
                                             **read_kwargs)

    if output_directory is None:
        output_directory = os.path.dirname(os.path.commonprefix(directories))

    __convert_experiment_set(experiment_set,
                             output_directory,
                             allow_overwrite,
                             flatten_to_single_file,
                             create_metadata_filenames,
                             suppress_condition_names,
                             suppress_replicate_index,
                             details_to_include)


def __convert_experiment_set(experiment_set,
                             output_directory: str,
                             allow_overwrite: bool = False,
                             flatten_to_single_file: bool = False,
                             create_metadata_filenames: bool = True,
                             suppress_condition_names: bool = False,
                             suppress_replicate_index: bool = True,
                             details_to_include: List[str] = None):
    """Convert an experiment set to a set of .csv files.

    Parameters
    ----------
    experiment_set : ExperimentSet
        The experiment set whose datasets are to be converted.
    output_directory : str
        The path to the directory in which to place the converted data.
    allow_overwrite : bool
        Whether to allow output files to overwrite existing files.
        Default is false for user safety.
    flatten_to_single_file : bool
        Whether to flatten the whole experiment set to a single csv.
        The alternative is to write one file per dataset.
    suppress_condition_names : bool
        Whether to prevent writing the metadata keys to the filename.
    details_to_include : list of str
        Names of details to include in created metadata filenames.
    """
    expand_replicates = True
    if details_to_include is None:
        details_to_include = []

    def validate_output_filepath(filepath: str) -> str:
        """Ensure that the output filepath does not overwrite an existing file.

        Parameters
        ----------
        filepath : str
            The desired output filepath.
        Returns
        -------
        str
            A valid, non-overwriting filepath.
        """
        filepath = os.path.abspath(filepath)
        head, extension = os.path.splitext(filepath)
        pattern = f"{head}*{extension}"
        number_of_matches = len(glob(pattern))
        if number_of_matches == 0:
            suffix = ""
        else:
            suffix = f"({number_of_matches})"

        return f"{head}{suffix}{extension}"

    def create_output_filepath(create_metadata_filenames_: bool,
                               filename_conditions_: List[str],
                               suppress_condition_names_: bool,
                               suppress_replicate_index_: bool,
                               output_directory_: str):
        """Create a filepath for a dataset to be output to.

        Parameters
        ----------
        create_metadata_filenames_ : bool
        filename_conditions_ : list of str
        suppress_condition_names_ : bool
        suppress_replicate_index_ : bool
        output_directory_ : str

        Returns
        -------
        output_filepath_ : str
        """

        if not create_metadata_filenames_:
            output_filename = dataset.source
            output_filename = f"{os.path.splitext(output_filename)[0]}.csv"
        else:
            output_filename = ""
            for condition_name in filename_conditions_:
                if suppress_condition_names_:
                    output_filename += str(experiment.conditions[condition_name]) + "_"
                else:
                    output_filename += condition_name + "=" + str(experiment.conditions[condition_name]) + "_"

            if suppress_replicate_index_:
                output_filename = output_filename[:-1] + ".csv"
            else:
                output_filename += f"{replicate_index + 1}.csv"

        output_filepath_ = os.path.join(output_directory_, output_filename)
        if not allow_overwrite:
            output_filepath_ = validate_output_filepath(output_filepath_)

        return output_filepath_

    if not flatten_to_single_file:

        for detail_name in details_to_include:
            experiment_set.detail_to_condition(detail_name)

        for experiment in experiment_set:

            filename_conditions = copy.deepcopy(experiment_set.equal_condition_names)
            filename_conditions.update(experiment_set.varying_conditions)
            filename_conditions -= set(details_to_include)
            filename_conditions = list(filename_conditions)

            for detail_name in details_to_include:
                filename_conditions.append(detail_name)

            if create_metadata_filenames:
                if not filename_conditions:
                    create_metadata_filenames = False

            if expand_replicates:
                for replicate_index, replicate in enumerate(experiment.datasets):
                    for dataset in replicate:

                        output_filepath = create_output_filepath(create_metadata_filenames,
                                                                 filename_conditions,
                                                                 suppress_condition_names,
                                                                 suppress_replicate_index,
                                                                 output_directory)

                        dataset.to_csv(output_filepath)

                        print(f"writing:\t{output_filepath}")
            else:
                # can use the mean of each dataset
                raise NotImplementedError("Collapsing replicates is not yet supported.")
