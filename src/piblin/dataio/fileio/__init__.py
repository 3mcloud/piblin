"""Modules for reading/writing experimental measurements from/to files.

A set of modules for reading and writing experimental measurement data and metadata from and to files, along with
convenience functions for interacting with file systems to generate lists of paths to files for reading. The majority of
modules are specific to reading a particular file format, and these follow the naming convention manufacturer_extension.
The remainder provide additional functionality related to reading files. All classes which read a specific format are
subclasses of the ABC FileReader and provide the data_from_filepath(str) method for reading sets of measurements from
files.
All classes which write a specific format are subclasses of the ABC FileWriter and provide the write_data_collection
method for writing data collection classes to the appropriately formatted files.

Modules
-------
autodetect - Functions for reading arbitrary files in supported formats.
batch_conversion - Functions for performing conversion of one or ore files to specified formats.
binary - Functions for reading typed values from binary data.

read
file_reader - Abstract base class definition for all file reader classes.

generic
    generic_file_reader - Abstract base class definition for all generic file reader classes.
    generic_csv - Reading of 2-column .csv files.
    generic_txt - Reading of 2-column .txt files.

multifile
    multifile_reader - Abstract base class definition for all multifile reader classes.

specific
    agilent_ch
    agilent_gpc_xlsx
    ait_dat
    anasys_axd
    aramis_csv
    chemstation_d
    chemstation_txt
    filemetrics_csv
    GUI_kinetics_xslx
    icpms_txt
    keyence_vk4
    keyence_vk6
    kmac_csv
    manufacturer_esr - Reading of EPR data in .esr format.
    mettler_dsc_txt
    nicolet_spa - Reading of Nicolet's .spa file format.
    nova_txt
    numpy_npy
    pyris_dsc
    pyspm_bruker
    renishaw_txt - Reading of Renishaw's .txt format.
    resmap_csv
    scikit_image_imread
    shimadzu_spc - Reading of Shimadzu's .spc file format.
    spachuta_rwx - Reading of ToF-SIMS data in Steven Pachuta's .rwx formats.
    thermofisher_spc - Reading of Thermo Fisher's .spc file format.
    trios_rheo_txt
    trios_thermal_txt
    unknown_asc
    unknown_xrdml
    vision_ascii
    vkanalyzer_csv
    wyko_opd
    xrf_csv

summary
    summary_file_reader - Abstract base class definition for all summary file reader classes.
    unknown_ini
    avizo_info

write
    file_writer - Abstract base class definition for all file writer classes.


Functions
---------
list_all_files_in_directory - Create a list of the paths to all files in a given directory.
list_files_in_directory - Create a list of paths to files with a given extension in directory.
count_files_in_directory_with_extensions - Count the files present in a directory with any of the given extensions.
find_image_in_directory_with_extensions - Find the first image present in a given directory.

Packages
--------
read - Modules for reading experimental measurements from files.
write - Modules for writing experiment measurements to files.
"""
import os
from typing import List, Set
from os.path import isfile, join


def list_all_files_in_directory(directory: str,
                                recursive: bool = False) -> List[str]:
    """Create a list of the paths to all files in a given directory.

    Parameters
    ----------
    directory : str
        The directory to inspect for files.
    recursive : bool
        Whether to find files in subdirectories of the specified directory.

    Returns
    -------
    list of str
        A list of files in the specified directory and optionally its subdirectories.
    """
    if not recursive:
        return [join(directory, f) for f in os.listdir(directory) if isfile(join(directory, f))]
    else:
        filepaths = []
        for root, _, files in os.walk(directory):
            for filename in files:
                filepaths.append(os.path.join(root, filename))

        return filepaths


def list_files_in_directory(directory: str,
                            extension: str = None,
                            case_sensitive: bool = False,
                            recursive: bool = False) -> List[str]:
    """Create a list of paths to files with a given extension in directory.

    Parameters
    ----------
    directory : str
        The directory to inspect for files.
    extension : str
        The extension of interest, default is to list all files.
    case_sensitive : bool
        Whether to treat the extension as case-sensitive.
    recursive : bool
        Whether to find files in subdirectories of the specified directory.

    Returns
    -------
    filepaths : list of str
        List of all files (with extension of interest) in directory.
    """
    all_files = list_all_files_in_directory(directory, recursive)

    if extension is None:
        return all_files

    filepaths = []
    for filepath in all_files:

        if case_sensitive:
            if filepath.endswith(extension):
                filepaths.append(filepath)
        else:
            if filepath.lower().endswith(extension.lower()):
                filepaths.append(filepath)

    return filepaths


def count_files_in_directory_with_extensions(directory: str,
                                             extensions: Set[str]) -> int:
    """Count the files present in a directory with any of the given extensions.

    Parameters
    ----------
    directory : str
        The directory to count files in.
    extensions : list of str
        The list of extensions.

    Returns
    -------
    count : int
        The number of files present in the directory with any of the given extensions.
    """
    all_files = os.listdir(directory)

    count = 0
    for filename in all_files:
        if os.path.splitext(filename)[1] in extensions:
            count += 1

    return count


def find_image_in_directory_with_extensions(directory: str, extensions: Set[str]) -> str:
    """Find the first image present in a given directory.

    Parameters
    ----------
    directory : str
        The directory to check for images.
    extensions : set of str
        The set of image extensions to find.

    Returns
    -------
    filename : str
        The path to the first image with a specified extension located in the directory.
    """
    all_files = os.listdir(directory)
    for filename in all_files:
        if os.path.splitext(filename)[1] in extensions:
            return os.path.join(directory, filename)
