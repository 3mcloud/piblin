"""Test module for functions in the fileio package.

cralds.fileio contains 2 functions:
list_files_in_directory
list_files_in_directory_with_ext

Test files for these functions are located in tests/fileio/
There is no need to deal with actual measurement data in this test module as
files are never read, the functionality is simply to return lists of files.
The test files are therefore empty, and are given the extensions ega and egb in
order to provide differentiable examples.
"""
import os
import pytest
from piblin.dataio.fileio import list_files_in_directory

listing_cases = [["lower_case", {"dummy.ega", "dummy.egb"}],
                 ["upper_case", {"dummy.EGA", "dummy.EGB"}],
                 ["mixed_case", {"dummy.Ega", "dummy.Egb"}]]

extension_cases = [["lower_case", "ega", False, ["dummy.ega"]],
                   ["lower_case", "egb", False, ["dummy.egb"]],
                   ["upper_case", "ega", False, ["dummy.EGA"]],
                   ["upper_case", "egb", False, ["dummy.EGB"]],
                   ["mixed_case", "ega", False, ["dummy.Ega"]],
                   ["mixed_case", "egb", False, ["dummy.Egb"]],
                   ["lower_case", "EGA", False, ["dummy.ega"]],
                   ["lower_case", "EGB", False, ["dummy.egb"]],
                   ["upper_case", "EGA", False, ["dummy.EGA"]],
                   ["upper_case", "EGB", False, ["dummy.EGB"]],
                   ["mixed_case", "EGA", False, ["dummy.Ega"]],
                   ["mixed_case", "EGB", False, ["dummy.Egb"]],
                   ["lower_case", "Ega", False, ["dummy.ega"]],
                   ["lower_case", "Egb", False, ["dummy.egb"]],
                   ["upper_case", "Ega", False, ["dummy.EGA"]],
                   ["upper_case", "Egb", False, ["dummy.EGB"]],
                   ["mixed_case", "Ega", False, ["dummy.Ega"]],
                   ["mixed_case", "Egb", False, ["dummy.Egb"]],

                   ["lower_case", "ega", True, ["dummy.ega"]],
                   ["lower_case", "egb", True, ["dummy.egb"]],
                   ["upper_case", "ega", True, []],
                   ["upper_case", "egb", True, []],
                   ["mixed_case", "ega", True, []],
                   ["mixed_case", "egb", True, []],
                   ["lower_case", "EGA", True, []],
                   ["lower_case", "EGB", True, []],
                   ["upper_case", "EGA", True, ["dummy.EGA"]],
                   ["upper_case", "EGB", True, ["dummy.EGB"]],
                   ["mixed_case", "EGA", True, []],
                   ["mixed_case", "EGB", True, []],
                   ["lower_case", "Ega", True, []],
                   ["lower_case", "Egb", True, []],
                   ["upper_case", "Ega", True, []],
                   ["upper_case", "Egb", True, []],
                   ["mixed_case", "Ega", True, ["dummy.Ega"]],
                   ["mixed_case", "Egb", True, ["dummy.Egb"]]]


@pytest.fixture(params=listing_cases)
def listing_case(datadir, request):

    location = datadir.join(request.param[0])
    fileset = request.param[1]
    return location, fileset


@pytest.fixture(params=extension_cases)
def extension_case(datadir, request):

    location = datadir.join(request.param[0])
    extension = request.param[1]
    case_sensitive = request.param[2]

    correct_result = request.param[3]

    return location, extension, case_sensitive, correct_result


def test_list_files(listing_case):
    """Ensure that listing directories returns all contents."""
    file_list = list_files_in_directory(listing_case[0])

    assert len(file_list) == 2

    for filepath in file_list:
        filename = os.path.basename(filepath)
        assert filename in listing_case[1]


def test_list_files_with_ext(extension_case):
    """Ensure that listing with specified extension returns a single file."""

    filepaths = list_files_in_directory(extension_case[0],
                                        extension_case[1],
                                        extension_case[2])
    filenames = []
    for file in filepaths:
        filenames.append(os.path.basename(file))

    assert filenames == extension_case[3]
