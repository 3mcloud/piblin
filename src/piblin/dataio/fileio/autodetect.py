"""Automatic detection of file reader classes for different file formats.

This module contains a function that, given a file extension, determines the set of implementations of the file reader
abstract base class which advertise that they can potentially read files with that extension. The order of the list of
file readers is key; specific file readers are placed before generic file readers which are placed before summary file
readers. This ensures that a generic reader does not incorrectly "successfully" read a file in a format for which there
is a specific reader. This is a particular problem with .csv files, where manufacturers will utilize a more restricted
.csv format for their specific data output that can still be read, e.g., by a generic .csv parser.

Functions
---------
potential_file_readers_by_extension -> List of Type of FileReader
    Determine which file readers advertise that they may read a file with a given extension.
"""
from typing import List, Type
import importlib
from types import ModuleType
from typing import Set, TypeVar
import inspect
import pkgutil
import piblin.dataio.fileio.read.file_reader as file_reader
import piblin.dataio.fileio.read.generic.generic_file_reader as generic_file_reader
import piblin.dataio.fileio.read.summary.summary_file_reader as summary_file_reader

import piblin.dataio.fileio.read.specific as base_specific
import piblin.dataio.fileio.read.generic as base_generic
import piblin.dataio.fileio.read.summary as base_summary

has_cralds = True
try:
    import cralds.dataio.fileio.read.specific as cralds_specific
    import cralds.dataio.fileio.read.generic as cralds_generic
    import cralds.dataio.fileio.read.summary as cralds_summary
except ImportError:
    has_cralds = False

# Create a type variable for type annotations. This allows the get_subclasses function to specify
# a return type that matches the input. bound=type restricts the accepted values to classes.
T = TypeVar('T', bound=type)


def get_subclasses(primary_class: T, module: ModuleType) -> Set[T]:
    """Finds subclasses of primary_class in provided module.

    Parameters
    ----------
    primary_class : T
        Super class object to find subclasses of.
    module : ModuleType
        `cralds` module to recursively search for subclasses in.

    Returns
    -------
    Set[T]
        Set of subclasses of primary class.

    Raises
    ------
    ValueError
        If `module` is not a module.
    """
    if not inspect.ismodule(module):
        raise ValueError(f'{module} is not a module')

    subclasses = set()

    for _, member in inspect.getmembers(module, inspect.isclass):
        # Find all classes in `module` that are a subclass of `primary_class`.
        # noinspection PyTypeHints
        if issubclass(member, primary_class) and not inspect.isabstract(member):
            # Get the class directly from the module, which provides full package/module path
            # The member object only uses the base name of the member, which can cause conflicts with
            # classes with the same name that exist in different modules.
            resolved_class = getattr(module, member.__name__)
            subclasses.add(resolved_class)

    if hasattr(module, '__path__'):
        # Module is a package. Get all submodules of the package and call this function recursively.
        submodules = pkgutil.iter_modules(module.__path__)
        for mod_info in submodules:
            try:
                submodule = importlib.import_module(f'{module.__name__}.{mod_info[1]}')
                subclasses.update(get_subclasses(primary_class, submodule))
            except ModuleNotFoundError:
                # Any module that utilizes an uninstalled optional
                # dependency will raise a ModuleNotFoundError. This doesn't
                # indicate an actual error, but does mean that the given
                # module and any submodules should be considered
                # unavailable here.
                pass

    return subclasses


base_specific_file_readers = tuple(get_subclasses(
    file_reader.FileReader,
    base_specific)
)

base_generic_file_readers = tuple(get_subclasses(
    generic_file_reader.GenericFileReader,
    base_generic)
)

base_summary_file_readers = tuple(get_subclasses(
    summary_file_reader.SummaryFileReader,
    base_summary)
)

if has_cralds:

    cralds_specific_file_readers = tuple(
        get_subclasses(file_reader.FileReader,
                       cralds_specific)
    )

    cralds_generic_file_readers = tuple(
        get_subclasses(file_reader.FileReader,
                       cralds_generic)
    )

    cralds_summary_file_readers = tuple(
        get_subclasses(file_reader.FileReader,
                       cralds_summary)
    )

else:
    cralds_specific_file_readers = ()
    cralds_generic_file_readers = ()
    cralds_summary_file_readers = ()

file_readers = base_specific_file_readers + \
               cralds_specific_file_readers + \
               base_generic_file_readers + \
               cralds_generic_file_readers + \
               base_summary_file_readers + \
               cralds_summary_file_readers


def potential_file_readers_by_extension(extension: str) -> \
        List[Type[file_reader.FileReader]]:
    """Determine which file readers advertise that they may read a file
    with a given extension.

    Parameters
    ----------
    extension : str
        The extension of the file which is to be read.

    Returns
    -------
    list of FileReader
        A list of all file readers that support the requested extension.
    """
    return [r for r in file_readers if r().supports_extension(extension)]

