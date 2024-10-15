"""Conversion between strings and dictionary metadata forms.

Module containing functionality for converting between strings (usually
filenames) and condition metadata for measurements. This provides the ability
to include information about measurement conditions in filenames so that data
can be organized into an experimental hierarchy.

It is possible to serialize these dictionaries alongside data files so that
limitations inherent to storing metadata in filenames are avoided. This is a
likely future direction for storing metadata in an ad-hoc fashion.
It is also likely that in future metadata will come from a database alongside
the location of the file containing the measurement data.

Classes
-------
MetadataConverter - Convert between string and dictionary forms of metadata.
"""
from typing import Union


class MetadataConverter(object):
    """Convert between string and dictionary forms of metadata.

    Parameters
    ----------
    separator : str
        The string to use to separate metadata key, value pairs.
        Default is underscore, i.e. key-a=a_key-b=b where the conditions are
        key-a and key-b with values a and b respectively.
    assignment : str
        The string to use to assign key, value pair relationships.
        Default is equals sign, i.e. key-a=a.

    Notes
    -----
    Metadata can be stored in string form or dictionary form provided a method
    of conversion exists. This class defines the conversion process, allowing
    string representations (usually from filenames) of metadata to be converted
    to dictionary representations (useful in programs).
    """
    def __init__(self,
                 separator: str = "_",
                 assignment: str = "="):

        super().__init__()
        self.__separator = separator
        self.__assignment = assignment

    def dict_to_string(self, metadata: dict) -> str:
        """Convert metadata dictionary to string representation.

        Parameters
        ----------
        metadata : dict
            The metadata dictionary to convert to string representation.

        Returns
        -------
        str_ : str
            The string representation of the metadata dictionary.
        """
        str_ = ""
        for key, value in metadata.items():
            str_ += key + self.__assignment + str(value) + self.__separator
        return str_[:-len(self.__separator)]

    def string_to_dict(self, metadata: str) -> dict:
        """Convert metadata string representation to dictionary.

        Parameters
        ----------
        metadata : str
            The string representation to convert to a dictionary.

        Returns
        -------
        dict_ : dict
            The dictionary representation of the metadata string.
        """
        dict_ = {}
        metadata_split = metadata.split(self.__separator)
        for entry in metadata_split:
            key_value_pair = entry.split(self.__assignment)
            if len(key_value_pair) == 2:
                dict_[key_value_pair[0]] = self.parse_string(key_value_pair[1])

        return dict_

    @staticmethod
    def parse_string(str_: str) -> Union[bool, int, float, str]:
        """Convert a string to a bool, int, double or just return the string.

        Metadata strings by definition only contain str parts, however these
        strings can represent any python object and need to be type-converted
        when the conversion to a metadata dictionary is performed. This
        implementation only covers bool, int and float values.

        Parameters
        ----------
        str_ : str
            The string to be converted to another type.

        Returns
        -------
        value
            The string value converted to the appropriate type.
        """
        if str_ == "False":
            return False
        elif str_ == "True":
            return True

        try:
            value = int(str_)
        except ValueError:
            try:
                value = float(str_)
            except ValueError:
                value = str_

        return value
