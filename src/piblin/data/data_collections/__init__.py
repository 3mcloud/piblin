"""
Package: piblin.data.data_collections - Hierarchical organization of measurements (data and metadata).

Modules
-------
experiment - Classes abstracting the concept of a scientific experiment.
measurement - Classes abstracting the concept of a scientific measurement.
measurement_set - Collections of scientific measurements and experiments.

Functions
---------
dict_repr - Return eval()-able string representation of a dict.
dict_str - Return human-readable representation of a dict.
"""


def dict_repr(dict_):
    """Return eval()-able string representation of a dict.

    Parameters
    ----------
    dict_ : dict
        The dictionary to convert to an eval()-able string.

    Notes
    -----
    Expects dictionary keys to be strings.
    """
    output = "{"
    for key, value in dict_.items():

        key_str = "\"" + key + "\""

        if isinstance(value, str):
            value_str = "\"" + value + "\""
        else:
            value_str = str(value)

        output += key_str + ": " + value_str + ", "
    return output[:-2] + "}"


def dict_str(dict_, title="", whitespace=None):
    """Return human-readable representation of a dict.

    Parameters
    ----------
    dict_ : dict
        The dictionary to convert to a human-readable representation.
    title : str
        A string to use for the table title.
    whitespace : str, None
        Whether to add a tab in front of each key-value pair string.

    Returns
    -------
    output : str
        A human-readable representation of the dict.
    """
    # output = title + "\n" + "-"*len(title) + "\n"
    output = f"{title}:\n\n"
    for key, value in dict_.items():
        if whitespace:
            output += whitespace
        output += str(key) + " = " + str(value) + "\n"
    return output
