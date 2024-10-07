"""Functions for reading typed values from binary data.

The use of the struct module for interpreting binary data results in code
that is repetitive and difficult to read. This module collects functions which
wrap the functionality of the struct module in more readable form.

Functions
---------
byte_to_bool_flags - Convert a byte to a list of 8 boolean values.
bytes_to_string - Convert a bytearray to an ASCII string.
read_short - Read a short integer (2 bytes) from a bytearray at an offset.
read_ushort - Read an unsigned short integer (2 bytes) from a bytearray at an offset.
read_int - Read an int integer (4 bytes) from a bytearray at an offset.
read_long - Read a long integer (4 bytes) from a bytearray at an offset.
read_float - Read a float (4 bytes) from a bytearray at an offset.
read_double - Read a float (8 bytes) from a bytearray at an offset.
read_byte - Read an unsigned char int (1 byte) from a bytearray at an offset.
"""
from typing import List
import struct


def byte_to_bool_flags(value: int) -> List[bool]:
    """Convert a byte to a list of 8 boolean values.

    Parameters
    ----------
    value : int
        Byte to be converted to flag array.

    Returns
    -------
    list of bool
        List of each bit in supplied byte as bools.
    """
    bit_string = bin(value).lstrip("0b")
    bool_list = []
    for char in bit_string:
        bool_list.append(bool(int(char)))
    return bool_list


def bytes_to_string(bytes_in: bytes) -> str:
    """ Convert a bytearray to an ASCII string.

    Parameters
    ----------
    bytes_in : buffer
        Bytes to be converted to the ASCII representation.

    Returns
    -------
    str
        The ASCII representation of the bytes.
    """
    str_out = ""
    for byte in bytes_in:
        if byte != 0:
            str_out += chr(byte)
    return str_out


def read_short(file_contents, offset: int, first_character="<") -> int:
    """Read a short integer (2 bytes) from a bytearray at an offset.

    Parameters
    ----------
    file_contents : buffer
        The contents of the binary file.
    offset : int
        The number of bytes to skip at the start of the file.
    first_character : str
        The first character of the format string.

    Returns
    -------
    int
        The short value represented by an int.
    """
    return struct.unpack_from(first_character + "h",
                              file_contents,
                              offset=offset)[0]


def read_ushort(file_contents, offset: int, first_character="<") -> int:
    """Read an unsigned short integer (2 bytes) from a bytearray at an offset.

    Parameters
    ----------
    file_contents : buffer
        The contents of the binary file.
    offset : int
        The number of bytes to skip at the start of the file.
    first_character : str
        The first character of the format string.

    Returns
    -------
    int
        The unsigned short value represented by an int.
    """
    return struct.unpack_from(first_character + "H",
                              file_contents,
                              offset=offset)[0]


def read_int(file_contents, offset: int, first_character="<") -> int:
    """Read an int integer (4 bytes) from a bytearray at an offset.

    Parameters
    ----------
    file_contents : buffer
        The contents of the binary file.
    offset : int
        The number of bytes to skip at the start of the file.
    first_character : str
        The first character of the format string.

    Returns
    -------
    int
        The int value represented by an int.
    """
    return struct.unpack_from(first_character + "i",
                              file_contents,
                              offset=offset)[0]


def read_uint(file_contents, offset: int, first_character="<") -> int:
    """Read an unsigned int integer (4 bytes) from a bytearray at an offset.
    Parameters
    ----------
    file_contents : buffer
        The contents of the binary file.
    offset : int
        The number of bytes to skip at the start of the file.
    first_character : str
        The first character of the format string.

    Returns
    -------
    int
        The unsigned integer value represented by an int.
    """
    return struct.unpack_from(first_character + "I",
                              file_contents,
                              offset=offset)[0]


def read_long(file_contents, offset: int, first_character="<") -> int:
    """Read a long integer (4 bytes) from a bytearray at an offset.

    Parameters
    ----------
    file_contents : buffer
        The contents of the binary file.
    offset : int
        The number of bytes to skip at the start of the file.
    first_character : str
        The first character of the format string.

    Returns
    -------
    int
        The long value represented by an int.
    """
    return struct.unpack_from(first_character + "l",
                              file_contents,
                              offset=offset)[0]


def read_ulong(file_contents, offset: int, first_character="<") -> int:
    """Read an unsigned long integer (4 bytes) from a bytearray at an offset.

    Parameters
    ----------
    file_contents : buffer
        The contents of the binary file.
    offset : int
        The number of bytes to skip at the start of the file.
    first_character : str
        The first character of the format string.

    Returns
    -------
    int
        The unsigned long value represented by an int.
    """
    return struct.unpack_from(first_character + "L",
                              file_contents,
                              offset=offset)[0]


def read_long_long(file_contents, offset: int) -> int:
    """Read a long long integer (8 bytes) from a bytearray at an offset.

    Parameters
    ----------
    file_contents : buffer
        The contents of the binary file.
    offset : int
        The number of bytes to skip at the start of the file.

    Returns
    -------
    int
        The long long value represented by an int.
    """
    return struct.unpack_from("<q",
                              file_contents,
                              offset=offset)[0]


def read_ulong_long(file_contents, offset: int) -> int:
    """Read a ulong long integer (8 bytes) from a bytearray at an offset.

    Parameters
    ----------
    file_contents : buffer
        The contents of the binary file.
    offset : int
        The number of bytes to skip at the start of the file.

    Returns
    -------
    int
        The ulong long value represented by an int.
    """
    return struct.unpack_from("<Q",
                              file_contents,
                              offset=offset)[0]


def read_float(file_contents, offset: int, first_character="<") -> float:
    """Read a float (4 bytes) from a bytearray at an offset.

    Parameters
    ----------
    file_contents : buffer
            The contents of the binary file.
    offset : int
        The number of bytes to skip at the start of the file.
    first_character
        The first character of the format string.

    Returns
    -------
    float
        The float value represented by a float.
    """
    return struct.unpack_from(first_character + "f",
                              file_contents,
                              offset=offset)[0]


def read_float16(file_contents, offset: int, first_character="<") -> float:
    """Read a float16 from a bytearray at an offset.

    Parameters
    ----------
    file_contents : buffer
            The contents of the binary file.
    offset : int
        The number of bytes to skip at the start of the file.
    first_character : str
        The first character of the format string.

    Returns
    -------
    float
        The float value represented by a float.
    """
    return struct.unpack_from(first_character + "e",
                              file_contents,
                              offset=offset)[0]


def read_double(file_contents, offset: int, first_character="<") -> float:
    """Read a float (8 bytes) from a bytearray at an offset.

    Parameters
    ----------
    file_contents : buffer
        The contents of the binary file.
    offset : int
        The number of bytes to skip at the start of the file.
    first_character : str
        The first character of the format string.

    Returns
    -------
    float
        The double value represented by a float.
    """
    return struct.unpack_from(first_character + "d",
                              file_contents,
                              offset=offset)[0]


def read_byte(file_contents, offset: int) -> bytes:
    """Read an unsigned char int (1 byte) from a bytearray at an offset.

    Parameters
    ----------
    file_contents : buffer
        The contents of the binary file.
    offset : int
        The number of bytes to skip at the start of the file.

    Returns
    -------
    int
        The byte value represented by an int.
    """
    return struct.unpack_from("<B",
                              file_contents,
                              offset=offset)[0]
