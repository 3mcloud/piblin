class FileParsingException(Exception):
    """Raised when a cralds file reader fails to convert a file to a measurement set.

    Each file reader implementation advertises that it can read files with one or more extensions. However, multiple
    file formats share extensions, so the ultimate determination whether a given reader can parse a given file can only
    be made at execution time. This exception is to be raised whenever a file reader encounters an internal error trying
    to parse a file that it should be able to parse based on the file's extension.
    """
