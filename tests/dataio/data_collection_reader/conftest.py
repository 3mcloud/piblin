import os
from distutils import dir_util
from typing import List
import copy

import numpy as np
import pytest


@pytest.fixture()
def datadir(tmpdir, request):

    # locate the directory containing test files
    filename = request.module.__file__
    test_dir, _ = os.path.splitext(filename)

    if os.path.isdir(test_dir):
        dir_util.copy_tree(test_dir, str(tmpdir))

    return tmpdir