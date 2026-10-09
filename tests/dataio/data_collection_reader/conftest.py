import os
import shutil
import pytest


@pytest.fixture()
def datadir(tmpdir, request):

    # locate the directory containing test files
    filename = request.module.__file__
    test_dir, _ = os.path.splitext(filename)

    if os.path.isdir(test_dir):
        shutil.copytree(test_dir, str(tmpdir), dirs_exist_ok=True)

    return tmpdir
