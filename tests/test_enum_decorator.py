"""Test that we can apply a decorator on a class."""

import os
import sys

# NOTE (mristin, 2024-04-5):
# os.getcwd() is not available on micropython 1.17 available on Ubuntu 22.04.
sys.path.insert(0, os.getenv("AAS_CORE3_MICROPYTHON_REPO"))

import aas_core3.enum


@aas_core3.enum.decorator
class SomeEnum:
    LITERAL_1 = "literal 1"
    LITERAL_2 = "literal 2"


if __name__ == "__main__":
    assert SomeEnum.LITERAL_1.value == "literal 1"
    assert SomeEnum.LITERAL_2.value == "literal 2"
