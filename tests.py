import importlib
import os

import pytest

os.environ["ROBOT_TEST_MODE"] = "1"
# NOTE: you must import local module after this line, to ensure test variables are used
from constants import Constants


def test_demo() -> None:
    assert mt() is True
