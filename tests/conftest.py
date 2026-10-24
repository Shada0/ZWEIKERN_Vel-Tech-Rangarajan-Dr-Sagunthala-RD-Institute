import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import pytest
from src.data.preprocessing import Data

@pytest.fixture(scope="session")
def D():
    return Data()
