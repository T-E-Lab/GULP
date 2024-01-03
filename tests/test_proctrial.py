from gulp2p.preproc.proctrial import ProcTrial
from pathlib import Path
import datetime
import pytest

def test_create_proctrial():
    trial = ProcTrial(path=None)
    assert trial is not None