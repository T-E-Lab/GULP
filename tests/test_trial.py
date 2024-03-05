from pathlib import Path
import datetime
import pytest

from gulp2p.preproc.trial import Trial

TEST_TIFF_PATH = Path(r"Z:\2PImaging\Kerstin\MIMS\20240215\20240215_DR018xiGluSnFR_Fly01_00001.tif")

def test_create_trial():
    trial = Trial(TEST_TIFF_PATH)
    assert trial is not None
