# test_experiment.py

from pathlib import Path
import datetime
import pytest

from gulp2p.preproc import utils
from gulp2p.preproc.tiff import Tiff
from gulp2p.preproc.trial import Trial
from gulp2p.preproc.experiment import Experiment

TRIAL_TIFF_PATHS = [Path(r"Z:\2PImaging\Kerstin\MIMS\20240124\20240124_DR019xiGluSnFR_Fly01_00003.tif"),
                    Path(r"Z:\2PImaging\Kerstin\MIMS\20240124\20240124_DR019xiGluSnFR_Fly01_00004.tif"),
                    Path(r"Z:\2PImaging\Kerstin\MIMS\20240124\20240124_DR019xiGluSnFR_Fly01_00005.tif")]

def test_create_experiment():
    # Load trials
    trials = []
    for tiff_path in TRIAL_TIFF_PATHS:
        tiff = Tiff(tiff_path)
        trials.append(utils.load_trial(tiff))
    # Create experiment object
    expt = Experiment(trials)
    assert expt.synced_df is not None

