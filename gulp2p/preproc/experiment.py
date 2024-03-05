# experiment.py

from pathlib import Path
import pickle
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Rectangle
from matplotlib.colors import CenteredNorm
import numpy as np
import pandas as pd
import pingouin

from unityvr.preproc import logproc as lp
from unityvr.analysis import posAnalysis, align2img
from gulp2p.preproc import utils as utils

class Experiment():
    def __init__(self, trials):
        """_summary_

        Args:
            trials (List[Trial]): List of all trials from one fly.
        """
        # TODO: stitch trials together into one large synced_df
        pass

    def stitch_trials(self):
        pass
