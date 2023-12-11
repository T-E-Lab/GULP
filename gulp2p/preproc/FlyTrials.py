# FlyTrials.py

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
from gulp2p import utils as utils
from gulp2p import ImagingPreProc as iPP
from gulp2p import ROIs as ROIs

class FlyTrials():
    def __init__(self, path):
        """_summary_

        Args:
            path (Path): path to pickle file of fly trials
        """
        pass

    def load_data(self):
        pass