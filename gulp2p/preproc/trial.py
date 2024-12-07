# trial.py

from pathlib import Path, PurePath
import logging
import pickle
import re
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.colors import CenteredNorm
import numpy as np
import pandas as pd
import pingouin

# from gulp2p import preproc
# from gulp2p.preproc import utils, imaging
from gulp2p.config import TRIAL_PICKLE_DIR, BHV_DATA_RAW_DIR

logger = logging.getLogger(__name__)

class Trial():
    def __init__(self, path, bhv_paths, tiff_metadata, mip_frame, slices, mc_obj, rois, masks, raw_flor, synced_df, rois_per_arm=8):
        assert path is not None
        self.path = PurePath(path) # path (Path): path to tiff file of fly trial
        self.name = path.stem
        self.bhv_paths = bhv_paths
        self.tiff_metadata = tiff_metadata
        self.slices = slices
        self.rois_per_arm = rois_per_arm
        self.synced_df = synced_df # update to include column for each channel

        self.channels = []

        self.mip_frame = mip_frame # 1 per channel (can allow multiple channels in stack)
        self.mc_obj = mc_obj # 1 per channel? or just use both channels to create it
        self.rois = rois # 1 per channel
        self.masks = masks # 1 per channel
        self.raw_flor = raw_flor # 1 per channel

    def get_metadata_length(self):
        length = self.tiff_metadata['length']
        return length

    def get_synced_length(self):
        length = self.synced_df['posTime'].iloc[-1] - self.synced_df['posTime'].iloc[0]
        return length

    def get_length(self, mode='synced'):
        if mode == 'synced':
            return self.get_synced_length()
        if mode == 'metadata':
            return self.get_metadata_length()

    @property
    def num_rois(self):
        return len(self.rois)

    @property
    def deltaf(self):
        return np.array(self.synced_df.iloc[:, 0:self.num_rois])

    @property
    def upper_deltaf(self):
        self.rois_per_arm = 8
        return self.deltaf[:, self.rois_per_arm:self.num_rois]

    @property
    def lower_deltaf(self):
        self.rois_per_arm = 8
        return self.deltaf[:, 0:self.rois_per_arm]

    def shift_elements(self, arr, num, fill_value=np.nan):
        arr = np.roll(arr,num)
        if num < 0:
            arr[num:] = fill_value
        elif num > 0:
            arr[:num] = fill_value
        return arr

    # Analysis Functions

    def correlate_with_lag(self, a, v, lags):
        corrs = []
        for lag in lags:
            # Lag second array
            if lag < 0:
                lagged_a = a[:lag]
                lagged_v = v[-lag:]
            elif lag == 0:
                lagged_a = a
                lagged_v = v
            else:
                lagged_a = a[lag:]
                lagged_v = v[:-lag]

            corr = np.corrcoef(lagged_a, lagged_v)[0,1]
            corrs.append(corr)
        return corrs


def parse_trial_name(trial_path, cell_types, genetic_tools):
    # Return fields defined in trials name
    # ex: 20240215_DR018xiGluSnFR_Fly02_00005.tif
    # regex patterns built with https://regexr.com/

    # Define regex patterns
    patterns = {
        "date": r"^(\d{8})_",
        "line": r"([^_]+x[^_]+)",
        "cell_type": r"([^_]+)x[^_]+",
        "genetic_tool": r"[^_]+x([^_]+)",
        "fly": r"fly(\d+)",
        "trial": r"_(\d{5})\."
    }

    # Get info from trial name with regex
    metadata = {}
    for name, pattern in patterns.items():
        match = re.search(pattern, trial_path.name,
                          flags=re.IGNORECASE)
        if match is None:
            metadata[name] = None
        else:
            metadata[name] = match.group(1)

    # Assign default values
    # Convert from strings and set default values.
    if metadata['date'] is None:
        # Get date from path name if missing.
        metadata['date'] = Path(trial_path).parent.name
    if metadata['cell_type'] is not None:
        cell_type = metadata['cell_type']
        # Replace DRo with DR0
        cell_type = cell_type.replace("DRo", "DR0")
        # Correct case and zero-padding of cell_type
        match = re.search(r"\d", cell_type)
        if match is not None:
            prefix = cell_type[:match.start()]
            suffix = int(cell_type[match.start():])
            cell_type = f"{prefix}{suffix:03d}"
        try:
            find_idx = list(map(str.lower, cell_types)).index(cell_type.lower())
            metadata['cell_type'] = cell_types[find_idx]
        except:
            pass
    if metadata['genetic_tool'] is not None:
        # Correct case of genetic_tool
        try:
            find_idx = list(map(str.lower, genetic_tools)).index(metadata['genetic_tool'].lower())
            metadata['genetic_tool'] = genetic_tools[find_idx]
        except:
            pass
    if metadata['fly'] is not None:
        # Convert fly number to int.
        metadata['fly'] = int(metadata['fly'])
    else:
        # Set default value of 1.
        metadata['fly'] = 1
    if metadata['trial'] is not None:
        # Convert trial number to int.
        metadata['trial'] = int(metadata['trial'])

    return metadata

def create_trial_df(root_tiff_folder, cell_types, genetic_tools):
    trial_df_dicts = []
    for path in root_tiff_folder.rglob('*.tif'):
        trial_metadata = parse_trial_name(path, cell_types, genetic_tools)

        trial_df_dicts.append({
            "path": PurePath(path),
            **trial_metadata
        })
    trial_df = pd.DataFrame(trial_df_dicts)
    return trial_df
