# experiment.py

from pathlib import Path
import pickle
import logging
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Rectangle
from matplotlib.colors import CenteredNorm
import numpy as np
import pandas as pd
import pingouin

from unityvr.preproc import logproc as lp
from unityvr.analysis import posAnalysis, align2img
from gulp2p.preproc import utils, trial as tr
from gulp2p.preproc.tiff import Tiff

logger = logging.getLogger(__name__)

class Experiment():
    def __init__(self, trials, rois_per_arm=8):
        """_summary_

        Args:
            trials (List[Trial]): List of all trials from one fly.
        """
        self.trials = self.filter_trials(trials)
        self.rois_per_arm = rois_per_arm
        self.synced_df = self.get_stitched_trial_df()
        self.num_rois = self.get_num_rois()
        self.trial_start_frames = self.get_trial_start_frames()

        # TODO: stitch rawf from each trial together

    @property
    def deltaf(self):
        return np.array(self.synced_df.iloc[:, 0:self.num_rois])

    @property
    def upper_deltaf(self):
        return self.deltaf[:, self.rois_per_arm:self.num_rois]

    @property
    def lower_deltaf(self):
        return self.deltaf[:, 0:self.rois_per_arm]


    def get_expt_name(self, cell_types, genetic_tools):
        trial_info = tr.parse_trial_name(self.trials[0].path, cell_types, genetic_tools)
        expt_name = f"{trial_info['date']}_{trial_info['line']}_Fly{trial_info['fly']}"
        return expt_name

    def filter_trials(self, trials):
        # Remove trials that are missing data
        filtered_trial_list = []
        for trial in trials:
            if (trial.synced_df is None) or (len(trial.synced_df) == 0):
                logger.info(f"Trial {trial.name} is missing synced_df, removing from Experiment.")
                continue
            filtered_trial_list.append(trial)
        return filtered_trial_list

    def get_stitched_trial_df(self):
        if len(self.trials) == 0:
            return None
        all_synced_dfs = []
        start_datetime = self.trials[0].tiff_metadata['date']
        for index, trial in enumerate(self.trials):
            # TODO: check for trials with missing data (no synced_df)
            if index != 0:
                trial_datetime = trial.tiff_metadata['date']
                time_diff = (trial_datetime - start_datetime).total_seconds()
                trial.synced_df['posTime'] += time_diff
            trial.synced_df['trial'] = index
            all_synced_dfs.append(trial.synced_df)

        synced_df = pd.concat(all_synced_dfs, ignore_index=True)
        return synced_df

    def get_num_rois(self):
        if len(self.trials) == 0:
            return None
        all_roi_lengths = [len(trial.rois) for trial in self.trials]
        # Check if all trials have the same number of rois
        if np.all(all_roi_lengths == all_roi_lengths[0]):
            return all_roi_lengths[0]
        else:
            # https://stackoverflow.com/questions/6252280/find-the-most-frequent-number-in-a-numpy-array
            counts = np.bincount(all_roi_lengths)
            return np.argmax(counts)

    def get_trial_start_frames(self):
        if len(self.trials) == 0:
            return None
        trial_start_frames = []
        for i in range(len(self.trials)):
            trial_start_frames.append(self.synced_df.trial.searchsorted(i, side='left'))
        return trial_start_frames


def add_expt_metadata(expt_df, cell_types, genetic_tools):
    # https://stackoverflow.com/questions/16236684/apply-pandas-function-to-column-to-create-multiple-new-columns
    applied_df = expt_df.apply(lambda row: tr.parse_trial_name(row['trial_paths'][0], cell_types, genetic_tools), axis='columns', result_type='expand')
    applied_df = applied_df.drop('trial', axis='columns')
    expt_df = pd.concat([expt_df, applied_df], axis='columns')
    return expt_df

def add_trials(trial_paths):
    # Load in trials and add them to the expt_df
    # Series apply function
    # Ex: expt_df['trials'] = expt_df['trial_paths'].apply(add_trials)
    trials = []
    for trial_path in trial_paths:
        trial = utils.load_trial(Tiff(trial_path))
        if trial is not None:
            trials.append(trial)
    return trials

def create_expt_df(trial_df, cell_types, genetic_tools):
    # Each unique combination of date and fly number is an experiment.
    # Since the line will also be the same, I will include it in the experiment name for clarity.
    expt_dict = {}
    for index, row in trial_df.iterrows():
        expt_name = f"{row['date']}_{row['line']}_Fly{row['fly']}"
        try:
            expt_dict[expt_name].append(row['path'])
        except KeyError:
            expt_dict[expt_name] = [row['path']]
    expt_df = pd.DataFrame(expt_dict.items(), columns=['expt_name', 'trial_paths'])
    expt_df = add_expt_metadata(expt_df, cell_types, genetic_tools)
    # Load trials
    expt_df['trials'] = expt_df['trial_paths'].apply(add_trials)
    # Add number of processed trials
    expt_df['proc_trial_count'] = expt_df['trials'].apply(len)
    return expt_df
