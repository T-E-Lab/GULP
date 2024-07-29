# behavior.py

from pathlib import Path
import pandas as pd
import pickle
import json
import os
import warnings
import logging

from unityvr.preproc import logproc as lp
from unityvr.analysis import posAnalysis, align2img
from gulp2p.config import BHV_DATA_PICKLE_DIR

logger = logging.getLogger(__name__)


def get_bhv_pickle_path(path):
    return Path(BHV_DATA_PICKLE_DIR, path.with_suffix(".pickle").name)

def heal_json_file(path):
    """Heal an incomplete behavioral json file.
    Behavioral json files can be incomplete due to unity crashing.
    This causes the json files to be unreadable due to missing closing brackets.

    Args:
        path (Path): path to behavioral json file.
    """
    # Check for empty files
    if path.stat().st_size == 0:
        warnings.warn(f"{path} is empty, unable to heal.\n")
        return

    # Get last json object
    # Open file as binary to allow negative seeks
    with open(path, 'rb') as file:
        # block counter will be multiplied by buffer
        # to get the block size from the end
        block_counter = -1
        _buffer = 4098

        # Search for the last json object (last data entry from unity)
        object_captured = False
        while not object_captured:
            try:
                file.seek(block_counter * _buffer, os.SEEK_END)
            except IOError:  # File is smaller than the buffer size
                file.seek(0)
                tail = file.read().decode(encoding="utf-8")
                break

            tail = file.read().decode(encoding="utf-8")
            # Check for {
            if '\n{' in  tail:
                object_captured = True

            # decrement the block counter to get the
            # next X bytes
            block_counter -= 1

    # TODO: Address case where file ends midline
    ''' 
    {
    "timeSecs": 4.598742485046387,
    "frame": 
    '''

    # Close missing brackets
    with open(path, 'a') as file:
        left_curly_idx = tail.rfind('{')
        right_curly_idx = tail.rfind('}')
        if left_curly_idx > right_curly_idx:
            # Object is not complete need to add closing bracket
            file.seek(0, os.SEEK_END)
            file.write("\n}\n")

        # Add final closing square bracket
        file.seek(0, os.SEEK_END)
        file.write("]\n")

def parse_bhv_data(path):
    try:
        uvr = lp.constructUnityVRexperiment(str(path.parent), path.name)
    except json.decoder.JSONDecodeError:
        print(f"Failed to read {path}, json is broken.")
        print("healing behavior file.")
        heal_json_file(path)
        uvr = lp.constructUnityVRexperiment(str(path.parent), path.name)
    return uvr

def get_bhv_data(path, from_pickle=True):
    bhv_pickle_path = get_bhv_pickle_path(path)
    if from_pickle:
        # Check if uvr pickle has been created
        if bhv_pickle_path.exists():
            return pd.read_pickle(bhv_pickle_path)
        else:
            print("Could not find preprocessed behavior data, parsing it now.")
    
    uvr = parse_bhv_data(path)
    with open(bhv_pickle_path, "wb") as file:
        pickle.dump(uvr, file)
    return uvr

def load_bhv_data(paths, from_pickle=True):
    # Given a list of paths to bhv data, return a uvr object with all the data.
    if (isinstance(paths, Path)):
        return get_bhv_data(paths, from_pickle)
    elif (len(paths) == 1):
        return get_bhv_data(paths[0], from_pickle)
    else:
        raise NotImplementedError("Loading multiple behavior files into one uvr object not implemented yet.")
