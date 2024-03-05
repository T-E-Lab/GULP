# behavior.py

from pathlib import Path
import pickle

from unityvr.preproc import logproc as lp
from unityvr.analysis import posAnalysis, align2img
from gulp2p.config import BHV_DATA_PICKLE_DIR


def get_bhv_pickle_path(path):
    return Path(BHV_DATA_PICKLE_DIR, path.with_suffix(".pickle").name)

def get_bhv_data(path):
    # Check if uvr pickle has been created
    bhv_pickle_path = get_bhv_pickle_path(path)
    if bhv_pickle_path.exists():
        with open(bhv_pickle_path, "rb") as file:
            return pickle.load(file)
    else:
        print("Could not find preprocessed behavior data, parsing it now.")
        uvr = lp.constructUnityVRexperiment(path.parent, path.name)
        with open(bhv_pickle_path, "wb") as file:
            pickle.dump(uvr, file)
        return uvr

def load_bhv_data(paths):
    # Given a list of paths to bhv data, return a uvr object with all the data.
    if len(paths) == 1:
        return get_bhv_data(paths[0])
    else:
        raise NotImplementedError("Loading multiple behavior files into one uvr object not implemented yet.")
