import tkinter as tk
from tkinter import filedialog
import os
from os import listdir
from os.path import sep
import pickle
import tifffile as tf
from datetime import datetime
from pathlib import Path



def loadFileNames(single_file=False):
    """Prompt user to select one or multiple files

    Returns:
        list: list of filenames of trials
    """

    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", 1)

    if single_file:
        trial_file_nms = filedialog.askopenfilename(title="Select files")
    else:
        trial_file_nms = filedialog.askopenfilenames(title="Select files")
    return trial_file_nms

def loadTrialInfo(rootDirs):
    """
    Gets the experiment and trial information for calcium imaging experiments

    Arguments:
        rootDir = the root directories. Each directory should contain a series of subdirectories
        with the dates of the data collection

    Returns:
        trials = a dictionary of all of the trial information keyed by the name of each experiment
    """

    trials = dict()
    for d in rootDirs:
        dates = listdir(d)
        for dt in dates:
            if dt == '.DS_Store':
                continue
            files_here = listdir(sep.join([d,dt]))
            expts_here = list(set(
                ["_".join(sep.join([d,dt,f]).split('_')[0:-2]) for f in files_here if ('tif' in f)]))
            for e in expts_here:
                trials[e] = sorted([sep.join([d,dt,f]) for f in files_here if (('tif' in f) & (e.split(sep)[-1] in f))])

    return trials


def formatDate(year, month):
    """Formats year and month together into form YYYY_MM

    Args:
        year (int)
        month (int)

    Returns:
        str: formated year and month string
    """
    formatted_date = f"{year}_{month:02d}"
    return formatted_date

def loadProcData(proc_data_fn):
    """Return processed data

    Args:
        proc_data_fn (str): path to processed data

    Returns:
        dict: dictionary of trial data
    """
    assert os.path.isfile(proc_data_fn)
    with open(proc_data_fn, 'rb') as infile:
        data = pickle.load(infile)
    return data

def saveTrials(expt_dat, folderNm):
    # Given a dictionary of trials,
    # save each trial in a seperate pickle file
    for trialNm, trial in expt_dat.items():
        trial_date = getDate(trialNm)
        year_month = formatDate(trial_date.year, trial_date.month)
        dirPath = os.path.join(folderNm, year_month)
        os.makedirs(dirPath, exist_ok=True)

        timestamp = trial_date.strftime("%Y%m%d-%H%M")
        basename = timestamp + '_' + Path(trialNm).stem + ".pickle"
        fullPath = os.path.join(dirPath, basename)
        with open(fullPath, 'wb') as outfile:
            pickle.dump(trial, outfile)

def getDate(path):
    """Get the date for a tiff file

    Args:
        path (str or pathlib.Path): path to tiff file

    Returns:
        datetime: time tiff was captured
    """
    # Load metadata
    with tf.TiffFile(path) as tif:
        imagej_metadata = tif.imagej_metadata

    # Search metadata for date
    info = imagej_metadata.get('Info', None)
    searchstr = "[Acquisition Parameters Common] ImageCaputreDate ="
    date_str = None
    for line in info.splitlines():
        if searchstr in line:
            date_str = line.split("=")[-1].strip()
            date_str = date_str.replace('\'', '')
            break
    assert date_str is not None

    date = datetime.fromisoformat(date_str)

    return date


def getPicklePath(trialNm, folderNm):
    """Create path for pickle file. 
    Under the folder given, the pickle file is stored in a year and month folder (year_month)

    Args:
        folder (str): path to folder to store pickle file in
        trialNm (str): filename of trial

    Returns:
        str: path of pickle file
    """
    trial_date = getDate(trialNm)
    year_month = formatDate(trial_date.year, trial_date.month)
    dirPath = Path(folderNm, year_month)

    timestamp = trial_date.strftime("%Y%m%d-%H%M")
    baseNm = timestamp + '_' + Path(trialNm).stem + ".pickle"
    fullPath = Path(dirPath, baseNm)
    return fullPath


def saveDFDat(fileNm, expt, expt_dat):
    """
    Save a dictionary of the processed data

    Arguments:
        fileNm = the file name
        expt = the name of the experiment
        expt_dat = the processed experimental data
    """
    allDat = dict()

    # Open the previously saved data
    if os.path.isfile(fileNm):
        infile = open(fileNm,'rb')
        allDat = pickle.load(infile)
        infile.close()

    # Add the new data
    allDat[expt] = expt_dat

    # Save the data
    with open(fileNm, 'wb') as outfile:
        pickle.dump(allDat, outfile)