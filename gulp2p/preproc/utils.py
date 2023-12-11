import tkinter as tk
from tkinter import filedialog
import os
from os import listdir
from os.path import sep
import pickle
import tifffile as tf
from datetime import datetime
from pathlib import Path
import warnings



def loadFileNames(single_file=False, title=None):
    """Prompt user to select one or multiple files

    Returns:
        list: list of filenames of trials
    """
    if title is None:
        title = "select files"

    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", 1)

    if single_file:
        trial_file_nms = filedialog.askopenfilename(title=title)
    else:
        trial_file_nms = filedialog.askopenfilenames(title=title)
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

    print(path)
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

def get_bhv_path(trial_tiff_path, bhv_data_folder, max_creation_delay=360, min_file_mb=2.5, creation_offset=0, filter_date='20231201'):
    # Given the path to a trials tiff file (trial_tiff_path)
    # Find the corresponding behavioral data file in the bhv_data_folder

    # Get creation date of tiff file using os.path.getctime(path)
    # Convert to datetime object
    # For json file in bhv_data_folder
    #   get datetime of date in json filename
    #   get the difference between tiff time and json time
    # print the min time 

    tiff_unix_time = min(os.path.getctime(trial_tiff_path), os.path.getmtime(trial_tiff_path))

    if tiff_unix_time < datetime.strptime(filter_date, '%Y%m%d').timestamp():
                # print(datetime.fromtimestamp(tiff_unix_time))
                creation_offset += 360

    # print(trial_tiff_path.name)
    # print(tiff_date_time)
    # print()

    min_diff = None
    min_diff_path = None
    
    for bhv_path in bhv_data_folder.rglob('*'):
        if bhv_path.is_file() and (bhv_path.suffix == '.json'):
            # Log_2023-11-29_13-45-03.json
            date_str = "_".join(bhv_path.stem.split('_')[1:])
            date_fmt = '%Y-%m-%d_%H-%M-%S'
            bhv_date_time = datetime.strptime(date_str, date_fmt).timestamp()

            diff = bhv_date_time - tiff_unix_time + creation_offset
            abs_diff = abs(diff)
            
            file_mb = bhv_path.stat().st_size / (2**20)
            if file_mb < min_file_mb:
                # print(f"{bhv_path} is {file_mb} MB, which is smaller than min file size of {min_file_mb} MB")
                continue

            if min_diff is None:
                min_diff = abs_diff
                min_diff_path = bhv_path
            
            if abs_diff < min_diff:
                min_diff = abs_diff
                min_diff_path = bhv_path

    # Pick smallest positive date as behavioral data as long as it is smaller than the creation delay.
    if min_diff > max_creation_delay:
        warnings.warn((f"Closest behavioral file\n"
                       f"{min_diff_path}\n"
                       f"is {diff} seconds from the creation of the tiff file.\n"
                       f"This is above the max delay of {max_creation_delay}."),
                       stacklevel=2)
        return None
    return min_diff_path
