import tkinter as tk
from tkinter import filedialog
import os
from os import listdir
from os.path import sep
import pickle
import tifffile as tf
from datetime import datetime, timedelta
from pathlib import Path
import warnings
import re
import cv2

from gulp2p.config import TRIAL_PICKLE_DIR, BHV_DATA_RAW_DIR, FICTRAC_DIR
from gulp2p.preproc.tiff import Tiff

def load_file_names(single_file=False, title=None):
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

def select_file_path(prompt="Select File", initialdir=None):
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", 1)

    file_path_str = filedialog.askopenfilename(title=prompt, initialdir=initialdir)
    if file_path_str == "":
        print("File selection canceled")
        return None
    file_path = Path(file_path_str)
    return file_path

def select_folder_path(prompt="Select Folder", initialdir=None):
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", 1)

    dir_path_str = filedialog.askdirectory(title=prompt, initialdir=initialdir)
    if dir_path_str == "":
        print("Folder selection canceled")
        return None
    dir_path = Path(dir_path_str)
    return dir_path

def load_trial_info(rootDirs):
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

def get_timestamp_from_string(string):
    date_pattern = r"\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}"
    date_fmt = '%Y-%m-%d_%H-%M-%S'
    date_match = re.search(date_pattern, string)
    if date_match is None:
        alt_date_pattern = r"\d{8}_\d{6}"
        date_fmt = '%Y%m%d_%H%M%S'
        date_match = re.search(alt_date_pattern, string)
    date_str = date_match.group()
    timestamp = datetime.strptime(date_str, date_fmt).timestamp()
    return timestamp

def format_date(year, month):
    """Formats year and month together into form YYYY_MM

    Args:
        year (int)
        month (int)

    Returns:
        str: formated year and month string
    """
    formatted_date = f"{year}_{month:02d}"
    return formatted_date

def load_proc_data(proc_data_fn):
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

def save_trials(expt_dat, trial_date, subfolder=None):
    # Given a dictionary of trials,
    # save each trial in a seperate pickle file
    for trialNm, trial in expt_dat.items():
        if subfolder is None:
            year_month = format_date(trial_date.year, trial_date.month)
            subfolder = year_month
        dirPath = Path(TRIAL_PICKLE_DIR, year_month)
        os.makedirs(dirPath, exist_ok=True)

        timestamp = trial_date.strftime("%Y%m%d-%H%M")
        basename = timestamp + '_' + Path(trialNm).stem + ".pickle"
        fullPath = Path(dirPath, basename)
        with open(fullPath, 'wb') as outfile:
            pickle.dump(trial, outfile)

def get_trial_pickle_path(path, trial_date):
    """Create path for pickle file. 
    Under the folder given, the pickle file is stored in a year and month folder (year_month)

    Args:
        folder (str): path to folder to store pickle file in
        path (Path): Path of tiff

    Returns:
        Path: path of pickle file
    """
    year_month = format_date(trial_date.year, trial_date.month)
    subfolder = year_month
    dir_path = Path(TRIAL_PICKLE_DIR, subfolder)
    timestamp = trial_date.strftime("%Y%m%d-%H%M%S")
    base_name = f"{timestamp}_{path.stem}.pickle"
    full_path = Path(dir_path, base_name)
    return full_path

def save_dat(fileNm, expt, expt_dat):
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


def tail(file_path, lines=1, _buffer=4098):
    """Tail a file and get X lines from the end
    https://stackoverflow.com/questions/136168/get-last-n-lines-of-a-file-similar-to-tail
    """
    # Open file as binary to allow negative seeks
    with open(file_path, 'rb') as file:
        # place holder for the lines found
        lines_found = []

        # block counter will be multiplied by buffer
        # to get the block size from the end
        block_counter = -1

        # loop until we find X lines
        while len(lines_found) < lines:
            try:
                file.seek(block_counter * _buffer, os.SEEK_END)
            except IOError:  # either file is too small, or too many lines requested
                file.seek(0)
                lines_found = [line.decode('utf-8') for line in file.readlines()]
                break

            lines_found = [line.decode('utf-8') for line in file.readlines()]

            # decrement the block counter to get the
            # next X bytes
            block_counter -= 1

    return lines_found[-lines:]

def get_bhv_length(bhv_path):
    num_lines = 10
    while True:
        end_lines = tail(bhv_path, num_lines)
        for line in end_lines[::-1]:
            if "timeSecs" in line:
                return float(line.split(':')[-1].strip(" ,\n"))
        # Searched whole file and length not found.
        # This works since tail will return all the lines in a file if you ask for more than is in the file.
        if num_lines > len(end_lines):
            return None
        num_lines *= 2

def get_vid_length(vid_path):
    # https://stackoverflow.com/questions/3844430/how-to-get-the-duration-of-a-video-in-python
    import ffmpeg
    info=ffmpeg.probe(vid_path.as_posix())
    try:
        duration = float(info['format']['duration'])
    except KeyError:
        # Video metadata needs to be fixed by ffmpeg
        raise NotImplementedError("Video metadata needs to be fixed by ffmpeg")
    return duration

def get_creation_time(file_path):
    # If the tiff was copied to another system its creation date will get reset but not other metadata.
    # In these cases use modification time - estimated length as tiff creation time.
    # Length of trial can be estimated with fm_interval and num of frames.

    # Modified before created
    # This means creation date is innacurate and needs to be estimated
    if file_path.suffix == ".tif":
        return Tiff(file_path).metadata['data'].timestamp()
    elif file_path.suffix == ".json":
        return get_timestamp_from_string(file_path.name)

    file_stats = file_path.stat()
    ctime = file_stats.st_ctime
    mtime = file_stats.st_mtime
    if ctime < mtime:
        # Expected result
        return ctime
    else:
        error_str = "Creation date is innacurate, and there is currently no method to estimate the time it took to record data for this file"
        raise RuntimeError(error_str)

def get_overlap(range1, range2, relative=False):
    """Get the overlap of 2 ranges.
    https://stackoverflow.com/questions/3269434/whats-the-most-efficient-way-to-test-if-two-ranges-overlap

    Args:
        range1 (Tuple[float, float]): First range
        range2 (Tuple[float, float]): Second range
        relative (bool): If True the returned range is relative to the range1.

    Returns:
        Tuple[float, float]: Overlap of the ranges.
    """
    start1, end1 = range1
    start2, end2 = range2
    # ranges must go from left to right
    assert end1 - start1 >=0
    assert end2 - start2 >=0

    if start1 < start2:
        overlap_start = start2
    else:
        overlap_start = start1
    if end1 > end2:
        overlap_end = end2
    else:
        overlap_end = end1

    if relative:
        overlap_start -= start1
        overlap_end -= start1

    if overlap_start < overlap_end:
        return [overlap_start, overlap_end]

def get_tiff_bhv_overlap(tiff, bhv_path):
    """Get the overlap of the behavior and tiff file, relative to the tiff.

    Args:
        tiff (Tiff): tiff object. Function uses the tiff object since it caches length.
        bhv_path (Path): Path to the bhv file.

    Returns:
        Tuple[float, float]: Overlap range relative to tiff.
    """
    tiff_start_time = tiff.metadata['date'].timestamp()
    tiff_length = tiff.length
    tiff_range = (tiff_start_time, tiff_start_time + tiff_length)

    bhv_start_time = get_timestamp_from_string(bhv_path.name)
    bhv_length = get_bhv_length(bhv_path)
    bhv_range = (bhv_start_time, bhv_start_time + bhv_length)

    # print(f"tiff_range: {tiff_range[0]-tiff_range[0]} <-> {tiff_range[1]-tiff_range[0]}")
    # print(f"bhv_range: {bhv_range[0]-tiff_range[0]} <-> {bhv_range[1]-tiff_range[0]}")

    return get_overlap(tiff_range, bhv_range, relative=True)

def get_tiff_vid_overlap(tiff, vid_path):
    """Get the overlap of the fictrac video and tiff file, relative to the tiff.

    Args:
        tiff (Tiff): tiff object. Function uses the tiff object since it caches length.
        vid_path (Path): Path to the fictrac video.

    Returns:
        Tuple[float, float]: Overlap range relative to tiff.
    """
    tiff_start_time = tiff.metadata['date'].timestamp()
    tiff_length = tiff.length
    tiff_range = (tiff_start_time, tiff_start_time + tiff_length)

    vid_start_time = get_timestamp_from_string(vid_path.name)
    vid_length = get_vid_length(vid_path)
    vid_range = (vid_start_time, vid_start_time + vid_length)

    return get_overlap(tiff_range, vid_range, relative=True)

def get_bhv_paths(tiff, bhv_data_folder=None):
    """Find the corresponding behavioral data file(s) of a tiff file.
    There can be multiple files if unityVR stimuli were run multiple times during one tiff acquisition.

    Args:
        tiff (Tiff): tiff object. Function uses the tiff object since it caches length.
        bhv_data_folder (Path): Path to behavioral data folder

    Returns:
        list[Path]: Paths of behavioral data files that overlapped.
    """
    if bhv_data_folder is None:
        bhv_data_folder = BHV_DATA_RAW_DIR

    overlapping_bhv_paths = []
    bhv_paths = []

    tiff_ctime = tiff.metadata['date'].timestamp()

    # Select only json files within 12 hours of tiff creation
    for bhv_path in bhv_data_folder.rglob('*'):
        if not bhv_path.is_file():
            continue
        if bhv_path.suffix != '.json':
            continue
        if bhv_path.name == "SessionParameters.json":
            continue
        bhv_ctime = get_timestamp_from_string(bhv_path.name)
        if bhv_ctime is None:
            continue
        if abs(bhv_ctime - tiff_ctime) < 12*60*60: # 0.5 days in seconds
            bhv_paths.append(bhv_path)
    # Search for overlapping behavioral files.
    for bhv_path in bhv_paths:
        # Skip duplicate files (have the same name but different folder)
        if bhv_path.name in [path.name for path in overlapping_bhv_paths]:
            continue
        if get_tiff_bhv_overlap(tiff, bhv_path) is not None:
            overlapping_bhv_paths.append(bhv_path)
    return overlapping_bhv_paths

def get_vid_paths(tiff, fictrac_video_folder=None, vid_style="dbg"):
    """Find the corresponding fictrac video(s) of a tiff file.
    There can be multiple files if fictrac ran multiple times during one tiff acquisition.

    Args:
        tiff (Tiff): tiff object. Function uses the tiff object since it caches length.
        fictrac_video_folder (Path): Path to fictrac video folder

    Returns:
        list[Path]: Paths of fictrac video files that overlapped.
    """
    if fictrac_video_folder is None:
        fictrac_video_folder = FICTRAC_DIR

    overlapping_vid_paths = []
    vid_paths = []

    tiff_ctime = tiff.metadata['date'].timestamp()

    # Select only avi files within 12 hours of tiff creation
    for vid_path in fictrac_video_folder.rglob('*'):
        if not vid_path.is_file():
            continue
        if vid_path.suffix != '.avi':
            continue
        if vid_style not in vid_path.stem:
            continue
        if vid_path.stat().st_size <= 100:
            continue
        vid_ctime = get_timestamp_from_string(vid_path.name)
        if vid_ctime is None:
            continue
        if abs(vid_ctime - tiff_ctime) < 12*60*60: # 0.5 days in seconds
            vid_paths.append(vid_path)
    # Search for overlapping video files.
    for vid_path in vid_paths:
        if get_tiff_vid_overlap(tiff, vid_path) is not None:
            overlapping_vid_paths.append(vid_path)
    return overlapping_vid_paths

def save_trial(trial):
    pickle_path = get_trial_pickle_path(trial.path, trial.tiff_metadata.date)
    with open(pickle_path, 'wb') as file:
        pickle.dump(trial, file)
