# head_direction.py

import numpy as np
import pingouin

import logging
logger = logging.getLogger(__name__)

## bump alignment functions
def get_circ_mean_rois(deltaf, precise=False):
    num_frames, num_rois = deltaf.shape
    circ_means = get_circ_means(deltaf)
    roi_angle = angle_radians_to_roi(circ_means, num_rois) - 0.5
    if not precise:
        roi_angle = np.round(roi_angle).astype(np.int32)
        roi_angle = roi_angle % num_rois
    return roi_angle

def align_bumps(deltaf_df, mean_rois=None, min_r = 0, offset=0, shift_peak_flor=False, use_max_as_peak=False):
    aligned_bumps = deltaf_df.copy()

    if mean_rois is None:
        mean_rois = get_circ_mean_rois(aligned_bumps)

    if min_r > 0:
        circ_rs = get_circ_rs(aligned_bumps)
        filter_circ_means(mean_rois, circ_rs)

    for frame, mean_roi in enumerate(mean_rois):

        roll_amount = -1 * mean_roi + offset
        aligned_bump = np.roll(aligned_bumps[frame, :], roll_amount)
        if shift_peak_flor:
            if use_max_as_peak:
                peak_deltaf = np.max(aligned_bump)
            else:
                peak_deltaf = aligned_bump[offset]
            aligned_bump += 1 - peak_deltaf
        aligned_bumps[frame, :] = aligned_bump

    return aligned_bumps

def normalize_bumps(aligned_bumps):
    normalized_bumps = np.empty(shape=aligned_bumps.shape)
    for index, bump in enumerate(aligned_bumps):
        min_val = np.min(aligned_bumps[index])
        max_val = np.max(aligned_bumps[index])
        normalized_bumps[index] = (bump - min_val)/(max_val - min_val)
    return normalized_bumps

## Cosine fit functions
def standard_cos_func(x, A, B, C, D):
    y = A*np.cos(B*(x-C)) + D
    return y

def get_r2(cos_func, xdata, ydata, parameters):
    # https://stackoverflow.com/questions/19189362/getting-the-r-squared-value-using-curve-fit
    residuals = ydata - cos_func(xdata, *parameters)
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((ydata-np.mean(ydata))**2)
    r_squared = 1 - (ss_res / ss_tot)
    return r_squared

def get_cos_fit(mean_bump, cos_func=standard_cos_func):
    # Returns best fit parameters for cos_func and 
    # https://education.molssi.org/python-data-analysis/03-data-fitting/index.html
    from scipy.optimize import curve_fit

    repeats = 4
    # Repeat bump to connect ends of bump together for better cosine fit.
    ydata = np.tile(mean_bump, repeats)
    xdata = range(len(ydata))

    initial_params = [1, 2*np.pi/8, 8/2, 0.5]
    parameters, covariance = curve_fit(cos_func, xdata, ydata,
                                       initial_params,
                                       method='lm')
    r_squared = get_r2(cos_func, xdata, ydata, parameters)
    return parameters, covariance, r_squared

## head direction functions
def get_equally_spaced_angles(num):
    """Given a number n, Return a list of n equally spaced angles in radians.
    For Example given n=4, returns [0, pi/2, pi, 3pi/2]

    Args:
        num (int): number of angles to space out.

    Returns:
        NDArray: List of equally spaced angles in radians. 
    """
    return np.linspace(0,(2*np.pi),num, endpoint=False)

def angle_radians_to_roi(rad_angle, num_roi):
    roi_angle = (rad_angle/(2*np.pi) * num_roi) % num_roi
    return roi_angle

def get_circ_mean(circular_weights):
    num_angle_bins = len(circular_weights)
    angles = get_equally_spaced_angles(num_angle_bins)
    circ_mean = pingouin.circ_mean(angles=angles, w=circular_weights)
    return circ_mean

def get_circ_means(timeseries):
    num_frames, num_angle_bins = timeseries.shape
    angles = np.repeat(np.expand_dims(get_equally_spaced_angles(num_angle_bins), axis=0), num_frames, axis=0)
    circ_means = pingouin.circ_mean(angles=angles, w=timeseries, axis=1)
    return circ_means

def get_circ_r(circular_weights):
    num_rois = len(circular_weights)
    angles = get_equally_spaced_angles(num_rois)
    return pingouin.circ_r(angles=angles, w=circular_weights)

def get_circ_rs(timeseries):
    num_frames, num_angle_bins = timeseries.shape
    angles = np.repeat(np.expand_dims(get_equally_spaced_angles(num_angle_bins), axis=0), num_frames, axis=0)
    return pingouin.circ_r(angles=angles, w=timeseries, axis=1)

def get_circ_var(angles):
    return 1 - pingouin.circ_r(angles)

def filter_circ_means(circ_means, circ_rs, min_r=0.15):
    # in place change
    if np.issubdtype(circ_means.dtype, np.integer):
        empty_value = -1
    else:
        empty_value = np.nan

    filtered_circ_means = np.copy(circ_means)

    mask = (circ_rs <  min_r)
    filtered_circ_means[mask] = empty_value
    if np.isnan(filtered_circ_means).all():
        logger.warning(f"No circular mean found with r >= {min_r}")
    return filtered_circ_means

def get_peak_angles(timeseries, min_r=0.3):
    # Get the peak angle of each timestep across a timeseries of shape (time, angle). Also works for DF/F treating each roi as a discrete angle.
    mean_rois = get_circ_means(timeseries)
    circ_rs = get_circ_rs(timeseries)

    filtered_circ_means = filter_circ_means(mean_rois, circ_rs, min_r=min_r)
    # Check if filtered_circ_means filtered out all points
    if np.isnan(filtered_circ_means).all():
        min_r = np.quantile(circ_rs, 1-0.1)
        logger.info(f"refiltering circular means with r >= {min_r}")
        filtered_circ_means = filter_circ_means(mean_rois, circ_rs, min_r=min_r)

    peak_angles = filtered_circ_means
    return peak_angles

def get_internal_head_direction(expt, mode='mean', min_r=0.3):
    """Calculate the internal head direction of an experiment.

    Args:
        expt (Experiment): Experiment object.
        mode (str): Must be 'upper', 'lower', or 'mean'. This determines how to estimate internal head direction.
        'upper' or 'lower' will use the respective PB arms based on the roi numbering, with ROI's 0-7 being the lower arm and 8-15 the upper arm.
        'mean' will estimate the head direction of both arms and average them.

    Returns:
        NDArray: Estimated internal head direction in radians.
    """

    assert mode in ['upper', 'lower', 'mean']

    if mode == 'upper':
        peak_angles = get_peak_angles(expt.upper_deltaf, min_r=min_r)
    if mode == 'lower':
        peak_angles = get_peak_angles(expt.lower_deltaf, min_r=min_r)
    if mode == 'mean':
        upper_peak_angles = get_peak_angles(expt.upper_deltaf, min_r=min_r)
        lower_peak_angles = get_peak_angles(expt.lower_deltaf, min_r=min_r)
        peak_angles = np.mean(np.array([upper_peak_angles, lower_peak_angles]), axis=0)

    # Convert from radians into roi angles
    int_head_direction = angle_radians_to_roi(peak_angles, expt.rois_per_arm)

    return int_head_direction

def get_hd_offset(expt, mode='mean'):
    """Calculate the head direction offset between the internal and external head direction in an experiment.

    Args:
        expt (Experiment): Experiment object.
        mode (str, optional): Must be 'upper', 'lower', or 'mean'. This determines how to estimate internal head direction.
        'upper' or 'lower' will use the respective PB arms based on the roi numbering, with ROI's 0-7 being the lower arm and 8-15 the upper arm.
        'mean' will estimate the head direction of both arms and average them.

    Returns:
        NDArray: Estimated head direction offset in radians.
    """

    int_head_direction = get_internal_head_direction(expt, mode=mode)
    ext_head_direction = pingouin.convert_angles(expt.synced_df.angle)

    hd_offset = (int_head_direction - ext_head_direction) % (2*np.pi)
    # Shift from range [0, 2pi) to [-pi, pi)
    hd_offset = pingouin.convert_angles(hd_offset, low=0, high=2*np.pi)
    return hd_offset
