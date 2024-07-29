import numpy as np
import pandas as pd
from skimage.registration import phase_cross_correlation
from scipy.ndimage import fourier_shift, gaussian_filter
import math
import napari
from napari.settings import SETTINGS # Changed from from napari.utils.settings import SETTINGS
SETTINGS.application.ipy_interactive = False
from matplotlib import pyplot as plt
import os
from pathlib import Path
import logging
import tempfile
import shapely.geometry as geo
import caiman as cm
from caiman.motion_correction import MotionCorrect

from gulp2p import preproc
from gulp2p.preproc import utils, behavior, draw, tiff as tf
from gulp2p.preproc.tiff import Tiff
from gulp2p.preproc.trial import Trial
from gulp2p.config import TRIAL_PICKLE_DIR, BHV_DATA_RAW_DIR, CONFIG_DICT
from unityvr.analysis import align2img

logger = logging.getLogger(__name__)

def plot_mean_plane(stack, col = 0, ncols = 4):
    """
    Plot the mean of each plane in a stack in a plot with ncol columns

    Arguments:
        stack = the imaging stack with dimensions:
            - plane, # of colors, # of pix in x, # of pix in y
        col = the color channel of interest
        ncol = the number of columns for the plot

    Returns:
        mean_fig = a figure with all of the mean planes
    """
    num_planes = stack.shape[1]
    num_plt_rows = math.ceil(num_planes/ncols)

    mean_stack = stack.mean(axis=0)

    mean_fig,axs = plt.subplots(nrows = num_plt_rows, ncols = ncols, squeeze=False)

    for plane in range(num_planes):
        axs[math.floor(plane/ncols), plane%ncols].imshow(mean_stack[plane,col,:,:])
        axs[math.floor(plane/ncols), plane%ncols].set_title('plane ' + str(plane))

    for row in range(num_plt_rows):
        for col in range(ncols):
            axs[row, col].set_axis_off()

    plt.show()

    return mean_fig

def stack_to_mip(stack, slices):
    """
    Convert a stack to a series of maximum intensity projections (MIPs),
    where the slices to be consider in each projection are specified
    in slices

    Arguments:
        stack = the imaging stack with dimensions:
            - plane, # of colors, # of pix in x, # of pix in y
        slices = the layers that set the bounds of each MIP

    Returns:
        div_stack_MIP = the MIPs from the stack
    """

    ### Need to add error checking in case the slices fall outside of the stack dimensions
    num_vols = len(slices)-1
    div_stack_MIP = np.zeros((stack.shape[0], num_vols) + stack.shape[2:])

    for v in range(num_vols):
        div_stack_MIP[:,v,:,:,:] = stack[:,slices[v]:slices[v+1],:,:,:].max(axis=1)

    return np.squeeze(div_stack_MIP)

def get_slices_from_stack():
    """
    Specify which slices to consider in the stack

    Returns:
        slices = the slice boundaries
    """

    num_vols = int(input('How many volumes?'))

    slices = [0]*(num_vols+1)

    # Specify the volume boundaries
    slices[0] = int(input('first slice to consider?'))
    for vol in range(num_vols):
        slices[vol+1] = int(input('last slice in volume ' + str(vol+1) + '?'))

    return slices

def get_mc_obj_params(mc):
    # Given a MotionCorrect object from caiman, return a dictionary of the relevant attributes.
    mc_params = {
        'border_nan': None,
        'dview': None,
        'fname': None,
        'gSig_filt': None,
        'indices': None,
        'is3D': None,
        'max_deviation_rigid': None,
        'max_shifts': None,
        'min_mov': None,
        'niter_rig': None,
        'nonneg_movie': None,
        'num_splits_to_process_els': None,
        'num_splits_to_process_rig': None,
        'overlaps': None,
        'pw_rigid': None,
        'shifts_opencv': None,
        'splits_els': None,
        'splits_rig': None,
        'strides': None,
        'upsample_factor_grid': None,
        'use_cuda': None,
        'var_name_hdf5:': None,
    }

    mc_outputs = {
        'border_to_0': None,
        'fname_tot_rig': None,
        'mmap_file': None,
        'shifts_rig': None,
        'templates_rig': None,
        'total_template_rig': None,
        'coord_shifts_els': None,
        'fname_tot_els': None,
        'template': None,
        'templates_els': None,
        'total_template_els': None,
        'x_shifts_els': None,
        'y_shifts_els': None,
    }

    mc_dict = {'mc_params': mc_params, 'mc_outputs': mc_outputs}

    raise NotImplementedError

def motion_correct(stack, mc_params=None):
    # Stack must be 3 dimensional (T,Y,X)

    default_mc_params = {
        'strides': (48, 48),       # maximum allowed rigid shift in pixels (view the movie to get a sense of motion)
        'overlaps': (24, 24),      # create a new patch every x pixels for pw-rigid correction
        'max_shifts': (6, 6),      # overlap between patches (size of patch strides+overlaps)
        'max_deviation_rigid': 3,  # maximum deviation allowed for patch with respect to rigid shifts
        'pw_rigid': False,         # flag for performing rigid or piecewise rigid motion correction
        'shifts_opencv': True,     # flag for correcting motion using bicubic interpolation (otherwise FFT interpolation is used)
        'border_nan': 'copy',       # replicate values along the boundary (if True, fill in with NaN) [True, False 'min', 'copy']
        }

    # Add default values to unspecified mc_params
    if mc_params is None:
        mc_params = default_mc_params
    else:
        for key, value in default_mc_params.items():
            if key not in mc_params.keys():
                mc_params[key] = value

    if stack.shape[0] < 30:
        logger.debug("stack is too small for default split size, setting it to 1")
        mc_params['splits_els'] = 1
        mc_params['splits_rig'] = 1

    # Start a cluster to use multiprocessing during motion correction
    c, dview, n_processes = cm.cluster.setup_cluster(backend='multiprocessing',
                                                     n_processes=None,
                                                     single_thread=False)

    # Set environment variables so caiman stores mmap files in temp directory instead of current directory.
    # TODO: Save existing value of environment variables and re-set them after
    tempdir = tempfile.TemporaryDirectory()
    os.environ['CAIMAN_NEW_TEMPFILE'] = "True"
    os.environ['CAIMAN_TEMP'] = tempdir.name

    # Save stack to temp file to pass to MotionCorrect
    temp_stack_file = Path(tempdir.name, "stack.hdf5")
    stack_movie = cm.movie(stack, file_name=temp_stack_file.name)
    stack_movie.save(temp_stack_file)

    # Create motion correction object
    mc = MotionCorrect(temp_stack_file, dview=dview, **mc_params)

    # correct for rigid motion correction
    mc.motion_correct()
    mc_obj = mc

    # Copy reshaped stack_mmap so that mmapped file can be deleted
    stack_mc = mc.apply_shifts_movie(temp_stack_file)

    cm.stop_server(dview=dview) # stop the server
    mc_obj.dview = None # Remove pool object from mc so it can be pickled.

    tempdir.cleanup()
    del os.environ['CAIMAN_NEW_TEMPFILE']
    del os.environ['CAIMAN_TEMP']

    return stack_mc, mc_obj

def tif_motion_correct(numRefImg, locRefImg, upsampleFactor, stack, sigma):
    """ Motion correct a tiff stack by using phase cross correlation
    
    Arguments:
        numRefImg = the number of images to average for the reference image
        locRefImg = the initial position in the stack to use for the reference
        upsampleFactor = how much to upsample the image in order to shift the image by less than one pixel
        stack = the stack to be registered
        sigma = the sigma to use in Gaussian filtering
        
    Returns:
        shift = the shift coordinates
        stackMC = the motion corrected stack
    """
    # Set defaults
    if numRefImg is None:
        numRefImg = 50
    if upsampleFactor is None:
        upsampleFactor = 20
    if sigma is None:
        sigma = 2

    # Generate reference image
    refImg = np.mean(stack[locRefImg:locRefImg+numRefImg,:,:],axis=0)

    # Gaussian filter the reference image
    refImgFilt = gaussian_filter(refImg, sigma=sigma)

    # Create empty arrays to hold the registration metrics
    shift = np.zeros((2, stack.shape[0]))
    error = np.zeros(stack.shape[0])
    diffphase = np.zeros(stack.shape[0])

    # Create an empty array to hold the motion corrected stack
    stackMC = np.ones(stack.shape).astype('int16')

    # Correct each volume
    for i in range(stack.shape[0]):
        # Get the current image
        shifImg = stack[i,:,:]

        # Filter it
        shifImgFilt = gaussian_filter(shifImg, sigma=sigma)

        # Find the cross correlation between the reference image and the current image
        shift[:,i], error[i], diffphase[i] = phase_cross_correlation(refImgFilt, shifImgFilt,
                                                                     upsample_factor = upsampleFactor)

        # Shift the image in Fourier space
        offset_image = fourier_shift(np.fft.fftn(shifImg), shift[:,i])

        # Convert back and save the motion corrected image
        stackMC[i,:,:] = np.fft.ifftn(offset_image).real.astype('int16')

    return [shift, stackMC]

def div_stack_mip(stack, col = 0):
    """
    Slice a stack into volumes and get the MIPs of those volumes

    Arguments:
        stack = the imaging stack with dimensions:
            - plane, # of colors, # of pix in x, # of pix in y
        col = the color channel to inspect

    Returns:
        slices = the volume boundaries
        div_stack_MIP = the divided, MIPed stack
    """
    # Plot the mean slices
    fig = plot_mean_plane(stack, col)

    # Specify the volume slices
    slices = get_slices_from_stack()

    # Calculate the MIPs for the stack
    div_stack_MIP = stack_to_mip(stack, slices)

    return [slices, div_stack_MIP]

def motion_correct_sliced_stack(div_stack_MIP, num_ref_img = 100, upsample_factor = 20, sigma = 2):
    """
    Motion correct each volume MIP in a sliced stack

    Arguments:
        div_stack_MIP = the MIPs of a sliced stack
        num_ref_img, upsample_factor, sigma = see tifMotionCorrect

    Returns:
        corrected_stacks = the motion corrected stacks
    """
    loc_ref_img = round(div_stack_MIP.shape[0]/12)

    shift_dat = []
    corrected_stack_1 = np.ones(div_stack_MIP.shape[0:2] + div_stack_MIP.shape[3:]).astype('int16')

    for vol in range(div_stack_MIP.shape[1]):
        [shift_dat_now, corrected_stack_1[:,vol,:,:]] = tif_motion_correct(num_ref_img, loc_ref_img, upsample_factor,
                                                                        div_stack_MIP[:,vol,0,:,:],sigma)
        shift_dat.append(shift_dat_now)

    if div_stack_MIP.shape[2] > 1:
        corrected_stack_2 = np.ones(div_stack_MIP.shape[0:2] + div_stack_MIP.shape[3:]).astype('int16')
        for vol in range(div_stack_MIP.shape[1]):
            for frame in range(0,shift_dat[vol].shape[1]):
                shif_img = np.squeeze(div_stack_MIP[frame,vol,1,:,:])

                # Shift the image in Fourier space
                offset_image = fourier_shift(np.fft.fftn(shif_img), shift_dat[vol][:,frame])

                # Convert back and save the motion corrected image
                corrected_stack_2[frame,vol,:,:] = np.fft.ifftn(offset_image).real.astype('uint16')

        corrected_stacks = np.concatenate((corrected_stack_1,corrected_stack_2),axis = 1)
    else:
        corrected_stacks = corrected_stack_1
    return corrected_stacks

def get_rois(mip_stack, roi_func, old_rois, old_type):
    """ Use napari to get ROIs from a stack, using a given ROI function
    """

    # Load the mean image in napari
    viewer = napari.Viewer()

    if CONFIG_DICT['show_full_stack']:
        viewer.add_image(mip_stack, name="full_stack",
                         colormap=CONFIG_DICT['napari_colormap'],
                         gamma=CONFIG_DICT['napari_gamma'])

    viewer.add_image(mip_stack.mean(axis=0),
                     colormap=CONFIG_DICT['napari_colormap'],
                     gamma=CONFIG_DICT['napari_gamma'])
    initial_rois = None
    initial_shape_type = 'polygon'
    if len(old_rois) > 0:
        initial_rois = old_rois
        initial_shape_type = old_type
    viewer.add_shapes(initial_rois, shape_type=initial_shape_type,
                      name='Shapes', opacity=CONFIG_DICT['napari_shape_opacity'])
    napari.run()

    # Use the ROIs that were drawn in napari to get image masks
    [_, rois, masks] = roi_func(viewer, mip_stack.mean(axis=0))

    return [_, rois, masks]

def create_dist_array(shapely_rois):
    # Create 2d array of distances
    dist_array = np.empty(shape=(len(shapely_rois), len(shapely_rois)))
    for roi_index in range(len(shapely_rois)):
        for other_index in range(len(shapely_rois)):
            if roi_index == other_index:
                dist_array[roi_index, other_index] = 0
            else:
                dist_array[roi_index, other_index] = shapely_rois[roi_index].distance(shapely_rois[other_index])
    return dist_array

def get_nearest_point_idx(point_idx, other_point_idxs, dist_array):
    min_dist = None
    min_idx = None
    for other_point_idx in other_point_idxs: 
        dist = dist_array[point_idx, other_point_idx]
        if min_dist is None:
            min_dist = dist
            min_idx = other_point_idx
        elif dist < min_dist:
            min_dist = dist
            min_idx = other_point_idx
    return min_idx

def get_best_path_idx(shapely_rois):
    dist_array = create_dist_array(shapely_rois)
    roi_indexes = list(range(len(shapely_rois)))

    start_roi_idx = roi_indexes[0]
    sorted_path = [start_roi_idx]
    roi_indexes.remove(start_roi_idx)

    # Build shortest path
    while len(sorted_path) < len(shapely_rois):
        # Get closest point to end of path
        path_end_roi_idx = sorted_path[-1]
        nearest_point_idx = get_nearest_point_idx(path_end_roi_idx, roi_indexes, dist_array)

        # Add point to closer end of the path
        if len(sorted_path) == 1:
            sorted_path.append(nearest_point_idx)
            roi_indexes.remove(nearest_point_idx)
            continue
        
        if (  dist_array[sorted_path[0], nearest_point_idx]
            < dist_array[sorted_path[-1], nearest_point_idx]):
            sorted_path.insert(0, nearest_point_idx)
        else:
            sorted_path.append(nearest_point_idx)
        roi_indexes.remove(nearest_point_idx)
    # Ensure path starts at a greater y value than it ends at.
    start_y = shapely_rois[sorted_path[0]].centroid.y
    end_y = shapely_rois[sorted_path[-1]].centroid.y
    if start_y >= end_y:
        return sorted_path
    else:
        return sorted_path[::-1]

def get_sorted_rois(rois, masks):
    shapely_rois = [geo.Polygon(roi) for roi in rois]
    best_path_idx = get_best_path_idx(shapely_rois)
    sorted_rois = [rois[idx] for idx in best_path_idx]
    sorted_masks = [masks[idx] for idx in best_path_idx]
    return sorted_rois, sorted_masks

def f_from_rois_div(stack, all_masks):
    """ Calculate the raw fluorescence in each ROI in all ROIS on the given stack
    for a stack with multiple volumes
    """
    num_frames = stack.shape[0]

    # Initialize the array to hold the fluorescence data
    rawF = np.zeros((num_frames,len(all_masks)))

    # Step through each frame in the stack
    for fm in range(0,num_frames):
        fmNow = stack[fm,:,:,:]

        # Find the sum of the fluorescence in each ROI for the given frame
        for roi in range(0,len(all_masks)):
            vol = all_masks[all_masks['roi'] == roi]['layer'][0]
            rawF[fm,roi] = np.multiply(fmNow[vol,:,:], np.transpose(all_masks[all_masks['roi'] == roi]['mask'][0])).sum()

    return rawF

def get_raw_flor(stack, masks, frame_dim=0):
    """Calculate the raw fluorescence in each ROI in all ROIS on the given stack

    Args:
        stack (NDArray[float64]): Image stack of shape TYX.
        masks (NDArray): list of masks.
        frame_dim (int, optional): Index of stack shape that stores frames. Defaults to 0.

    Returns:
        NDArray[float64]: ndarray raw florescence per frame and roi. shape = (# of frames, # of ROI's)
    """

    # Initialize the array to hold the fluorescence data
    raw_flor = np.zeros((stack.shape[frame_dim], len(masks)))

    # Step through each frame in the stack
    # https://stackoverflow.com/questions/1589706/iterating-over-arbitrary-dimension-of-numpy-array
    for fm_idx, frame in enumerate(np.moveaxis(stack, frame_dim, 0)):
        # Find the sum of the fluorescence in each ROI for the given frame
        for roi_idx in range(0, len(masks)):
            raw_flor[fm_idx, roi_idx] = np.multiply(np.squeeze(frame), masks[roi_idx].T).sum()

    return raw_flor

def get_delta_flor(raw_flor):
    """ Calculate the DF/F given a raw fluorescence signal
    The baseline fluorescence is the mean of the lowest 10% of fluorescence signals
    """

    # Initialize the array to hold the DF/F data
    delta_flor = np.zeros(raw_flor.shape)

    # Correct for negative rawf
    min_rawf = np.min(raw_flor)
    if min_rawf < 0:
        raw_flor += abs(min_rawf)

    # Calculate the DF/F for each ROI
    for roi in range(0,raw_flor.shape[1]):
        baseline_flor = np.sort(raw_flor[:,roi])[0:round(0.1*raw_flor.shape[0])].mean()
        delta_flor[:,roi] = (raw_flor[:,roi]/baseline_flor)-1

    return delta_flor

def get_delta_flor_from_first_fms(rawF, fm_interval, baseline_sec=10):
    """Calculate the DF/F given a raw fluorescence signal
    The baseline fluorescence is the mean of first 10 seconds of florescence

    Args:
        rawF (NDArray[float64]): ndarray of raw florescence over frame and roi.
                                 shape = (# of frames, # of ROI's)
        fm_interval (float): Time it takes to capture one frame (in seconds per frame)
        baseline_sec (float): How long into the trial to get the baseline from

    Returns:
        NDArray[float64]: ndarray of delta florescence over baseline florescence per frame and roi.
                          shape = (# of frames, # of ROI's)
    """

    # Initialize the array to hold the DF/F data
    DF = np.zeros(rawF.shape)

    # rawF axes: [frames, rois]
    baseline_sec = 10
    baseline_end_frame = round(baseline_sec / fm_interval)

    # Calculate the DF/F for each ROI
    for r in range(0, rawF.shape[1]):
        Fbaseline = rawF[0:baseline_end_frame, r].mean()
        DF[:, r] = rawF[:, r] / Fbaseline - 1

    return DF

def preprocess(path, tiff=None, bhv_paths=None, prev_rois=None, save=True, skip=False, use_full_volume=False):
    """Draw rois over brain regions in napari.
    Returns dictionary with florescence data.

    Args:
        path (Path): Path of tiff to preprocess.
        tiff (Tiff): Tiff object of tiff. If not given, created from path. Defaults to None.
        prev_rois (list): List of rois from previous napari session. Defaults to None.
        num_ref_img (int, optional): The number of images to average for the reference image. Defaults to 50.
        upsample_factor (int, optional): How much to upsample the image in order to shift the image by less than one pixel. Defaults to 20.
        sigma (int, optional): The sigma to use in Gaussian filtering. Defaults to 2.
        save (bool, optional): If True saves the processed trial as a pickle. Defaults to True.

    Returns:
        Trial: Trial object containing tiff and synced dataframe 
    """
    logger.info(f"Processing {path}")

    # Load the tiff
    if tiff is None:
        tiff = Tiff(path)
    stack = tiff.stack

    # Load old trials info
    old_trial = utils.load_trial(tiff)
    if old_trial is not None:
        logger.info(f"Trial {old_trial.name} was preprocessed. Loading rois and using volume slices {old_trial.slices}")
        old_rois = old_trial.rois
        slices = old_trial.slices

    # Load bhv object and concatenate if there are multiple
    uvr = behavior.load_bhv_data(bhv_paths)
    if uvr.nidDf.empty:
        logger.info("loading bhv file from json (skipping pickle since it is missing nidaq signal)")
        uvr = behavior.load_bhv_data(bhv_paths, from_pickle=False)

    # Plot the mean of each plane
    fig_mean_planes = plot_mean_plane(stack, col=0) # col=1 to plot the second channel if it exists

    if old_trial is None:
        # Specify the planes to use for the Maximum Intensity Projection (MIP)
        if use_full_volume:
            slices = [0, tiff.metadata['SizeZ']-1]
            logger.info(f"Using full volume as slice {slices}")
        else:
            slices = get_slices_from_stack()
            logger.info(f"Using slices: {slices}")

    # Calculate the MIP
    mip_stack = stack_to_mip(stack,slices)

    # Motion correct the MIP
    if hasattr(old_trial, 'mc_obj') and (old_trial.mc_obj is not None):
        logger.info("Motion correction has already been computed, applying shifts")

        tempdir = tempfile.TemporaryDirectory()

        # Save mip_stack to temp file to pass to MotionCorrect
        temp_stack_file = Path(tempdir.name, "mip_stack.hdf5")
        stack_movie = cm.movie(mip_stack, file_name=temp_stack_file.name)
        stack_movie.save(temp_stack_file)

        mc_obj = old_trial.mc_obj
        # TODO: Create new MotionCorrect object with settings from mc_obj
        stack_mc = mc_obj.apply_shifts_movie(temp_stack_file)
        tempdir.cleanup()
    else:
        stack_mc, mc_obj = motion_correct(mip_stack)

    # Plot the before and after
    fig, axs = plt.subplots(ncols = 2, nrows = 1, figsize = (6,2))
    axs[0].imshow(mip_stack.mean(axis=0))
    axs[0].axis('off')
    axs[1].imshow(stack_mc.mean(axis=0))
    axs[1].axis('off')

    if skip:
        rois = old_trial.rois
        masks = old_trial.masks
    else:
        # Get the ROIS - For EB wedges, there are a number of other possible ROI functions for different shapes
        initial_rois = []
        initial_roi_type = ''
        if old_trial is not None:
            initial_rois = old_rois
            initial_roi_type = 'polygon'
        elif prev_rois is not None:
            initial_rois = prev_rois
            initial_roi_type = 'polygon'
        [_, rois, masks] = get_rois(stack_mc, preproc.draw.PolyROIs,
                                    initial_rois, initial_roi_type)

    # Sort ROIs based on their position
    rois, masks = get_sorted_rois(rois, masks)

    # Get the raw fluorescence
    raw_flor = get_raw_flor(stack_mc, masks)

    # Get the DF/F
    delta_flor = get_delta_flor(raw_flor)

    # Syncronize behavioral and imaging data
    if not uvr.nidDf.empty:
        fpv = tiff.metadata.get('SizeZ')
        [imgInd, volFramePos] = align2img.findImgFrameTimes(uvr, fpv=fpv)
        if len(volFramePos) == 0: # Check if list is empty
            logger.warning("No nidaq frame signal found, unable to sync behavior and image data.")
        imgDat = pd.DataFrame(delta_flor).add_prefix("roi")
        synced_df = align2img.combineImagingAndPosDf(imgDat, uvr.posDf, volFramePos)
    else:
        logger.warning("No nidaq data in unity vr data.")
        synced_df = None

    # TODO: make mip_frame, raw_flor, and delta_flor multichannel if image is multichannel.
    trial = Trial(path = path,
                  tiff_metadata = tiff.metadata,
                  bhv_paths = bhv_paths,
                  mip_frame = stack_mc.mean(axis=0),
                  slices = slices,
                  mc_obj = mc_obj,
                  rois = rois,
                  masks = masks,
                  raw_flor = raw_flor,
                  synced_df = synced_df)

    #TODO: Add overwrite protection
    # Pickle and save the data
    if save:
        logger.info("Saving trial")
        utils.save_trial(trial)

    return trial

def process_all(path_list, save=True, skip=False, use_full_volume=False):
    prev_rois=None
    bhv_paths = utils.get_bhv_paths(path_list)
    for index, path in enumerate(path_list):
        trial = preprocess(path, prev_rois=prev_rois, bhv_paths=bhv_paths[index],
                save=save, skip=skip, use_full_volume=use_full_volume)
        prev_rois = trial.rois
