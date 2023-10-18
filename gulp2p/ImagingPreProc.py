from ScanImageTiffReader import ScanImageTiffReader
import tifffile as tf
import numpy as np
from skimage.registration import phase_cross_correlation
from scipy.ndimage import fourier_shift, gaussian_filter
import math
import napari
from napari.settings import SETTINGS # Changed from from napari.utils.settings import SETTINGS
SETTINGS.application.ipy_interactive = False
from matplotlib import pyplot as plt

def loadTif(path):
    """ Load in a Scan Image tiff from the specified path
    
    Returns:
        stack: the imaging data in dimensions of 
            0: # of volumes
            1: # of frames per volume
            2: # of channels
            3: width
            4: height
        nCh: # of channels
        nDiscardFBFrames: # of frames discarded during fly back
        fpv: frames per volume
    """

    # Make a tiff reader object
    mytiffreader = ScanImageTiffReader(str(path))

    # Get the metadata
    [nCh, discardFBFrames, nDiscardFBFrames, fpv, nVols] = tifMetadata(path)

    # Load the tif data
    vol = mytiffreader.data()

    # Reshape the volume to reflect the experimental parameters
    vol = vol.reshape((int(vol.shape[0]/(fpv*nCh)),fpv,nCh,vol.shape[1], vol.shape[2]))

    # Discard the flyback frames
    stack = vol[:,0:fpv-nDiscardFBFrames,:,:,:]

    return [stack, nCh, nDiscardFBFrames, fpv]


def tifMetadata(path):
    """ Load Scan Image tiff metadata from a Scan Image tiff reader object
    """

    mytiffreader = ScanImageTiffReader(str(path))
    metadat = mytiffreader.metadata()

    # If metadat is empty, it is not a ScanImage tiff file so use a general tiff metadata reader
    if not metadat:
        with tf.TiffFile(path) as tif:
            imagej_metadata = tif.imagej_metadata

            SizeC = int(imagej_metadata.get('channels', 1))
            SizeT = int(imagej_metadata.get('frames', 1))
            SizeZ = int(imagej_metadata.get('images', 1) / SizeC / SizeT)

            nCh = SizeC
            discardFBFrames = None
            nDiscardFBFrames = 0 # Assuming no flyback frames
            fpv = int(SizeT / SizeZ)
            nVols = SizeZ

    else:
        # Step through the metadata, extracting relevant parameters
        for i, line in enumerate(metadat.split('\n')):
            if not 'SI.' in line: continue

            # get channel info
            if 'channelSave' in line:
                if not '[' in line:
                    nCh = 1
                else:
                    nCh = len(line.split('=')[-1].split(sep=';'))

            if 'scanFrameRate' in line:
                fpsscan = float(line.split('=')[-1].strip())

            if 'discardFlybackFrames' in line:
                discardFBFrames = line.split('=')[-1].strip()

            if 'numDiscardFlybackFrames' in line:
                nDiscardFBFrames = int(line.split('=')[-1].strip())

            if 'numFramesPerVolume' in line:
                fpv = int(line.split('=')[-1].strip())

            if 'numVolumes' in line:
                nVols = int(line.split('=')[-1].strip())

    return [nCh, discardFBFrames, nDiscardFBFrames, fpv, nVols]


def getFmInterval(path):
    """Get the frame interval of a tiff file

    Args:
        path (str): path to tiff file

    Returns:
        float: frame interval
    """
    with tf.TiffFile(path) as tif:
        imagej_metadata = tif.imagej_metadata
    fm_interval = float(imagej_metadata.get("finterval"))
    return fm_interval


def getPixelDims(path):
    # Load metadata
    with tf.TiffFile(path) as tif:
        imagej_metadata = tif.imagej_metadata

    # Search metadata for pixel dimensions
    info = imagej_metadata.get('Info', None)
    for line in info.splitlines():
        if '[Reference Image Parameter] WidthConvertValue' in line:
            pixel_width = line.split("=")[-1].strip()

        if '[Reference Image Parameter] WidthUnit' in line:
            width_unit = line.split("=")[-1].strip()

        if '[Reference Image Parameter] HeightConvertValue' in line:
            pixel_height = line.split("=")[-1].strip()

        if '[Reference Image Parameter] HeightUnit' in line:
            height_unit = line.split("=")[-1].strip()

    pixel_dims = {'pixel_width': pixel_width,
                  'width_unit': width_unit,
                  'pixel_height': pixel_height,
                  'height_unit': height_unit,}

    return pixel_dims


def plotMeanPlane(stack, col = 0, ncols = 4):
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

    mean_fig,axs = plt.subplots(nrows = num_plt_rows, ncols = ncols)

    for plane in range(num_planes):
        axs[math.floor(plane/ncols), plane%ncols].imshow(mean_stack[plane,col,:,:])
        axs[math.floor(plane/ncols), plane%ncols].set_axis_off()
        axs[math.floor(plane/ncols), plane%ncols].set_title('plane ' + str(plane))

    plt.show()

    return mean_fig


def stackToMIP(stack, slices):
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


def getSlicesFromStack():
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


def tifMotionCorrect(numRefImg, locRefImg, upsampleFactor, stack, sigma):
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


def divStackMIP(stack, col = 0):
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
    fig = plotMeanPlane(stack, col)

    # Specify the volume slices
    slices = getSlicesFromStack()

    # Calculate the MIPs for the stack
    div_stack_MIP = stackToMIP(stack, slices)

    return [slices, div_stack_MIP]


def motionCorrectSlicedStack(div_stack_MIP, num_ref_img = 100, upsample_factor = 20, sigma = 2):
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
        [shift_dat_now, corrected_stack_1[:,vol,:,:]] = tifMotionCorrect(num_ref_img, loc_ref_img, upsample_factor,
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


def getROIs(stack, roiFN, oldROIs, oldType):
    """ Use napari to get ROIs from a stack, using a given ROI function
    """

    # Load the mean image in napari
    viewer = napari.Viewer()
    viewer.add_image(stack)
    if len(oldROIs) > 0:
        viewer.add_shapes(oldROIs, shape_type=oldType, name = 'Shapes')
    napari.run()

    # Use the ROIs that were drawn in napari to get image masks
    [napOut, allROIs, allMasks] = roiFN(viewer, stack)

    return [napOut, allROIs, allMasks]


def FfromROIsDiv(stack, all_masks):
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


def FfromROIs(stack, allMasks, frameIdx=0):
    """Calculate the raw fluorescence in each ROI in all ROIS on the given stack

    Args:
        stack (NDArray[float64]): Image stack.
        allMasks (NDArray): list of masks.
        frameIdx (int, optional): Index of stack shape that stores frames. Defaults to 0.

    Returns:
        NDArray[float64]: ndarray raw florescence per frame and roi. shape = (# of frames, # of ROI's)
    """

    # Initialize the array to hold the fluorescence data
    rawF = np.zeros((stack.shape[frameIdx],len(allMasks)))

    # Step through each frame in the stack
    # https://stackoverflow.com/questions/1589706/iterating-over-arbitrary-dimension-of-numpy-array
    for fm_num, frame in enumerate(np.moveaxis(stack, frameIdx, -1)):
        # Find the sum of the fluorescence in each ROI for the given frame
        for r in range(0,len(allMasks)):
            rawF[fm_num,r] = np.multiply(frame, allMasks[r]).sum()

    return rawF


def DFoF(rawF):
    """ Calculate the DF/F given a raw fluorescence signal
    The baseline fluorescence is the mean of the lowest 10% of fluorescence signals
    """

    # Initialize the array to hold the DF/F data
    DF = np.zeros(rawF.shape)

    # Calculate the DF/F for each ROI
    for r in range(0,rawF.shape[1]):
        Fbaseline = np.sort(rawF[:,r])[0:round(0.1*rawF.shape[0])].mean()
        DF[:,r] = rawF[:,r]/Fbaseline-1

    return DF


def DFoFfromfirstfms(rawF, fm_interval, baseline_sec=10):
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


def incr_bbox(bounding_box, scale_factor):
    """Scale a bounding box keeping it centered at the same spot

    Args:
        bounding_box (ndarray): Bounding box of shape (2,2): [x or y, min or max]
        scale_factor (float): Amount to scale each side of the bounding box by

    Returns:
        NDArray[float64]: scaled bounding box
    """
    view_box = np.empty(shape=(2, 2))
    for dim in range(2):
        for lim in range(2):
            if lim == 0:  # min
                sign = -1
            if lim == 1:  # max
                sign = 1
            length = bounding_box[dim, 1] - bounding_box[dim, 0]
            scale_amount = sign * (scale_factor - 1) / 2 * length
            view_box[dim, lim] = bounding_box[dim, lim] + scale_amount
    return view_box

def get_bbox(rois, scale_factor=1.5):
    """Given a list of rois, return a bounding box, a scale factor of 1 is a tight box

    Args:
        rois (list[ndarray]): List of rois, each roi is an ndarray of the points that make up the roi.
        scale_factor (float, optional): Amount to scale each side of the bounding box by. Defaults to 1.5.

    Returns:
        NDArray[float64]: Bounding box of shape (2,2): [x or y, min or max]
    """
    XCOL = 0
    YCOL = 1
    # roi_bound axes: [roi, x or y, min or max]
    roi_bounds = np.empty(shape=(len(rois), 2, 2))

    # Get min and max for each roi x and y
    for i, r in enumerate(rois):
        roi_bounds[i][0][0], roi_bounds[i][1][0] = r.min(axis=0)[XCOL : YCOL + 1]
        roi_bounds[i][0][1], roi_bounds[i][1][1] = r.max(axis=0)[XCOL : YCOL + 1]

    # Get the coords for the bounding box, using upper left corner to lower right
    # bounding_box axes: [x or y, min or max]
    bounding_box = np.empty(shape=(2, 2))
    bounding_box[:, 0] = roi_bounds[:, :, 0].min(axis=0)
    bounding_box[:, 1] = roi_bounds[:, :, 1].max(axis=0)

    # Create a larger bounding box to not cut off parts of the PB
    view_box = incr_bbox(bounding_box, scale_factor)
    return view_box