# Tiff.py
from ScanImageTiffReader import ScanImageTiffReader
import tifffile as tf
from datetime import datetime
import os.path
from pathlib import Path, PurePath
from functools import cached_property, cache
import numpy as np
import xarray as xr
import pickle
import logging

from gulp2p.config import TIFF_METADATA_DICT_PATH

logger = logging.getLogger(__name__)

reshape_order = 'TZCYX'

class Tiff:
    """A class to handle loading tiffs and metadata.
    Use Tiff.metadata to access metadata
    Use Tiff.stack to access the stack
    Both properties are loaded and cached the first time they are called.

    Usage:
        Create Tiff object:
            `tiff = Tiff(path_to_tiff)`
        Access stack:
            `stack = tiff.stack`
        Access metadata:
            `metadata = tiff.metadata`

    Attributes:
        path (os.PathLike): Path to tiff file.
        scanimage_tiff_reader (ScanImageTiffReader): ScanImageTiffReader of tiff.
        scanimage_metadata (str): ScanImage specific metadata of tiff.
        stack (np.ndarray): Numpy array of tiff data.
        metadata (dict): Dict of relevant tiff metadata.
            Dictionary keys defined in metadata function

    Methods:
        get_scanimage_metadata(self) -> dict
        get_imagej_metadata(self) -> dict
        load_stack(self) -> np.ndarray
    """
    def __init__(self, path, reload_cache=False) -> None:
        """Constructor to create Tiff object.

        Args:
            path (os.PathLike): Path to tiff file.
        """
        self.path = PurePath(path)
        self.reload_cache = reload_cache


    @cached_property
    def scanimage_tiff_reader(self) -> ScanImageTiffReader:
        return ScanImageTiffReader(str(self.path))

    @cached_property
    def scanimage_metadata(self) -> str:
        return self.scanimage_tiff_reader.metadata()

    @cached_property
    def is_scanimage(self) -> bool:
        if self.scanimage_metadata == '':
            # Not a ScanImage tiff
            return False
        else:
            # Is a ScanImage tiff
            return True

    def get_scanimage_metadata(self) -> dict:
        """Get metadata of scanimage tiff.
        Dictionary keys defined in metadata function

        Returns:
            dict: Dictionary of metadata.
        """
        logger.info("Parsing scanimage metadata")
        # Step through the metadata, extracting relevant parameters
        metadata_dict = {}
        for line in self.scanimage_metadata.split('\n'):
            if not 'SI.' in line:
                continue
            line_value = line.split('=')[-1].strip()

            # Dimension info
            if 'channelSave' in line:
                if not '[' in line:
                    metadata_dict['SizeC'] = 1
                else:
                    metadata_dict['SizeC'] = len(line_value.split(sep=' '))
            if 'numVolumes' in line:
                metadata_dict['SizeT'] = int(line_value)
            if 'numFramesPerVolumeWithFlyback' in line:
                metadata_dict['SizeZ'] = int(line_value)
            if 'linesPerFrame' in line:
                metadata_dict['SizeY'] = int(line_value)
            if 'pixelsPerLine' in line:
                metadata_dict['SizeX'] = int(line_value)
            if 'hStackManager.zs =' in line:
                metadata_dict['z_coords'] = [float(num_str) for num_str in line_value.strip('[]').split(' ')]

            # Imaging parameters
            if 'scanFrameRate' in line:
                metadata_dict['frame_rate'] = float(line_value)
            if 'scanFramePeriod' in line:
                metadata_dict['frame_interval'] = float(line_value)
            if 'scanVolumeRate' in line:
                metadata_dict['volume_rate'] = float(line_value)
            if 'scanZoomFactor' in line:
                metadata_dict['zoom_factor'] = float(line_value)
            if 'discardFlybackFrames' in line:
                # https://stackoverflow.com/questions/715417/converting-from-a-string-to-boolean-in-python
                metadata_dict['discard_fb_frames'] = (line_value == "true")
            if 'numDiscardFlybackFrames' in line:
                metadata_dict['num_fb_frames'] = int(line_value)
            if 'flybackTime' in line:
                metadata_dict['flyback_time'] = float(line_value)
            if 'powerFractions' in line:
                metadata_dict['laser_power'] = float(line_value)
            if 'pixelBinFactor' in line:
                metadata_dict['pixel_bin_factor'] = int(line_value)

        # C, Z, and T Dimensions are grouped together in sequential order
        # stack needs to be reshaped using dimension info
        # since the stack is reshaped during access, ignore the original order
        metadata_dict['original_dim_order'] = 'TYX'
        metadata_dict['dimension_order'] = reshape_order

        # TODO: Verify dimension lengths are correct using len(tif.pages) aka total number of frames

        with tf.TiffFile(self.path) as tiff:
            total_num_frames = len(tiff.pages)

        if total_num_frames != metadata_dict['SizeC'] * metadata_dict['SizeZ'] * metadata_dict['SizeT']:
            # update SizeT to match number of frames
            logger.debug("Resizing dimensions to match size of tiff")
            metadata_dict['SizeT'] = int(total_num_frames / metadata_dict['SizeC'] / metadata_dict['SizeZ'])

        # Calculate length
        metadata_dict['length'] = (metadata_dict['SizeC']
                                   * metadata_dict['SizeZ']
                                   * metadata_dict['SizeT']
                                   * metadata_dict['frame_interval'])

        # Get file size and date
        # date can vary depending on how it is saved
        ctime = Path(self.path).stat().st_ctime
        mtime = Path(self.path).stat().st_mtime
        # Estimate creation time if modified time is earlier than creation time
        if ctime >= mtime:
            logger.debug("tiff modification time is earlier than creation time")
            ctime = mtime - metadata_dict['length']
        metadata_dict['date'] = datetime.fromtimestamp(ctime)

        metadata_dict['file_size'] = os.path.getsize(self.path)

        # Get pixel resolution in micrometers
        with tf.TiffFile(self.path) as tiff:
            resolution = tiff.pages.first.get_resolution(unit=tf.RESUNIT.MICROMETER.value)

        # Resolution is given in pixels per micrometer, so the inverse is the pixel width/height
        metadata_dict['pixel_width'] = 1 / resolution[0]
        metadata_dict['width_unit'] = 'um'
        metadata_dict['pixel_height'] = 1 / resolution[1]
        metadata_dict['height_unit'] = 'um'

        return metadata_dict

    def get_imagej_metadata(self) -> dict:
        """Get metadata of imagej tiff.
        Dictionary keys defined in metadata function.

        Returns:
            dict: Dictionary of metadata.
        """
        with tf.TiffFile(self.path) as tiff:
            imagej_metadata = tiff.imagej_metadata

        # TODO: Fix for when Info key is not in metadata (will need to pull dimension information from other keys like frames and images) 
        metadata_dict = {}
        for line in imagej_metadata['Info'].splitlines():
            line_value = line.split('=')[-1].strip()
            # Dimension info
            if 'DimensionOrder' in line:
                metadata_dict['original_dim_order'] = line_value
                metadata_dict['dimension_order'] = reshape_order
            if 'SizeC' in line:
                metadata_dict['SizeC'] = int(line_value)
            if 'SizeT' in line:
                metadata_dict['SizeT'] = int(line_value)
            if 'SizeZ' in line:
                metadata_dict['SizeZ'] = int(line_value)
            if 'SizeY' in line:
                metadata_dict['SizeY'] = int(line_value)
            if 'SizeX' in line:
                metadata_dict['SizeX'] = int(line_value)

            # File data
            if '[Acquisition Parameters Common] ImageCaputreDate =' in line:
                date_str = line_value.strip('\'')
                metadata_dict['date'] = datetime.fromisoformat(date_str)
            if 'TotalFileSize' in line:
                metadata_dict['file_size'] = int(line_value)
            
            # Imaging parameters
            if 'ZoomValue' in line:
                metadata_dict['zoom_factor'] = float(line_value)
            if '[Reference Image Parameter] WidthConvertValue' in line:
                metadata_dict['pixel_width'] = float(line_value)
            if '[Reference Image Parameter] WidthUnit' in line:
                metadata_dict['width_unit'] = line_value
            if '[Reference Image Parameter] HeightConvertValue' in line:
                metadata_dict['pixel_height'] = float(line_value)
            if '[Reference Image Parameter] HeightUnit' in line:
                metadata_dict['height_unit'] = line_value
        
        frame_interval = float(imagej_metadata['finterval'])
        metadata_dict['frame_interval'] = frame_interval
        metadata_dict['frame_rate'] = 1 / frame_interval

        # Calculate length
        metadata_dict['length'] = (metadata_dict['SizeZ']
                                   * metadata_dict['SizeC']
                                   * metadata_dict['SizeT']
                                   * metadata_dict['frame_interval'])

        # Assume no flyback frames if not in scanimage format
        metadata_dict['discard_fb_frames'] = False
        metadata_dict['num_fb_frames'] = 0
        metadata_dict['volume_rate'] = None
        metadata_dict['flyback_time'] = None
        metadata_dict['laser_power'] = None
        metadata_dict['pixel_bin_factor'] = None

        return metadata_dict

    @cached_property
    def metadata(self) -> dict:
        """Load tiff metadata
        ScanImage tiffs will be processed with ScanImageTiffReader.
        Other tiffs will be processed with tiffile.TiffFile.imagej_metadata.
        For ScanImage tiffs, pixel resolution is called from tiffile.
        This is because there is no method in ScanImageTiffReader to get the resolution info.

        Args:
            self (obj): Class object.

        Returns:
            dict: Dictionary of metadata.
            Keys:
                SizeC (int): Number of channels.
                SizeT (int): Number of frames.
                SizeZ (int): Number of slices.
                SizeY (int): Height of each frame.
                SizeX (int): Width of each frame.
                frame_rate (float): Frame rate of tiff.
                frame_interval (float): Frame interval of tiff.
                volume_rate (float): Volume rate of tiff. Only defined for scanimage tiffs.
                length (float): Length of tiff in seconds.
                zoom_factor (float): Optical zoom amount.
                discard_fb_frames (bool): True if flyback frames need to be discarded. 
                num_fb_frames (int): Number of flyback frames per volume.
                flyback_time (float): Time it takes for Z motor to move back to start position. Only defined for scanimage tiffs.
                laser_power (float): Percent of laser power used during acqusition. Only defined for scanimage tiffs.
                pixel_bin_factor (int): bin size used during acquisition. Only defined for scanimage tiffs.
                date (datetime): Time of acquisition.
                file_size (int): Size of tiff in bytes.
                dimension_order (str): Dimension order.
                pixel_width (float): width of pixels in units from width_unit.
                width_unit (str): units used to define width of pixels.
                pixel_height (float): height of pixels in units from width_unit.
                height_unit (str): units used to define height of pixels.
                z_coords (List[float]): coordinates for each z position. Only defined for scanimage tiffs.
        """
        # Check if the metadata has already been parsed and saved.
        if not self.reload_cache:
            tiff_metadata_dict = get_tiff_metadata_dict()
            metadata = tiff_metadata_dict.get(self.path)
            if metadata is not None:
                return metadata

        # If it isn't, parse it and save it.
        if self.is_scanimage:
            metadata = self.get_scanimage_metadata()
        else:
            metadata = self.get_imagej_metadata()
        save_tiff_metadata(self.path, metadata)
        return metadata

    def load_stack(self) -> np.ndarray:
        """ Load in the tiff stack from self.path
        
        Returns:
            stack (np.ndarray): the imaging data in dimensions of 
                0: # of volumes
                1: # of frames per volume
                2: # of channels
                3: width
                4: height
        """
        logger.info("loading tiff stack")

        # Get the metadata
        metadata = self.metadata
        size_t = metadata['SizeT']
        size_z = metadata['SizeZ']
        size_c = metadata['SizeC']
        size_y = metadata['SizeY']
        size_x = metadata['SizeX']
        discard_fb_frames = metadata['discard_fb_frames']
        num_fb_frames = metadata['num_fb_frames']

        # Load the tiff data
        stack = np.copy(self.scanimage_tiff_reader.data())

        # Reshape the volume to reflect the experimental parameters
        if (np.size(stack) != size_t * size_z * size_c * size_y * size_x):
            logger.debug("Tiff stack doesn't match shape given in metadata. Adjusting metadata to actual shape")
            num_remainder_frames = np.size(stack) % (size_z * size_c * size_y * size_x)
            size_t = (np.size(stack) // (size_z * size_c * size_y * size_x))
            self.metadata['SizeT'] = size_t
            stack = stack.flatten()[:stack.size-num_remainder_frames].reshape(size_t, size_z, size_c, size_y, size_x)
        else:
            stack = stack.reshape(size_t, size_z, size_c, size_y, size_x)
        self.metadata['dimension_order'] = 'TZCYX'

        # Convert stack to xarray object
        dims = tuple(dim for dim in self.metadata['dimension_order'])
        stack = xr.DataArray(data=stack, dims=dims)
        
        # Discard the flyback frames
        if discard_fb_frames:
            stack = stack.sel(Z=slice(0, size_z-num_fb_frames))
            self.metadata['SizeZ'] = size_z - num_fb_frames
        logger.debug(f"Tiff shape: {stack.shape}")

        # Define stack coordinates
        t_coords = np.arange(0, stack.sizes['T']) / self.metadata['volume_rate']
        logger.debug(f"z_coords: {self.metadata.get('z_coords')}")
        if self.metadata.get('z_coords') is not None:
            z_coords = self.metadata.get('z_coords')
        else:
            z_coords = None
        logger.debug(f"z_coords: {z_coords}")
        y_coords = np.arange(0, stack.sizes['Y']) * self.metadata['pixel_height']
        x_coords = np.arange(0, stack.sizes['X']) * self.metadata['pixel_width']
        
        stack = stack.assign_coords(T=t_coords, Z=z_coords, Y=y_coords, X=x_coords)
        if z_coords is not None:
            stack.Z.attrs["units"] = 'um'
        stack.Y.attrs["units"] = self.metadata['height_unit']
        stack.X.attrs["units"] = self.metadata['width_unit']

        stack.attrs["long_name"] = "Raw Florescence"

        # Correct for negative raw florescence
        min_rawf = np.min(stack)
        if min_rawf < 0:
            logger.debug(f"correcting for negative raw florescence. min flor: {min_rawf}")
            stack += abs(min_rawf)

        return stack

    @cached_property
    def stack(self) -> np.ndarray:
        return self.load_stack()

    @ cached_property
    def length(self):
        """Length of the tiff in seconds

        Returns:
            float: length of tiff in seconds
        """
        return self.metadata['length']

    def get_dim_axis(self, dim):
        index = self.metadata['dimension_order'].index(dim)
        return index

    @cache
    def get_mip_stack(self, motion_correct=False):
        from gulp2p.preproc import imaging
        zaxis = self.get_dim_axis('Z')
        mip_stack = np.squeeze(np.max(self.stack, axis=zaxis))
        if motion_correct:
            mip_stack, template = imaging.motion_correct_mip(mip_stack)
        return mip_stack

    def get_creation_date(self):
        # If the tiff was copied to another system its creation date will get reset but not other metadata.
        # In these cases use modification time - estimated length as tiff creation time.
        # Length of trial can be estimated with fm_interval and num of frames.
        file_stats = Path(self.path).stat()
        ctime = file_stats.st_ctime
        mtime = file_stats.st_mtime
        
        # Modified before created
        # This means creation date is innacurate and needs to be estimated
        if mtime < ctime:
            ctime = mtime - self.length
        return datetime.fromtimestamp(ctime)

### Non-Class functions

def get_tiff_metadata_dict():
    # Return a dictionary of all tiff metadata.
    # Create dictionary if not found.
    # dictionary keys are tiff paths and values are tiff metadata.
    # During trial processing save the tiff path and metadata in dictionary.
    # Dictionary is used for quick access to tiff metadata such as date
    # so that the pickle file can be found since it requires the date and time of a tiff
    # to decide where it is stored and how it is named.

    # Create pickle file if it does not exit
    if not Path(TIFF_METADATA_DICT_PATH).exists():
        tiff_metadata_dict = {}
        with open(TIFF_METADATA_DICT_PATH, 'wb+') as file:
            pickle.dump(tiff_metadata_dict, file)

    # Load tiff
    with open(TIFF_METADATA_DICT_PATH, 'rb+') as file:
        tiff_metadata_dict = pickle.load(file)

    # Return dict
    return tiff_metadata_dict

def set_tiff_metadata_dict(tiff_metadata_dict):
    # Save dict
    with open(TIFF_METADATA_DICT_PATH, 'wb+') as file:
        pickle.dump(tiff_metadata_dict, file)

def save_tiff_metadata(path, metadata):
    # Save tiff metadata in tiff_metadata_dict pickle file
    tiff_metadata_dict = get_tiff_metadata_dict()
    tiff_metadata_dict[path] = metadata
    set_tiff_metadata_dict(tiff_metadata_dict)
