# Tiff.py
from ScanImageTiffReader import ScanImageTiffReader
import tifffile as tf
from datetime import datetime
import os.path
from pathlib import Path
from functools import cached_property, cache
import numpy as np
import xarray as xr
import pickle

from gulp2p.config import TIFF_METADATA_DICT_PATH

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
        load_tiff(self) -> np.ndarray
    """
    def __init__(self, path) -> None:
        """Constructor to create Tiff object.

        Args:
            path (os.PathLike): Path to tiff file.
        """
        self.path = Path(path)


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

        # Get file size and date
        # date can vary depending on how it is saved
        metadata_dict['date'] = datetime.fromtimestamp(self.path.stat().st_ctime)
        metadata_dict['file_size'] = os.path.getsize(self.path)

        # C, Z, and T Dimensions are grouped together in sequential order
        # stack needs to be reshaped using dimension info
        # since the stack is reshaped during access, ignore the original order
        metadata_dict['original_dim_order'] = 'TYX'
        metadata_dict['dimension_order'] = reshape_order

        # Get pixel resolution in micrometers
        with tf.TiffFile(self.path) as tiff:
            resolution = tiff.pages.first.get_resolution(unit=tf.RESUNIT.MICROMETER.value)

        metadata_dict['pixel_width'] = resolution[0]
        metadata_dict['width_unit'] = 'um'
        metadata_dict['pixel_height'] = resolution[1]
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
        """
        # Check if the metadata has already been parsed and saved.
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

    def load_tiff(self) -> np.ndarray:
        """ Load in the tiff from self.path
        
        Returns:
            stack (np.ndarray): the imaging data in dimensions of 
                0: # of volumes
                1: # of frames per volume
                2: # of channels
                3: width
                4: height
        """

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
        stack = stack.reshape(size_t, size_z, size_c, size_y, size_x)
        self.metadata['dimension_order'] = 'TZCYX'

        # Discard the flyback frames
        if discard_fb_frames:
            stack = stack[:,0:size_z-num_fb_frames,:,:,:]

        return stack

    @cached_property
    def stack(self) -> np.ndarray:
        return self.load_tiff()

    @ cached_property
    def length(self):
        """Length of the tiff in seconds

        Returns:
            float: length of tiff in seconds
        """
        length = (self.metadata['SizeZ']
                  * self.metadata['SizeC']
                  * self.metadata['SizeT']
                  * self.metadata['frame_interval'])
        return length

    def get_dim_axis(self, dim):
        index = self.metadata['dimension_order'].index(dim)
        return index

    @cache
    def get_mip_stack(self, motion_correct=True):
        from gulp2p.preproc import imaging
        zaxis = self.get_dim_axis('Z')
        mip_stack = np.squeeze(np.max(self.stack, axis=zaxis))
        if motion_correct:
            numRefImg=50
            upsampleFactor=20
            sigma=2
            locRefImg = round(mip_stack.shape[0]/12)# the initial position in the stack to use for the reference. (~1/12 through the video)
            [shift, mc_stack] = imaging.tif_motion_correct(numRefImg, locRefImg, upsampleFactor, mip_stack, sigma)
            mip_stack = mc_stack
        return mip_stack

def get_tiff_metadata_dict():
    # Return a dictionary of all tiff metadata.
    # Create dictionary if not found.
    # dictionary keys are tiff paths and values are tiff metadata.
    # During trial processing save the tiff path and metadata in dictionary.
    # Dictionary is used for quick access to tiff metadata such as date
    # so that the pickle file can be found since it requires the date and time of a tiff
    # to decide where it is stored and how it is named.

    # Create pickle file if it does not exit
    if not TIFF_METADATA_DICT_PATH.exists():
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
