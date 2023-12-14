from gulp2p.preproc.tiff import Tiff
from pathlib import Path
import datetime
import pytest

class TestTiff:
    scanimage_tiff_path = Path(r"Z:\2PImaging\Kerstin\MIMS\20231207\20231207_DR019xiGluSnFr_Fly1_00001.tif")
    olympus_2p_tiff_path = Path(r"Z:\2PImaging\Kerstin\HD7-wtb\20230223_HD7-WTB\tiffs\20230223_HD7-WTB_brain1_standard.tif")

    def test_create_Tiff(self):
        tiff = Tiff(self.scanimage_tiff_path)
        assert tiff is not None
    
    def test_is_scanimage(self):
        tiff = Tiff(self.scanimage_tiff_path)
        assert tiff.is_scanimage is not None

    def test_create_scanimage_tiff(self):
        tiff = Tiff(self.scanimage_tiff_path)
        assert tiff is not None
        assert tiff.is_scanimage is True
        assert tiff.stack is not None
        assert tiff.metadata is not None

    def test_create_olympus_tiff(self):
        tiff = Tiff(self.olympus_2p_tiff_path)
        assert tiff is not None
        assert tiff.is_scanimage is False
        assert tiff.stack is not None
        assert tiff.metadata is not None

# def test_scanimagetiff_pixelunits():
#     scanimage_tiff_path = Path(r"Z:\2PImaging\Kerstin\MIMS\20231207\20231207_DR019xiGluSnFr_Fly1_00001.tif")
