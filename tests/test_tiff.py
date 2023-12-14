from gulp2p.preproc.tiff import Tiff
from pathlib import Path
import datetime
import pytest


# Create tiffs
@pytest.fixture(scope='module')
def scanimage_tiff_path():
    # Shape is (2350, 8, 1, 256, 128)
    return Path(r"Z:\2PImaging\Kerstin\MIMS\20231207\20231207_DR019xiGluSnFr_Fly1_00001.tif")

SCANIMAGE_TIFF_SHAPE = (2350, 8, 1, 256, 128)

@pytest.fixture(scope='module')
def olympus_tiff_path():
    # Shape is (1653, 1, 2, 256, 128)
    return Path(r"Z:\2PImaging\Kerstin\HD7-wtb\20230223_HD7-WTB\tiffs\20230223_HD7-WTB_brain1_standard.tif")

OLYMPUS_TIFF_SHAPE = (1653, 1, 2, 256, 128)

@pytest.fixture(scope='module')
def scanimage_tiff(scanimage_tiff_path):
    return Tiff(scanimage_tiff_path)

@pytest.fixture(scope='module')
def olympus_tiff(olympus_tiff_path):
    return Tiff(olympus_tiff_path)

# Test scanimage tiff
def test_create_scanimage_Tiff(scanimage_tiff):
    assert scanimage_tiff is not None

def test_is_scanimage(scanimage_tiff):
    assert scanimage_tiff.is_scanimage is True

def test_load_scanimage_tiff(scanimage_tiff):
    assert scanimage_tiff.stack is not None
    assert scanimage_tiff.metadata is not None

def test_scanimage_tiff_shape(scanimage_tiff):
    assert scanimage_tiff.stack.shape == SCANIMAGE_TIFF_SHAPE

def test_scanimage_tiff_pixelunits(scanimage_tiff):
    for metadata_key in ["pixel_width", "pixel_height", "width_unit", "height_unit"]:
        value = scanimage_tiff.metadata[metadata_key]
        # print(f"{metadata_key}: {value}")
        assert value is not None
        
# Test olympus tiff
def test_create_olympus_Tiff(olympus_tiff):
    assert olympus_tiff is not None

def test_is_not_scanimage(olympus_tiff):
    assert olympus_tiff.is_scanimage is False

def test_load_olympus_tiff(olympus_tiff):
    assert olympus_tiff.stack is not None
    assert olympus_tiff.metadata is not None

def test_olympus_tiff_shape(olympus_tiff):
    assert olympus_tiff.stack.shape == OLYMPUS_TIFF_SHAPE

def test_olympus_tiff_pixelunits(olympus_tiff):
    for metadata_key in ["pixel_width", "pixel_height", "width_unit", "height_unit"]:
        value = olympus_tiff.metadata[metadata_key]
        # print(f"{metadata_key}: {value}")
        assert value is not None
