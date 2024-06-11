# config.py

from pathlib import Path
import strictyaml
import logging

logger = logging.getLogger(__name__)

DEFAULT_CONFIG_FILE = Path(Path(__file__).parent, "..", "settings.yaml").resolve()

def load_config(config_file=None):
    if config_file is None:
        config_file = DEFAULT_CONFIG_FILE
    schema = strictyaml.Map({"main_pickle_dir": strictyaml.Str(),
                             "bhv_data_dir": strictyaml.Str(),
                             "fictrac_dir": strictyaml.Str(),
                             "napari_gamma": strictyaml.Float(),
                             "napari_shape_opacity": strictyaml.Float(),
                             "napari_colormap": strictyaml.Str(),
                             "show_full_stack": strictyaml.Bool()})

    yaml_text = config_file.read_text()
    config_dict = strictyaml.load(yaml_text, schema).data
    logger.info("config loaded")
    return config_dict

CONFIG_DICT = load_config()
MAIN_PICKLE_DIR = Path(CONFIG_DICT['main_pickle_dir'])
MAIN_BHV_DATA_DIR = Path(CONFIG_DICT['bhv_data_dir'])

TRIAL_PICKLE_DIR = Path(MAIN_PICKLE_DIR, "preproc")
TIFF_METADATA_DICT_PATH = Path(TRIAL_PICKLE_DIR, "tiff_metadata.pickle")

BHV_DATA_RAW_DIR = Path(MAIN_BHV_DATA_DIR, "raw data")
BHV_DATA_PICKLE_DIR = Path(MAIN_BHV_DATA_DIR, "pickles")

FICTRAC_DIR = Path(CONFIG_DICT['fictrac_dir'])
