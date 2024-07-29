# config.py

from pathlib import Path
import strictyaml
from strictyaml import Map, Optional, Str, Float, Bool, Seq
import logging

logger = logging.getLogger(__name__)

DEFAULT_CONFIG_FILE = Path(Path(__file__).parent, "..", "settings.yaml").resolve()

def load_user_data_table(file_path):
    # Given the path to the user_data_table
    # Return the data in the table

    schema = Seq(Map({"member": Str(),
                      "data_dirs":Seq(Str()) | strictyaml.EmptyList()}))
    yaml_text = Path(file_path).read_text()
    user_data_table = strictyaml.load(yaml_text, schema).data

    return user_data_table


def load_config(config_file=None):
    if config_file is None:
        config_file = DEFAULT_CONFIG_FILE
    schema = Map({"user_data_table_path": Str(),
                  "main_pickle_dir": Str(),
                  "bhv_data_dir": Str(),
                  "fictrac_dir": Str(),
                  "napari_gamma": Float(),
                  "napari_shape_opacity": Float(),
                  "napari_colormap": Str(),
                  "show_full_stack": Bool()})

    yaml_text = config_file.read_text()
    config_dict = strictyaml.load(yaml_text, schema).data

    # Add user data table info
    user_data_table = load_user_data_table(config_dict['user_data_table_path'])
    config_dict['user_data_table'] = user_data_table

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
