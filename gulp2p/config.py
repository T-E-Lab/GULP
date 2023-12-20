# config.py

from pathlib import Path
import strictyaml

DEFAULT_CONFIG_FILE = Path(Path(__file__).parent, "..", "..", "settings.yaml").resolve()

def load_config(config_file=None):
    if config_file is None:
        config_file = DEFAULT_CONFIG_FILE
    schema = strictyaml.Map({"proc_dat_folder": strictyaml.Str(),
                             "napari_gamma": strictyaml.Float(),
                             "napari_shape_opacity": strictyaml.Float(),
                             "napari_colormap": strictyaml.Str(),
                             "show_full_stack": strictyaml.Bool()})

    yaml_text = config_file.read_text()
    config_dict = strictyaml.load(yaml_text, schema).data
    return config_dict
