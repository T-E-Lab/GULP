# test_behavior.py

from pathlib import Path
from gulp2p import config
import pytest


USER_DATA_TABLE_PATH = Path(r"Z:\GULP\user_data_table.yaml")

def test_load_user_data_table():
    user_data_table = config.load_user_data_table(USER_DATA_TABLE_PATH)
    print("\nUser data table:")
    for index, item in enumerate(user_data_table):
        print(f"item #{index}")
        print(f"member: {item['member']}")
        print(f"data_dirs: {item['data_dirs']}")
        print()

    assert user_data_table is not None


def test_load_config():
    config_dict = config.load_config()
    print("\nConfig Dict:")
    for k,v in config_dict.items():
        print(f"{k}: {v}")
    assert config_dict is not None

# def test_config():
#     pass
