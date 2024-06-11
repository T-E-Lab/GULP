# test_behavior.py

from pathlib import Path
import json

from unityvr.preproc import logproc as lp
from gulp2p.preproc import behavior

import pytest

BROKEN_JSON = (
"""[
{
    "timeSecs": 0.0,
    "frame": 0.0,
    "timeSecsAfterSplash": 0.0,
    "frameAfterSplash": 0.0,
    "sessionParameters": [
        "timeoutSecs: 360",
        "backgroundCylinderTexture: "
    ]
},
{
    "timeSecs": 0.019999999552965165,
    "frame": 2.0,
    "timeSecsAfterSplash": 0.019999999552965165,
    "frameAfterSplash": 1.0,
    "numberOfEntriesWritten": 32
""")

JSON_FILENAME = "Log_2023-12-12_15-20-38.json"

def test_heal_json_file(tmp_path):
    # Create json file
    file_path = Path(tmp_path, JSON_FILENAME)
    file_path.write_text(BROKEN_JSON)

    # Heal file
    behavior.heal_json_file(file_path)
    print(file_path.read_text())

    # Load file
    with open(file_path, 'r') as file:
        data = json.load(file)

