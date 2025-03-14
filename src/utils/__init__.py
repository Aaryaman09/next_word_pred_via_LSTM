import json
from typing import Dict

def read_json(path)->Dict:
    with open(path, "r") as file:
        data = json.load(file)  # Load JSON into a Python dictionary

    return data
