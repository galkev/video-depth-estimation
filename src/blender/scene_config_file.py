import json


scene_config_file = None


def load(filename):
    global scene_config_file

    if scene_config_file is None:
        with open(filename, "r") as f:
            scene_config_file = json.load(f)
        print("Loaded", filename)
    else:
        print("Scene config already loaded")
