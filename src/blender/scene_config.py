import json
import os


class SceneConfig(object):
    filename = os.path.expanduser("~/Documents/master-thesis/scenes/dining_room/config{:04}.json")

    def __init__(self):
        self.config = None

    def _apply_rec(self, obj, values):
        for k, v in values.items():
            if isinstance(v, dict):
                self._apply_rec(getattr(obj, k), v)
            else:
                setattr(obj, k, v)

    def _get_rec(self, obj, values):
        import mathutils

        for k, v in values.items():
            if isinstance(v, dict):
                self._get_rec(getattr(obj, k), v)
            else:
                obj_val = getattr(obj, k)
                if type(obj_val) == mathutils.Vector or type(obj_val) == mathutils.Euler:
                    obj_val = list(obj_val)

                values[k] = obj_val

    def _apply(self, obj, config):
        for k, v in config.items():
            setattr(obj, k, v)

    def set(self, config):
        import bpy

        self.config = config

        for k, v in config.items():
            self._apply_rec(bpy.data.objects[k], v)

    def get(self):
        return self.config

    def exists(self, i):
        return os.path.isfile(self.filename.format(i))

    def load(self, i):
        with open(self.filename.format(i), "r") as f:
            self.set(json.load(f))

    def update(self):
        import bpy

        bpy.context.scene.frame_set(0)

        for k, v in self.config.items():
            self._get_rec(bpy.data.objects[k], v)

        # check if correct
        self.set(self.config)

    def save(self):
        i = 0
        while os.path.exists(self.filename.format(i)):
            i += 1

        print("Config", i)

        with open(self.filename.format(i), "w") as f:
            json.dump(self.config, f, indent=4)

        return i
