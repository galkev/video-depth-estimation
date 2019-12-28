import os
import sys

from torch import optim
import data
import net
import trainer

module_root = sys.path[0]
project_home = os.path.join(module_root, *[".."]*2)

base_path = None

paths = {
    "module": os.path.join("code", "mt-project"),
    "scripts": os.path.join("code", "mt-project", "scripts"),
    "datasets": "datasets",
    "models": "models",
    "logs": "logs",
    "pretrained": "pretrained",
    "crawl": "crawl",
    "fonts": "fonts",
    "export": "export"
}


def set_base_path(bp):
    global base_path
    base_path = bp


def use_slurm_system():
    path = os.path.join("/", "storage", "slurm", "galim")
    if os.path.isdir(path):
        print("Use slurm paths")
        set_base_path(path)
        return True
    else:
        return False


def proj_dir(*dirs):
    if len(dirs) > 1:
        subpath = os.path.join(paths[dirs[0]], *dirs[1:])
    else:
        subpath = paths[dirs[0]]

    path = os.path.join(project_home, subpath)

    if base_path is not None and not subpath.startswith("logs"):
        path = os.path.join(base_path, os.path.relpath(path, os.path.expanduser("~")))

    return path


def set_proj_home(new_home):
    global project_home
    project_home = new_home


def get_class(module_name, class_name):
    return getattr(sys.modules[module_name], class_name)


def create_component(comp_type, name, **kwargs):
    return get_class(comp_type, name)(**kwargs)
