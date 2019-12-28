import logging
import os

import importlib
importlib.reload(logging)

fh, sh = None, None

"""
def create_logger(file=None):
    global fh, sh

    logger_inst = logging.getLogger()

    for handler in [fh, sh]:
        if handler is not None:
            #handler.stream.close()
            logger_inst.removeHandler(handler)

    logger_inst.setLevel(logging.NOTSET)

    log_fmt = logging.Formatter("[%(asctime)s] [%(filename)s:%(lineno)d] %(levelname)s: %(message)s")

    if file is not None:
        directory = os.path.dirname(file)
        if not os.path.exists(directory):
            os.makedirs(directory)

        fh = logging.FileHandler(file, mode="w")
        fh.setLevel(logging.INFO)
        fh.setFormatter(log_fmt)

        logger_inst.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setLevel(logging.NOTSET)
    sh.setFormatter(log_fmt)

    logger_inst.addHandler(sh)

    return logger_inst


logger = create_logger()
"""