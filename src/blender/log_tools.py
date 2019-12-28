import logging
import os

fh, sh = None, None


def create_logger(file):
    global fh, sh

    logger = logging.getLogger()

    for handler in [fh, sh]:
        if handler is not None:
            #handler.stream.close()
            logger.removeHandler(handler)

    logger.setLevel(logging.INFO)

    log_fmt = logging.Formatter("[%(asctime)s] %(message)s", "%Y-%m-%d %H:%M:%S")

    directory = os.path.dirname(file)
    if not os.path.exists(directory):
        os.makedirs(directory)

    fh = logging.FileHandler(file, mode="w")
    fh.setLevel(logging.INFO)
    fh.setFormatter(log_fmt)

    sh = logging.StreamHandler()
    sh.setLevel(logging.INFO)
    sh.setFormatter(log_fmt)

    logger.addHandler(fh)
    logger.addHandler(sh)

    return logger
