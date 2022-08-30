import logging
import os

def log_setup():
    formatter = logging.Formatter(
        '{levelname[0]} {message:<60s} {name}',
        style='{')

    fh = logging.FileHandler('debug.log')
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)

    LOGLEVEL = os.environ.get('LOGLEVEL', 'INFO').upper()
    ch = logging.StreamHandler()
    ch.setLevel(LOGLEVEL)
    ch.setFormatter(formatter)

    logging.basicConfig(level=logging.DEBUG, handlers=[fh, ch])
    root_logger = logging.getLogger('')
    root_logger.addHandler(ch)
    root_logger.addHandler(fh)

    return root_logger

def get_logger(name):
    return logging.getLogger(name)
