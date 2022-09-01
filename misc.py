from datetime import datetime
import os
import logging
log = logging.getLogger(__name__)

def now_ts():
    return datetime.now().strftime('%Y-%m-%dT%H-%M-%S.%fZ')

def now_unix():
    return datetime.now().strftime('%s.%f')


def _env_var(name, default):
    val = os.environ.get(name, None)
    if val is None:
        return default
    if isinstance(default, bool):
        return bool(int(val))
    else:
        return type(default)(val)

default_sentinel = 'f0829b62-84b4-408d-90ef-b3fdca567232'
def env_var(name, default=default_sentinel, do_log=True):
    res = _env_var(name, default)
    if res == default_sentinel:
        raise RuntimeError(f'Required env_var {name} not set.')
    if do_log:
        log.info(f'CONFIG: Using {name}=`{repr(res)}`.')
    return res
