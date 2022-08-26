def key2attr(key):
    return f'__mlucy_utils_{key}__'

def with_tag(key, x):
    set_tag(x, key)
    return x

def set_tag(x, key, val=True):
    setattr(x, key2attr(key), val)

def get_tag(x, key, default_val=False):
    return getattr(x, key2attr(key), default_val)
