class Config:
    def __init__(self, obj, presets=[], required=[], preset_map={}, defaults={}):
        self._obj = obj
        self._defaults = defaults

        for preset in presets:
            if preset not in preset_map:
                raise RuntimeError(
                    f'Unrecognized preset `{preset}`.  Legal: {set(preset_map)}')
            for k, v in preset_map[preset].items():
                if k in self._obj:
                    raise RuntimeError(
                        f'Conflict between user setting `{k}={self._obj[k]}` and '+
                        f'preset `{preset}.{k}={v}`.')
                else:
                    self._obj[k] = v

        legal_configs = set(required).union(set(defaults))
        for k in self._obj:
            if k not in legal_configs:
                raise RuntimeError(
                    f'Unrecognized config `{k}`.  Legal: {legal_configs}')
        for k in required:
            if k not in self._obj:
                raise RuntimeError(f'Missing required config `{k}`.')

    def __getattr__(self, field):
        if field in ['_obj', '_defaults']:
            raise AttributeError(f'bad config field {field}')
        if field in self._obj:
            return self._obj[field]
        elif field in self._defaults:
            return self._defaults[field]
        else:
            raise AttributeError(f'no config field {field}')
