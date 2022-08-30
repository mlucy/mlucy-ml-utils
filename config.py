class Config:
    def __init__(self, obj, presets=[], required=[], preset_map={}, defaults={}):
        self.obj = obj
        self.defaults = defaults

        for preset in presets:
            if preset not in preset_map:
                raise RuntimeError(
                    f'Unrecognized preset `{preset}`.  Legal: {set(preset_map)}')
            for k, v in preset_map[preset].items():
                if k in self.obj:
                    raise RuntimeError(
                        f'Conflict between user setting `{k}={self.obj[k]}` and '+
                        f'preset `{preset}.{k}={v}`.')

        legal_configs = set(required).union(set(defaults))
        for k in self.obj:
            if k not in legal_configs:
                raise RuntimeError(
                    f'Unrecognized config `{k}`.  Legal: {legal_configs}')
        for k in required:
            if k not in self.obj:
                raise RuntimeError(f'Missing required config `{k}`.')

    def __getattr__(self, field):
        if field in self.obj:
            return self.obj[field]
        else:
            return self.defaults[field]
