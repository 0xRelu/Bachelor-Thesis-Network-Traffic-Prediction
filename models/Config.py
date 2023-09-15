

class Config:
    def __init__(self, config_dict):
        self._config_dict = config_dict

    def __getattr__(self, name):
        if name in self._config_dict:
            return self._config_dict[name]
        else:
            raise AttributeError(f"'Config' object has no attribute '{name}'")


