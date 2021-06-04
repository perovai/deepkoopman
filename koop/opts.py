import addict


class Opts(addict.Dict):
    def __missing__(self, key):
        raise KeyError(key)
