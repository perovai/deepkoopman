import addict


class Opts(addict.Dict):
    def __missing__(self, key):
        if key == "_repr_mimebundle_":
            return True
        raise KeyError(key)
