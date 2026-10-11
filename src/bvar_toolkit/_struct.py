"""Struct - a dict whose keys can also be read as attributes, standing in for a MATLAB
struct. Functions that take a struct read it by key, so a plain dict works as input."""


class Struct(dict):
    """Dictionary with attribute access: s.beta0 and s['beta0'] are the same entry."""

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name) from None

    def __setattr__(self, name, value):
        self[name] = value

    def __delattr__(self, name):
        try:
            del self[name]
        except KeyError:
            raise AttributeError(name) from None

    def __dir__(self):
        return list(self.keys())

    def __repr__(self):
        if not self:
            return 'Struct()'
        width = max(len(k) for k in self)
        rows = []
        for k, v in self.items():
            shape = getattr(v, 'shape', None)
            desc = f'array of shape {shape}' if shape is not None and shape != () else repr(v)
            rows.append(f'  {k.rjust(width)}: {desc}')
        return 'Struct(\n' + '\n'.join(rows) + '\n)'
