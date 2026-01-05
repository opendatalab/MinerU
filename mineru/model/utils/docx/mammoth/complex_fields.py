class unknown(object):
    pass


class Begin:
    def __init__(self, *, fld_char):
        self.fld_char = fld_char


def begin(*, fld_char):
    return Begin(fld_char=fld_char)


class Hyperlink(object):
    def __init__(self, kwargs):
        self.kwargs = kwargs


def hyperlink(kwargs):
    return Hyperlink(kwargs=kwargs)


class Checkbox:
    def __init__(self, *, checked):
        self.checked = checked


def checkbox(*, checked):
    return Checkbox(checked=checked)
