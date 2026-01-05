import sys


def flatten(values):
    return flat_map(lambda x: x, values)


def unique(values):
    output = []
    seen = set()
    for value in values:
        if value not in seen:
            seen.add(value)
            output.append(value)
    return output


def flat_map(func, values):
    return [
        element
        for value in values
        for element in func(value)
    ]


def find_index(predicate, values):
    for index, value in enumerate(values):
        if predicate(value):
            return index


if sys.version_info[0] == 2:
    map = map
    filter = filter
else:
    import builtins
    def map(*args, **kwargs):
        return list(builtins.map(*args, **kwargs))
    def filter(*args, **kwargs):
        return list(builtins.filter(*args, **kwargs))
