def float_gt(a, b):
    if 0.0001 >= abs(a -b):
        return False
    return a > b
    