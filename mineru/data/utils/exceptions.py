# Copyright (c) Opendatalab. All rights reserved.

class FileNotExisted(Exception):

    def __init__(self, path):
        self.path = path

    def __str__(self):
        return f'File {self.path} does not exist.'


class InvalidConfig(Exception):
    def __init__(self, msg):
        self.msg = msg

    def __str__(self):
        return f'Invalid config: {self.msg}'


class InvalidParams(Exception):
    def __init__(self, msg):
        self.msg = msg

    def __str__(self):
        return f'Invalid params: {self.msg}'


class EmptyData(Exception):
    def __init__(self, msg):
        self.msg = msg

    def __str__(self):
        return f'Empty data: {self.msg}'

class CUDA_NOT_AVAILABLE(Exception):
    def __init__(self, msg):
        self.msg = msg

    def __str__(self):
        return f'CUDA not available: {self.msg}'