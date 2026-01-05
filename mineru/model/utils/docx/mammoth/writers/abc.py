from __future__ import absolute_import

import abc


class Writer(object):
    __metaclass__ = abc.ABCMeta
    
    @abc.abstractmethod
    def text(self, text):
        pass
    
    @abc.abstractmethod
    def start(self, name, attributes=None):
        pass

    @abc.abstractmethod
    def end(self, name):
        pass
    
    @abc.abstractmethod
    def self_closing(self, name, attributes=None):
        pass
    
    @abc.abstractmethod
    def append(self, html):
        pass
    
    @abc.abstractmethod
    def as_string(self):
        pass
