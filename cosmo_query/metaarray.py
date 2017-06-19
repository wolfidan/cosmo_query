#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 13 10:33:20 2017

@author: wolfensb
"""
import numpy as np
import collections

class MetaArray(np.ndarray):
    """
    CLASS:
        metaarray = MetaArray(array)
    
    PURPOSE:
         Creates an array which possesses one additional variable, a dict 
         called 'metadata' and to which any new variable can be added 
         
    INPUTS:
        array : a numpy array
    
    OUTPUTS:
        metaarray : an array with an additional variable called metadata 
            (accessible through metaarray.metadata). Note that you can 
            add any new variable to this metaarray
    """    

    def __new__(cls, array, dtype=None, order=None, **kwargs):
        obj = np.asarray(array, dtype=dtype, order=order).view(cls)                                 
        obj.metadata = kwargs
        return obj

    def __array_finalize__(self, obj):
        if obj is None: return
        self.metadata = getattr(obj, 'metadata', None)
        
class MetaDict(collections.MutableMapping):
    """A dictionary that applies an arbitrary key-altering
       function before accessing the keys"""

    def __init__(self, *args, **kwargs):
        self.store = dict()
        self.update(dict(*args, **kwargs))  # use the free update to set keys

    def __getitem__(self, key):
        return self.store[self.__keytransform__(key)]

    def __setitem__(self, key, value):
        self.store[self.__keytransform__(key)] = value

    def __delitem__(self, key):
        del self.store[self.__keytransform__(key)]

    def __iter__(self):
        return iter(self.store)

    def __len__(self):
        return len(self.store)

    def __keytransform__(self, key):
        return key
