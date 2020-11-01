# -*- coding: utf-8 -*-
"""
Created on Sun Nov  1 16:46:12 2020

@author: Mylène
"""
import numpy as np

def ReduceSkewness(data_set,column):
    data_set_transformed = np.log1p(data_set[['capital-gain','capital-loss']])
    data_set[['capital-gain','capital-loss']] = data_set_transformed
    return data_set