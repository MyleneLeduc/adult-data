# -*- coding: utf-8 -*-
"""
Created on Sun Nov  1 16:35:34 2020

@author: Mylène
"""
import pandas as pd
import os

def read_data_file(file_name):
    file_path = os.path.join('data', file_name)
    columns = [
        'age', 
        'workclass',
        'fnlwgt',
        'education',
        'education-num',
        'marital-status',
        'occupation',
        'relationship',
        'race',
        'sex',
        'capital-gain',
        'capital-loss',
        'hours-per-week',
        'native-country',
        'salary'
    ]
    df = pd.read_csv(file_path, names=columns, header=0)
    return df