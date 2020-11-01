# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import pandas as pd
import csv
import matplotlib.pyplot as plt
from csv import reader


table = pd.read_csv('surveys.csv')

### Exploring Data type ###

'''
print(type(table))
print(table.dtypes)

print(table.columns)

print(pd.unique(table['species_id']))
'''

### Plot Data ###

weight = table['weight']
'''
plt.hist(weight, bins=50)
plt.xlabel('Weight')
plt.ylabel('Number of animals')
plt.title('Exemple d\'histogramme poid des animaux')
'''

hindfoot_length = table['hindfoot_length']
plt.scatter(weight,hindfoot_length)
plt.xlabel('Weight')
plt.ylabel('hindfoot_length')
plt.show()

'''
# iterating the columns 
for col in table.columns: 
    print(col) 
'''

