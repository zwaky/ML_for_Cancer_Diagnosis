import numpy as np
import pandas as pd

file = pd.read_csv('data.csv', delimiter=',')

dicts = file.to_dict('records')

# print(dicts('id'))


