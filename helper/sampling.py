from numpy.random import random
import numpy as np
import math

def sample_train(x, y, size=50):
    res_x = []
    res_y = []
    limit = len(x)
    while len(res_x) < size:
        idx = math.floor(random() * limit)
        elt_x = x[idx]
        elt_y = y[idx]
        res_x.append(elt_x)
        res_y.append(elt_y)
    output_x = np.array(res_x)
    output_y = np.array(res_y)
    return output_x, output_y

def partition_data(data):
    raise ValueError("not implemented")
