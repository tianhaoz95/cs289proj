import numpy as np
from .filters import *

class FilterWrapper():
    def __init__(self):
        self.filters = {}

    def add(self, filter_name, filter_func):
        self.filters[filter_name] = filter_func

    def run(self, data):
        size = -1
        for f in data:
            if size == -1:
                size = len(data[f])
            else:
                if len(data[f]) != size:
                    raise ValueError("Features have different sizes")
        res = np.zeros((size, 1))
        for f in data:
            filterFunc = self.filters[f] if f in self.filters else BasicFilterFunc()
            skip, col = filterFunc.run(data[f])
            if not skip:
                res = np.hstack((res, col))
        output = res[:,1:]
        return output
