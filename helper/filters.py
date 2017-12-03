import numpy as np

# other filter functions can derive from this base class
class BasicFilterFunc():
    def run(self, col):
        res = []
        for r in col:
            d = self.prep(r)
            res.append([d])
        output = np.array(res)
        return False, output

    def prep(self, d):
        return d

class CategoricalFilterFunc(BasicFilterFunc):
    def __init__(self):
        self.dict = {}

    def prep(self, d):
        if d not in self.dict:
            self.dict[d] = len(self.dict) + 1
        return self.dict[d]

class SkipFilterFunc():
    def run(self, col):
        return True, None
