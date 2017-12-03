import numpy as np

# other filter functions can derive from this base class
class BasicFilterFunc():
    def run(self, col):
        res = []
        params = self.preprocess(col)
        for r in col:
            d = self.prep(r, params)
            res.append([d])
        output = np.array(res)
        return False, output

    def preprocess(self, col):
        return None

    def prep(self, d, params):
        return d

class CategoricalFilterFunc(BasicFilterFunc):
    def __init__(self):
        self.dict = {}

    def prep(self, d, params):
        if d not in self.dict:
            self.dict[d] = len(self.dict) + 1
        return self.dict[d]

class ShrinkFilterFunc(BasicFilterFunc):
    def preprocess(self, col):
        max_elt = -float("inf")
        for r in col:
            max_elt = max(max_elt, abs(r))
        return {"max_elt": max_elt}

    def prep(self, d, params):
        max_elt = params["max_elt"]
        return d / max_elt

class SkipFilterFunc():
    def run(self, col):
        return True, None
