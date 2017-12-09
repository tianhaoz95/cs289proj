import numpy as np
import math

# other filter functions can derive from this base class
class BasicFilterFunc():
    def run(self, col):
        res = []
        params = self.preprocess(col)
        for r in col:
            d = self.prep(r, params)
            res.append(d)
        output = np.array(res)
        return output

    def preprocess(self, col):
        return None

    def prep(self, d, params):
        return [abs(float(d))]

class CategoricalOneHotFilterFunc(BasicFilterFunc):
    def __init__(self, verbose=False):
        self.dict = {}
        self.verbose = verbose

    def preprocess(self, col):
        if self.verbose:
            print("verbose mode, printing ...")
        for r in col:
            if type(r) == 'str':
                r = r.strip()
            if r != 'NaN' and r != 'nan' and r not in self.dict:
                if self.verbose:
                    print(r)
                self.dict[r] = len(self.dict)

    def prep(self, d, params):
        category_cnt = len(self.dict)
        res = [0 for i in range(category_cnt)]
        if type(d) == 'str':
            d = d.strip()
        if d != 'NaN' and d != 'nan':
            res[self.dict[d]] = 1
        return res

class CategoricalFilterFunc(BasicFilterFunc):
    def __init__(self, verbose=False):
        self.dict = {}
        self.verbose = verbose

    def prep(self, d, params):
        if self.verbose:
            print("verbose mode, printing ...")
        if d not in self.dict:
            if self.verbose:
                print(d)
            self.dict[d] = len(self.dict) + 1
        return [self.dict[d]]

class ShrinkFilterFunc(BasicFilterFunc):
    def __init__(self, limit=1):
        self.val_limit = limit

    def preprocess(self, col):
        max_elt = -float("inf")
        for r in col:
            if type(r) != 'str' or r != "NaN":
                max_elt = max(max_elt, abs(float(r)))
        return {"max_elt": max_elt}

    def prep(self, d, params):
        res = 0
        max_elt = params["max_elt"]
        if type(d) != 'str' or d != "NaN":
            res = abs(float(d) / max_elt * self.val_limit)
        return [res]
