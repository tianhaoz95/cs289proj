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

class SkipFilterFunc():
    def run(self, col):
        return True, None
