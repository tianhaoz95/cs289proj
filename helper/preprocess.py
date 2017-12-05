import numpy as np

class FilterWrapper():
    def __init__(self):
        self.filters = {}

    def add(self, filter_name, filter_func):
        if filter_name not in self.filters:
            self.filters[filter_name] = []
        self.filters[filter_name].append(filter_func)

    def run(self, data, label_name):
        size = -1
        for f in data:
            if size == -1:
                size = len(data[f])
            else:
                if len(data[f]) != size:
                    raise ValueError("Features have different sizes")
        res_x = np.zeros((size, 1))
        res_y = None
        for f in data:
            if f in self.filters:
                filter_funcs = self.filters[f]
                col = data[f]
                for filter_func in filter_funcs:
                    col = filter_func.run(col)
                print("sample after preprocessing feature ", f, ": ")
                print(col[0,:])
                if f == label_name:
                    res_y = col
                else:
                    res_x = np.hstack((res_x, col))
        output_x = res_x[:,1:]
        output_y = res_y
        return output_x, output_y
