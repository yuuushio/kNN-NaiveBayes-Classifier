"""
KNN classification algorithm implementation in 40 sloc, where the caculations are
comprised of mostly vectorized operations from numpy and python, therefore; faster vs. for-loops.
This is a less-readable version of the implemenatation, which I did for FUN after implemenating the 
proper version (`knn.py`), to see how much I could compact/reduce it.
"""

import numpy as np
import pandas as pd
import sys


class Example:

    def __init__(self, v):
        self.value = v


class KNN:

    def __init__(self, training_df, testing_df, k):
        self._training_df = training_df
        self._testing_df = testing_df
        self.k = k
        self._training_data = training_df.iloc[:, [
            i for i in range(len(training_df.columns) - 1)
        ]].to_numpy()
        self._testing_data = [Example(r) for r in testing_df.to_numpy()]
        self._init_data(training_df, testing_df)

    def classify(self):
        for eg in self._testing_data:

            dist_class_list = [(d, c) for d, c in zip(
                np.sqrt(
                    np.square(self._training_data - eg.vector).sum(axis=1)),
                [v[0] for v in self._training_df.iloc[:, [-1]].values])
            ].sort(key=lambda x: x[0])

            class_dict = {"yes": 0, "no": 0}
            for j in range(self.k):
                class_dict[dist_class_list[j][1]] += 1

            yield "no" if class_dict["no"] > class_dict["yes"] else "yes"


def main():
    knn = KNN(pd.read_csv(sys.argv[1], header=None),
              pd.read_csv(sys.argv[2], header=None), int(sys.argv[3]))
    for c in knn.classify():
        print(c)


if __name__ == "__main__":
    main()
