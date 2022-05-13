import numpy as np
import pandas as pd
import sys


class Example:

    def __init__(self, v):
        self._value = v

    @property
    def classification(self):
        return self._classification

    @classification.setter
    def classification(self, c):
        self._classification = c

    @property
    def vector(self):
        return self._value

    @vector.setter
    def vector(self, v):
        self._value = v


class Classifier:

    def __init__(self, training_df, testing_df, k):
        self._training_df = training_df
        self._testing_df = testing_df
        self.k = k
        self._init_data(training_df, testing_df)

    def _init_data(self, training, testing):
        self._training_data = self._create_training_matrix(training)

        # List of example objects
        self._testing_data = [
            self._create_test_matrix(r) for r in testing.to_numpy()
        ]

    # Returns a numpy matrix given pd.dataframe
    # Training data will have the last column as a class
    def _create_training_matrix(self, dataframe):
        # ":" for all rows, then provide a list of indexes for columns
        # Here we want to take all columns except the last one and transform it into
        # a numpy matrix
        df_matrix = dataframe.iloc[:, [
            i for i in range(len(dataframe.columns) - 1)
        ]].to_numpy()
        return df_matrix

    # Creates matrix for test (incoming) data
    def _create_test_matrix(self, df_row):
        eg = Example(df_row)
        return eg

    def knn(self):
        for eg in self._testing_data:
            # Example's vector is broadcasted to the testing_data matrix
            sq_difference = np.square(self._training_data - eg.vector)

            # Square root each value to get the distance vector
            distance_matrix = np.sqrt(sq_difference.sum(axis=1))

            # Match the class with the corresponding distance before we sort
            # the distances.
            # [:, [-1]] means [all rows, last column]
            class_list = [v[0] for v in self._training_df.iloc[:, [-1]].values]

            dist_class_list = [(d, c)
                               for d, c in zip(distance_matrix, class_list)]

            # Sort the tuples by the distance. Ascending order by default
            dist_class_list.sort(key=lambda x: x[0])

            # Keep count of yes'es and no's for first k items
            class_dict = {"yes": 0, "no": 0}
            for j in range(self.k):
                # 1st index of the j'th item is its class; increment count of that class
                class_dict[dist_class_list[j][1]] += 1

            # Since whenever there's a tie, we choose "yes", therefore we're
            # only choosing "no" whenever "no" count is strictly greater
            eg.classification = "no" if class_dict["no"] > class_dict["yes"] else "yes"

            yield eg.classification

    def naive_bayes(self):
        # Get the string value of the different types of classes
        classes = [c[0] for c in self._training_df.iloc[:,[-1]].drop_duplicates().to_numpy()]
        # Array of indices to get all columns except the last
        indexes = [i for i in range(len(self._training_df.columns)-1)]

        # For each class, create their corressponding dataframe
        df_li = [self._training_df[self._training_df.iloc[:,-1] == c].iloc[:,indexes] for c in classes]

        yield 0


def main():
    df_training = pd.read_csv(sys.argv[1], header=None)
    df_testing = pd.read_csv(sys.argv[2], header=None)
    k = int(sys.argv[3])

    knn = Classifier(df_training, df_testing, k)

    for c in knn.naive_bayes():
        print(c)


if __name__ == "__main__":
    main()
