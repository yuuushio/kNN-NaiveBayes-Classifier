import numpy as np
import pandas as pd
import sys

class Example:
    def __init__(self):
        pass

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

class KNN:
    def __init__(self, training_df, testing_df, k):
        self._training_df = training_df
        self._testing_df = testing_df
        self.k = k
        self.init_data(training_df, testing_df)

    def init_data(self, training, testing):
        self._training_data = self.create_training_matrix(training)
        
        # List of example objects
        self._testing_data = [self.create_test_matrix(r) for r in testing.to_numpy()]


    # Returns a numpy matrix given pd.dataframe
    # Training data will have the last column as a class
    def create_training_matrix(self, dataframe):
        # ":" for all rows, then provide a list of indexes for columns
        # Here we want to take all columns except the last one and transform it into
        # a numpy matrix
        df_matrix = dataframe.iloc[:,[i for i in range(len(dataframe.columns)-1)]].to_numpy()    
        return df_matrix

    # Creates matrix for test (incoming) data
    def create_test_matrix(self, df_row):
        eg = Example()
        eg.vector = df_row
        return eg

    def classify(self):
        for eg in self._testing_data:
            # Example's vector is broadcasted to the testing_data matrix
            difference = self._training_data - eg.vector

            sq_difference = np.square(difference)
            attribute_sum = sq_difference.sum(axis=1)
            # Square root each value to get the distance vector
            distance_matrix = np.sqrt(attribute_sum)
            
            # Match the class with the corresponding distance before we sort
            # the distances.
            # [:, [-1]] means [all rows, last column]
            #print(self._training_df.iloc[1:,[-1]].values[i][0])

            class_list = [v[0] for v in self._training_df.iloc[:,[-1]].values]

            dist_class_list = [(d, c) for d,c in zip(distance_matrix, class_list)]

            # Sort the tuples by the distance. Ascending order by default
            dist_class_list.sort(key=lambda x:x[0])
            # Keep count of yes'es and no's of the first k items
            class_dict = {"yes":0, "no":0}
            for j in range(self.k):
                print(dist_class_list[j])
                # 1st index of the j'th item is its class; increment that class
                class_dict[dist_class_list[j][1]] += 1

            # Since whenever there's a tie, we choose "yes", therefore we're
            # only choosing "no" whenever "no" count is strictly greater
            eg.classification = "no" if class_dict["no"] > class_dict["yes"] else "yes"

            yield eg.classification

            

def main():
    df_training = pd.read_csv(sys.argv[1], header=None)
    df_testing = pd.read_csv(sys.argv[2], header=None)
    k = int(sys.argv[3])

    knn = KNN(df_training, df_testing, k)

    for c in knn.classify():
        print(c)


if __name__ == "__main__":
    main()
