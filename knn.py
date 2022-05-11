import numpy as np
import pandas as pd

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

    @value.vector(self, v):
        self._value = v

class KNN:
    def __init__(self, training_df, testing_df):
        self.init_data(training_df, testing_df)

    def init_data(self, training, testing):
        self._training_data = self.create_test_matrix(training)

        # List of example objects
        self._testing_data = [create_test_matrix(r) for r in testing]


    # Returns a numpy matrix given pd.dataframe
    # Training data will have the last column as a class
    def create_training_matrix(self, dataframe):

        # ":" for all rows, then provide a list of indexes for columns
        # Here we want to take all columns except the last one and transform it into
        # a numpy matrix
        df_matix = df.iloc[:,[i for i in range(len(df.columns)-1)]].to_numpy()    
        return return df_matrix

    # Creates matrix for test (incoming) data
    def create_test_matrix(self, df_row):
        eg = Example()
        eg.vector = df_row.to_numpy()
        return eg

    def classify(self):
        for eg in self._testing_data:
            # Example's vector is broadcasted to the testing_data matrix
            difference = self._testing_data - eg.vector

            sq_difference = np.square(difference)
            attribute_sum = sq_difference.sum()
            # Square root each value to get the distance vector
            distance_matrix = np.sqrt(attribute_sum)
            
            # Match the class with the corresponding distance before we sort
            # the distances.
            # [:, [-1]] means [all rows, last column]
            dist_class_list = [(d, c) for d,c in zip(distance_matrix, df.iloc[:,[-1]]]

            # Sort the tuples by the distance. Ascending order by default
            dist_class_list.sort(key=lambda x:x[0])
            
            # Keep count of yes'es and no's of the first k items
            class_dict = {"yes":0, "no":0}
            for j in range(k):
                # 1st index of the j'th item is its class; increment that class
                class_dict[dist_class_list[j][1]]

            # Since whenever there's a tie, we choose "yes", therefore we're
            # only choosing "no" whenever "no" count is strictly greater
            eg.classification = "no" if class_dict["no"] > class_dict["yes"] else "yes"

            

            

def main():
    df = pd.read_csv("pima.csv", header=None)

    

    for eg in test_data:
        take_away_matrix = df_matix - eg
        sq_matrix = np.square(take_away_matrix)
        summed_matrix = sq_matrix.sum()
        distance_matrix = np.sqrt(summed_matrix)
        tp_li = [(d, i) for d,i in zip(distance_matrix, df.iloc[:,[-1]])]
        tp_li.sort(ley=lambda x:x[0])
        class_dict = {"yes":0, "no":0}
        for j in range(k):
            class_dict[tp_li[j][1]] += 1

        # Takes care of a tie when we use else statement
        classification = 
        yield classification


if __name__ == "__main__":
    main()
