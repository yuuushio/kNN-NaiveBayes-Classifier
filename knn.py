import numpy as np
import pandas as pd

def main():
    df = pd.read_csv("pima.csv", header=None)

    # ":" for all rows, then provide a list of indexes for columns
    # Here we want to take all columns except the last one and transform it into
    # a numpy matrix
    df_matix = df.iloc[:,[i for i in range(len(df.columns)-1)]].to_numpy()

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
        classification = "no" if class_dict["no"] > class_dict["yes"] else "yes"
        yield classification


if __name__ == "__main__":
    main()
