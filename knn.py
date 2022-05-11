import numpy as np
import pandas as pd

def main():
    df = pd.read_csv("pima.csv", header=None)

    # ":" for all rows, then provide a list of indexes for columns
    # Here we want to take all columns except the last one and transform it into
    # a numpy matrix
    df_matix = df.iloc[:,[i for i in range(len(df.columns)-1)]].to_numpy()

if __name__ == "__main__":
    main()
