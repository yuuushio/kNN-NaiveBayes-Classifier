# kNN & Naive Bayes Classifier
Implementation of K-Nearest Neighbour and Naive Bayes algorithms to evaluate them on a real dataset and classify incoming examples. The k-NN algorithm is implemented for any `k` value and uses Euclidean distance as the distance measure. And since we're working with numeric attributes, the probabilty density function is used to evaluate the probabilities for Naive Bayes algorithm.

The implementation only contains 1 for-loop for *each* algorithm, with rest of the calculations done in a data-science oriented manner: pandas dataframe, to read and filter data; whereafter the data is transformed into matricies - making extensive use of numpy broadcasting to perform efficient, vector operations. 

The original dataset was normalized using Weka.

*Currently only works for two classes, as it's based on a pre-defined specification for analyzing/classfying Pima Indian Diabetes dataset.*

### Usage
```bash
python classifier.py <training-set> <testing-set> <algorithm>
```

E.g., running the kNN classifier:

```bash
python classifier.py pima.csv test.csv 3NN
```

The *3* can be replaced with any integer to specify the number of neighbours.

E.g., running the Naive Bayes classifier:

```bash
python classifier.py pima.csv test.csv nb
```

### TODO
- Parallelize calculations of each class and then join the results.
- Extend to work on arbitary number of classes.
