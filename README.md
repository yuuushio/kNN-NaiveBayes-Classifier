# kNN & Naive Bayes Classifier
Implementation of K-Nearest Neighbor and Naive Bayes algorithms to assess their performance on a real dataset and classify incoming data. 

The k-NN algorithm was implemented for any `k` value and employed Euclidean distance as the distance measure. Since the analysis involved numeric attributes, the probability density function was used to evaluate probabilities for the Naive Bayes algorithm.

The implementation incorporated a data-science oriented approach, with only one for-loop per algorithm and most of the calculations executed through pandas to read, manipulate, and filter the data. Following this, the data was transformed into matrices, leveraging numpy broadcasting to perform efficient vector operations.

To ensure consistency and accuracy, the original dataset was normalized using Weka.

*Currently only works for two classes (binary classification), as it's based on a pre-defined specification for analyzing/classfying Pima Indian Diabetes dataset.*

### Usage
```bash
python main.py <training-set> <testing-set> <algorithm>
```

E.g., running the kNN classifier:

```bash
python main.py pima.csv test.csv 3NN
```

The *3* can be replaced with any integer to specify the number of neighbours.

E.g., running the Naive Bayes classifier:

```bash
python main.py pima.csv test.csv nb
```
