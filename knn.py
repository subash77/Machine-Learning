import numpy as np
from collections import Counter

def Euclidean_distance(x1,x2):
    return np.sqrt(np.sum(x1-x2)**2)

class KNN:
    def __init__(self, k=3):
        self.k = k

    def fit(self, X, y):
        self.X = X_train
        self.y = y_train

    def predict(self, x):
        y_pred = [self._predict(x) for x in X]
        return np.array(y_pred)

    def _predict(self, x):
        distances = [Euclidean_distance(x, x_train) for x_train in self.X_train]
        k_idx = np.argsort(distances)[: self.k]
        k_nearest_labels = [self.y_train[i] for i in k_idx]
        most_common = Counter(k_nearest_labels).most_common()
        return most_common[0][0]


if __name__ == "__main__":
    from matplotlib.colors import ListedColormap
    from sklearn import datasets
    from sklearn.model_selection import train_test_split

    cmap = ListedColormap(["#FF0000","#00FF00","0000FF"])

def accuracy_score(y_true, y_pred):
    accuracy = np.sum(y_true == y_pred) / len(y_true)
    return accuracy

    iris = datasets.load_iris()
    X,y = iris.data, iris.target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

    k=3

    clf = KNN(k=k)
    clf.fit(X_train, y_train)
    predictions = clf.predict(X_test)

    print("KNN classification Accuracy", accuracy(y_test, predictions))



    


