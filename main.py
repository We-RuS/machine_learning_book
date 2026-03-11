from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from log import LogisticRegressionGD

iris = load_iris()
X = iris.data[:100, :2]
y = iris.target[:100]

X_train, X_test, y_train, y_test = train_test_split(X , y, random_state=42,
                                                    stratify=y, test_size=0.3)
model = LogisticRegressionGD(learning_rate=0.1, n_iter=10)
model.fit(X_train, y_train)