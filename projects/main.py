from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix

dataset = load_iris()
X = dataset.data
y = dataset.target
x_train, x_test, y_train, y_test = \
    train_test_split(X, y,
                     test_size=0.3,
                     random_state=42,
                     stratify=y)
scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
svc_rbf = SVC(kernel='rbf', C=1, gamma=0.1)
svc_rbf.fit(x_train, y_train)
y_pred = svc_rbf.predict(x_test)
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))