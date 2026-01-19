import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

df = pd.read_csv("simple_data.csv")
X = df.drop("Target", axis=1)
y = df["Target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

w = np.zeros(X_train.shape[1] + 1)
lr = 0.01

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

for _ in range(500):
    z = w[0] + X_train @ w[1:]
    y_hat = sigmoid(z)
    w[0] -= lr * (y_hat - y_train).mean()
    w[1:] -= lr * (X_train.T @ (y_hat - y_train)) / len(y_train)

gd_pred = (sigmoid(w[0] + X_test @ w[1:]) >= 0.5).astype(int)

lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
lr_pred = (lr_model.predict(X_test) >= 0.5).astype(int)

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)
knn_pred = knn.predict(X_test)

dt = DecisionTreeClassifier()
dt.fit(X_train, y_train)
dt_pred = dt.predict(X_test)

print("Gradient Descent :", accuracy_score(y_test, gd_pred))
print("Linear Regression:", accuracy_score(y_test, lr_pred))
print("KNN             :", accuracy_score(y_test, knn_pred))
print("Decision Tree   :", accuracy_score(y_test, dt_pred))

compare_df = pd.DataFrame({
    "Actual": y_test.values[:5],
    "GD": gd_pred[:5],
    "LinearReg": lr_pred[:5],
    "KNN": knn_pred[:5],
    "DecisionTree": dt_pred[:5]
})

print(compare_df)
