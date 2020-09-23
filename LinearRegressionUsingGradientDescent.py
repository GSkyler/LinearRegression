import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (12.0, 9.0)

data = pd.read_csv('Test.csv')
X = data.iloc[:, 0]
Y = data.iloc[:, 1]

m = 0
b = 0

L = 0.0001
epochs = 1000

n = float(len(X))

for i in range(epochs):
    Y_pred = m * X + b
    D_m = (-2 / n) * sum(X * (Y - Y_pred))
    D_b = (-2 / n) * sum(Y - Y_pred)
    m = m - L * D_m
    b = b - L * D_b

print(m, b)

Y_pred = m*X + b

plt.scatter(X, Y)
plt.plot([min(X), max(X)], [min(Y_pred), max(Y_pred)], color='red') # predicted
plt.show()
