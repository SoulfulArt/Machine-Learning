from sklearn import datasets
from sklearn import linear_model
import matplotlib.pyplot as plt

digits = datasets.load_digits()

X_digits = digits.data
y_digits = digits.target

n_points = len(X_digits)

X_train = X_digits[:int(0.7*n_points)]
y_train = y_digits[:int(0.7*n_points)]
X_test = X_digits[int(0.7*n_points):]
y_test = y_digits[int(0.7*n_points):]

#model representation
logistic = linear_model.LogisticRegression()
logistic.fit(X_train, y_train)

print("LR accuracy", logistic.score(X_test, y_test))

plt.figure(figsize=(3, 3))
plt.imshow(digits.images[10], cmap = plt.cm.gray_r)