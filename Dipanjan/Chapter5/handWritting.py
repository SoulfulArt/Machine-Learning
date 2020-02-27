from sklearn import datasets
import matplotlib.pyplot as plt

digits = datasets.load_digits()

plt.figure(figsize=(3, 3))
plt.imshow(digits.images[10], cmap = plt.cm.gray_r)
plt.show()