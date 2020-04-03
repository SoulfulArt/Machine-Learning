from sklearn.datasets import load_breast_cancer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt
from model_evaluation_utils import display_confusion_matrix
from model_evaluation_utils import plot_model_roc_curve
from scipy.cluster.hierarchy import fcluster
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from numpy import round
from sklearn import metrics

data = load_breast_cancer()

X = data.data
y = data.target

#model representation. Clustering are non supervised models. We don't need to specify the labels (y, output) to create a model
"""
km = KMeans(n_clusters = 2)
km.fit(X)
labels = km.labels_
centers = km.cluster_centers_

pca = PCA (n_components = 2)
bc_pca = pca.fit_transform(X) #non supervised function

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8,4))
fig.suptitle('Breast cancer')
fig.subplots_adjust(top = 0.85, wspace = 0.5)
ax1.set_title('Actual Labels')
ax2.set_title('Clustered Labels')

for i in range (len(y)):
	
	if y[i] == 0:
		c1 = ax1.scatter(bc_pca[i, 0], bc_pca[i, 1], c='g', marker = '.')

	if y[i] == 1:
		c2 = ax1.scatter(bc_pca[i, 0], bc_pca[i, 1], c='r', marker = '.')

	if labels[i] == 1:
		c3 = ax2.scatter(bc_pca[i, 0], bc_pca[i, 1], c='g', marker = '.')

	if labels[i] == 0:
		c4 = ax2.scatter(bc_pca[i, 0], bc_pca[i, 1], c='r', marker = '.')

l1 = ax1.legend([c1, c2], ['0', '1'])
l2 = ax2.legend([c3, c4], ['0', '1'])

Z = linkage(X, 'ward')

max_dist = 10000
hc_labels = fcluster(Z, max_dist, criterion='distance')

plt.figure(figsize=(8,3))
plt.title('Hirarchical Clustering Dendrogram')
plt.xlabel('Data point')
plt.ylabel('Distance')
dendrogram(Z)
plt.axhline(y = 1000, c = 'k', 	ls = '--', lw = 0.5)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
fig.suptitle('Visualizing breast cancer clusters')
fig.subplots_adjust(top=0.85, wspace=0.5)
ax1.set_title('Actual Labels')
ax2.set_title('Hierarchical Clustered Labels')

for i in range(len(y)):
    if y[i] == 0:
        c1 = ax1.scatter(bc_pca[i,0], bc_pca[i,1],c='g', marker='.')
    if y[i] == 1:
        c2 = ax1.scatter(bc_pca[i,0], bc_pca[i,1],c='r', marker='.')
        
    if hc_labels[i] == 1:
        c3 = ax2.scatter(bc_pca[i,0], bc_pca[i,1],c='g', marker='.')
    if hc_labels[i] == 2:
        c4 = ax2.scatter(bc_pca[i,0], bc_pca[i,1],c='r', marker='.')

l1 = ax1.legend([c1, c2], ['0', '1'])
l2 = ax2.legend([c3, c4], ['1', '2'])
"""

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .3, random_state = 42)

linmodel = linear_model.LogisticRegression()
linmodel.fit(X_train, y_train)

y_pred = linmodel.predict(X_test)
cm_frame = display_confusion_matrix(true_labels = y_test, predicted_labels = y_pred, classes = [0, 1])

#true positive
TP = cm_frame.iloc[1,1]
TN = cm_frame.iloc[0,0]
FP = cm_frame.iloc[0,1]
FN = cm_frame.iloc[1,0]

#accuracy

acc = (TP + TN) / (TP + FP + TN + FN)

#precision it's related with the total correct positive predictions. It's used when we're more worried about finding positive predictions even if the total accuracy reduces. 

prec = TP / (TP + FP)

#recall is how good the model is to find all the positive possibilities, how complete the results are. From all the cases that the model should recognize, how much it really got, this is recall.

rec = TP / (TP + FN)

#F1 score is used when we want a balance between recall and precision

F1score = 2*prec*rec/(prec + rec)

print("Confusion Matrix")
print(cm_frame)

print("Confusion matrix performance indexes:")
print("Accuracy: ", acc)
print("Precision: ", prec)
print("Recall: ", rec)
print("F1 Score: ", F1score)

#print("Ploting ROC ")

#plot_model_roc_curve(clf = linmodel, features = X_test, true_labels = y_test)

km2 = KMeans(n_clusters = 2, random_state = 42).fit(X)
km2_labels = km2.labels_
km5 = KMeans(n_clusters = 5, random_state = 42).fit(X)
km5_labels = km5.labels_

km2_hcv = round(metrics.homogeneity_completeness_v_measure(y, km2_labels), 3)

km5_hcv = round(metrics.homogeneity_completeness_v_measure(y, km5_labels), 3)

print("HCV")
print(km2_hcv)
print(km5_hcv)

km2_silc = metrics.silhouette_score(X, km2_labels, metric = 'euclidean')
km5_silc = metrics.silhouette_score(X, km5_labels, metric = 'euclidean')

print("Silhouette")
print("Two clusters ", km2_silc)
print("Five clusters ", km5_silc)

km2_chs = metrics.calinski_harabaz_score(X, km2_labels)
km5_chs = metrics.calinski_harabaz_score(X, km5_labels)

print("Calinski Harabaz")
print("Two clusters ", km2_chs)
print("Five clusters ", km5_chs)