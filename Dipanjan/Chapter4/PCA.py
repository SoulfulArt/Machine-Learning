from sklearn.datasets import load_breast_cancer
from sklearn.feature_selection import chi2, SelectKBest
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from pandas import DataFrame
from numpy import array
from numpy import linalg
from numpy import round
from numpy import average
from sklearn.decomposition import PCA

bc_data = load_breast_cancer()
bc_features = DataFrame(bc_data.data, columns = bc_data.feature_names)
bc_classes = DataFrame(bc_data.target, columns = ['IsMalignant'])
bc_X = array(bc_features)
bc_Y = array(bc_classes).T[0]

#Reducing dimesion by using SVA

bc_XC = bc_X - bc_X.mean(axis = 0)

U, S, VT = linalg.svd(bc_XC)

PC = VT.T

PC3 = PC[:, 0:3]

bc_SVA = round(bc_XC.dot(PC3), 2)

pca = PCA(n_components = 3)

pca.fit(bc_X)

bc_pca = pca.transform(bc_X)

lr = LogisticRegression()

print (average(cross_val_score(lr, bc_pca, bc_Y, scoring = 'accuracy', cv = 5)))