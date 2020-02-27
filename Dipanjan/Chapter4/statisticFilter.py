from sklearn.datasets import load_breast_cancer
from sklearn.feature_selection import chi2, SelectKBest
from pandas import DataFrame
from numpy import array

bc_data = load_breast_cancer()
bc_features = DataFrame(bc_data.data, columns = bc_data.feature_names)
bc_classes = DataFrame(bc_data.target, columns = ['IsMalignant'])
bc_X = array(bc_features)
bc_Y = array(bc_classes).T[0]

#one way to decrease dimensions is by using chi2 SelectKBest

skb = SelectKBest(score_func = chi2, k = 15)

skb.fit(bc_X, bc_Y)

feature_scores = [(item, scope) for item, scope in zip(bc_data.feature_names, skb.scores_)]

print(sorted(feature_scores, key = lambda x: -x[1])[:10])