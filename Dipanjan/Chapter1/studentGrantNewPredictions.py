from pandas import read_csv
from pandas import get_dummies
from pandas import DataFrame
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.externals import joblib
import os
from numpy import array

model = joblib.load(r'Model/model.pickle')
scaler = joblib.load(r'Scaler/scaler.pickle')

new_data = DataFrame([{'Name': 'Nathan','OverallGrade':'F', 'Obedient': 'N', 'ResearchScore': 30, 'ProjectScore':20},{'Name': 'Thomaz','OverallGrade':'A', 'Obedient': 'Y', 'ResearchScore': 78, 'ProjectScore': 80}])

num_feat = ['ResearchScore', 'ProjectScore']
feature_names = ['OverallGrade','Obedient','ResearchScore','ProjectScore']
categorical_feat = ['OverallGrade', 'Obedient']

pred_feat = new_data[feature_names]

pred_feat[num_feat] = scaler.transform(pred_feat[num_feat])

print(pred_feat)

pred_feat = get_dummies(pred_feat, columns = categorical_feat)

pred_feat['OverallGrade_B'] = 0
pred_feat['OverallGrade_C'] = 0
pred_feat['OverallGrade_D'] = 0
pred_feat['OverallGrade_E'] = 0

prediction = model.predict(pred_feat)

new_data['Recommend'] = prediction

print(new_data)