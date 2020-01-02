from pandas import read_csv
from pandas import get_dummies
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.externals import joblib
import os
from numpy import array

df = read_csv('students_records.csv')

feature_names = ['OverallGrade','Obedient','ResearchScore','ProjectScore']

train_set = df[feature_names]
y_set_name = ['Recommend']
y_set_value = df[y_set_name]

df = read_csv('students_records_test.csv')
test_set = df[feature_names]

#separate numerical and text features

num_feat = ['ResearchScore', 'ProjectScore']

categorical_feat = ['OverallGrade', 'Obedient']

#normalize numerical data

ss = StandardScaler()



#fit scalars on ss model for test set

ss.fit(test_set[num_feat])

#scale numeric features for test set

test_set[num_feat] = ss.transform(test_set[num_feat])

#fit scalars on ss model for train set

ss.fit(train_set[num_feat])

#scale numeric features for train set

train_set[num_feat] = ss.transform(train_set[num_feat])

train_set = get_dummies(train_set, columns = categorical_feat)
test_set = get_dummies(test_set, columns = categorical_feat)

train_set['OverallGrade_D'] = 0
test_set['OverallGrade_E'] = 0
test_set['OverallGrade_F'] = 0

#modeling

lr = LogisticRegression()
model = lr.fit(train_set, array(y_set_value['Recommend']))

#model evaluation

pred_labels = model.predict(test_set)
actual_labels = array(y_set_value['Recommend'])

print(pred_labels)

print('Accuracy:', float(accuracy_score(actual_labels,pred_labels))*100, '%')

print(classification_report(actual_labels, pred_labels))

#save model on server for future prediction

if not os.path.exists('Model'):
	os.mkdir('Model')

if not os.path.exists('Scaler'):
	os.mkdir('Scaler')

joblib.dump(model, r'Model/model.pickle')
joblib.dump(ss, r'Scaler/scaler.pickle')