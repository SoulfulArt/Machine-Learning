from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.datasets import load_breast_cancer
from model_evaluation_utils import display_model_performance_metrics
from sklearn.model_selection import GridSearchCV

data = load_breast_cancer()

X = data.data
y = data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42)

def_SVC = SVC(random_state = 42)
def_SVC.fit(X_train, y_train)

def_y_pred = def_SVC.predict(X_test)

"""
print("Default mode stats: ")

display_model_performance_metrics(true_labels = y_test, predicted_labels = def_y_pred, classes = [0,1])"""

#setting the parameters grid

grid_parameters = {'kernel': ['linear','rbf'], 'gamma': [1e-3, 1e-4], 'C': [1, 10, 50, 100]}

print("# tuning hyper-parameters for accuracy \n")

clf = GridSearchCV(SVC(random_state = 42), grid_parameters, cv = 5, scoring = 'accuracy')

clf.fit(X_train, y_train)

print("Grid scores for all the models based on CV:\n")

means = clf.cv_results_['mean_test_score']
stds = clf.cv_results_['std_test_score']

for mean, std, params in zip(means, stds, clf.cv_results_['params']):print("%0.5f	(+/-%0.05f) for %r"%(mean, std*2, params))

print("\nBest parameters: ",clf.best_params_)
print("Best model validation: ", clf.best_score_)

tuned_y_pred = clf.best_estimator_.predict(X_test)

print("\nTuned model:\n")

display_model_performance_metrics(true_labels = y_test, predicted_labels = tuned_y_pred, classes = [0,1])