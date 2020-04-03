from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.datasets import load_breast_cancer
from model_evaluation_utils import display_model_performance_metrics
from sklearn.model_selection import RandomizedSearchCV
from scipy import stats

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

param_grid = {'C': stats.expon(scale=10), 'gamma': stats.expon(scale=.1), 'kernel': ['rbf', 'linear']}

random_search = RandomizedSearchCV(SVC(random_state=42),param_distributions=param_grid, n_iter=50, cv=5)
random_search.fit(X_train, y_train)
print("Best parameters set found on development set:")
random_search.best_params_