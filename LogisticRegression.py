from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
import numpy as np
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
class LogisticReg:
    def __init__(self, x_train, y_train, x_test, y_test):
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_Test = y_test
        self.y_pred = None
        self.model = LogisticRegression(penalty='l2')
        self.grid_params = {'C': np.logspace(-3, 3, 7), 'solver': ['newton-cg', 'sag', 'saga']}
        self.model = GridSearchCV(self.model, self.grid_params, cv=10)

    def train_model(self):
        self.model.fit(self.x_train, self.y_train)
        print('LR train')

    def predict_test(self):
        self.y_pred = self.model.predict(self.x_test)
        return self.y_pred

    def get_model(self):
        return self.model



