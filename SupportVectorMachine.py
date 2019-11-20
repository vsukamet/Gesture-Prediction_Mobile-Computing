from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

class SupportVectorMachine:

    def __init__(self, x_train, y_train, x_test, y_test):
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_Test = y_test
        self.y_pred = None
        self.model = SVC()
        self.grid_params = {'kernel': ['linear', 'rbf', 'poly'], 'gamma': ['auto', 'scale'],
                                   'degree': [3, 6, 8, 10]}
        self.model = GridSearchCV(self.model, self.grid_params, cv=10)

    def train_model(self):
        self.model.fit(self.x_train, self.y_train)
        print('SVM train')

    def predict_test(self, x_test = None):
        self.y_pred = self.model.predict(self.x_test)
        return self.y_pred

    def get_model(self):
        return self.model





