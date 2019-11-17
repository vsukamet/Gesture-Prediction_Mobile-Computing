from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier

class XGBoost:
    def __init__(self, x_train, y_train, x_test, y_test):
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_Test = y_test
        self.y_pred = None
        self.grid_params = {'objective': ['binary:logistic'], 'learning_rate': [0.05],
                                   'min_child_weight': [11],
                                   'max_depth': [6],
                                   'colsample_bytree': [0.7],
                                   'n_estimators': [100],
                                   'missing': [-999],
                                   'seed': [1337],
                                   'silent': [1],
                                   'nthread': [4],
                                   'subsample': [0.8]}
        self.model = XGBClassifier()
        self.model = GridSearchCV(self.model, self.grid_params, cv=10)

    def train_model(self):
        self.model.fit(self.x_train, self.y_train)
        print('xgb train')

    def predict_test(self, x_test = None):
        self.y_pred = self.model.predict(self.x_test)
        return self.y_pred

    def get_model(self):
        return self.model





