from sklearn.tree import DecisionTreeClassifier


class DecisionTree:
    def __init__(self, x_train, y_train, x_test, y_test):
        self.model = DecisionTreeClassifier(max_depth=10)
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_Test = y_test
        self.y_pred = None

    def train_model(self):
        self.model.fit(self.x_train, self.y_train)
        print('dt train')

    def predict_test(self):
        self.y_pred = self.model.predict(self.x_test)
        return self.y_pred

    def get_model(self):
        return self.model


