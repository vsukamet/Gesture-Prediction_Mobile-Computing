from sklearn.neural_network import MLPClassifier


class MultiLayerPerceptron:
    def __init__(self, x_train, y_train, x_test, y_test):
        self.model = MLPClassifier(hidden_layer_sizes=(10,),  activation='relu',
                                  solver='adam', alpha=0.001, batch_size='auto',
                                learning_rate='constant', learning_rate_init=0.01,
                                  power_t=0.5, max_iter=1000, shuffle=True,
                                    random_state=9, tol=0.0001, verbose=False,
                                  warm_start=False, momentum=0.9, nesterovs_momentum=True,
                                    early_stopping=False, validation_fraction=0.1,
                                  beta_1=0.9, beta_2=0.999, epsilon=1e-08)
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_Test = y_test
        self.y_pred = None

    def train_model(self):
        self.model.fit(self.x_train, self.y_train)

    def predict_test(self):
        self.y_pred = self.model.predict(self.x_test)
        return self.y_pred

    def get_model(self):
        return self.model




