from sklearn.model_selection import train_test_split
from DecisionTree import DecisionTree
from MultiLayerPerceptron import MultiLayerPerceptron
from LogisticRegression import LogisticReg
from XGBoost import XGBoost

class ModelClassification:

    def __init__(self, model_name, data_matrix, labels):
        self.x_train = None
        self.y_train = None
        self.x_test = None
        self.y_test = None
        self.data = data_matrix
        self.labels = labels
        self.model_name = model_name
        self.model = None
        self.split_test_train()

    """ Splitting the data, 80% to train and 20% to test the accyracy of trained model"""
    def split_test_train(self):
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.data, self.labels, test_size=0.2,
                                                                                shuffle=True, random_state=42)

    """ Model to call corresponding models """
    def get_trained_model(self):

        if self.model_name == 'MLP':
            self.model = MultiLayerPerceptron(self.x_train, self.y_train, self.x_test, self.y_test)
            self.model.train_model()


        elif self.model_name == 'LR':
            self.model = LogisticReg(self.x_train, self.y_train, self.x_test, self.y_test)
            self.model.train_model()


        elif self.model_name == 'DT':
            self.model = DecisionTree(self.x_train, self.y_train, self.x_test, self.y_test)
            self.model.train_model()


        elif self.model_name == 'XGB':
            self.model = XGBoost(self.x_train, self.y_train, self.x_test, self.y_test)
            self.model.train_model()

        return self.model.get_model()

    def get_model(self):
        return self.model.get_model()





