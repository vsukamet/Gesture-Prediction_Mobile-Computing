from sklearn.model_selection import train_test_split
from AdaBoost import AdaBoost
from DecisionTree import DecisionTree
from RandomForest import RandomForest
from MultiLayerPerceptron import MultiLayerPerceptron
from SupportVectorMachine import SupportVectorMachine
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

    def split_test_train(self):
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.data, self.labels, test_size=0.2,
                                                                                shuffle=True, random_state=42)
        print(self.y_train)

    def get_trained_model(self):

        if self.model_name == 'ADB':
            self.model = AdaBoost(self.x_train, self.y_train, self.x_test, self.y_test)
            self.model.train_model()
            print('adb')

        elif self.model_name == 'MLP':
            self.model = MultiLayerPerceptron(self.x_train, self.y_train, self.x_test, self.y_test)
            self.model.train_model()
            print('MLP')

        elif self.model_name == 'SVM':
            self.model = SupportVectorMachine(self.x_train, self.y_train, self.x_test, self.y_test)
            self.model.train_model()
            print('SVM')

        elif self.model_name == 'LR':
            self.model = LogisticReg(self.x_train, self.y_train, self.x_test, self.y_test)
            self.model.train_model()
            print('LR')

        elif self.model_name == 'DT':
            self.model = DecisionTree(self.x_train, self.y_train, self.x_test, self.y_test)
            self.model.train_model()
            print('dt')

        elif self.model_name == 'RF':
            self.model = RandomForest(self.x_train, self.y_train, self.x_test, self.y_test)
            self.model.train_model()
            print('rf')

        elif self.model_name == 'XGB':
            self.model = XGBoost(self.x_train, self.y_train, self.x_test, self.y_test)
            self.model.train_model()
            print('xgb')

        return self.model.get_model()

    def get_model(self):
        return self.model.get_model()





