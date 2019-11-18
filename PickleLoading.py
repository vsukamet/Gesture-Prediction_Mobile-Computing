from Model_Classification import ModelClassification
from DataPreprocess import get_data_label
import pickle
import numpy as np

"""Gettting the data which is extracted using the Function defined in DataPreprocess.py"""
data, labels = get_data_label()


""" Training the data on 'Decision-Tree' Classifier """
DT = ModelClassification('DT', data, labels)
DT.get_trained_model()
pickle.dump(DT.get_model(), open('DT.pkl', 'wb'))

""" Training the data on 'XGBoost' Classifier """
XGB = ModelClassification('XGB', np.array(data), np.array(labels))
XGB.get_trained_model()
pickle.dump(XGB.get_model(), open('XGB.pkl', 'wb'))

""" Training the data on 'Mutli-layer-Perceptron' model """
XGB = ModelClassification('MLP', np.array(data), np.array(labels))
XGB.get_trained_model()
pickle.dump(XGB.get_model(), open('MLP.pkl', 'wb'))

""" Training the data on 'Logistic Regression' modedl  """
LR = ModelClassification('LR', np.array(data), np.array(labels))
LR.get_trained_model()
pickle.dump(XGB.get_model(), open('LR.pkl', 'wb'))


# ADB = ModelClassification('ADB', np.array(data), np.array(labels))
# ADB.get_trained_model()
# pickle.dump(ADB.get_model(), open('ADB.pkl', 'wb'))

