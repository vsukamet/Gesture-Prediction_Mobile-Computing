
Mobile Computing Assignment-2


Before doing this, install libraries in requirements.txt

Contributions for developing Machine Learning Models:

SupportVectorMachine - Krishna Chaitanya Bogavalli
Logistic Regression - Itish
LinearDiscriminantAnalysis - Vinisha Sukameti
MultilayerPerceptron - Prasanth Reddy


1. To train the Machine Learning Models:
-->Run PickleLoading.py

2. On running this Code, Model directly split the given dataset (features extracted from the Posenet) into training and test sets in the ratio 70:30. Then all the models are trained using 80% training data. Before this, data preprocessing is done to extract the features (process is included in the report)

After training, all trained models are saved in pickle files
1.LDA_final.pkl
2.MLP_final.pkl
3.LR_final.pkl
4.SVM_final.pkl

3. To run the application of our service, use the trained models that are saved in pickle files mentioned above. 

4. Run 'app.py' to run the service in your Local-server(used Flask in back-end before running this python file. Required libraries are also mentioned in requirements.txt). Also, make sure that Datapreprocess.py is in the same directory before running this script. 

5. Service url: http://g32-gesture-prediction.herokuapp.com/












