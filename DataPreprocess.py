import pandas as pd
import os
import glob
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

'''This Function takes all the features(.csv files) of given 6 class labels'''

def get_data_label():

    ges_list = {1:'buy', 2:'communicate', 3:'fun', 4:'hope', 5:'mother', 6:'really'}

    data_frames = []

    for ges in ges_list:
        path = os.getcwd()

        """ Takes all features(.csv files) for each given gesture"""
        ges_path = path + '/' + ges_list[ges]
        files = glob.glob(ges_path + '/*.csv')
        df = pd.concat(map(pd.read_csv,files))
        label_values = [ges]*len(df)
        df = df.assign(Label = label_values)
        data_frames.append(df)
        del df

    """Combine all feature data frames obtained from 6 different classes """
    final_df = pd.concat(data_frames, axis=0, ignore_index=True)
    final_df = final_df.drop('Frames#', 1)

    labels = final_df.iloc[:,-1]
    data = final_df.iloc[:,:-1]

    """Applying dimensionality reduction technique to reduce the no of features from 52 to 15"""
    pca = PCA(n_components=15)
    data = pca.fit_transform(data)

    """Saving the trained pca-model to apply on test-data"""
    pca_filename = "pca_model.pkl"
    with open(pca_filename, 'wb') as file:
        pickle.dump(pca, file)

    """Applying Standard Scaling to normalize the values"""
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)
    pkl_filename = "Scaled_model1.pkl"
    with open(pkl_filename, 'wb') as file:
        pickle.dump(scaler, file)

    return scaled_data, labels

if __name__ == '__main__':

    data, labels = get_data_label()



















