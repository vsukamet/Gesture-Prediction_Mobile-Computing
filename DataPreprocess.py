import pandas as pd
import os
import glob
import pickle
from sklearn.preprocessing import StandardScaler



""" Combining all features of different gestures """
def get_data_label():

    ges_list = {1:'buy', 2:'communicate', 3:'fun', 4:'hope', 5:'mother', 6:'really'}
    data_frames = []
    for ges in ges_list:
        path = os.getcwd()
        ges_path = path + '/' + ges_list[ges]
        files = glob.glob(ges_path + '/*.csv')
        df = pd.concat(map(pd.read_csv,files))
        label_values = [ges]*len(df)
        df = df.assign(Label = label_values)
        data_frames.append(df)
        del df

    final_df = pd.concat(data_frames, axis=0, ignore_index=True)
    final_df = final_df.drop('Frames#', 1)
    labels = final_df.iloc[:,-1]
    data = final_df.iloc[:,:-1]
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)
    pkl_filename = "Scaled_model.pkl"

    with open(pkl_filename, 'wb') as file:
        pickle.dump(scaler, file)

    return scaled_data, labels


if __name__ == '__main__':

    data, labels = get_data_label()
    print(data)
    print(labels)


















