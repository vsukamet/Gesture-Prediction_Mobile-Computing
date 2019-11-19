import os
import glob
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import pickle

def get_data_label():

    ges_list = {1:'buy', 2:'communicate', 3:'fun', 4:'hope', 5:'mother', 6:'really'}

    data_frames = []
    splits = 10

    for ges in ges_list:
        ges_video_frame = []
        path = os.getcwd()
        ges_path = path + '/' + ges_list[ges]
        files = glob.glob(ges_path + '/*.csv')
        for video in files:
            video = pd.read_csv(video)
            video = video.iloc[:,1:35]
            df_mean = pd.DataFrame(index=range(splits), columns=video.columns)
            df_std = pd.DataFrame(index=range(splits), columns=video.columns)
            df_split = np.array_split(video,splits)

            for i in range(len(df_split)):
                df_mean.at[i] = np.mean(df_split[i])
                df_std.at[i] = np.std(df_split[i])
            feature_vector = np.concatenate((df_mean.values.flatten(), df_std.values.flatten()))
            feature_vector = feature_vector.reshape(1,feature_vector.shape[0])
            ges_df = pd.DataFrame(feature_vector)
            # ges_df = pd.concat([df_mean,df_std],axis=0)
            ges_video_frame.append(ges_df)
        sub_df = pd.concat(ges_video_frame,axis=0)
        sub_df['Label'] = ges
        data_frames.append(sub_df)

    final_df = pd.concat(data_frames,ignore_index=True,axis=0)


    labels = final_df.iloc[:,-1]
    data = final_df.iloc[:,:-1]

    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)
    pkl_filename = "Scaled_model_rp.pkl"


    with open(pkl_filename, 'wb') as file:
        pickle.dump(scaler, file)
    return scaled_data, labels

if __name__ == '__main__':
    get_data_label()
