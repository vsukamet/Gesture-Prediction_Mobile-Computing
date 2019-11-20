import os
import glob
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import pickle
from numpy.fft import fft



columns = ['nose_x', 'nose_y', 'leftShoulder_x', 'leftShoulder_y', 'rightShoulder_x', 'rightShoulder_y',
           'leftElbow_x','leftElbow_y',  'rightElbow_x', 'rightEar_x', 'rightEar_y',
           'leftEar_x', 'leftEar_y', 'leftEye_x', 'leftEye_y', 'rightEye_x', 'rightEye_y',
           'rightElbow_y', 'leftWrist_x', 'leftWrist_y',
           'rightWrist_x', 'rightWrist_y', 'leftHip_y', 'leftKnee_y']

def find_fft(vector):
    max_points_per_series = 10
    res = np.absolute(fft(vector, max_points_per_series))
    return res.tolist()


def std(vector):
    return np.std(vector)

def get_feature_vector(video):

    feature_vector = []

    nobject_x = video['nose_x'].values
    nobject_y = video['nose_y'].values

    nose_x_fft = find_fft(nobject_x)
    nose_y_fft = find_fft(nobject_y)
    feature_vector += nose_x_fft
    feature_vector += nose_y_fft

    leyobject_x = video['leftEye_x'].values
    leyobject_y = video['leftEye_y'].values
    left_eye_x_fft = find_fft(leyobject_x)
    left_eye_y_fft = find_fft(leyobject_y)
    feature_vector += left_eye_x_fft
    feature_vector += left_eye_y_fft


    reyobject_x = video['rightEye_x'].values
    reyobject_y = video['rightEye_y'].values
    right_eye_x_fft = find_fft(reyobject_x)
    right_eye_y_fft = find_fft(reyobject_y)
    feature_vector += right_eye_x_fft
    feature_vector += right_eye_y_fft

    leaobject_x = video['leftEar_x'].values
    leaobject_y = video['leftEar_y'].values
    left_ear_x_fft = find_fft(leaobject_x)
    left_ear_y_fft = find_fft(leaobject_y)
    feature_vector += left_ear_x_fft
    feature_vector += left_ear_y_fft


    reaobject_x = video['rightEar_x'].values
    reaobject_y = video['rightEar_y'].values
    right_ear_x_fft = find_fft(reaobject_x)
    right_ear_y_fft = find_fft(reaobject_y)
    feature_vector += right_ear_x_fft
    feature_vector += right_ear_y_fft


    lsobject_x = video['leftShoulder_x'].values
    lsobject_y = video['leftShoulder_y'].values
    left_shoulder_x_fft = find_fft(lsobject_x)
    left_shoulder_y_fft = find_fft(lsobject_y)
    feature_vector += left_shoulder_x_fft
    feature_vector += left_shoulder_y_fft


    rsobject_x = video['rightShoulder_x'].values
    rsobject_y = video['rightShoulder_y'].values
    right_shoulder_x_fft = find_fft(rsobject_x)
    right_shoulder_y_fft = find_fft(rsobject_y)
    feature_vector += right_shoulder_x_fft
    feature_vector += right_shoulder_y_fft

    leobject_x = video['leftElbow_x'].values
    leobject_y = video['leftElbow_y'].values
    left_elbow_x_fft = find_fft(leobject_x)
    left_elbow_y_fft = find_fft(leobject_y)
    feature_vector += left_elbow_x_fft
    feature_vector += left_elbow_y_fft

    reobject_x = video['rightElbow_x'].values
    reobject_y = video['rightElbow_y'].values
    right_elbow_x_fft = find_fft(reobject_x)
    right_elbow_y_fft = find_fft(reobject_y)
    feature_vector += right_elbow_x_fft
    feature_vector += right_elbow_y_fft

    lwobject_x = video['leftWrist_x'].values
    lwobject_y = video['leftWrist_y'].values
    left_wrist_x_fft = find_fft(lwobject_x)
    left_wrist_y_fft = find_fft(lwobject_y)
    feature_vector += left_wrist_x_fft
    feature_vector += left_wrist_y_fft

    rwobject_x = video['rightWrist_x'].values
    rwobject_y = video['rightWrist_y'].values
    right_wrist_x_fft = find_fft(rwobject_x)
    right_wrist_y_fft = find_fft(rwobject_y)
    feature_vector += right_wrist_x_fft
    feature_vector += right_wrist_y_fft


    feature_vector.append(std(nobject_x))
    feature_vector.append(std(nobject_y))
    feature_vector.append(std(leaobject_x))
    feature_vector.append(std(leaobject_y))
    feature_vector.append(std(reaobject_x))
    feature_vector.append(std(reaobject_y))
    feature_vector.append(std(leyobject_x))
    feature_vector.append(std(leyobject_y))
    feature_vector.append(std(reyobject_x))
    feature_vector.append(std(reyobject_y))
    feature_vector.append(std(lsobject_x))
    feature_vector.append(std(lsobject_y))
    feature_vector.append(std(rsobject_x))
    feature_vector.append(std(rsobject_y))
    feature_vector.append(std(leobject_x))
    feature_vector.append(std(leobject_y))
    feature_vector.append(std(reobject_x))
    feature_vector.append(std(reobject_y))
    feature_vector.append(std(lwobject_x))
    feature_vector.append(std(lwobject_y))
    feature_vector.append(std(rwobject_x))
    feature_vector.append(std(rwobject_y))



    feature_vector.append(np.mean(nobject_x))
    feature_vector.append(np.mean(nobject_y))
    feature_vector.append(np.mean(leaobject_x))
    feature_vector.append(np.mean(leaobject_y))
    feature_vector.append(np.mean(reaobject_x))
    feature_vector.append(np.mean(reaobject_y))
    feature_vector.append(np.mean(leyobject_x))
    feature_vector.append(np.mean(leyobject_y))
    feature_vector.append(np.mean(reyobject_x))
    feature_vector.append(np.mean(reyobject_y))
    feature_vector.append(np.mean(lsobject_x))
    feature_vector.append(np.mean(lsobject_y))
    feature_vector.append(np.mean(rsobject_x))
    feature_vector.append(np.mean(rsobject_y))
    feature_vector.append(np.mean(leobject_x))
    feature_vector.append(np.mean(leobject_y))
    feature_vector.append(np.mean(reobject_x))
    feature_vector.append(np.mean(reobject_y))
    feature_vector.append(np.mean(lwobject_x))
    feature_vector.append(np.mean(lwobject_y))
    feature_vector.append(np.mean(rwobject_x))
    feature_vector.append(np.mean(rwobject_y))


    feature_vector = np.asarray(feature_vector)
    feature_vector = feature_vector.reshape(1,feature_vector.shape[0])

    return feature_vector

'''This Function takes all the features(.csv files) of given 6 class labels'''
def get_data_label():

    ges_list = {1:'buy', 2:'communicate', 3:'fun', 4:'hope', 5:'mother', 6:'really'}

    data_frames = []

    """ Takes all features(.csv files) for each given gesture"""
    for ges in ges_list:
        ges_video_frame = []
        path = os.getcwd()
        ges_path = path + '/' + ges_list[ges]
        files = glob.glob(ges_path + '/*.csv')
        for video in files:
            video = pd.read_csv(video, usecols=columns)
            feature_vector = get_feature_vector(video)
            video = pd.DataFrame(feature_vector)
            ges_video_frame.append(video)

        ges_frame = pd.concat(ges_video_frame,axis=0)
        ges_frame['Label'] = ges
        data_frames.append(ges_frame)

    """Combine all feature data frames obtained from 6 different classes """
    final_df = pd.concat(data_frames,axis=0)
    # print(final_df)
    labels = final_df.iloc[:,-1]
    data = final_df.iloc[:,:-1]

    """Applying Standard Scaling to normalize the values"""
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)
    pkl_filename = "Scaled_model_final.pkl"

    with open(pkl_filename, 'wb') as file:
        pickle.dump(scaler, file)

    return scaled_data, labels



if __name__ == '__main__':
    get_data_label()


