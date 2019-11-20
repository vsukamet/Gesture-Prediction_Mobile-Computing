from collections import Counter
from flask import Flask, request, jsonify
import pandas as pd
import pickle
import numpy as np
from numpy.fft import fft
from DataPreprocessing import get_feature_vector


def compute_fft(vector):
    max_points_per_series = 10
    fft_res = np.absolute(fft(vector, max_points_per_series))
    return fft_res.tolist()


def compute_std(vector):
    return np.std(vector)


app = Flask(__name__)


columns =['score_overall', 'nose_score', 'nose_x', 'nose_y',
   'leftEye_score', 'leftEye_x', 'leftEye_y', 'rightEye_score',
   'rightEye_x', 'rightEye_y', 'leftEar_score', 'leftEar_x', 'leftEar_y',
   'rightEar_score', 'rightEar_x', 'rightEar_y', 'leftShoulder_score',
   'leftShoulder_x', 'leftShoulder_y', 'rightShoulder_score',
   'rightShoulder_x', 'rightShoulder_y', 'leftElbow_score', 'leftElbow_x',
   'leftElbow_y', 'rightElbow_score', 'rightElbow_x', 'rightElbow_y',
   'leftWrist_score', 'leftWrist_x', 'leftWrist_y', 'rightWrist_score',
   'rightWrist_x', 'rightWrist_y', 'leftHip_score', 'leftHip_x',
   'leftHip_y', 'rightHip_score', 'rightHip_x', 'rightHip_y',
   'leftKnee_score', 'leftKnee_x', 'leftKnee_y', 'rightKnee_score',
   'rightKnee_x', 'rightKnee_y', 'leftAnkle_score', 'leftAnkle_x',
   'leftAnkle_y', 'rightAnkle_score', 'rightAnkle_x', 'rightAnkle_y']

model1 = pickle.load(open('LDA_final.pkl', 'rb'))
model2 = pickle.load(open('LR_final.pkl', 'rb'))
model3 = pickle.load(open('MLP_final.pkl', 'rb'))
model4= pickle.load(open('SVM_final.pkl', 'rb'))
scale_model = pickle.load(open('Scaled_model_final.pkl','rb'))

ges_list = {1:'buy', 2:'communicate', 3:'fun', 4:'hope', 5:'mother', 6:'really'}


def json_to_dataframe(packet):

    all_frames = []
    for i in range(len(packet)):
        temp = []
        current_frame = packet[i]
        temp.append(current_frame['score'])
        key_points = current_frame['keypoints']
        for j in range(len(key_points)):
            parts = key_points[j]
            temp.append(parts['score'])
            position = parts['position']
            x, y = list(position.values())
            temp.append(x)
            temp.append(y)
        all_frames.append(temp)
    data_frame = pd.DataFrame(all_frames, columns=columns)
    return data_frame



def max_count(predictions):
    res_dict = Counter(predictions)
    res_max = max(res_dict.values())
    result = [i for i in res_dict if res_dict[i] == res_max]
    return result


@app.route('/',methods=['POST'])
def predict_api():
    json_data = request.json
    video = json_to_dataframe(json_data)
    test_data = get_feature_vector(video)
    test_data = scale_model.transform(test_data)
    model1_predictions = model1.predict(test_data)
    model2_predictions = model2.predict(test_data)
    model3_predictions = model3.predict(test_data)
    model4_predictions = model4.predict(test_data)

    output = {'1':ges_list[model1_predictions[0]],'2':ges_list[model2_predictions[0]],
              '3':ges_list[model3_predictions[0]],'4':ges_list[model4_predictions[0]]}

    return jsonify(output)


if __name__ == "__main__":
    app.run(port=3000, debug=True)

