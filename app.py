# from collections import Counter
# from flask import Flask, request, jsonify
# import pandas as pd
# import pickle
# import numpy as np
#
# app = Flask(__name__)
#
# """ Getting all the pickle files(must present in same directory)
#     which gives the trained Machine-learning Models """
#
# # model1 = pickle.load(open('DT.pkl', 'rb'))
# # model2 = pickle.load(open('XGB.pkl', 'rb'))
# # model3 = pickle.load(open('MLP.pkl', 'rb'))
# # model4= pickle.load(open('LR.pkl', 'rb'))
# # # pca_model = pickle.load(open('pca_model.pkl','rb'))
# # scale_model = pickle.load(open('Scaled_model.pkl','rb'))
#
#
#
# ges_list = {1:'buy', 2:'communicate', 3:'fun', 4:'hope', 5:'mother', 6:'really'}
#
#
# """ Function to convert the given json-object to data-frame """
# def json_to_dataframe(packet):
#     all_frames = []
#     for i in range(len(packet)):
#         temp = []
#         current_frame = packet[i]
#         temp.append(current_frame['score'])
#         key_points = current_frame['keypoints']
#         for j in range(len(key_points)):
#             parts = key_points[j]
#             temp.append(parts['score'])
#             position = parts['position']
#             x, y = list(position.values())
#             temp.append(x)
#             temp.append(y)
#         all_frames.append(temp)
#     data_frame = pd.DataFrame(all_frames)
#     return data_frame
#
#
# def max_count(predictions):
#     res_dict = Counter(predictions)
#     res_max = max(res_dict.values())
#     result = [i for i in res_dict if res_dict[i] == res_max]
#     return result
#
#
# """ Function to predict the given test-json obtained using POST request """
# @app.route('/',methods=['POST'])
# def predict_api():
#     """Getting the given json-object"""
#     json_data = request.get_json()
#
#     """Convert json to numpy array"""
#     data_frame = json_to_dataframe(json_data)
#     test_data = data_frame.values
#
#     """Applying trained PCA model and Scaled-model on test-data"""
#     # test_data = pca_model.transform(test_data)
#     test_data = scale_model.transform(test_data)
#
#     """Getting Predictions to test-data by using 4-trained Machine Learning Models """
#     model1_predictions = model1.predict(test_data)
#     model2_predictions = model2.predict(test_data)
#     model3_predictions = model3.predict(test_data)
#     model4_predictions = model4.predict(test_data)
#     model1_res = max_count(model1_predictions)
#     model2_res = max_count(model2_predictions)
#     model3_res = max_count(model3_predictions)
#     model4_res = max_count(model4_predictions)
#
#     """Storing the output in dictionary"""
#     output = {'1':ges_list[model1_res[0]], '2':ges_list[model2_res[0]],
#               '3':ges_list[model3_res[0]], '4':ges_list[model4_res[0]]}
#
#     return jsonify(output)
#
# # @app.route('/',methods=['POST'])
# # def predict_api():
# #     splits = 10
# #     json_data = request.json
# #     data_frame = json_to_dataframe(json_data)
# #     data_frame = data_frame.iloc[:,1:35]
# #     df_mean = pd.DataFrame(index=range(splits), columns=data_frame.columns)
# #     df_std = pd.DataFrame(index=range(splits), columns=data_frame.columns)
# #     df_split = np.array_split(data_frame,splits)
# #     for i in range(len(df_split)):
# #         df_mean.at[i] = np.mean(df_split[i])
# #         df_std.at[i] = np.std(df_split[i])
# #     feature_vector = np.concatenate((df_mean.values.flatten(), df_std.values.flatten()))
# #     test_data = feature_vector.reshape(1,feature_vector.shape[0])
# #     # test_data = pca_model.transform(test_data)
# #     test_data = scale_model.transform(test_data)
# #     model1_predictions = model1.predict(test_data)
# #     model2_predictions = model2.predict(test_data)
# #     model3_predictions = model3.predict(test_data)
# #     model4_predictions = model4.predict(test_data)
# #     # print(model4_predictions)
# #     # model1_res = max_count(model1_predictions)
# #     # model2_res = max_count(model2_predictions)
# #     # model3_res = max_count(model3_predictions)
# #     # model4_res = max_count(model4_predictions)
# #     # output = {'1':ges_list[model1_res[0]], '2':ges_list[model2_res[0]],
# #     #           '3':ges_list[model3_res[0]], '4':ges_list[model4_res[0]]}
# #     output = {'1':ges_list[model1_predictions[0]],'2':ges_list[model2_predictions[0]],
# #               '3':ges_list[model3_predictions[0]],'4':ges_list[model4_predictions[0]]}
# #     return jsonify(output)
#
#
# if __name__ == "__main__":
#     app.run(port=3000,debug=True)

from collections import Counter
from flask import Flask, request, jsonify, render_template
import pandas as pd
import pickle
import numpy as np
from numpy.fft import fft
import json
from sklearn.preprocessing import scale

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

model1 = pickle.load(open('SGD_final.pkl', 'rb'))
model2 = pickle.load(open('LR_final.pkl', 'rb'))
model3 = pickle.load(open('MLP_final.pkl', 'rb'))
model4= pickle.load(open('SVM_final.pkl', 'rb'))
# pca_model = pickle.load(open('pca_model_rp.pkl','rb'))
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
    splits = 10
    json_data = request.json
    print(json_data)

    with open('data.json', 'w') as f:
        json.dump(json_data, f)
    video = json_to_dataframe(json_data)
    # print(video)
    feature_vector = []

    # FFT
    lsobject_x = video['leftShoulder_x'].values
    lsobject_y = video['leftShoulder_y'].values
    left_shoulder_x_fft = compute_fft(lsobject_x)
    left_shoulder_y_fft = compute_fft(lsobject_y)
    feature_vector += left_shoulder_x_fft
    feature_vector += left_shoulder_y_fft


    rsobject_x = video['rightShoulder_x'].values
    rsobject_y = video['rightShoulder_y'].values
    right_shoulder_x_fft = compute_fft(rsobject_x)
    right_shoulder_y_fft = compute_fft(rsobject_y)
    feature_vector += right_shoulder_x_fft
    feature_vector += right_shoulder_y_fft

    leobject_x = video['leftElbow_x'].values
    leobject_y = video['leftElbow_y'].values
    left_elbow_x_fft = compute_fft(leobject_x)
    left_elbow_y_fft = compute_fft(leobject_y)
    feature_vector += left_elbow_x_fft
    feature_vector += left_elbow_y_fft

    reobject_x = video['rightElbow_x'].values
    reobject_y = video['rightElbow_y'].values
    right_elbow_x_fft = compute_fft(reobject_x)
    right_elbow_y_fft = compute_fft(reobject_y)
    feature_vector += right_elbow_x_fft
    feature_vector += right_elbow_y_fft

    lwobject_x = video['leftWrist_x'].values
    lwobject_y = video['leftWrist_y'].values
    left_wrist_x_fft = compute_fft(lwobject_x)
    left_wrist_y_fft = compute_fft(lwobject_y)
    feature_vector += left_wrist_x_fft
    feature_vector += left_wrist_y_fft

    rwobject_x = video['rightWrist_x'].values
    rwobject_y = video['rightWrist_y'].values
    right_wrist_x_fft = compute_fft(rwobject_x)
    right_wrist_y_fft = compute_fft(rwobject_y)
    feature_vector += right_wrist_x_fft
    feature_vector += right_wrist_y_fft

    # Variance
    feature_vector.append(compute_std(lsobject_x))
    feature_vector.append(compute_std(lsobject_y))
    feature_vector.append(compute_std(rsobject_x))
    feature_vector.append(compute_std(rsobject_y))
    feature_vector.append(compute_std(leobject_x))
    feature_vector.append(compute_std(leobject_y))
    feature_vector.append(compute_std(reobject_x))
    feature_vector.append(compute_std(reobject_y))
    feature_vector.append(compute_std(lwobject_x))
    feature_vector.append(compute_std(lwobject_y))
    feature_vector.append(compute_std(rwobject_x))
    feature_vector.append(compute_std(rwobject_y))

    # Mean
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
    test_data = feature_vector.reshape(1,feature_vector.shape[0])

    test_data = scale_model.transform(test_data)
    model1_predictions = model1.predict(test_data)
    model2_predictions = model2.predict(test_data)
    model3_predictions = model3.predict(test_data)
    model4_predictions = model4.predict(test_data)
    print(model4_predictions)

    # model1_res = max_count(model1_predictions)
    # model2_res = max_count(model2_predictions)
    # model3_res = max_count(model3_predictions)
    # model4_res = max_count(model4_predictions)
    # output = {'1':ges_list[model1_res[0]], '2':ges_list[model2_res[0]],
    #           '3':ges_list[model3_res[0]], '4':ges_list[model4_res[0]]}

    output = {'1':ges_list[model1_predictions[0]],'2':ges_list[model2_predictions[0]],
              '3':ges_list[model3_predictions[0]],'4':ges_list[model4_predictions[0]]}
    return jsonify(output)

# @app.route('/predict_api',methods=['POST'])
# def predict_api():
#     splits = 10
#     json_data = request.json
#     data_frame = json_to_dataframe(json_data)
#     data_frame = data_frame.iloc[:,1:35]
#     df_mean = pd.DataFrame(index=range(splits), columns=data_frame.columns)
#     df_std = pd.DataFrame(index=range(splits), columns=data_frame.columns)
#     df_split = np.array_split(data_frame,splits)
#     for i in range(len(df_split)):
#         df_mean.at[i] = np.mean(df_split[i])
#         df_std.at[i] = np.std(df_split[i])
#     feature_vector = np.concatenate((df_mean.values.flatten(), df_std.values.flatten()))
#     test_data = feature_vector.reshape(1,feature_vector.shape[0])
#     # test_data = pca_model.transform(test_data)
#     test_data = scale_model.transform(test_data)
#     model1_predictions = model1.predict(test_data)
#     model2_predictions = model2.predict(test_data)
#     model3_predictions = model3.predict(test_data)
#     model4_predictions = model4.predict(test_data)
#     print(model4_predictions)
#
#     # model1_res = max_count(model1_predictions)
#     # model2_res = max_count(model2_predictions)
#     # model3_res = max_count(model3_predictions)
#     # model4_res = max_count(model4_predictions)
#     # output = {'1':ges_list[model1_res[0]], '2':ges_list[model2_res[0]],
#     #           '3':ges_list[model3_res[0]], '4':ges_list[model4_res[0]]}
#
#     output = {'1':ges_list[model1_predictions[0]],'2':ges_list[model2_predictions[0]],
#               '3':ges_list[model3_predictions[0]],'4':ges_list[model4_predictions[0]]}
#     return jsonify(output)


if __name__ == "__main__":
    app.run(debug=True)

