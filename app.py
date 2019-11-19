from collections import Counter
from flask import Flask, request, jsonify
import pandas as pd
import pickle

app = Flask(__name__)

""" Getting all the pickle files(must present in same directory)
    which gives the trained Machine-learning Models """

model1 = pickle.load(open('DT.pkl', 'rb'))
model2 = pickle.load(open('XGB.pkl', 'rb'))
model3 = pickle.load(open('MLP.pkl', 'rb'))
model4= pickle.load(open('LR.pkl', 'rb'))
# pca_model = pickle.load(open('pca_model.pkl','rb'))
scale_model = pickle.load(open('Scaled_model1.pkl','rb'))

ges_list = {1:'buy', 2:'communicate', 3:'fun', 4:'hope', 5:'mother', 6:'really'}


""" Function to convert the given json-object to data-frame """
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
    data_frame = pd.DataFrame(all_frames)
    return data_frame


def max_count(predictions):
    res_dict = Counter(predictions)
    res_max = max(res_dict.values())
    result = [i for i in res_dict if res_dict[i] == res_max]
    return result


""" Function to predict the given test-json obtained using POST request """
@app.route('/',methods=['POST'])
def predict_api():
    """Getting the given json-object"""
    json_data = request.get_json()

    """Convert json to numpy array"""
    data_frame = json_to_dataframe(json_data)
    test_data = data_frame.values

    """Applying trained PCA model and Scaled-model on test-data"""
    # test_data = pca_model.transform(test_data)
    test_data = scale_model.transform(test_data)

    """Getting Predictions to test-data by using 4-trained Machine Learning Models """
    model1_predictions = model1.predict(test_data)
    model2_predictions = model2.predict(test_data)
    model3_predictions = model3.predict(test_data)
    model4_predictions = model4.predict(test_data)
    model1_res = max_count(model1_predictions)
    model2_res = max_count(model2_predictions)
    model3_res = max_count(model3_predictions)
    model4_res = max_count(model4_predictions)

    """Storing the output in dictionary"""
    output = {'1':ges_list[model1_res[0]], '2':ges_list[model2_res[0]],
              '3':ges_list[model3_res[0]], '4':ges_list[model4_res[0]]}

    return jsonify(output)


if __name__ == "__main__":
    app.run(debug=True)
