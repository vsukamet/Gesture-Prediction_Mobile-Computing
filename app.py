from collections import Counter
from flask import Flask, request, jsonify, render_template
import pandas as pd
import pickle
import ast



app = Flask(__name__)

model1 = pickle.load(open('DT.pkl', 'rb'))
model2 = pickle.load(open('XGB.pkl', 'rb'))
model3 = pickle.load(open('MLP.pkl', 'rb'))
model4= pickle.load(open('LR.pkl', 'rb'))
scale_model = pickle.load(open('Scaled_model.pkl','rb'))

ges_list = {1:'buy', 2:'communicate', 3:'fun', 4:'hope', 5:'mother', 6:'really'}


def json_to_dataframe(packet):
    all_frames = []
    for frame in packet:
        temp = []
        row = list(frame.values())
        overall_score = row[0]
        temp.append(overall_score)
        for key_points in row[1]:
            vals = list(key_points.values())
            temp.append(vals[0])
            vals[2] = str(vals[2])
            if type(vals[2]) is str:
                vals[2] = eval(vals[2])
            x, y = list(vals[2].values())
            temp.append(x)
            temp.append(y)
        all_frames.append(temp)
    # print(len(all_frames))
    data_frame = pd.DataFrame(all_frames)
    return data_frame

def max_count(predictions):
    res_dict = Counter(predictions)
    res_max = max(res_dict.values())
    result = [i for i in res_dict if res_dict[i] == res_max]
    return result

# @app.route('/',methods=['GET'] )
# def home():
#     return 'deployed'

# @app.route('/predict',methods=['POST'])
# def predict():
#     default_name = 'experience'
#     json_data = request.form.to_dict(flat=False)
#     data_frame = json_to_dataframe(json_data)
#     test_data = data_frame.values
#     test_data = scale_model.transform(test_data)
#     model1_predictions = model1.predict(test_data)
#     model2_predictions = model2.predict(test_data)
#     model3_predictions = model3.predict(test_data)
#     model1_res = max_count(model1_predictions)
#     model2_res = max_count(model2_predictions)
#     model3_res = max_count(model3_predictions)
#     output = {'1':ges_list[model1_res[0]], '2':ges_list[model2_res[0]], '3':ges_list[model3_res[0]]}
#     return render_template('home.html', prediction_text='Prediction is {}'.format(output))

@app.route('/',methods=['POST'])
def predict_api():
    json_data = request.get_json(force=True)
    # print(type(json_data))
    data_frame = json_to_dataframe(json_data)
    test_data = data_frame.values
    test_data = scale_model.transform(test_data)
    model1_predictions = model1.predict(test_data)
    model2_predictions = model2.predict(test_data)
    model3_predictions = model3.predict(test_data)
    model4_predictions = model4.predict(test_data)
    model1_res = max_count(model1_predictions)
    model2_res = max_count(model2_predictions)
    model3_res = max_count(model3_predictions)
    model4_res = max_count(model4_predictions)
    output = {'1':ges_list[model1_res[0]], '2':ges_list[model2_res[0]],
              '3':ges_list[model3_res[0]], '4':ges_list[model4_res[0]]}
    return jsonify(output)


if __name__ == "__main__":
    app.run(debug=True)
