import pandas as pd
import numpy as np
from flask import Flask, request, render_template, jsonify
import joblib

app = Flask(__name__)

df = pd.read_csv("Pos_Pakaging_Quan_Day.csv")


def process(InvID, quan_pos):
    result = []
    data = df[df["InvID"] == InvID]
    
    # Check if data for given InvID exists
    if not data.empty:
        result.append(data.iloc[0]["Differ_mean"])
        result.append(data.iloc[0]["Quan_pag_max_day"])
        
        if quan_pos < 1000:
            result.append(1)
        elif quan_pos < 5000:
            result.append(2)
        elif quan_pos < 20000:
            result.append(3)
        elif quan_pos < 40000:
            result.append(4)
        elif quan_pos < 60000:
            result.append(5)
        else:
            result.append(6)
    else:
        # If data for the given InvID is not found, return None or an appropriate value.
        # You can modify this as per your requirement.
        result = None
    
    return result
model = joblib.load('model.pkl')
@app.route('/')
def home():
    return render_template('index1.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        InvID = str(request.form['InvID'])
        Quan_pos = float(request.form['Pos_quan'])
        inp = process(InvID, Quan_pos)

        if inp is not None:
            differ_mean, quan_pag_max_day, label = inp
        else:
            differ_mean, quan_pag_max_day, label = None, None, None

        # Create a DataFrame with feature names
        input_data = pd.DataFrame({
            'Differ_mean': [differ_mean],
            'Quan_pag_max_day': [quan_pag_max_day],
            'label': [label]
        })

        prediction = model.predict(input_data)
        output = "{:.2f}".format(float(prediction[0]))
        differ_mean = inp[0]
        quan_pag_max_day = inp[1]
        label = inp[2]

        return render_template('index1.html', differ_mean=differ_mean, quan_pag_max_day=quan_pag_max_day, label=label, prediction_text='Leadtime should be {} days'.format(output))
    except ValueError:
        return render_template('index1.html', differ_mean=None, quan_pag_max_day=None, label=None, prediction_text='Invalid Input! Please enter valid numeric values.')

def predict_api():
    '''
    For direct API calls trought request
    '''
    InvID = str(request.form['InvID'])
    Quan_pos = float(request.form['Pos_quan'])
    inp = process(InvID, Quan_pos)

    if inp is not None:
        differ_mean, quan_pag_max_day, label = inp
    else:
        differ_mean, quan_pag_max_day, label = None, None, None

    # Create a DataFrame with feature names
    input_data = pd.DataFrame({
        'Differ_mean': [differ_mean],
        'Quan_pag_max_day': [quan_pag_max_day],
        'label': [label]
    })
    output = model.predict(input_data)
    return jsonify(output)

if __name__ == "__main__":
    app.run(debug= True)
