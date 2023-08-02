import pandas as pd
import numpy as np
from flask import Flask, request, render_template
from flask_ngrok import run_with_ngrok
from pyngrok import ngrok
import joblib
import webbrowser

app = Flask(__name__)
#ngrok.set_auth_token("2TBpvKypZVYnE4r6WT3fReIZ4KX_4jaqcbfmHaQJQ5ZWd3Ezy")
#run_with_ngrok(app)

#tunnel = ngrok.connect(5000)
#public_url = tunnel.public_url
model = joblib.load('model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        Differ_mean = float(request.form['Differ_mean'])
        Quan_pag_max_day = float(request.form['Quan_pag_max_day'])
        label = float(request.form['label'])

        # Create a DataFrame with feature names
        input_data = pd.DataFrame({
            'Differ_mean': [Differ_mean],
            'Quan_pag_max_day': [Quan_pag_max_day],
            'label': [label]
        })

        prediction = model.predict(input_data)
        output = "{:.2f}".format(float(prediction[0]))

        return render_template('index.html', prediction_text='Leadtime should be $ {}'.format(output))
    except ValueError:
        return render_template('index.html', prediction_text='Invalid Input! Please enter valid numeric values.')

if __name__ == "__main__":
    #webbrowser.open_new(public_url)
    app.run()
    #print("public url: {}".format(public_url))
