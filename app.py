import pickle
from flask import Flask, jsonify, render_template, request
import numpy as np
import pandas as pd

app = Flask(__name__)

# load model
model = pickle.load(open("model.pkl", "rb"))

# this API is resposible for collect data & do prediction
@app.route('/predict_api', methods=['POST'])
def predict_api():
    # collect data in json format
    data = request.json['data']
    print(data)
    # convert collected values into 2D array
    new_data = [list(data.values())]
    # prediction
    output = model.predict(new_data)[0]
    return jsonify(output)


if __name__=="__main__":
    app.run(debug=True)



