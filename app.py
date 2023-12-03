import pickle
from flask import Flask, jsonify, render_template, request, url_for
import numpy as np
import pandas as pd

app = Flask(__name__)

# load model
model = pickle.load(open("model.pkl", "rb"))


@app.route('/')
def home():
    #return 'Hello World'
    return render_template('home.html')

# this API is resposible for collect data & do prediction in postman
@app.route('/predict_postman', methods=['POST'])
def predict_postman():
    # collect data in json format
    data = request.json['data']
    print(data)
    # convert collected values into 2D array
    new_data = [list(data.values())]
    # prediction
    output = model.predict(new_data)[0]
    return jsonify(output)


# this API is resposible for collect data & do prediction in webapp
@app.route('/predictWebapp', methods=['POST'])
def predictWebapp():
    # collect values from webapp
    data = [float(x) for x in request.form.values()]
    # convert collected values into 2D array
    final_features = [np.array(data)]
    print(data)
    # prediction
    output=model.predict(final_features)[0]
    print(output)
    #output = round(prediction[0], 2)
    return render_template('home.html', prediction_text="Airfoil pressure is  {}".format(output))


if __name__=="__main__":
    app.run(debug=True)



