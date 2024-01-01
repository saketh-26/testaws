# pip install numpy scikit-learn flask 
# pip install pipreqs

from flask import Flask, render_template, request
import numpy as np
import pickle

application = Flask(__name__)

with open('House_Price.pkl', 'rb') as f:
    model = pickle.load(f)

@application.route('/')
def home():
    return render_template('index.html')

@application.route('/predict', methods=['POST'])
def predict():
    Rooms = int(request.form['bedrooms'])
    Bathrooms= int(request.form['bathrooms'])
    Place = int(request.form['location'])
    Area = int(request.form['area'])
    Status = int(request.form['status'])
    Facing = int(request.form['facing'])
    P_Type = int(request.form['type'])
    
    input_data = np.array([[Place,Area,Status,Rooms,Bathrooms,Facing,P_Type]])
    
    prediction = model.predict(input_data)[0]
    
    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    application.run()
