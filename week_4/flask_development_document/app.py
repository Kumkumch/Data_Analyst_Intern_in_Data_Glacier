from flask import Flask, render_template, request
import joblib
import numpy as np

# Load the trained model
model = joblib.load('iris_model.pkl')

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if request.method == 'POST':
            # Get data from form
            sepal_length = float(request.form['sepal_length'])
            sepal_width = float(request.form['sepal_width'])
            petal_length = float(request.form['petal_length'])
            petal_width = float(request.form['petal_width'])

            # Make prediction
            data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
            prediction = model.predict(data)
            flower = ['Setosa', 'Versicolor', 'Virginica'][prediction[0]]

            return render_template('index.html', prediction_text=f'Predicted Iris Class: {flower}')
    except Exception as e:
        return render_template('index.html', prediction_text=f'Error: {e}')

if __name__ == '__main__':
    print("✅ Flask app is starting...")
    app.run(debug=True)



