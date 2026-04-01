from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load model safely
def load_model(filename):
    with open(filename, 'rb') as file:
        model = pickle.load(file)
    return model

model = load_model('linear_regression_model.pkl')

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    error = None

    if request.method == 'POST':
        try:
            # Get input values from form
            open_val = float(request.form['Open'])
            high_val = float(request.form['High'])
            low_val = float(request.form['Low'])
            adj_close_val = float(request.form['Adj_Close'])
            volume_val = float(request.form['Volume'])
            sp_open_val = float(request.form['SP_open'])
            sp_high_val = float(request.form['SP_high'])
            sp_low_val = float(request.form['SP_low'])
            sp_close_val = float(request.form['SP_close'])

            # Arrange features
            features = np.array([[
                open_val,
                high_val,
                low_val,
                adj_close_val,
                volume_val,
                sp_open_val,
                sp_high_val,
                sp_low_val,
                sp_close_val
            ]])

            # Predict
            result = model.predict(features)
            prediction = round(float(result[0]), 2)

        except Exception as e:
            error = str(e)

    return render_template('index.html', prediction=prediction, error=error)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
