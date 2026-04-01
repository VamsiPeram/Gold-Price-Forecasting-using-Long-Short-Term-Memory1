from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load model
def load_model(filename):
    with open(filename, 'rb') as file:
        model = pickle.load(file)
    return model

model = load_model('linear_regression_model.pkl')

# Home route
@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    if request.method == 'POST':
        try:
            # Extract input values from form
            features = [
                float(request.form.get('Open')),
                float(request.form.get('High')),
                float(request.form.get('Low')),
                float(request.form.get('Adj_Close')),
                float(request.form.get('Volume')),
                float(request.form.get('SP_open')),
                float(request.form.get('SP_high')),
                float(request.form.get('SP_low')),
                float(request.form.get('SP_close'))
            ]
            input_array = np.array(features).reshape(1, -1)
            prediction = round(model.predict(input_array)[0], 2)
        except Exception as e:
            prediction = f"Error: {str(e)}"
    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True,host='0.0.0.0',port=5000)
