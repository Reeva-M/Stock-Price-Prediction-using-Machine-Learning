from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl' , 'rb'))


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    val1 = float(request.form['bedrooms'])
    val2 = float(request.form['bathrooms'])
    val3 = float(request.form['floors'])
    val4 = float(request.form['yr_built'])
    arr = np.array([val1, val2, val3, val4])
# Assuming pred is a 1D array
    pred = model.predict(arr.reshape(1, -1))[0]

# Verify pred's shape (optional)
    print("Shape of pred:", pred.shape)

# Ensure pred is a scalar (if needed)
    scalar_pred = float(pred[0])
    data=scalar_pred
# Return the result
    return render_template('index.html', data=scalar_pred)


if __name__ == '__main__':
    app.run(debug=True)
