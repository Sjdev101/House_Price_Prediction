from flask import Flask, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)

# Load the pre-trained model (replace with the correct path)
with open('model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

@app.route('/')
def home():
    return "Welcome to the ML Model API!"

@app.route('/predict', methods=['POST'])
def predict():
    # Get the input data from the POST request
    data = request.get_json()  # Assuming JSON input

    # Extract features (assuming input is a list of features, e.g., [feature1, feature2])
    features = np.array(data['features']).reshape(1, -1)

    # Make prediction
    prediction = model.predict(features)

    # Return the prediction as a JSON response
    return jsonify({'prediction': prediction[0]})

if __name__ == '__main__':
    app.run(debug=True)
