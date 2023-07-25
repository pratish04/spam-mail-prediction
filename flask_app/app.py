from flask import Flask, request, jsonify, g
from flask_cors import CORS
import os
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer


app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# load the trained ml model
model = joblib.load('trained_model.joblib')

@app.before_request
def before_request():
    # Load the fitted feature extraction vectorizer
    if 'vectorizer' not in g:
        g.vectorizer = joblib.load('fitted_vectorizer.joblib') 

@app.route('/', methods=['POST'])
def handle_post_request():
    if request.method == 'POST':
        data = request.get_json()
        name = data.get('name')
        # Process the data as needed
        print("Received name:", name)

        # # using the ml model for prediction
        input_mail_features = g.vectorizer.transform([name])
        prediction = model.predict(input_mail_features)
        print('prediction: ', prediction)

        return jsonify({"message": "Data received successfully!", "prediction": int(prediction[0])})

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
