from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

# Load the sentiment analysis model
with open('sentiment_analysis_model.pkl', 'rb') as f:
    model = pickle.load(f)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    review = request.form['review']

    # Preprocess the review text if necessary

    # Pass the review to the sentiment analysis model for prediction
    sentiment_prediction = model.predict([review])

    # Interpret the model prediction
    sentiment = "Positive" if sentiment_prediction[0] == 1 else "Negative"

    return render_template('result.html', result=sentiment)

if __name__ == '__main__':
    app.run(debug=True)




