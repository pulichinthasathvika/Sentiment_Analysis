from flask import Flask, render_template, request
import joblib
import re

app = Flask(__name__)

# Loading of model
best_model = joblib.load('Data/svm_model.pkl')


def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text

@app.route('/')
def index():
    return render_template('home.html')
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        review = request.form['review']
        processed_review = preprocess_text(review)
        prediction = best_model.predict([processed_review])[0]
        sentiment = "Positive" if prediction == 'Positive' else "Negative"
        return render_template('result.html', sentiment=sentiment, review=review)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)