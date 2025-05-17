# Sentiment Analysis Project (End-to-End)

# Step 1: Import Libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Step 2: Load Dataset
data = pd.DataFrame({
    'text': [
        "I love this product",
        "This is the worst experience ever",
        "Okay service",
        "Amazing quality!",
        "Not satisfied"
    ],
    'sentiment': ["positive", "negative", "neutral", "positive", "negative"]
})

# Step 3: Preprocess Text
def preprocess(text):
    return text.lower()

data['text'] = data['text'].apply(preprocess)

# Step 4: Vectorize Text
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(data['text'])
y = data['sentiment']

# Step 5: Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 6: Train Model
model = MultinomialNB()
model.fit(X_train, y_train)

# Step 7: Evaluate
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Report:\n", classification_report(y_test, y_pred))

# Step 8: Save Model and Vectorizer
joblib.dump(model, "sentiment_model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")

# Step 9: Flask Web UI
from flask import Flask, render_template, request

# Load saved model
model = joblib.load("sentiment_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

app = Flask(__name__)

def predict_sentiment(text):
    text = preprocess(text)
    vector = vectorizer.transform([text])
    return model.predict(vector)[0]

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    text = request.form['text']
    prediction = predict_sentiment(text)
    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
