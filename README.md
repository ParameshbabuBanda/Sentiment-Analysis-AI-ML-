# Sentiment-Analysis-AI-ML-
Sentiment Analysis (AI / ML)

# 💬 Sentiment Analysis Web App — Flask + Machine Learning (End-to-End Project)

This is a complete end-to-end **Sentiment Analysis Web Application** developed using **Python (Flask)** and **Scikit-learn**. The application accepts user text input and predicts the sentiment as **positive**, **negative**, or **neutral**. It demonstrates a full ML pipeline: data preprocessing, model training, evaluation, and deployment through a web interface.

---

## 🎯 Project Objective

To classify user-input text based on sentiment using a trained machine learning model and make it accessible through a simple web app interface.

---

## 🔧 Technologies Used

- **Language**: Python 3
- **Libraries**: Scikit-learn, Pandas, Joblib
- **ML Algorithm**: Multinomial Naive Bayes
- **Text Preprocessing**: CountVectorizer (Bag of Words)
- **Framework**: Flask (Web App)
- **Interface**: HTML + Jinja2 templates

---

## 🛠️ Key Features

- ✅ Simple web interface to input text
- ✅ ML model classifies sentiment as positive, negative, or neutral
- ✅ Automatic text preprocessing (lowercasing)
- ✅ Vectorization using Bag-of-Words
- ✅ Accuracy & classification report printed on model training
- ✅ Model and vectorizer stored and reloaded using `joblib`
- ✅ Real-time prediction with Flask routes

---

## 🧠 Skills Demonstrated

- Natural Language Processing (NLP) fundamentals
- Machine Learning model training and evaluation
- Web development using Flask
- Handling user input and real-time prediction
- Saving and loading models with `joblib`
- Integration of ML with web application

---

## 📁 Project Structure

sentiment-analysis/
├── sentiment_analysis_project.py # Main app: data, model, web
├── sentiment_model.pkl # Trained Naive Bayes model
├── vectorizer.pkl # CountVectorizer object
├── templates/
│ └── index.html # HTML form and output
├── requirements.txt # Required Python packages
└── README.md # Project overview


---
## 🚀 How to Run
Clone repo:
git clone https://github.com/your-username/sentiment-analysis-webapp.git

Install dependencies:
pip install flask scikit-learn pandas joblib

Run the app:
python sentiment_analysis_project.py

Access in browser:
http://127.0.0.1:5000

The model and vectorizer (sentiment_model.pkl, vectorizer.pkl) will be created and saved automatically during the first run.


## OUTPUT

![Image](https://github.com/user-attachments/assets/5cfec36e-d576-4c5e-9788-ed4ea3b83c65)
