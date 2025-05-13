import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template
import joblib
from sklearn.feature_extraction.text import CountVectorizer
import re
import nltk
nltk.download('stopwords')
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords

app = Flask(__name__, template_folder='templates')

# Load models (replace with actual model files for each type)
review_models = {
    "vectorizer": joblib.load(open('models/Natural Language Processing (NLP)/vectorizer.pkl', 'rb')),
    "naive_bayes": joblib.load(open('models/Natural Language Processing (NLP)/naive_bayes_model.pkl', 'rb')),
    "logistic_regression": joblib.load(open('models/Natural Language Processing (NLP)/logistic_regression_model.pkl', 'rb')),
    "knn": joblib.load(open('models/Natural Language Processing (NLP)/knn_model.pkl', 'rb')),
    "svm": joblib.load(open('models/Natural Language Processing (NLP)/svm_model.pkl', 'rb')),
    "kernel_svm": joblib.load(open('models/Natural Language Processing (NLP)/kernel_svm_model.pkl', 'rb')),
    "decision_tree": joblib.load(open('models/Natural Language Processing (NLP)/decision_tree_model.pkl', 'rb')),
    "random_forest": joblib.load(open('models/Natural Language Processing (NLP)/random_forest_model.pkl', 'rb')),
}

income_models = {
    # "multiple_regression": pickle.load(open('multiple_regression_model.pkl', 'rb')),
    # "polynomial_regression": pickle.load(open('polynomial_regression_model.pkl', 'rb')),
    # "decision_tree_regression": pickle.load(open('decision_tree_regression_model.pkl', 'rb')),
    # "random_forest_regression": pickle.load(open('random_forest_regression_model.pkl', 'rb')),
    # "support_vector_regression": pickle.load(open('support_vector_regression_model.pkl', 'rb')),
}
def preprocess_review(review):
    """
    Preprocess the review text
    """
    # Clean the text
    review_cleaned = re.sub('[^a-zA-Z]', ' ', review).lower().split()

    # If stemming/lemmatization was used, apply it here
    # Example: 
    ps = PorterStemmer()

    all_stop_words = stopwords.words('english')
    unwanted_words=['not','against','no','nor',"don't","aren't","didn'","didn't",'doesn', "doesn't",'hadn',
                    "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't",'mightn', "mightn't", 'mustn', 
                    "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 
                    'weren', "weren't", "won't", 'wouldn', "wouldn't",'couldn',"couldn't"]
    all_stop_words = [ele for ele in all_stop_words if ele not in unwanted_words]

    review_cleaned = [ps.stem(word) for word in review_cleaned if not word in set(all_stop_words)]

    # Join the words back into a single string
    review_cleaned = ' '.join(review_cleaned)

    # Transform the review using the same vectorizer (e.g., CountVectorizer or TfidfVectorizer)
    # Wrap the cleaned review in a list
    cv = joblib.load(open('models/Natural Language Processing (NLP)/vectorizer.pkl', 'rb'))
    review_transformed = cv.transform([review_cleaned]).toarray()

    return review_transformed

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict_review', methods=['POST'])
def predict_review():
    """
    API for predicting review classification
    """
    model_name = request.form['Model']
    review = request.form['Review']
    
    # Preprocess the review
    review_features = preprocess_review(review)
    # Example: Convert review to a feature vector
    # Replace this with actual preprocessing logic
    
    # Get the selected model
    model = review_models.get(model_name)
    if not model:
        return jsonify({"error": "Invalid model selected"}), 400
    
    
    # Predict using the selected model
    prediction = model.predict(review_features)
    # Map prediction to 'good' or 'bad'
    prediction_text = f"Review Prediction: {'Good' if prediction[0] == 1 else 'Bad'} (Model: {model_name})"
    
    return render_template('index.html', prediction_text=f'{prediction_text}')
@app.route('/predict_income', methods=['POST'])
def predict_income():
    """
    API for predicting income based on recency
    """
    model_name = request.form['Model']
    recency = float(request.form['Recency'])
    
    # Preprocess the input (if necessary)
    # Example: Convert recency to a feature vector
    # Replace this with actual preprocessing logic
    income_features = np.array([[recency]])  # Dummy example
    
    # Get the selected model
    model = income_models.get(model_name)
    if not model:
        return jsonify({"error": "Invalid model selected"}), 400
    
    # Predict using the selected model
    prediction = model.predict(income_features)
    return render_template('index.html', prediction_text=f'Predicted Income: ${prediction[0]:.2f}')

if __name__ == '__main__':
    app.run(host='0.0.0.0')