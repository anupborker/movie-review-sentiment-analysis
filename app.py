from flask import Flask, request, render_template
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string

# Initialize Flask app
app = Flask(__name__)

# Load the trained model and TF-IDF vectorizer
model = pickle.load(open('sentiment_model.pkl', 'rb'))
tfidf_vectorizer = pickle.load(open('tfidf_vectorizer.pkl', 'rb'))

# Define text cleaning function
def clean_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    stop_words = set(stopwords.words('english'))
    tokens = word_tokenize(text)
    cleaned_text = ' '.join([word for word in tokens if word not in stop_words])
    return cleaned_text

# Define the home route
@app.route('/')
def home():
    return render_template('index.html')

# Define the prediction route
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get the review text from the form
        review = request.form['review']
        
        # Clean and transform the review text
        cleaned_review = clean_text(review)
        vectorized_review = tfidf_vectorizer.transform([cleaned_review])
        
        # Predict sentiment using the loaded model
        prediction = model.predict(vectorized_review)[0]
        
        # Return the prediction result
        return render_template('index.html', prediction=f'The sentiment is: {prediction}')

# Run the Flask app
if __name__ == '__main__':
    nltk.download('punkt')
    nltk.download('stopwords')
    app.run(debug=True)
