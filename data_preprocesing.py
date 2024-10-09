
'''
## Team Members

- **Tanaya Chari**
- **Anup Borker**
- **Tanavi  Nipanicar**

'''



# Import necessary libraries
import pandas as pd
import numpy as np
import nltk
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Download necessary NLTK data
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string

# Load dataset (assume CSV file with 'review' and 'sentiment' columns)
# You can replace the dataset with IMDb reviews or any other movie reviews dataset.
data = pd.read_csv('IMDB Dataset.csv')  # Replace with the actual dataset file

# Inspect data (optional)
print(data.head())

# Define a function for text cleaning
def clean_text(text):
    # Convert text to lowercase
    text = text.lower()
    
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = word_tokenize(text)
    cleaned_text = ' '.join([word for word in tokens if word not in stop_words])
    
    return cleaned_text

# Apply the text cleaning function to the dataset
data['cleaned_reviews'] = data['review'].apply(clean_text)

# Vectorize the cleaned text using TF-IDF
tfidf = TfidfVectorizer(max_features=5000)  # You can change max_features to a different number
X = tfidf.fit_transform(data['cleaned_reviews'])

# Define the target variable
y = data['sentiment']  # Assuming sentiment is labeled as 'positive' and 'negative'

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train a Naive Bayes model
model = MultinomialNB()
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# Optional: Save the trained model and TF-IDF vectorizer
import pickle
pickle.dump(model, open('sentiment_model.pkl', 'wb'))
pickle.dump(tfidf, open('tfidf_vectorizer.pkl', 'wb'))

# Optional: Predict sentiment for new reviews
def predict_sentiment(new_review):
    # Clean the new review
    cleaned_review = clean_text(new_review)
    
    # Convert the cleaned review into the TF-IDF vector
    vectorized_review = tfidf.transform([cleaned_review])
    
    # Predict sentiment using the trained model
    prediction = model.predict(vectorized_review)
    
    return prediction[0]

# Test prediction on new review
new_review = "This movie was fantastic! I loved every bit of it."
predicted_sentiment = predict_sentiment(new_review)
print(f"\nPredicted sentiment for new review: {predicted_sentiment}")
