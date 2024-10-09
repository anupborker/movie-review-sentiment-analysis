# Movie Review Sentiment Analysis

This project implements a sentiment analysis model to classify movie reviews as either positive or negative using Natural Language Processing (NLP) techniques.

## Team Members

- **Tanaya  Chali**
- **Anup  Borker**
- **Tanavi  Nipanicar**

## Project Structure

- **IMDB Dataset.csv**: The dataset containing movie reviews and their corresponding sentiment labels (positive/negative).
- **app.py**: The Flask application file for serving the model and handling user input.
- **data_preprocessing.py**: The script for data preprocessing, training the model, and saving it.
- **sentiment_model.pkl**: The saved trained Naive Bayes model for sentiment classification (optional).
- **tfidf_vectorizer.pkl**: The saved TF-IDF vectorizer used for transforming the text data (optional).

## Requirements

To run this project, you will need the following Python packages:

- **pandas**
- **numpy**
- **nltk**
- **scikit-learn**
- **matplotlib**
- **seaborn**
- **Flask**

You can install the required packages using pip:

```bash
pip install pandas numpy nltk scikit-learn matplotlib seaborn Flask
```

### Additional Setup for NLTK

Make sure to download the necessary NLTK data for stopwords:

```python
import nltk
nltk.download('stopwords')
```

## Usage

1. Ensure that the IMDB Dataset.csv file is in the same directory as the scripts.
1. Run the data_preprocessing.py script to preprocess the data, train the model, and save the trained model.
1. Execute the app.py file to start the Flask application, which allows users to input movie reviews and get sentiment predictions.
1. Open your web browser and go to http://127.0.0.1:5000 to access the sentiment analysis interface.

## Conclusion

This project demonstrates the application of sentiment analysis on movie reviews using machine learning and NLP techniques. You can further enhance the model by experimenting with different algorithms and tuning hyperparameters.