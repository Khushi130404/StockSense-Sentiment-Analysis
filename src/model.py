import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
import joblib

# Load and prepare dataset
def load_data(csv_file='data/aapl_tweets.csv'):
    df = pd.read_csv(csv_file)
    df.dropna(inplace=True)
    X = df['tweet']
    y = df['action'].map({'Buy': 1, 'Sell': 0})
    return train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
def train_model():
    X_train, X_test, y_train, y_test = load_data()
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1,2))
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_vec, y_train)

    # Save model & vectorizer
    joblib.dump(model, 'src/aapl_sentiment_model.pkl')
    joblib.dump(vectorizer, 'src/tfidf_vectorizer.pkl')

    print("Model trained and saved.")
    return model, vectorizer, X_test_vec, y_test

# Evaluate model
def evaluate_model(model, X_test_vec, y_test):
    y_pred = model.predict(X_test_vec)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))

# Predict new tweets
def predict_sentiment(text_list):
    vectorizer = joblib.load('src/tfidf_vectorizer.pkl')
    model = joblib.load('src/aapl_sentiment_model.pkl')
    text_vec = vectorizer.transform(text_list)
    preds = model.predict(text_vec)
    label_map = {1: 'Buy', 0: 'Sell'}
    return [label_map[p] for p in preds]

# Example usage
if __name__ == "__main__":
    model, vectorizer, X_test_vec, y_test = train_model()
    evaluate_model(model, X_test_vec, y_test)

    sample_tweets = [
        "$AAPL looks bullish after strong earnings!",
        "Apple facing challenges in China, stock might drop."
    ]
    predictions = predict_sentiment(sample_tweets)
    for tweet, pred in zip(sample_tweets, predictions):
        print(f"Tweet: {tweet} -> Prediction: {pred}")
