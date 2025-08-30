import pandas as pd
import random
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List

# --- Existing Request Model ---
class TweetRequest(BaseModel):
    tweets: List[str]

app = FastAPI()

# --- Your existing sentiment predictor ---
def predict_sentiment(tweets: List[str]):
    # Replace this with your actual model logic
    return ["buy" if "buy" in tweet.lower() else "sell" for tweet in tweets]

# --- POST endpoint for direct prediction ---
@app.post("/predict/")
def predict(request: TweetRequest):
    predictions = predict_sentiment(request.tweets)
    return {"predictions": predictions}

# --- NEW endpoint to randomly generate tweets and predict ---
@app.get("/sentimentTrigger/")
def sentiment_trigger(num_samples: int = 5):
    """
    Randomly selects tweets from dataset and returns sentiment predictions.
    """
    # Load dataset
    df = pd.read_csv("data/aapl_tweets.csv", header=None, names=["label", "tweet"])
    
    # Randomly select tweets
    sampled_tweets = df["tweet"].sample(n=num_samples, random_state=random.randint(0, 1000)).tolist()
    
    # Get predictions
    predictions = predict_sentiment(sampled_tweets)
    
    # Return both tweets and predictions
    return {
        "tweets": sampled_tweets,
        "predictions": predictions
    }
