import pandas as pd
import random
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List

# Create FastAPI app
app = FastAPI()

# --- Add CORS middleware ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8080"],  # React dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Existing Request Model ---
class TweetRequest(BaseModel):
    tweets: List[str]

# --- Your existing sentiment predictor ---
def predict_sentiment(tweets: List[str]):
    # Replace this with your actual model logic
    return ["buy" if "buy" in tweet.lower() else "sell" for tweet in tweets]

# --- POST endpoint for direct prediction ---
@app.post("/predict/")
def predict(request: TweetRequest):
    predictions = predict_sentiment(request.tweets)
    return {"predictions": predictions}

# --- GET endpoint to randomly generate tweets and predict ---
@app.get("/sentimentTrigger/")
def sentiment_trigger():
    """
    Randomly selects one tweet from dataset and returns its sentiment prediction.
    """
    df = pd.read_csv("data/aapl_tweets.csv", header=None, names=["label", "tweet"])
    
    # Randomly select 1 tweet
    sampled_tweet = df["tweet"].sample(n=1, random_state=random.randint(0, 1000)).iloc[0]
    
    # Get prediction
    prediction = predict_sentiment([sampled_tweet])[0]
    
    return {
        "tweet": sampled_tweet,
        "prediction": prediction
    }
