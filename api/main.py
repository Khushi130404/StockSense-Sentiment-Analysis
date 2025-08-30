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
def sentiment_trigger():
    """
    Randomly selects one tweet from dataset and returns its sentiment prediction.
    """
    import pandas as pd
    import random

    # Load dataset
    df = pd.read_csv("data/aapl_tweets.csv", header=None, names=["label", "tweet"])
    
    # Randomly select 1 tweet
    sampled_tweet = df["tweet"].sample(n=1, random_state=random.randint(0, 1000)).iloc[0]
    
    # Get prediction (returns a list, take first element)
    prediction = predict_sentiment([sampled_tweet])[0]
    
    return {
        "tweet": sampled_tweet,
        "prediction": prediction
    }
