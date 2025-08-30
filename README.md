# StockSense-Sentiment-Analysis

python -m venv venv
venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
python src/model.py
uvicorn api.main:app --reload
http://127.0.0.1:8000/docs
