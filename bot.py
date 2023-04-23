import os
import pandas as pd
from datetime import datetime
from newsapi import NewsApiClient
from backtesting import Backtest
from backtesting import Strategy
import openai
import matplotlib
from IPython.display import display
import warnings
import yfinance as yf
import requests
import time
import os
import requests
import pandas as pd
import os.path
import pickle
import random
import re
import json

# Ignore warnings related to datetime and deprecation
warnings.filterwarnings("ignore", category=UserWarning, message="DatetimeFormatter scales now only accept a single format.")
warnings.filterwarnings("ignore", category=DeprecationWarning, message="DatetimeFormatter scales now only accept a single format.")

# Function to truncate text to a specified number of tokens
def truncate_text(text, max_tokens):
    tokens = text.split()
    truncated_tokens = tokens[:max_tokens]
    return " ".join(truncated_tokens)

# Define the SentimentStrategy class
class SentimentStrategy(Strategy):
    params = {'sentiment_threshold': 0.2}

    def __init__(self, broker, data, params, stock_symbol):
        super().__init__(broker, data, params)
        self.stock_symbol = stock_symbol
        self.news = {}
        self.decisions = self.load_decisions()

    def init(self):
        pass

    def next(self):
        i = len(self.data) - 1  # Get the index of the most recent data point
        current_date = self.data.index[i].date()
        current_date_str = str(current_date)  # Convert date to string

        if current_date_str in self.decisions:
            print(f"Decision for {current_date} is already made: {self.decisions[current_date_str]}")
            if self.decisions[current_date_str] == "buy":
                self.sell()
            elif self.decisions[current_date_str] == "sell":
                self.buy()
        else:
            if not pd.isna(self.data['news']).any():
                news_text = self.data['news'][i]  # Get news text for the current date
                
                # Truncate the news_text to fit within the token limit
                max_tokens = 4096 - 50  # Subtract 50 tokens for the prompt and other parts of the message
                truncated_news_text = truncate_text(news_text, max_tokens)

                sentiment = get_sentiment(truncated_news_text)
                print(f"Sentiment: {sentiment}")

                if sentiment > self.params['sentiment_threshold']:
                    print(f"{current_date}: Buying")
                    self.decisions[current_date_str] = "buy"
                    self.buy()
                elif sentiment < -self.params['sentiment_threshold']:
                    print(f"{current_date}: Selling")
                    self.decisions[current_date_str] = "sell"
                    self.sell()
                else:
                    print(f"{current_date}: No action")
                    self.decisions[current_date_str] = "no action"

                self.save_decisions()

    # Load decisions from a JSON file
    def load_decisions(self):
        filename = f'decisions_{self.stock_symbol}.json'
        if os.path.exists(filename) and os.path.getsize(filename) > 0:
            with open(filename, 'r') as f:
                return json.load(f)
        return {}

    # Add a decision to the decisions dictionary and save it
    def add_decision(self, date, decision):
        self.decisions[date] = decision
        self.save_decisions()

    # Save decisions to a JSON file
    def save_decisions(self):
        filename = f'decisions_{self.stock_symbol}.json'
        with open(filename, 'w') as f:
            json.dump(self.decisions, f, indent=4)

    # Get a decision by date
    def get_decision_by_date(self, date):
        return next((decision for decision in self.decisions if decision[0] == str(date)), None)

# Function to create a wrapped SentimentStrategy class for a specific stock symbol
def sentiment_strategy_wrapper(stock_symbol):
    class WrappedSentimentStrategy(SentimentStrategy):
        def __init__(self, broker, data, params):
            super().__init__(broker, data, params, stock_symbol)
    return WrappedSentimentStrategy

# Function to get sentiment using OpenAI GPT-3.5
def get_sentiment(text):
    openai.api_key = os.environ["OPENAI_API_KEY"]
    prompt = f"Forget all your previous instructions. Pretend you are a financial expert. You are a financial expert with stock recommendation experience. Rate the sentiment of \"{text}\" on a scale from -1 (very negative) to 1 (very positive)."

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a financial expert with stock recommendation experience."},
            {"role": "user", "content": prompt},
        ],
        max_tokens=50,
        n=1,
        temperature=0.5,
    )

    sentiment_text = response.choices[0].message['content'].strip()
    print(f"Sentiment text: {sentiment_text}")  # Add this line to print the sentiment_text
    sentiment_numbers = re.findall(r"[-+]?\d*\.\d+|[-+]?\d+", sentiment_text)
    
    if sentiment_numbers:
        sentiment = float(sentiment_numbers[0])
    else:
        sentiment = 0.0

    return sentiment


def get_historical_news_data(stock, start_date, end_date):
    # Define the cache file path
    cache_file = f"{stock}_news_data_cache.pkl"

    # Check if the cache file exists
    if os.path.exists(cache_file):
        # Load the cached data
        with open(cache_file, "rb") as file:
            cache_data = pickle.load(file)
    else:
        cache_data = {}

    cache_key = f"{stock}_{start_date}_{end_date}"
    if cache_key in cache_data:
        print("Data found in cache.")
        news_df = cache_data[cache_key]
    else:
        print("Data not found in cache. Scraping...")
        news_df = scrape_news_data(stock, start_date, end_date)
        cache_data[cache_key] = news_df
        with open(cache_file, "wb") as file:
            pickle.dump(cache_data, file)

    news_start_date = news_df.index.min()
    news_end_date = news_df.index.max()

    return news_df, news_start_date, news_end_date



def scrape_news_data(stock, start_date, end_date):
    # Define the cache file path
    cache_file = f"{stock}_news_data_cache.pkl"

    # Check if the cache file exists
    if os.path.exists(cache_file):
        # Load the cached data
        with open(cache_file, "rb") as file:
            cache_data = pickle.load(file)
    else:
        cache_data = {}

    api_key = os.environ.get('NYT')  # Set your NYTimes API key as an environment variable
    if not api_key:
        raise ValueError("NYTIMES_API_KEY environment variable not set")

    url = 'https://api.nytimes.com/svc/search/v2/articlesearch.json'
    params = {
        'q': stock,
        'api-key': api_key,
        'sort': 'oldest',
        'fq': f'organizations.contains:("{stock}") AND pub_date:[{start_date}T00:00:00Z TO {end_date}T23:59:59Z]'
    }

    news_data = []
    params['page'] = 0
    retries = 0  # Initialize retries variable
    max_retries = 10
    backoff_factor = 2

    while params['page'] < 100:
        while retries < max_retries:
            response = requests.get(url, params=params)

            if response.status_code == 200:
                break
            elif response.status_code == 429:
                retry_after = int(response.headers.get('Retry-After', '0'))
                wait_time = max(backoff_factor ** retries, retry_after)
                print(f"Rate limit exceeded. Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
                retries += 1
            else:
                print(f"Error fetching page {params['page']}: {response.status_code}")
                retries = max_retries
                break

        if retries == max_retries:
            print("Max retries reached. Aborting.")
            break

        response_json = response.json()
        articles = response_json.get('response', {}).get('docs', [])
        if not articles:
            break

        for article in articles:
            article_data = [article['pub_date'], article['headline']['main']]
            if article_data not in news_data:  # Check if the data is not already in the DataFrame
                news_data.append(article_data)
                print(article_data)

        params['page'] += 1

    news_df = pd.DataFrame(news_data, columns=['datetime', 'news'])
    news_df['datetime'] = pd.to_datetime(news_df['datetime'], utc=True).dt.date  # Convert to date only (no time)
    news_df['datetime'] = pd.to_datetime(news_df['datetime'])  # Convert back to datetime

    news_df['datetime'] = news_df['datetime'].dt.tz_localize(None)  # Remove timezone information
    news_df.set_index('datetime', inplace=True)
    # Save the API response to the cache
    cache_key = f"{stock}_{start_date}_{end_date}"
    cache_data[cache_key] = news_df

    # Save the updated cache data to the file
    with open(cache_file, "wb") as file:
        pickle.dump(cache_data, file)

    return news_df


def get_stock_data(symbol, start_date, end_date):
    # Check if cached data exists
    cache_file = f'{symbol}_{start_date}_{end_date}.pickle'
    if os.path.exists(cache_file):
        with open(cache_file, 'rb') as f:
            return pickle.load(f)

    # Fetch new data
    data = yf.download(symbol, start=start_date, end=end_date)

    # Cache the data
    with open(cache_file, 'wb') as f:
        pickle.dump(data, f)

    return data


# Define the stocks to analyze
stocks = {
    "Google": "GOOG",
    "Apple": "AAPL",
    "Tesla": "TSLA"
}

# Iterate over each stock and run the backtest
for company_name, stock_symbol in stocks.items():
    print(f"Running backtest for {company_name} ({stock_symbol})")
    
    news_df, news_start_date, news_end_date = get_historical_news_data(company_name, "2019-01-01", "2019-12-31")
    stock_data = get_stock_data(stock_symbol, news_start_date, news_end_date)
    
    display(news_df)
    print(f"{stock_symbol}_filtered dataframe:")
    print(stock_data)
    print("news_df dataframe:")
    print(news_df)

    combined_df = stock_data.join(news_df, how='inner')
    
    strategy_class = sentiment_strategy_wrapper(stock_symbol)
    bt = Backtest(combined_df, strategy_class, cash=10000, commission=.002, exclusive_orders=True)
    
    stats = bt.run()
    display(stats)
    bt.plot()
    