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
import time
from openai.error import RateLimitError

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
    start_date = datetime.strptime(start_date, "%Y-%m-%dT%H:%M:%S").strftime('%Y%m%d')
    end_date = datetime.strptime(end_date, "%Y-%m-%dT%H:%M:%S").strftime('%Y%m%d')
    params = {
    'q': stock,
    'api-key': api_key,
    'sort': 'oldest',
    'begin_date': start_date,
    'end_date': end_date,
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
                print(response.text)
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
    
    # Aggregate headlines by date
    news_df = news_df.groupby('datetime').agg({'news': ' '.join})
    
    # Save the API response to the cache
    cache_key = f"{stock}_{start_date}_{end_date}"
    cache_data[cache_key] = news_df

    # Save the updated cache data to the file
    with open(cache_file, "wb") as file:
        pickle.dump(cache_data, file)

    return news_df

print(scrape_news_data('Google', '2019-01-01T00:00:00', '2019-12-31T23:59:59'))
