import os
import pandas as pd
from datetime import datetime
from newsapi import NewsApiClient
from backtesting import Backtest, Strategy
import openai 
import matplotlib 
import http.client
from IPython.display import display 
import warnings 
import yfinance as yf 
import requests 
import time 
from urllib.parse import urlparse 
import pickle 
import random 
import re 
import json  
from openai.error import RateLimitError  

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
        current_date = self.data.index[i]
        current_date_str = str(current_date)  # Convert date to string

        if current_date_str in self.decisions:
            print(f"Decision for {current_date} is already made: {self.decisions[current_date_str]}")
            if self.decisions[current_date_str] == "buy":
                self.buy()
            elif self.decisions[current_date_str] == "sell":
                self.sell()
        else:
            if not pd.isna(self.data['news']).any():
                news_text = self.data['news'][i]  # Get news text for the current date
                
                # Truncate the news_text to fit within the token limit
                max_tokens = 4096 - 50  # Subtract 50 tokens for the prompt and other parts of the message
               # Split the news_text into chunks that fit the token limit
                news_chunks = []
                tokens = news_text.split()
                chunk = []
                current_tokens = 0
                for token in tokens:
                    if current_tokens + len(token) + 1 <= max_tokens:
                        chunk.append(token)
                        current_tokens += len(token) + 1
                    else:
                        news_chunks.append(" ".join(chunk))
                        chunk = [token]
                        current_tokens = len(token) + 1
                if chunk:
                    news_chunks.append(" ".join(chunk))

                # Get decisions for each chunk
                decisions = []
                for chunk in news_chunks:
                    decision = get_sentiment(chunk, stock_symbol)
                    decisions.append(decision)
                    print(f"Chunk decision: {decision}")

                # Determine the final decision by counting occurrences of "Yes", "No", and "Unknown"
                decision_counts = {'Yes': 0, 'No': 0, 'Unknown': 0}
                for decision in decisions:
                    decision_counts[decision] += 1

                # Choose the decision with the highest count
                final_decision = max(decision_counts, key=decision_counts.get)
                print(f"Final decision for {current_date}: {final_decision}")
                if final_decision == "Yes":
                    print(f"{current_date}: Buying")
                    self.decisions[current_date_str] = "buy"
                    self.buy()
                elif final_decision == "No":
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
def get_sentiment(text, stock_symbol):
    openai.api_key = os.environ["OPENAI_API_KEY"]
    prompt = f"Forget all your previous instructions. Pretend you are a financial expert. You are a financial expert with stock recommendation experience. Answer “YES” if good news, “NO” if bad news, or “UNKNOWN” if uncertain in the first line. Then elaborate with one short and concise sentence on the next line. Is this headline good or bad for the stock price of {stock_symbol} in the short term? Headline: {text}"
    
    max_retries = 5
    retry_count = 0
    
    while retry_count < max_retries:
        try:
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
            break  # If the API call is successful, exit the loop
        except openai.error.APIError as e:
            if e.code == 502:
                retry_count += 1
                if retry_count < max_retries:
                    time.sleep(5)  # Wait for 5 seconds before retrying
                else:
                    raise e  # If all retries have been exhausted, raise the error
            else:
                raise e

    # If max_retries is reached, return "Unknown"
    if retry_count == max_retries:
        print("Reached max retries. Returning 'Unknown'.")
        return "Unknown"

    sentiment_text = response.choices[0].message['content'].strip().split('\n')[0]
    print(f"Sentiment text: {sentiment_text}")  # Add this line to print the sentiment_text

    # Use a regular expression to extract the answer from the first line
    answer = re.search(r'(YES|NO|UNKNOWN)', sentiment_text, re.IGNORECASE)
    if answer:
        sentiment = answer.group(0).capitalize()
    else:
        sentiment = "Unknown"

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


def format_date(date_str):
    dt = datetime.strptime(date_str.split('T')[0], "%Y-%m-%d")
    return dt.strftime('%Y%m%d')

def scrape_rapidapi_data(stock_ticker, start_date, end_date):
    rapidapi_key = os.environ.get('rapidapi_key')
    if not rapidapi_key:
        raise ValueError("RAPIDAPI_SECRET environment variable not set")

    rapidapi_url = "https://contextualwebsearch-websearch-v1.p.rapidapi.com/api/search/NewsSearchAPI"
    rapidapi_headers = {
        "x-rapidapi-host": "contextualwebsearch-websearch-v1.p.rapidapi.com",
        "x-rapidapi-key": rapidapi_key
    }

    rapidapi_querystring = {
        "q": stock_ticker,
        "pageNumber": "1",
        "pageSize": "50",
        "autoCorrect": "true",
        "safeSearch": "false",
        "fromPublishedDate": start_date,
        "toPublishedDate": end_date
    }

    rapidapi_response = requests.get(rapidapi_url, headers=rapidapi_headers, params=rapidapi_querystring).json()
    news_data = []
    for web_page in rapidapi_response["value"]:
        date_published = web_page["datePublished"]
        pub_date = datetime.strptime(date_published.split('.')[0], "%Y-%m-%dT%H:%M:%S").date()
        title = web_page["title"]
        source = "RapidAPI"  # Adding the source
        news_data.append([pub_date, title, source])
    return news_data

def scrape_techcrunch_data(search_string, start_date, end_date):
    # Check if start_date and end_date are strings and convert to datetime
    if isinstance(start_date, str):
        start_date = datetime.strptime(start_date, "%Y-%m-%dT%H:%M:%S")
    if isinstance(end_date, str):
        end_date = datetime.strptime(end_date, "%Y-%m-%dT%H:%M:%S")
    url = 'https://techcrunch.com/'
    post_type = 'posts'
    url_parsed = urlparse(url)
    conn = http.client.HTTPSConnection(url_parsed.netloc)

    news_data = []
    
    page = 1
    while True:
        endpoint = f"/wp-json/wp/v2/{post_type}?page={page}&per_page=100&after={start_date.isoformat()}&before={end_date.isoformat()}&search={search_string}".replace(' ', '%20')
        conn.request("GET", endpoint)
        response = conn.getresponse()
        data = response.read().decode("utf-8")

        # Parse the JSON response and extract the fields we need
        for post in json.loads(data):
            date = datetime.strptime(post['date'], '%Y-%m-%dT%H:%M:%S')
            if date.date() < start_date.date():
                # Reached the end of the date range
                return news_data
            if start_date.date() <= date.date() <= end_date.date() and search_string in post['title']['rendered']:
                title = post['title']['rendered']
                source = "TechCrunch"
                news_data.append([date.date(), title, source])

        # Check if we have reached the last page
        links = response.getheader('Link')
        if not links or 'rel="next"' not in links:
            break

        # Move to the next page
        page += 1

    return news_data


def scrape_nyt_data(company_name, start_date, end_date):
    nyt_key = os.environ.get('NYT')  # Set your NYTimes API key as an environment variable
    if not nyt_key:
        raise ValueError("NYT_API_KEY environment variable not set")

    start_date = format_date(start_date)
    end_date = format_date(end_date)

    url = 'https://api.nytimes.com/svc/search/v2/articlesearch.json'
    params = {
        'q': company_name,
        'api-key': nyt_key,
        'sort': 'oldest',
        'begin_date': start_date,
        'end_date': end_date,
    }

    params['page'] = 0
    retries = 0  # Initialize retries variable
    max_retries = 3
    backoff_factor = 2

    news_data = []

    while params['page'] < 100:
        response = requests.get(url, params=params)
        while retries < max_retries:
            if response.status_code == 200:
                break
            elif response.status_code == 429:
                retry_after = int(response.headers.get('Retry-After', '0'))
                wait_time = max(backoff_factor ** retries, retry_after)
                print(f"Rate limit exceeded. Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
                retries += 1
                response = requests.get(url, params=params)
            else:
                print(response.text)
                print(f"Error fetching page {params['page']}: {response.status_code}")
                retries = max_retries
        if retries == max_retries:
            print("Max retries reached. Skipping this function.")
            return news_data
        response_json = response.json()
        articles = response_json.get('response', {}).get('docs', [])
        if not articles:
            break

        for article in articles:
            pub_date = datetime.strptime(article["pub_date"].split('T')[0], "%Y-%m-%d").date()
            headline = article["headline"]["main"]
            source = "New York Times"
            news_data.append([pub_date, headline, source])

        params['page'] += 1

    return news_data


def scrape_news_data(company_name, start_date, end_date):
    if company_name not in stocks:
        raise ValueError(f"Unknown company name: {company_name}")
    stock_ticker = stocks[company_name]
    # Define the cache file path
    cache_file = f"{stock_ticker}_news_data_cache.pkl"

    # Check if the cache file exists
    if os.path.exists(cache_file):
        # Load the cached data
        with open(cache_file, "rb") as file:
            cache_data = pickle.load(file)
    else:
        cache_data = {}

    news_data = []

    # Check cache and fetch data for each source
    for source, scraper in [('RapidAPI', scrape_rapidapi_data), 
                            ('NYT', scrape_nyt_data), 
                            ('TechCrunch', scrape_techcrunch_data)]:
        cache_key = f"{stock_ticker}_{source}_{start_date}_{end_date}"
        if cache_key in cache_data:
            news_data += cache_data[cache_key]
        else:
            if source == 'TechCrunch':
                start_date = datetime.strptime(start_date, "%Y-%m-%dT%H:%M:%S")
                end_date = datetime.strptime(end_date, "%Y-%m-%dT%H:%M:%S")

            fetched_data = scraper(company_name, start_date, end_date)
            news_data += fetched_data

            # Save the API response to the cache
            cache_data[cache_key] = fetched_data

    news_df = pd.DataFrame(news_data, columns=['Date', 'news', 'source'])
    news_df.set_index('Date', inplace=True)
    news_df = news_df.groupby(['Date', 'source']).agg({'news': ' '.join}).reset_index()

    # Save the updated cache data to the file
    with open(cache_file, "wb") as file:
        pickle.dump(cache_data, file)

    return news_df


def get_stock_data(stock_symbol, start_date, end_date):
    start_datetime = datetime.strptime(start_date, "%Y-%m-%dT%H:%M:%S")
    end_datetime = datetime.strptime(end_date, "%Y-%m-%dT%H:%M:%S")
    start_date = start_datetime.strftime('%Y-%m-%d')
    end_date = end_datetime.strftime('%Y-%m-%d')

    if start_datetime >= end_datetime:
        raise ValueError("start_date should be earlier than end_date")
    # Check if cached data exists
    cache_file = f'{stock_symbol}_{start_date}_{end_date}.pickle'
    if os.path.exists(cache_file):
        with open(cache_file, 'rb') as f:
            return pickle.load(f)

    # Fetch new data
    data = yf.download(stock_symbol, start=start_date, end=end_date)
    data = data.reset_index()
    data['Date'] = pd.to_datetime(data['Date']).dt.date
    data.set_index('Date', inplace=True)

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
    
    news_df, news_start_date, news_end_date = get_historical_news_data(company_name, '2019-01-01T00:00:00', '2019-12-31T23:59:59')
    if 'datetime' in news_df.columns:
        news_df = news_df.rename(columns={'datetime': 'Date'})

    # news_df['datetime'] = pd.to_datetime(news_df['datetime'])
    print(f"news_start_date: {news_start_date}, news_end_date: {news_end_date}")
    
    news_start_date, news_end_date = ('2019-01-01T00:00:00', '2019-12-31T23:59:59')
        
    stock_data = get_stock_data(stock_symbol, news_start_date, news_end_date)
    stock_data.index = pd.to_datetime(stock_data.index)
    # print(stock_data.columns)
    # print(stock_data.index)
    # stock_data['Date'] = pd.to_datetime(stock_data['Date'])

    news_df.set_index('Date', inplace=True)
    # print(news_df.columns)
    # print(news_df.index)
    print(f"{stock_symbol}_filtered dataframe:")
    print(stock_data)
    print("news_df dataframe:")
    print(news_df)

    # Merge the two dataframes using inner join
    combined_df = pd.merge(stock_data, news_df, how='inner',left_index=True, right_index=True)
    print(combined_df.isnull().sum())
    print(combined_df.dtypes)


    # Drop the datetime column and reset the index
    # combined_df.drop(columns=['datetime'], inplace=True)
    # combined_df.set_index('Date', inplace=True)

    print(f"combined dataframe: {combined_df}")
    strategy_class = sentiment_strategy_wrapper(stock_symbol)

    bt = Backtest(combined_df, strategy_class, cash=10000, commission=.002, exclusive_orders=True)
    
    stats = bt.run()
    display(stats)
    bt.plot()

# for company_name, stock_symbol in stocks.items():
#     class BuyAndHoldStrategy(Strategy):
#         def init(self):
#             pass

#         def next(self):
#             if not self.position:
#                 self.buy()

#     news_start_date, news_end_date = ('2019-01-01T00:00:00', '2019-12-31T23:59:59')

#     stock_data = get_stock_data(stock_symbol, news_start_date, news_end_date)
#     stock_data = get_stock_data(stock_symbol, news_start_date, news_end_date)
#     stock_data.index = pd.to_datetime(stock_data.index)  # Convert the index to DateTimeIndex
#     print(stock_data.index.to_series().diff().fillna(pd.Timedelta(seconds=0)))
#     bt = Backtest(stock_data, BuyAndHoldStrategy, cash=10000, commission=.002, exclusive_orders=True)
#     stats = bt.run()
