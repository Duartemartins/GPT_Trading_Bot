import os
import time
import pickle
import requests
import pandas as pd
import json
import http.client
from urllib.parse import urlparse
from datetime import datetime

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

    news_df = pd.DataFrame(news_data, columns=['datetime', 'news', 'source'])
    news_df.set_index('datetime', inplace=True)
    news_df = news_df.groupby(['datetime', 'source']).agg({'news': ' '.join}).reset_index()

    # Save the updated cache data to the file
    with open(cache_file, "wb") as file:
        pickle.dump(cache_data, file)

    return news_df

stocks = {
    "Google": "GOOG",
    "Apple": "AAPL",
    "Tesla": "TSLA"
}

print(scrape_news_data('Google', '2019-01-01T00:00:00', '2019-12-31T23:59:59'))