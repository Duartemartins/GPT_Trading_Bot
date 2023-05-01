import requests
import os

API_KEY = os.environ.get('PERIGON')
url = f"https://api.goperigon.com/v1/all?companySymbol=AAPL&from=2019-01-01&sortBy=date&apiKey={API_KEY}"

resp = requests.get(url)
article = resp.json()["articles"][0]

print(article["title"])
print(article["pubDate"])
