# Stock Sentiment AI Trading Bot

This is a stock sentiment trading bot that uses news articles and GPT-3.5-turbo from OpenAI to analyze stock sentiment and make buy/sell decisions based on the sentiment score. It currently supports backtesting for Google, Apple, and Tesla.
Requirements

- Python 3.7 or higher
- pip for installing dependencies
- OpenAI API key for GPT-3.5-turbo
- NYTimes API key for fetching news articles
- Yahoo Finance for fetching stock data

## Installation

Clone this repository:

```bash

git clone https://github.com/yourusername/stock-sentiment-trading-bot.git
cd stock-sentiment-trading-bot

```

Install the required dependencies:
```
pip install -r requirements.txt
```
Set up environment variables for your OpenAI API key and NYTimes API key:

```

export OPENAI_API_KEY="your_openai_api_key"
export NYT_API_KEY="your_nyt_api_key"
```
## Usage
```
Open the bot.py file and ensure the stock symbols and company names are defined in the stocks dictionary.
```
```python

stocks = {
"Google": "GOOG",
"Apple": "AAPL",
"Tesla": "TSLA"
}
```
You can add or remove stock symbols and company names as needed.

Run the script:
```
python bot.py
```
The script will fetch historical news data and stock data, perform backtesting, and display the results. It will also cache the fetched data to speed up future runs.

## Customization

You can customize the bot's behavior by modifying the following parameters:

- sentiment_threshold: Adjust this value in the SentimentStrategy class to change the threshold for buy/sell decisions based on sentiment scores. A higher value will make the bot more selective in its buy/sell decisions, while a lower value will make it more responsive to sentiment changes.

```python

class SentimentStrategy(Strategy):
params = {'sentiment_threshold': 0.2} # ...
```
- cash: Adjust the initial cash available for the backtest in the Backtest instantiation.

```python

bt = Backtest(combined_df, strategy_class, cash=10000, commission=.002, exclusive_orders=True)
```
- commission: Adjust the commission per trade in the Backtest instantiation.

```python

bt = Backtest(combined_df, strategy_class, cash=10000, commission=.002, exclusive_orders=True)
```
## Troubleshooting

If you encounter any issues or errors, please double-check your API keys and ensure you have the latest versions of the required dependencies. If the issue persists, create an issue on this repository with a detailed description of the problem, and we will assist you.


This project is licensed under the MIT License. See the LICENSE file for details.
