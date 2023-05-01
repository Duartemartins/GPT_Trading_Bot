from yahooquery import Ticker

# Create a Ticker object for the desired ticker symbol
goog = Ticker('goog')

# Fetch financial data
financial_data = goog.financial_data
news = goog.technical_insights
# Print the financial data
print(financial_data)