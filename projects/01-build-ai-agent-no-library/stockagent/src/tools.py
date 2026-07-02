def get_stock_price(ticker: str) -> str:
    """
    Get the current stock price for a given ticker symbol.

    In production, this would call a real stock API like Alpha Vantage,
    Yahoo Finance, or any financial data provider. For this demo, we
    use hardcoded data to keep things simple and self-contained.
    """
    fake_stocks = {
        "AAPL": "187.32 USD",
        "GOOGL": "142.58 USD",
        "MSFT": "378.91 USD",
        "TSLA": "248.50 USD",
        "AMZN": "178.22 USD",
    }

    return fake_stocks.get(
        ticker.upper(),
        f"No stock price found for {ticker}",
    )