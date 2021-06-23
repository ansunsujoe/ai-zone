import requests, json
from ai_zone.config import Config
from definitions import DATASET_DIR
import pandas as pd
from pathlib import Path

def read_intraday(c:Config, ticker):
    url = "https://alpha-vantage.p.rapidapi.com/query"
    querystring = {
        "interval": "5min",
        "function": "TIME_SERIES_INTRADAY",
        "symbol": ticker.upper(),
        "datatype": "json",
        "output_size": "compact"
    }
    headers = {
        'x-rapidapi-key': c.get("alphavantage_key"),
        'x-rapidapi-host': "alpha-vantage.p.rapidapi.com"
    }
    response = requests.request("GET", url, headers=headers, params=querystring)
    
    return response.json()["Time Series (5min)"].keys()

def read_daily(c:Config, ticker):
    # Make the request and get the response
    url = 'https://www.alphavantage.co/query'
    params = {
        "function": "TIME_SERIES_DAILY_ADJUSTED",
        "symbol": ticker,
        "apikey": c.get("alphavantage_key"),
        "outputsize": "full"
    }
    response = requests.request("GET", url, params=params)
    
    # Create the dataframe
    columns = ["Open", "High", "Low", "Close", "Adjusted Close", "Volume", 
               "Dividend Amount", "Split Coefficient"]
    array = []
    
    # Iterate through timeseries
    timeseries = response.json()["Time Series (Daily)"]
    for key in timeseries:
        day = timeseries[key]
        array.append([day["1. open"], day["2. high"], day["3. low"], day["4. close"],
                      day["5. adjusted close"], day["6. volume"], day["7. dividend amount"],
                      day["8. split coefficient"]])
    
    df = pd.DataFrame(data=array[::-1], columns=columns)
        
    return df


if __name__ == "__main__":
    # File paths needed
    tickers_path = Path("additional_files/tickers.txt")
    daily_data_folder = Path(DATASET_DIR) / "stock-data" / "daily"
    c = Config()
    
    # Define tickers
    tickers = []
    with open(tickers_path) as f:
        tickers = f.readlines()
    tickers = [x.strip() for x in tickers] 
    
    # Print
    for ticker in tickers:
        df = read_daily(c, ticker)
        ticker_file = daily_data_folder / f'{ticker}.csv'
        if not ticker_file.exists():
            print(f'Generating file for ticker {ticker}')
            df.to_csv(ticker_file)