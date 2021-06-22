import requests, json
from ai_zone.config import Config
from definitions import CONFIG_PATH
import pandas as pd

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
    url = 'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol=IBM&apikey=demo'
    params = {
        "function": "TIME_SERIES_DAILY_ADJUSTED",
        "symbol": "IBM",
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
    # Read json file
    data = json.loads(open(CONFIG_PATH, "r").read())
    
    # Config
    c = Config()
    c.add("alphavantage_key", data["alphavantage"]["apiKey"])
    
    # Print
    print(read_daily(c, "MSFT"))