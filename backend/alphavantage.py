import requests

url = "https://alpha-vantage.p.rapidapi.com/query"

querystring = {"interval":"5min","function":"TIME_SERIES_INTRADAY","symbol":"MSFT","datatype":"json","output_size":"compact"}

headers = {
    'x-rapidapi-key': "07dc42e472msh9e9c34431ad54f0p1cc110jsn709b2fde1d1c",
    'x-rapidapi-host': "alpha-vantage.p.rapidapi.com"
    }

response = requests.request("GET", url, headers=headers, params=querystring)

print(response.text)