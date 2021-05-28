import requests

url = "https://apidojo-yahoo-finance-v1.p.rapidapi.com/stock/v3/get-historical-data"

querystring = {"symbol":"AMZN","region":"US"}

headers = {
    'x-rapidapi-key': "07dc42e472msh9e9c34431ad54f0p1cc110jsn709b2fde1d1c",
    'x-rapidapi-host': "apidojo-yahoo-finance-v1.p.rapidapi.com"
    }

response = requests.request("GET", url, headers=headers, params=querystring)

print(response.text)
# NL9B3FEJRMVDW0XH