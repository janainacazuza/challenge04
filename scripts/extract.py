import sys
import os
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
env_path = os.path.join(BASE_DIR, ".env")
with open(env_path) as f:
    for line in f:
        key, value = line.strip().split("=", 1)
        os.environ[key] = value
import pandas as pd
import requests
from bs4 import BeautifulSoup
import yfinance as yf



def extract_ipeadata():
    url = "http://www.ipeadata.gov.br/ExibeSerie.aspx?serid=1650971490&module=M"
    response = requests.get(url)
    soup = BeautifulSoup(response.text, "html.parser")
    table = soup.find("table", {"class": "dxgvTable"})

    data = []
    for line in table.find_all("tr")[1:]:
        columns = line.find_all("td")
        if len(columns) >= 2:
            date = columns[0].text.strip()
            value = columns[1].text.strip()
            data.append([date, value])

    df_ipea = pd.DataFrame(data)
    df_ipea = df_ipea.drop([0, 1]).reset_index(drop=True)
    df_ipea.columns = ["date", "brent_price"]
    df_ipea["date"] = pd.to_datetime(df_ipea["date"], errors="coerce")
    df_ipea["date"] = df_ipea["date"].dt.strftime("%Y-%m-%d")
    df_ipea["brent_price"] = df_ipea["brent_price"].str.replace(",", ".")
    df_ipea["brent_price"] = pd.to_numeric(df_ipea["brent_price"], errors="coerce")
    df_ipea = df_ipea.sort_values(by="date", ascending=True)
    df_ipea.set_index("date", inplace=True)
    df_ipea.to_csv("/home/naina/challenge04/data/processed/data_brent_clean.csv")


def extract_production_oil():
    api_key = os.getenv("API_KEY")
    base_url = "https://api.eia.gov/v2/steo/data/"
    params = {
        "api_key": api_key,
        "frequency": "monthly",
        "data[0]": "value",
        "facets[seriesId][]": ["COPR_NONOPEC", "COPR_OPEC"],
        "sort[0][column]": "period",
        "sort[0][direction]": "desc",
        "start": "2000-01",
        "end": "2025-02",
        "length": 5000,
    }

    all_data = []
    offset = 0
    while True:
        params["offset"] = offset
        response = requests.get(base_url, params=params)
        data = response.json()
        if "response" in data and "data" in data["response"]:
            records = data["response"]["data"]
            all_data.extend(records)
            if len(records) < params["length"]:
                break
            offset += params["length"]
        else:
            break

    df_production = pd.DataFrame(all_data)
    df_production.columns = ["date", "origin_oil", "description", "production_value", "unit"]
    df_production = df_production.drop(columns=["description", "unit"])
    df_production["date"] = pd.to_datetime(df_production["date"], errors="coerce")
    df_production["date"] = df_production["date"].dt.strftime("%Y-%m-%d")
    df_production["production_value"] = pd.to_numeric(df_production["production_value"], errors="coerce")
    df_production_opec_total = (
        df_production[df_production["origin_oil"] == "COPR_OPEC"]
        .rename(columns={"production_value": "production_value_opec"})
        .drop(columns=["origin_oil"])
        .sort_values(by="date", ascending=True)
        .reset_index()
    )
    df_production_no_opec_total = (
        df_production[df_production["origin_oil"] == "COPR_NONOPEC"]
        .rename(columns={"production_value": "production_value_no_opec"})
        .drop(columns=["origin_oil"])
        .sort_values(by="date", ascending=True)
        .reset_index()
    )

    df_production_no_opec_total["date"] = pd.to_datetime(df_production_no_opec_total["date"])
    df_production_opec_total["date"] = pd.to_datetime(df_production_opec_total["date"])
    df_production_no_opec_total.set_index("date", inplace=True)
    df_production_opec_total.set_index("date", inplace=True)
    df_production_no_opec_total.drop(columns=["index"], inplace=True)
    df_production_opec_total.drop(columns=["index"], inplace=True)
    df_production_opec_total_day = df_production_opec_total.resample("D").ffill()
    df_production_no_opec_total_day = df_production_no_opec_total.resample("D").ffill()

    df_production_opec_total_day.to_csv(
        "/home/naina/challenge04/data/processed/opec_total_production.csv"
    )
    df_production_no_opec_total_day.to_csv(
        "/home/naina/challenge04/data/processed/no_opec_total_production.csv"
    )


def extract_dxy():
    dxy = yf.download("DX-Y.NYB", start="2000-01-01", end="2025-02-01", interval="1d")
    df_dxy = dxy.loc[:, [("Close", "DX-Y.NYB")]].reset_index()
    df_dxy.columns = ["date", "dxy_value_close"]
    df_dxy.set_index("date", inplace=True)
    df_dxy.to_csv("/home/naina/challenge04/data/processed/dxy.csv")


if __name__ == "__main__":
    extract_ipeadata()
    extract_production_oil()
    extract_dxy()
