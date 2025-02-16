import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def train_xgboost(df):
    df_merged = pd.read_csv("data/df_merged.csv")
    df_merged["date"] = pd.to_datetime(df_merged["date"])
    df_merged.set_index("date", inplace=True)
    df_merged["target"] = df_merged["brent_price"].shift(-1)
    df_merged.dropna(inplace=True)

    X = df_merged.drop(columns=["target"])
    y = df_merged["target"]

    model = xgb.XGBRegressor(objective="reg:squarederror", n_estimators=100)
    model.fit(X, y)

    return model

# Salva o modelo localmente
def save_model_local(model):
    model.save_model("modelo_brent.json")

# Salva o modelo no S3
def save_model_s3(model):
    model.save_model("s3://datalake-igti-2021-06-14/modelo_brent.json")

# Carrega o modelo localmente
def load_model_local():
    model = xgb.XGBRegressor()
    model.load_model("modelo_brent.json")
    return

# Carrega o modelo do S3
def load_model_s3():
    model = xgb.XGBRegressor()
    model.load_model("s3://datalake-igti-2021-06-14/modelo_brent.json")
    return

