import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def merging_data(df1, df2, df3, df4):
    df = pd.concat(
    [df1, df2, df3, df4],
    axis=1,  # Junta ao longo das colunas
    join="outer")

    df_merged = df[['brent_price', 'production_value_opec', 'production_value_no_opec', 'dxy_value_close']].dropna()

def correlation_matrix(df):
    corr = df.corr()
    sns.heatmap(corr, cmap='coolwarm', fmt=".2f", annot=True)
    plt.show()

def plot_heatmap(df):
    plt.figure(figsize=(12, 8))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
    plt.show()
