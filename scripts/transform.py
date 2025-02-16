
def transform_data(df):
    df["date"] = pd.to_datetime(df["date"])
    df.set_index("date", inplace=True)
    return df

def transform_data_yearly(df):
    df["date"] = pd.to_datetime(df["date"])
    df.set_index("date", inplace=True)
    return df


def select_data(df, start, end):
    df = df.loc[start:end]
    return df

def merge_data(df_brent, df_opec_production, df_no_opec_production, df_dxy):
    df_brent_copy = df_brent.copy()
    df_opec_production_copy = df_opec_production.copy()
    df_no_opec_production_copy = df_no_opec_production.copy()
    df_dxy_copy = df_dxy.copy()

    df_brent_copy = transform_data(df_brent_copy)
    df_opec_production_copy = transform_data(df_opec_production_copy)
    df_no_opec_production_copy = transform_data(df_no_opec_production_copy)
    df_dxy_copy = transform_data(df_dxy_copy)

    df_brent_copy = select_data(df_brent_copy, "2000-01-01", "2025-02-01")
    df_opec_production_copy = select_data(df_opec_production_copy, "2000-01-01", "2025-02-01")
    df_no_opec_production_copy = select_data(df_no_opec_production_copy, "2000-01-01", "2025-02-01")
    df_dxy_copy = select_data(df_dxy_copy, "2000-01-01", "2025-02-01")

    df_brent_copy = transform_data_yearly(df_brent_copy)
    df_opec_production_copy = transform_data_yearly(df_opec_production_copy)
    df_no_opec_production_copy = transform_data_yearly(df_no_opec_production_copy)
    df_dxy_copy = transform_data_yearly(df_dxy_copy)

    df_brent_copy = df_brent_copy.resample("D").interpolate()
    df_opec_production_copy = df_opec_production_copy.resample("D").interpolate()
    df_no_opec_production_copy = df_no_opec_production_copy.resample("D").interpolate()
    df_dxy_copy = df_dxy_copy.resample("D").interpolate()

    df_merged = pd.concat(
        [df_brent_copy, df_opec_production_copy, df_no_opec_production_copy, df_dxy_copy],
        axis=1,  # Junta ao longo das colunas
        join="outer",  # Mantém todos os índices, preenchendo com NaN se necessário
    )

    df_merged.to_csv("data/processed/df_merged.csv")

if __name__ == "__main__":
    df_brent = pd.read_csv("data/processed/brent.csv")
    df_opec_production = pd.read_csv("data/processed/opec_total_production.csv")
    df_no_opec_production = pd.read_csv("data/processed/no_opec_total_production.csv")
    df_dxy = pd.read_csv("data/processed/dxy.csv")

    merge_data(df_brent, df_opec_production, df_no_opec_production, df_dxy)

    