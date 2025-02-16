import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Carregar os dados
df_merged = pd.read_csv("../data/processed/df_merged.csv", index_col=0, parse_dates=True)

# Criar variáveis de defasagem (lags) para capturar dependências temporais
for lag in [1, 2, 3, 5, 7, 14]:
    df_merged[f"lag_{lag}"] = df_merged["brent_price"].shift(lag)

# Criar médias móveis para capturar tendências de curto prazo
df_merged["rolling_mean_7"] = df_merged["brent_price"].rolling(7).mean()

# Criar variações percentuais para capturar mudanças relativas
df_merged["pct_change_1"] = df_merged["brent_price"].pct_change(1)
df_merged["dxy_change"] = df_merged["dxy_value_close"].pct_change(1)

# Criar variável para indicar se o dia é um final de semana
df_merged["is_weekend"] = df_merged.index.weekday >= 5

# Remover valores NaN gerados pelas operações anteriores
df_merged = df_merged.dropna()

# Definir variáveis independentes (X) e variável alvo (y)
features = [col for col in df_merged.columns if col != "brent_price"]
X = df_merged[features]
y = df_merged["brent_price"]

# Dividir os dados em conjunto de treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Definir modelo XGBoost e grade de hiperparâmetros para otimização
param_grid = {
    "n_estimators": [100, 200],  # Número de árvores
    "max_depth": [3, 5, 7],  # Profundidade máxima das árvores
    "learning_rate": [0.01, 0.05, 0.1],  # Taxa de aprendizado
    "subsample": [0.8, 1.0],  # Fração dos dados usada em cada árvore
}

xgb_model = xgb.XGBRegressor(objective="reg:squarederror", eval_metric="rmse")

# Otimização de hiperparâmetros usando GridSearchCV
grid_search = GridSearchCV(xgb_model, param_grid, cv=3, scoring="neg_mean_squared_error", verbose=1)
grid_search.fit(X_train, y_train)

# Selecionar o melhor modelo ajustado
tuned_model = grid_search.best_estimator_

# Fazer previsões no conjunto de teste
y_pred = tuned_model.predict(X_test)

# Calcular métricas de avaliação do modelo
mae = mean_absolute_error(y_test, y_pred)  # Erro absoluto médio
mse = mean_squared_error(y_test, y_pred)  # Erro quadrático médio
rmse = np.sqrt(mse)  # Raiz do erro quadrático médio
mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100  # Erro percentual absoluto médio

# Exibir os resultados das métricas
print(f"MAE: {mae:.2f}")
print(f"MSE: {mse:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"MAPE: {mape:.2f}%")

# Salvar as métricas em um arquivo de texto
with open("../data/processed/xgboost_metrics.txt", "w") as f:
    f.write(f"MAE: {mae:.2f}\n")
    f.write(f"MSE: {mse:.2f}\n")
    f.write(f"RMSE: {rmse:.2f}\n")
    f.write(f"MAPE: {mape:.2f}%\n")

print("Métricas salvas em 'data/processed/xgboost_metrics.txt'")

# Salvar o modelo ajustado
tuned_model.save_model("../models/modelo_brent.json")
print("Modelo salvo em 'models/modelo?brent.json'")

#salvando o resultado do modelo para ser usado no dashboard
df_merged["brent_price_pred"] = tuned_model.predict(X)
df_merged.to_csv("../data/processed/predicted.csv")
print("Previsões salvas em 'data/processed/predicted.csv'")