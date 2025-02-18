# Previsão do Preço do Petróleo Brent

## Sobre o Projeto

### O problema: Você foi contratado(a) para uma consultoria, e seu trabalho envolve analisar os dados de preço do petróleo brent, que pode ser encontrado no site do ipea. Essa base de dados histórica envolve duas colunas: data e preço (em dólares).
### Um grande cliente do segmento pediu para que a consultoria desenvolvesse um dashboard interativo para gerar insights relevantes para tomada de decisão. Além disso, solicitaram que fosse desenvolvido um modelo de Machine Learning para fazer o forecasting do preço do petróleo.

### Objetivos:
#### 🔹 Criar um dashboard interativo com ferramentas à sua escolha.
#### 🔹 Seu dashboard deve fazer parte de um storytelling que traga insights relevantes sobre a variação do preço do petróleo.
#### 🔹 Criar um modelo de Machine Learning que faça a previsão do preço do petróleo diariamente.O modelo deve estar no storytelling e deve conter o código que você trabalhou e as performances do modelo.
#### 🔹 Criar um plano para fazer o deploy em produção do modelo, com as ferramentas que são necessárias.
#### 🔹 Faça um MVP do seu modelo em produção utilizando o Streamlit.
#### 🔹 Elabore um vídeo de até 5 (cinco) minutos explicando todo o desenvolvimento do seu projeto.


## Funcionalidades

#### 🔹 Dashboard interativo com gráficos e análises do preço do Brent.
#### 🔹 Análise de fatores como produção da OPEP, taxa de câmbio do dólar e eventos geopolíticos.
#### 🔹 Modelo de previsão utilizando XGBoost, comparado com SARIMAX.
#### 🔹 Atualização automática dos dados e re-treinamento semanal do modelo.
#### 🔹 Deploy no Streamlit Cloud com integração AWS S3 para armazenamento do modelo.
#### 🔹 Vídeo explicativo na última aba do dashboard.

## Tecnologias Utilizadas

#### 🔹 Linguagens: Python

#### 🔹 Bibliotecas: pandas, requests, os, sys, json, BeautifulSoup, yfinance, numpy, matplotlib, seaborn, plotly, xgboost, sklearn, streamlit, datetime (vide requirements.txt)

#### 🔹 Infraestrutura: Streamlit Cloud, git

#### 🔹Automação: CRON Jobs, GitHub Actions (implementaçẽs futuras)


## Estrutura do Projeto e Arquivos

```bash
├── data/                   # Dados processados e originais
│   ├── raw/                # Dados brutos
│   ├── processed/          # Dados limpos e transformados
│
├── models/                 # Modelos de Machine Learning
│   ├── modelo_brent.json   # Modelo XGBoost salvo
│   ├── metrics/            # Arquivos com métricas de avaliação
│       ├── metrics.json    # Métricas de avaliação do modelo
│
├── scripts/                # Scripts de processamento e atualização
│   ├── update_model.py     # Atualiza dados e re-treina modelo
│
├── dashboard.py            # Código principal do Streamlit
├── requirements.txt        # Dependências do projeto
├── README.md               # Documentação do projeto
```

## Como Executar o Projeto

### 🔹 Passo 1: Clone o Repositório

```bash
git clone https://github.com/seu_usuario/previsao-brent.git
cd previsao-brent
```

### 🔹 Passo 2: Crie um Ambiente Virtual e Instale as Dependências
```bash
python -m venv venv
source venv/bin/activate  # Linux/macOS
venv\Scripts\activate  # Windows
pip install -r requirements.txt
```

### 🔹 Passo 3: Execute o Dashboard no Streamlit
```bash
streamlit run dashboard.py
```

## Deploy e Automação

### O projeto está hospedado no Streamlit Cloud e possui um pipeline de automação - Link do dashboard :[https://challenge04cazuza.streamlit.app/]

#### 1. Coleta de dados do IPEA, EIA e FRED semanalmente.

#### 2. Treinamento automático do modelo XGBoost.

#### 3. Armazenamento do modelo em cloud storage.

#### 4. Atualização do dashboard automaticamente via Streamlit Cloud.

## Vídeo Explicativo

### Acesse a aba "Vídeo" no dashboard para assistir à apresentação do projeto.

## Próximos Passos

Automatizar toda a pipeline de ETL

Refatorar o código para melhorar a manutenção e escalabilidade.

Ligar o dashboard a uma base de dados para armazenar métricas e logs.

Aprimorar o modelo com novos indicadores macroeconômicos.

Automatizar o deploy com GitHub Actions.

Manter o modelo atualizado com novos dados e eventos.

Testar outras arquiteturas, como LSTM e Prophet.

Melhorar a interface do dashboard com mais interatividade.


📩 Contato: [janainamartinscazuza@gmail.com] | 🌐 GitHub: [https://github.com/janainacazuza]
LinkedIn: [https://www.linkedin.com/in/janainacazuza/]
```