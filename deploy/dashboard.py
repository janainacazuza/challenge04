import pandas as pd
import requests
import os
import sys
import json
#BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
#sys.path.append(BASE_DIR)
#from scripts import extract
from bs4 import BeautifulSoup
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from datetime import date, datetime
from scipy.signal import find_peaks
import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error
import streamlit as st
from datetime import datetime, timedelta

# Configurações do Streamlit

st.set_page_config(page_title="Análise do Brent", layout="wide", initial_sidebar_state="auto", menu_items=None)
st.title(" **Análise do Preço do Petróleo Brent** ")

aba1, aba2, aba3, aba4, aba5, aba6, aba7, aba8, aba9 = st.tabs(
    [
        "Pétroleo Brent",
        "Produção de Petróleo",
        "Dólar e Petróleo",
        "Correlações",
        "Insights",
        "Plano de deploy",
        "Modelo Preditivo - IA",
        "Previsões do Brent",
        "Vídeo",
    ])

# Carregando os dados
df_brent = pd.read_csv("data/processed/data_brent_clean.csv")
df_brent_y = pd.read_csv("data/processed/data_brent_yearly.csv")
df_brent["date"] = pd.to_datetime(df_brent["date"])
df_brent_copy = df_brent.copy()
df_brent_y.set_index("date", inplace=True)

df_opec_production = pd.read_csv("data/processed/opec_total_production.csv", index_col="date", parse_dates=True)
df_opec_production_copy = df_opec_production.copy()
df_opec_production_y = df_opec_production.copy()
df_opec_production_y = df_opec_production_y.resample("YE").mean()
df_opec_production_y.index = pd.to_datetime(df_opec_production_y.index)
df_opec_production_y = df_opec_production_y[df_opec_production_y.index.year >= 2000]
df_opec_production_y = df_opec_production_y[df_opec_production_y.index.year <= 2025]

df_no_opec_production = pd.read_csv("data/processed/no_opec_total_production.csv", index_col="date", parse_dates=True)
df_no_opec_production_copy = df_no_opec_production.copy()
df_no_opec_production_y = df_no_opec_production.copy()
df_no_opec_production_y = df_no_opec_production_y.resample("YE").mean()
df_no_opec_production_y.index = pd.to_datetime(df_no_opec_production_y.index)
df_no_opec_production_y = df_no_opec_production_y[df_no_opec_production_y.index.year >= 2000]
df_no_opec_production_y = df_no_opec_production_y[df_no_opec_production_y.index.year <= 2025]

df_dxy = pd.read_csv("data/processed/dxy.csv", index_col="date", parse_dates=True)
df_dxy_copy = df_dxy.copy()
df_dxy_y = df_dxy.copy()
df_dxy_y["date"] = pd.to_datetime(df_dxy_y.index)
df_dxy_y.set_index("date", inplace=True)
df_dxy_y = df_dxy_y.resample("YE").mean()

#abas

with aba1:
    st.write(
        "## ***O petróleo Brent é uma referência internacional para os preços do petróleo bruto.É extraído do Mar do Norte e é usado como referência para o preço do petróleo em todo o mundo.***"
    )
    st.write(
        "#### **Vamos explorar a série histórica do preço do petróleo Brent, identificando tendências e eventos marcantes que afetaram o mercado de energia nos últimos anos.**"
    )

     # Gráfico
    fig = px.line(
        df_brent_y,
        x=df_brent_y.index,
        y="brent_price",
        title="Preço Médio Anual do Brent",
        labels={"brent_price": "Preço Médio (USD)", "date": "Ano"},
    )

    # Eventos marcantes no gráfico
    eventos = {
        "2008-12-31": """O preço do barril de petróleo Brent atingiu seu pico histórico em 2008, ultrapassando os 90 USD. Esse aumento foi impulsionado por uma demanda global crescente, especialmente de economias emergentes como China e Índia, que elevaram o consumo de petróleo a níveis sem precedentes. Além disso, a Organização dos Países Exportadores de Petróleo (OPEP) implementou cortes na produção, restringindo a oferta e contribuindo para a alta dos preços.""",
        "2009-12-31": """Após o pico em 2008/2009, o preço do barril de petróleo Brent caiu bruscamente em 2010, atingindo valores mínimos. Essa queda foi resultado da recessão global de 2008-2009, que reduziu significativamente a demanda por petróleo. A desaceleração econômica levou a uma diminuição no consumo de energia, pressionando os preços para baixo.""",
        "2011-12-31": """Em 2012, o preço do barril de petróleo Brent voltou a subir, atingindo um novo pico. Esse aumento foi influenciado por tensões geopolíticas no Oriente Médio e na África do Norte, que afetaram a oferta de petróleo. Além disso, a recuperação econômica de algumas economias desenvolvidas contribuiu para o aumento da demanda por energia, elevando os preços.""",
        "2015-12-31": """Após alguns anos de estabilização, houve uma nova queda brusca em 2015. A queda foi causada por um excesso de oferta no mercado, com aumento na produção de petróleo de xisto nos Estados Unidos e a decisão da OPEP de não reduzir a produção. Além disso, a desaceleração econômica global e a valorização do dólar também contribuíram para a queda dos preços.""",
        "2016-12-31": """Em 2016/2017, o preço do petróleo Brent caiu para valores mínimos. Essa queda foi atribuída a um aumento na produção de petróleo de xisto nos Estados Unidos, que superou as expectativas de oferta, e a uma desaceleração na demanda global por energia. Além disso, a OPEP e outros produtores não conseguiram chegar a um acordo para reduzir a produção, o que contribuiu para a queda dos preços.""",
        "2018-12-31": """Em 2018, o preço do petróleo Brent experimentou uma recuperação, impulsionada por cortes na produção pela OPEP e outros produtores, bem como pela crescente demanda por energia. A recuperação foi sustentada por fatores geopolíticos, como as sanções dos Estados Unidos ao Irã e a crise na Venezuela, que afetaram a oferta de petróleo.""",
        "2020-12-31": """De 2019 a 2021, o preço do petróleo Brent experimentou uma queda significativa. A pandemia de COVID-19 levou a uma redução drástica na demanda por petróleo devido a lockdowns e desaceleração econômica global. Além disso, a guerra comercial entre os Estados Unidos e a China criou incertezas nos mercados de energia, impactando negativamente os preços.""",
        "2022-12-31": """A partir de 2021, o mercado de petróleo iniciou um período de recuperação, atingindo um novo pico em 2023. Essa recuperação foi impulsionada pela reabertura econômica pós-pandemia, aumento na demanda por energia e cortes na produção pela OPEP+. No entanto, a recuperação também enfrentou desafios, como a escassez de oferta e tensões geopolíticas que afetaram a estabilidade do mercado.""",
        "2023-12-31": """Após o pico em 2023, o mercado de petróleo pode estar entrando em um período de estabilização. Fatores como o aumento da produção de petróleo de xisto nos Estados Unidos, a transição para fontes de energia renováveis e políticas ambientais mais rigorosas podem influenciar a oferta e a demanda, levando a uma estabilização nos preços do petróleo Brent. No entanto, o mercado de energia é volátil e sujeito a mudanças rápidas, o que torna difícil prever com precisão o futuro do preço do petróleo.""",
    }

    # Criar marcadores para os eventos
    marcadores = go.Scatter(
        x=list(eventos.keys()),
        y=[df_brent_y.loc[data, "brent_price"] for data in eventos.keys()],
        mode="markers+text",
        marker=dict(size=9, color="red", symbol="arrow-bar-down", opacity=0.8),
        text=[data[:4] for data in eventos.keys()],  # Exibir apenas o ano nos marcadores
        textposition="top center",
        hoverinfo="text",
    )
    fig.add_trace(marcadores)

    # Ajustar tamanho da fonte do título
    fig.update_traces(line=dict(color="blue"))
    fig.update_layout(
        title={
            "text": "Preço Médio Anual do Brent com Eventos Marcantes",
            "x": 0,
            "font": dict(size=20, color="black"),
        },
        xaxis_title="Ano",
        yaxis_title="Preço Médio (USD)",
        template="plotly_white",
    )
    # Exibir o gráfico
    st.plotly_chart(fig, use_container_width=True)
    st.write(
        " \n ***A partir dos dados disponibilizados no site do IPEA - Instituto de Pesquisa Econômica Aplicada,**\n  **foi possível extrair a série histórica do preço do barril de petróleo Brent***"
    )

    # Criar colunas para os botões e as explicações

    col1, col2 = st.columns([1, 2])  # Define a proporção de tamanho das colunas

    with col1:
        st.write("### Eventos Marcantes")
        for key, descricao in eventos.items():
            ano = key[:4]  # Extrai apenas o ano
            if st.button(ano):  # Botão exibe só o ano
                st.session_state["evento_selecionado"] = descricao  # Armazena o evento clicado

    with col2:
        st.write("### Detalhes do Evento")
        if "evento_selecionado" in st.session_state:
            st.markdown(f"**{st.session_state['evento_selecionado']}**")  # Exibe diretamente o evento
with aba2:
    st.write(
        "## **O principal indicador de oferta de petróleo é a produção de petróleo bruto, que mede a quantidade de petróleo extraída e produzida por um país ou região.**"
    )
    st.write(
        "#### **A OPEP é uma organização formada por 13 países produtores de petróleo, responsável por regular a produção e os preços do petróleo no mercado internacional.**"
    )
    st.write(
        "#### ***A produção de petróleo bruto pela OPEP é um indicador importante para entender a dinâmica do mercado de energia e as políticas de produção dos países membros.***"
    )


    fig = px.line(
        df_opec_production_y,
        x=df_opec_production_y.index,
        y="production_value_opec",
        title="Produção Anual de Petróleo Bruto pela OPEP",
        labels={"production_value_opec": "Produção Anual (milhões de barris/dia)", "date": "Ano"},
    )
    fig.update_traces(line=dict(color="green"))
    fig.update_layout(
        title={"text": "Produção Anual de Petróleo Bruto pela OPEP", "x": 0, "font": dict(size=20, color="black")},
        xaxis_title="Ano",
        yaxis_title="Produção Anual (milhões de barris/dia)",
        template="plotly_white",
    )
    st.plotly_chart(fig, use_container_width=True)
    st.write(
        " \n ***A partir dos dados disponibilizados pela EIA - Energy Information Administration, foi possível extrair a série histórica da produção de petróleo bruto pela OPEP***"
    )


    fig = px.line(
        df_no_opec_production_y,
        x=df_no_opec_production_y.index,
        y="production_value_no_opec",
        title="Produção Anual de Petróleo Bruto por Países Não-OPEP",
        labels={"production_value_no_opec": "Produção Anual (milhões de barris/dia)", "date": "Ano"},
    )
    fig.update_traces(line=dict(color="purple"))
    fig.update_layout(
        title={
            "text": "Produção Anual de Petróleo Bruto por Países Não-OPEP",
            "x": 0,
            "font": dict(size=20, color="black"),
        },
        xaxis_title="Ano",
        yaxis_title="Produção Anual (milhões de barris/dia)",
        template="plotly_white",
    )
    st.plotly_chart(fig, use_container_width=True)
    st.write(
        " \n ***A partir dos dados disponibilizados pela EIA - Energy Information Administration, foi possível extrair a série histórica da produção de petróleo bruto por países não-OPEP***"
    )

    st.write(
        "### **Os países não-OPEP são aqueles que não fazem parte da OPEP e têm autonomia para definir suas políticas de produção de petróleo.**"
    )

    st.write("#### ***Nota-se pelo gráfico que a produção de petróleo bruto pelos países da OPEP tem uma correlação mais evidente com o preço do petróleo Brent, enquanto a produção de petróleo por países não-OPEP tem uma correlação mais fraca.*** ")

with aba3:
    st.write(
        "### **O preço do petróleo Brent é influenciado por diversos fatores, incluindo a taxa de câmbio do dólar americano.**"
    )

    fig = px.line(
        df_dxy_y,
        x=df_dxy_y.index,
        y="dxy_value_close",
        title="Taxa de Câmbio do Dólar Americano",
        labels={"dxy_value_close": "Taxa de Câmbio", "date": "Ano"},
    )
    fig.update_traces(line=dict(color="red"))
    fig.update_layout(
        title={"text": "Taxa de Câmbio do Dólar Americano", "x": 0, "font": dict(size=20, color="black")},
        xaxis_title="Ano",
        yaxis_title="Taxa de Câmbio",
        template="plotly_white",
    )
    st.plotly_chart(fig, use_container_width=True)

    st.write(
        " \n ***A partir dos dados disponibilizados pelo FRED - Federal Reserve Economic Data, foi possível extrair a série histórica da taxa de câmbio do dólar americano***"
    )

    st.write(
        "### **A taxa de câmbio do dólar americano é um indicador importante para entender a relação entre o petróleo Brent e o mercado de câmbio.**"
    )
    st.write(
        "#### ***A valorização ou desvalorização do dólar americano pode afetar o preço do petróleo Brent e a economia global.***"
    )

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df_brent_y.index,
            y=df_brent_y["brent_price"],
            mode="lines",
            name="Preço Médio Anual do Brent",
            line=dict(color="blue"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df_dxy_y.index,
            y=df_dxy_y["dxy_value_close"],
            mode="lines",
            name="Taxa de Câmbio do Dólar Americano",
            line=dict(color="red"),
        )
    )
    fig.update_layout(
        title="Preço Médio Anual do Brent vs. Taxa de Câmbio do Dólar Americano",
        xaxis_title="Ano",
        yaxis_title="Valor",
        template="plotly_white",
    )
    st.plotly_chart(fig, use_container_width=True)
    st.write(
        "### **A partir do gráfico acima, podemos observar a relação entre o preço do petróleo Brent e a taxa de câmbio do dólar americano ao longo dos anos.**"
    )
    st.write(
        "#### ***A análise da relação entre esses indicadores nos mostra uma correlação negativa entre o dollar index - DYX e o valor em dolar do barril de Brent de forma que quanto mais valorizada a moeda Norte Americana menores são os preços do Brent, tendo em vista que o mesmo é cotado em dollar.***"
    )

with aba4:
    st.write(
        "### **O preço do petróleo Brent está correlacionado com diversos indicadores econômicos e financeiros, como o índice de ações, o preço do ouro e a taxa de juros.**"
    )

    df_brent_copy.index = pd.to_datetime(df_brent_copy.index)
    df_opec_production_copy.index = pd.to_datetime(df_opec_production_copy.index)
    df_no_opec_production_copy.index = pd.to_datetime(df_no_opec_production_copy.index)
    df_dxy_copy.index = pd.to_datetime(df_dxy_copy.index)

    df_brent_copy = df_brent_copy.resample("D").interpolate()
    df_opec_production_copy = df_opec_production_copy.resample("D").interpolate()
    df_no_opec_production_copy = df_no_opec_production_copy.resample("D").interpolate()
    df_dxy_copy = df_dxy_copy.resample("D").interpolate()

    df_merged = pd.concat(
        [df_brent_copy, df_opec_production_copy, df_no_opec_production_copy, df_dxy_copy],
        axis=1,  # Junta ao longo das colunas
        join="outer",  # Mantém todos os índices, preenchendo com NaN se necessário
    )

    df_merged = pd.read_csv("data/processed/df_merged.csv")
    df_merged["date"] = pd.to_datetime(df_merged["date"])
    df_merged.set_index("date", inplace=True)
    df_corr = df_merged.corr()
    # traduzindo as correlações

    df_corr.columns = [
        "Preço do Petróleo Brent",
        "Produção de Petróleo pela OPEP",
        "Produção de Petróleo por Países Não-OPEP",
        "Taxa de Câmbio do Dólar Americano",
    ]
    df_corr.index = df_corr.columns  # Ajusta os índices para manter os mesmos nomes
    fig = go.Figure(
        data=go.Heatmap(
            z=df_corr.values,
            x=df_corr.columns,
            y=df_corr.index,
            colorscale="Viridis",
            text=df_corr.round(2).values,  # Adiciona os valores numéricos formatados
            texttemplate="%{text}",  # Mostra os valores diretamente no gráfico
            showscale=True,  # Mantém a barra de escala de cores
        )
    )

    # Ajustando layout do gráfico
    fig.update_layout(
        title="Correlações do Preço do Petróleo Brent com Indicadores Quantitativos",
        xaxis_title="Indicadores Quantitativos",
        yaxis_title="Indicadores Quantitativos",
        template="plotly_white",
    )

    # Exibindo no Streamlit
    st.plotly_chart(fig, use_container_width=True)

    st.write(
        "#### ***A partir do gráfico acima, podemos observar as correlações do preço do petróleo Brent com outros indicadores financeiros, como a produção de petróleo pela OPEP, a produção de petróleo por países não-OPEP e a taxa de câmbio do dólar americano. Podemos destacar que existem correlações positivas como a produção de petróleo e correlação negativa como o valor do Dollar Index***"
    )

    st.write(
        "#### ***Eventos geopolíticos como a guerra na Ucrânia, a crise na Venezuela e a pandemia de COVID-19 também podem afetar o preço do petróleo Brent, mas não estão refletidos nas correlações acima. Mas depois de vários testes de significância levamos em consideração a Guerra da Ucrânia x Rússia como principal fator ocorrendo no momento para que haja uma real correlação a ser testada***"
    )

    df_merged["war_ukraine"] = 0
    df_merged.loc[df_merged.index >= "2022-02-24", "war_ukraine"] = 1
    df_corr2 = df_merged.corr()
    df_corr2.columns = [
        "Preço do Petróleo Brent",
        "Produção de Petróleo pela OPEP",
        "Produção de Petróleo por Países Não-OPEP",
        "Taxa de Câmbio do Dólar Americano",
        "Guerra da Ucrânia x Rússia",
    ]
    df_corr2.index = df_corr2.columns  # Ajusta os índices para manter os mesmos nomes

    fig = go.Figure(
        data=go.Heatmap(
            z=df_corr2.values,
            x=df_corr2.columns,
            y=df_corr2.index,
            colorscale="Viridis",
            text=df_corr2.round(2).values,  # Adiciona os valores numéricos formatados
            texttemplate="%{text}",  # Mostra os valores diretamente no gráfico
            showscale=True,  # Mantém a barra de escala de cores
        )
    )

    # Ajustando layout do gráfico
    fig.update_layout(
        title="Correlações do Preço do Petróleo Brent com Indicadores Quantitativos e Guerra da Ucrânia x Rússia",
        xaxis_title="Indicadores Quantitativos",
        yaxis_title="Indicadores Quantitativos",
        template="plotly_white",
    )

    # Exibindo no Streamlit
    st.plotly_chart(fig, use_container_width=True)

with aba5:
    st.markdown("### **Insights:**")
    st.markdown("""
    #### ***Monitorar decisões da OPEP+: As políticas de produção da OPEP+ têm impacto direto nos preços do petróleo. Decisões sobre cortes ou aumentos na produção podem influenciar a oferta e, consequentemente, os preços.***

    #### ***Atenção às tensões geopolíticas: Conflitos em regiões produtoras de petróleo, como o Oriente Médio, podem afetar a oferta e a estabilidade dos preços. Investidores devem acompanhar eventos geopolíticos que possam impactar o mercado de petróleo.***

    #### ***Considerar a transição energética: O movimento global em direção a fontes de energia renováveis pode reduzir a demanda por petróleo no longo prazo. Investidores devem avaliar como essa transição pode afetar o mercado de petróleo e os preços do Brent.***

    #### ***Avaliar a produção de petróleo de xisto nos EUA: O aumento na produção de petróleo de xisto nos Estados Unidos pode influenciar a oferta global e os preços do Brent. Investidores devem monitorar a atividade de perfuração e produção nos EUA.***


    #### ***Observar a demanda global por energia: Crescimentos ou desacelerações econômicas em grandes economias, como China e Índia, podem afetar a demanda por petróleo e influenciar os preços. Investidores devem acompanhar indicadores econômicos dessas regiões.*** """)

with aba6:
    col1, col2 = st.columns([1, 1])
    with col1:
            st.markdown("### **Plano de Deploy**")
            st.markdown(
                "#### ***Para o deploy do projeto, consideramos as seguintes etapas:***"
            )
            st.markdown(
                "#### ***1. Estrategia de Deploy***"
            )
            st.markdown("##### ***1.1 Hospedagem do Dashboard***"
            )
            st.markdown("##### ***O dashboard interativo desenvolvido com Streamlit sera hospedado no Streamlit Cloud (alternativamente, pode ser usado Render ou uma instância AWS EC2). A escolha do Streamlit Cloud facilita a implantação sem necessidade de gestão manual de infraestrutura.***"
            )
            st.markdown("##### **Passos para deploy no Streamlit Cloud:** ")
            st.markdown("##### ***1. Criar um repositório no GitHub contendo todo o código do dashboard.***")
            st.markdown("##### ***2. Criar uma conta no Streamlit Cloud (https://share.streamlit.io/).***")
            st.markdown("#####***3. Conectar o repositório e configurar o arquivo requirements.txt.***")
            st.markdown("##### ***4. Definir a variável de ambiente do modelo no Streamlit Cloud (caso haja credenciais sensíveis).***")
            st.markdown("##### ***5. Fazer o deploy e testar a aplicação.***")

            st.markdown("##### ***1.2 Infraestrutura para Atualização do Modelo - a ser implementado***")

            st.markdown("#### ***O modelo XGBoost precisa ser atualizado regularmente com novos dados. Para isso, será criado um job automatizado que:***")
            st.markdown("##### ***1. Baixa os dados mais recentes do IPEA.***")
            st.markdown("##### ***2. Atualiza os dados externos usados no modelo (Produção da OPEP/EIA, Taxa de câmbio do DXY/FRED).***")
            st.markdown("##### ***3. Atualiza o dataset.***")
            st.markdown("##### ***4. Re-treina o modelo XGBoost.***")
            st.markdown("##### ***5. Avalia o desempenho do novo modelo comparado ao anterior.***")
            st.markdown("##### ***6. Substitui o modelo salvo na AWS S3.***")
            st.markdown("##### ***7. Reinicia o Streamlit para carregar o modelo atualizado.***")

    with col2:
        st.markdown("### **2. Automação da Atualização do Modelo**")
        st.markdown("#### ***2.1 Fluxo de Atualização***")
        st.markdown("##### ***A cada domingo às 02:00 da manhã, um script Python será executado automaticamente via CRON em um servidor (EC2, GitHub Actions ou um container Docker). Esse script:***")
        st.markdown("##### ***1. Faz scraping dos novos dados do site do IPEA.***")
        st.markdown("##### ***2. Baixa os dados externos necessários (Produção da OPEP, Taxa de câmbio DXY).***")
        st.markdown("##### ***3. Atualiza os arquivos de dados.***")
        st.markdown("##### ***4. Re-treina o modelo XGBoost com os dados atualizados.***")
        st.markdown("##### ***5. Avalia o desempenho do novo modelo comparado ao anterior.***")
        st.markdown("##### ***6. Substitui o modelo salvo na AWS S3.***")
        st.markdown("##### ***7. Reinicia a aplicação Streamlit para carregar o modelo atualizado.***")

        st.markdown("#### ***Conclusão***")
        st.markdown("##### ***O dashboard estará acessível via Streamlit Cloud.***")
        st.markdown("##### ***O modelo será atualizado automaticamente toda semana.***")
        st.markdown("##### ***O modelo treinado será salvo na AWS S3.***")
        st.markdown("##### ***O script também atualizará os dados externos usados no treinamento.***")
        st.markdown("##### ***Um script agendado via CRON garantirá a automação.***")
        st.markdown("##### ***Dessa forma, a previsão do preço do Brent será sempre baseada nos dados mais recentes, sem necessidade de intervenção manual.***")


with aba7:
    st.markdown(
        "### Efetuamos testes com dois modelos que consideram fatores externos e avaliam séries temporais. O SARIMAX e o XGBoost.")
    st.write("Avaliamos as métricas MAE - Erro Médio Absoluto  , o MSE - Erro Quadático médio, RMSE - Raiz Quadrada do Erro Médio e o MAPE - Percentual Absoluto do erro,  para avaliar a performance dos modelos."
    )

    file_path = "models/metrics/sarimax_metrics.txt"
    with open(file_path, "r") as file:
        sarimax_metrics = file.read()
        st.markdown("#### ***Modelo SARIMAX:***")
        st.markdown(f"""{sarimax_metrics}""")

    file_path = "models/metrics/xgboost_metrics.txt"
    with open(file_path, "r") as file:
        xgboost_metrics = file.read()
        st.markdown("#### ***Modelo XGBoost:***")
        st.markdown(f"""{xgboost_metrics}""")

    st.write(
        "#### **Com base nos resultados, o modelo XGBoost apresentou melhor desempenho na previsão do preço do petróleo Brent.**"
    )
    st.write(
        "##### ***O modelo XGBoost é um algoritmo de aprendizado de máquina baseado em árvores de decisão que pode lidar com dados não lineares e capturar relações complexas entre as variáveis.***"
    )

    st.markdown("#### **Para apreciação do código, ajustes e treinamento do modelo XGBoost, clique no botão abaixo:**")
    with st.expander("Código do Modelo XGBoost"):
        with open("models/modelo.py", "r") as file:
            st.code(file.read(), language="python")

    # grafico do modelo
    previsoes = pd.read_csv("data/processed/predicted.csv")
    previsoes["date"] = pd.to_datetime(previsoes["date"])
    previsoes.set_index("date", inplace=True)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=previsoes.index,
            y=previsoes["brent_price"],
            mode="lines",
            name="Preço Real",
            line=dict(color="blue"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=previsoes.index,
            y=previsoes["brent_price_pred"],
            mode="lines",
            name="Preço Previsto",
            line=dict(color="red"),
        )
    )
    fig.update_layout(
        title="Preço Real vs. Preço Previsto do Petróleo Brent",
        xaxis_title="Data",
        yaxis_title="Preço (USD)",
        template="plotly_white",
    )
    st.plotly_chart(fig, use_container_width=True)

    # tabela com as previsões e valores reais somente com as colunas date, brent_price e brent_price_pred dos últimos 10 registros renomeadas para "Data", "Preço Real" e "Preço Previsto" e fazer expandir
    st.write("#### **Previsões e Valores Reais do Preço do Petróleo Brent**")
    st.write(
        "##### ***A tabela abaixo exibe os últimos 10 registros com as previsões e valores reais do preço do petróleo Brent.***"
    )
    st.write("##### ***Os valores estão em dólares americanos (USD).***")
    with st.expander("Tabela de Previsões e Valores Reais"):
        st.write(
            previsoes[["brent_price", "brent_price_pred"]]
            .rename(columns={"brent_price": "Preço Real", "brent_price_pred": "Preço Previsto"})
            .tail(10)
        )

# Caminho do modelo salvo
MODEL_PATH = "models/modelo_brent.json"


# Função para carregar o modelo XGBoost
def load_xgb_model():
    with open(MODEL_PATH, "r") as f:
        model_json = json.load(f)
    model = xgb.XGBRegressor()
    model.load_model(MODEL_PATH)
    return model


# Função para criar os dados futuros para previsão
def create_future_features(start_date, df_merged):
    future_dates = pd.date_range(start=start_date, periods=7, freq="D")
    future_data = pd.DataFrame(index=future_dates)

    # Criando as mesmas features usadas no treinamento
    for lag in [1, 2, 3, 5, 7, 14]:
        future_data[f"lag_{lag}"] = df_merged["brent_price"].shift(lag).iloc[-1]

        future_data["rolling_mean_7"] = df_merged["brent_price"].rolling(7).mean().iloc[-1]
        future_data["pct_change_1"] = df_merged["brent_price"].pct_change(1).iloc[-1]
        future_data["dxy_change"] = df_merged["dxy_value_close"].pct_change(1).iloc[-1]
        future_data["is_weekend"] = future_data.index.weekday >= 5

    # Preencher com os últimos valores conhecidos
    for col in ["production_value_opec", "production_value_no_opec", "dxy_value_close"]:
        future_data[col] = df_merged[col].iloc[-1]

    # Garantir que as colunas estejam na ordem correta
    expected_features = [
        "production_value_opec",
        "production_value_no_opec",
        "dxy_value_close",
        "lag_1",
        "lag_2",
        "lag_3",
        "lag_5",
        "lag_7",
        "lag_14",
        "rolling_mean_7",
        "pct_change_1",
        "dxy_change",
        "is_weekend",
    ]

    future_data = future_data[expected_features]

    return future_data


# Carregar os dados históricos
df_merged = pd.read_csv("data/processed/df_merged.csv")

# Carregar modelo
model = load_xgb_model()

# Criando uma aba de previsão no Streamlit
with aba8:
    st.write("### **Previsão do Preço do Petróleo Brent**")

    # Input da data de previsão
    selected_date = st.date_input("Selecione a data inicial para previsão", datetime.today())

    # Botão para gerar a previsão
    if st.button("Prever Preço"):
        future_data = create_future_features(selected_date, df_merged)
        predictions = model.predict(future_data)

        # Criar DataFrame com resultados
        result_df = pd.DataFrame({"Data": future_data.index, "Previsão Brent": predictions})

        # Exibir tabela
        st.write("### Previsão do Preço do Brent para os próximos 7 dias")
        st.dataframe(result_df)
with aba9:
    st.write("### Apresentação do Projeto")

    st.video("https://youtu.be/6v8LGPlXsdM")

    st.write("Neste vídeo, explico o todo o projeto.")
