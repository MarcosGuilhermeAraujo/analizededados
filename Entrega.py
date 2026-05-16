import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix,
    classification_report
)

from datetime import datetime

# Leitura do CSV

df = pd.read_csv("Trabalho em Grupo\cobranca_banco.csv")

bins_valor = [0, 5000, 15000, float('inf')]
labels_valor = ['Baixo', 'Médio', 'Alto']

df['faixa_valor'] = pd.cut(
    df['valor_devido'],
    bins=bins_valor,
    labels=labels_valor
)

# Tempo de inadimplência

bins_atraso = [0, 30, 90, float('inf')]
labels_atraso = ['Curto', 'Médio', 'Longo']

df['faixa_atraso'] = pd.cut(
    df['dias_atraso'],
    bins=bins_atraso,
    labels=labels_atraso
)

# Contato anterior efetivo

df['contato_anterior_efetivo'] = (
    (df['historico_contato'] > 0) &
    (df['canal_efetivo'] != 'nenhum')
).astype(int)

# Média de atraso por ramo

df['media_dias_ramo'] = (
    df.groupby("ramo_atividade")["dias_atraso"]
    .transform("mean")
)

# Média de valor por ramo

df['media_valor_ramo'] = (
    df.groupby("ramo_atividade")["valor_devido"]
    .transform("mean")
)

# Relação entre atraso e valor

df['relacao_dia_valor'] = (
    df["media_dias_ramo"] /
    df["media_valor_ramo"]
)

# Cálculo vetorizado

df['calculo_linha_por_linha'] = (
    df["dias_atraso"] *
    df["relacao_dia_valor"]
)

# Arredondamento

df['relacao_dia_valor'] = np.round(
    df["relacao_dia_valor"],
    6
)

df['calculo_linha_por_linha'] = np.round(
    df["calculo_linha_por_linha"],
    2
)

# Análise da dívida

df['analise_divida'] = np.where(
    df['valor_devido'] > df['calculo_linha_por_linha'],
    "Acima do esperado",

    np.where(
        df['valor_devido'] < df['calculo_linha_por_linha'],
        "Abaixo do esperado",

        "Dentro do padrão"
    )
)

# Gráfico de canais

canais_validos = ["telefone", "sms", "email"]

df_filtrado = df[
    df["canal_efetivo"].isin(canais_validos)
]

eficacia = (
    df_filtrado
    .groupby("canal_efetivo")["recuperou"]
    .sum()
)

plt.figure(figsize=(7, 7))

plt.pie(
    eficacia,
    labels=eficacia.index,
    autopct='%1.2f%%',
    startangle=90
)

plt.title("Eficácia dos Canais de Contato")

plt.show()

# Boxplot

bins = [0, 30, 60, 90, 120, 180, 360]

labels = [
    '1–30',
    '31–60',
    '61–90',
    '91–120',
    '121–180',
    '181–360'
]

df['faixa_atraso_boxplot'] = pd.cut(
    df['dias_atraso'],
    bins=bins,
    labels=labels
)

df_rec = df[
    df['recuperou'] == 1
]

plt.figure(figsize=(12, 6))

sns.boxplot(
    data=df_rec,
    x='faixa_atraso_boxplot',
    y='valor_devido'
)

plt.title('Valor Recuperado por Faixa de Dias em Atraso')

plt.xlabel('Dias em Atraso')

plt.ylabel('Valor (R$)')

plt.tight_layout()

plt.show()

# Prioridade por valor

df["prioridade_valor_devido"] = pd.cut(
    df["valor_devido"],
    bins=[0, 5000, 15000, float('inf')],
    labels=["baixo", "medio", "alto"]
)

# Prioridade por atraso

df["prioridade_dias_atraso"] = pd.cut(
    df["dias_atraso"],
    bins=[0, 30, 90, float('inf')],
    labels=["baixo", "medio", "alto"]
)

# Regra de negócio

def prioridade_final(linha):

    valor_devido = linha["prioridade_valor_devido"]
    dias_atraso = linha["prioridade_dias_atraso"]

    if valor_devido == 'baixo' and dias_atraso == 'baixo':
        return 'baixo'

    elif valor_devido == 'medio' and dias_atraso == 'medio':
        return 'medio'

    elif valor_devido == 'alto' and dias_atraso == 'alto':
        return 'alto'

    elif valor_devido == 'baixo' and dias_atraso == 'medio':
        return 'baixo'

    elif valor_devido == 'medio' and dias_atraso == 'baixo':
        return 'medio'

    elif valor_devido == 'baixo' and dias_atraso == 'alto':
        return 'medio'

    elif valor_devido == 'alto' and dias_atraso == 'baixo':
        return 'alto'

    elif valor_devido == 'medio' and dias_atraso == 'alto':
        return 'alto'

    elif valor_devido == 'alto' and dias_atraso == 'medio':
        return 'alto'

    else:
        return 'baixo'

# Aplicando a prioridade

df["prioridade_final"] = df.apply(
    prioridade_final,
    axis=1
)

# Treinamento da IA

X = df[[
    "valor_devido",
    "dias_atraso"
]]

y = df["prioridade_final"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=42
)

modelo = DecisionTreeClassifier()

modelo.fit(X_train, y_train)

y_pred = modelo.predict(X_test)

print(confusion_matrix(y_test, y_pred))

print(classification_report(y_test, y_pred))

# Ordenação por prioridade

df["prioridade_final"] = pd.Categorical(
    df["prioridade_final"],
    categories=["alto", "medio", "baixo"],
    ordered=True
)

df_ordenado = df.sort_values("prioridade_final")

# Ação sugerida

df_ordenado['acao_sugerida'] = np.where(
    (
        (df_ordenado['prioridade_final'] == 'medio') |
        (df_ordenado['prioridade_final'] == 'alto')
    ),
    'Cobrar cliente',
    'Nao cobrar cliente'
)

# Seleção final

df_final = df_ordenado[
    [
        "id_devedor",
        "valor_devido",
        "dias_atraso",
        "prioridade_final",
        "acao_sugerida"
    ]
].copy()

# Nome do arquivo

data = (datetime.now()).strftime("%Y-%m-%d")

nome_arquivo = f"cobranca_priorizada_{data}.csv"

# Exportação do CSV

df_final.to_csv(
    nome_arquivo,
    index=False
)