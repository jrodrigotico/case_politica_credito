import streamlit as st
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from imblearn.over_sampling import SMOTE

coef = {
    'const': 0.567964,
    'genero_Masculino': 0.162199,
    'nivel_escolaridade_Médio': -0.208041,
    'nivel_escolaridade_Pós-graduação': -0.584570,
    'Faixa_renda_4001-6000': -0.271105,
    'Faixa_renda_6001-10000': -0.226520,
    'Faixa_renda_Até 2000': -0.192529,
    'Faixa_idade_36-50': 0.338755,
    'Faixa_idade_51-70': 0.302021,
    'Faixa_Score_501-600': -0.364174,
    'Faixa_Score_601-700': -0.685427,
    'Faixa_Score_701-800': -0.838719,
    'Faixa_Score_801-850': -0.961575,
    'Faixa_Score_Até 400': 0.117366}



# Interface do usuário no Streamlit
st.title('Previsão de Inadimplência')

# Inputs do usuário
genero = st.selectbox('Gênero:', ['Masculino', 'Feminino'])
nivel_escolaridade = st.selectbox('Nível de Escolaridade:', ['Médio', 'Superior', 'Pós-graduação'])
faixa_renda = st.selectbox('Faixa de Renda:', ['Até 2000', '2001-4000', '4001-6000', '6001-10000'])
faixa_idade = st.selectbox('Faixa de Idade:', ['18-25', '26-35', '36-50', '51-70'])
faixa_score = st.selectbox('Faixa de Score de Crédito:', ['Até 400', '401-600', '601-700', '701-800', '801-850'])

# Criar o dicionário com as entradas do cliente
novo_cliente = {
    'genero_Masculino': 1 if genero == 'Masculino' else 0,
    'nivel_escolaridade_Médio': 1 if nivel_escolaridade == 'Médio' else 0,
    'nivel_escolaridade_Pós-graduação': 1 if nivel_escolaridade == 'Pós-graduação' else 0,
    'Faixa_renda_4001-6000': 1 if faixa_renda == '4001-6000' else 0,
    'Faixa_renda_6001-10000': 1 if faixa_renda == '6001-10000' else 0,
    'Faixa_renda_Até 2000': 1 if faixa_renda == 'Até 2000' else 0,
    'Faixa_idade_36-50': 1 if faixa_idade == '36-50' else 0,
    'Faixa_idade_51-70': 1 if faixa_idade == '51-70' else 0,
    'Faixa_Score_501-600': 1 if faixa_score == '501-600' else 0,
    'Faixa_Score_601-700': 1 if faixa_score == '601-700' else 0,
    'Faixa_Score_701-800': 1 if faixa_score == '701-800' else 0,
    'Faixa_Score_801-850': 1 if faixa_score == '801-850' else 0,
    'Faixa_Score_Até 400': 1 if faixa_score == 'Até 400' else 0
}

# logit(p)
logit = coef['const']
for feature, value in novo_cliente.items():
    if feature in coef:
        logit += coef[feature] * value

# probabilidade de inadimplência
probabilidade = 1 / (1 + np.exp(-logit))

# Exibir a probabilidade de inadimplência
st.write(f"Probabilidade de inadimplência do novo cliente: {probabilidade*100:.4}%")

# Threshold para classificar como inadimplente ou adimplente
threshold = 0.4
classe = 1 if probabilidade >= threshold else 0
if classe == 1:
    st.write("O cliente será INADIMPLENTE.")
else:
    st.write("O cliente será ADIMPLENTE.")