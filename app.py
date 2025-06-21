import streamlit as st
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from imblearn.over_sampling import SMOTE

# Função para carregar e pré-processar os dados
def carregar_dados():
    # Carregar o DataFrame com os dados (você pode substituir com o caminho do seu arquivo)
    df2 = pd.read_excel('dados_case.xlsx', sheet_name='dados')

    # Definir colunas numéricas e categóricas
    numeric_features = ['contas_bancarias']
    categorical_feats = ['genero', 'localizacao', 'nivel_escolaridade', 'Faixa_renda', 'Faixa_idade', 'Faixa_Score']

    # Separar variáveis independentes e dependentes
    var_x = df2[numeric_features + categorical_feats]
    var_y = df2['Atraso_apos_6meses']

    # Dividir em treino e teste
    X_train, X_test, y_train, y_test = train_test_split(var_x, var_y, test_size=0.2, random_state=123, stratify=var_y)

    # Pipeline de pré-processamento
    preprocessor = ColumnTransformer([
        ('n', MinMaxScaler(), numeric_features),
        ('c', OneHotEncoder(sparse_output=False, handle_unknown='ignore', drop='first'), categorical_feats),
    ])

    # Aplicando o pré-processamento nos dados de treino e teste
    X_train_p = preprocessor.fit_transform(X_train)
    X_test_p = preprocessor.transform(X_test)

    # Aplicando SMOTE no conjunto de treino
    smote = SMOTE(sampling_strategy='auto', random_state=123)
    X_train_res, y_train_res = smote.fit_resample(X_train_p, y_train)

    return preprocessor, X_train_res, y_train_res, X_test_p, y_test

# Função para treinar o modelo
def treinar_modelo(X_train_res, y_train_res, preprocessor):
    # Nomes das variáveis transformadas
    feat_names = preprocessor.get_feature_names_out()

    # Treinando o modelo Logit com os dados balanceados após o SMOTE
    X_train_res_sm = pd.DataFrame(X_train_res, columns=feat_names)
    X_train_res_sm = sm.add_constant(X_train_res_sm)  # Adicionar constante (intercepto)
    
    # Modelo Logit
    logit_model = sm.Logit(y_train_res, X_train_res_sm).fit(disp=False)

    return logit_model

# Função para prever a inadimplência de um novo cliente
def prever_inadimplencia(cliente_info, logit_model):
    logit = logit_model.params['const']
    for feature, value in cliente_info.items():
        if feature in logit_model.params:
            logit += logit_model.params[feature] * value

    probabilidade = 1 / (1 + np.exp(-logit))  
    threshold = 0.6
    classe = 1 if probabilidade >= threshold else 0
    
    return probabilidade, classe

# ------------------------ Layout Streamlit ------------------------ #
st.title('Previsão de Inadimplência de Clientes')

# Carregar e treinar o modelo
preprocessor, X_train_res, y_train_res, X_test_p, y_test = carregar_dados()
logit_model = treinar_modelo(X_train_res, y_train_res, preprocessor)

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

# Botão para gerar a previsão
if st.button('Gerar Previsão'):
    # Prever a inadimplência do novo cliente
    probabilidade, classe = prever_inadimplencia(novo_cliente, logit_model)

    # Exibir a probabilidade de inadimplência e a classificação
    st.write(f'A probabilidade de inadimplência do cliente é: {probabilidade:.2f}')
    if classe == 1:
        st.write("O cliente será INADIMPLENTE.")
    else:
        st.write("O cliente será ADIMPLENTE.")