import streamlit as st
import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
import shap

# Configurar Seaborn para gráficos
sns.set(style="whitegrid")

# Carregar modelos e objetos de preprocessamento
rf_model = joblib.load("random_forest_model.pkl")
lr_model = joblib.load("logistic_regression_model.pkl")
scaler = joblib.load("scaler.pkl")
imputer = joblib.load("imputer.pkl")
model_columns = joblib.load("model_columns.pkl")

# Função para preprocessamento dos dados de entrada
def preprocess_data(data):
    data = data.drop(columns=['Attrition'], errors='ignore')
    data = pd.get_dummies(data, drop_first=True)
    data = data.reindex(columns=model_columns, fill_value=0)
    data = imputer.transform(data)
    data = scaler.transform(data)
    return data

# Função para exibir métricas de avaliação
def calculate_metrics(y_true, y_pred):
    return {
        "Acurácia": accuracy_score(y_true, y_pred),
        "Precisão": precision_score(y_true, y_pred),
        "Revocação": recall_score(y_true, y_pred),
        "F1-Score": f1_score(y_true, y_pred)
    }

# Interface do Streamlit
st.title("Employee Attrition Prediction WebApp")
st.write("Este app prevê a chance de um funcionário sair da empresa.")

# Upload do arquivo CSV
uploaded_file = st.file_uploader("Escolha um arquivo CSV para prever", type="csv")

if uploaded_file is not None:
    input_data = pd.read_csv(uploaded_file)
    st.write("Dados Carregados:")
    st.write(input_data.head())

    # Filtragem de Dados
    st.sidebar.write("### Filtrar Dados")
    department_filter = st.sidebar.multiselect("Filtrar por Departamento", options=input_data['Department'].unique())
    jobrole_filter = st.sidebar.multiselect("Filtrar por Função", options=input_data['JobRole'].unique())
    
    filtered_data = input_data.copy()
    if department_filter:
        filtered_data = filtered_data[filtered_data['Department'].isin(department_filter)]
    if jobrole_filter:
        filtered_data = filtered_data[filtered_data['JobRole'].isin(jobrole_filter)]
    
    if len(filtered_data) == 0:
        st.write("Nenhum dado corresponde aos filtros aplicados.")
    else:
        st.write("Dados Filtrados:")
        st.write(filtered_data.head())

    # Opções de visualização
    view_options = st.sidebar.radio("Escolha o que deseja visualizar:", ["Previsões", "Métricas de Desempenho", "Distribuição de Dados"])

    # Preprocessar dados filtrados
    X_input = preprocess_data(filtered_data)

    # Fazer previsões
    rf_pred = rf_model.predict(X_input)
    lr_pred = lr_model.predict(X_input)

    # Adicionar previsões ao DataFrame de entrada
    filtered_data['RandomForest_Prediction'] = rf_pred
    filtered_data['LogisticRegression_Prediction'] = lr_pred

    # Opções de visualização
    if view_options == "Previsões":
        st.write("Previsões:")
        model_choice = st.selectbox("Escolha o Modelo para Visualizar:", ["Random Forest", "Logistic Regression", "Ambos"])
        if model_choice == "Ambos":
            st.write(filtered_data[['RandomForest_Prediction', 'LogisticRegression_Prediction']])
        else:
            model_col = 'RandomForest_Prediction' if model_choice == "Random Forest" else 'LogisticRegression_Prediction'
            st.write(filtered_data[[model_col]])

    elif view_options == "Métricas de Desempenho":
        st.write("### Comparação de Métricas de Desempenho")
        if 'Attrition' in input_data.columns:
            rf_metrics = calculate_metrics(input_data['Attrition'], rf_pred)
            lr_metrics = calculate_metrics(input_data['Attrition'], lr_pred)
            metrics_df = pd.DataFrame([rf_metrics, lr_metrics], index=['Random Forest', 'Logistic Regression'])
            st.write(metrics_df)

    elif view_options == "Distribuição de Dados":
        st.write("### Visualização da Distribuição")
        chart_type = st.selectbox("Escolha o Tipo de Gráfico:", ["Histograma de Atributos", "Dispersão entre Atributos"])
        
        if chart_type == "Histograma de Atributos":
            attribute = st.selectbox("Escolha o Atributo:", input_data.columns)
            fig, ax = plt.subplots()
            sns.histplot(input_data[attribute].dropna(), kde=True, ax=ax)
            ax.set_title(f"Distribuição de {attribute}")
            st.pyplot(fig)
        
        elif chart_type == "Dispersão entre Atributos":
            x_axis = st.selectbox("Escolha o Atributo para o Eixo X:", input_data.columns)
            y_axis = st.selectbox("Escolha o Atributo para o Eixo Y:", input_data.columns)
            fig, ax = plt.subplots()
            sns.scatterplot(x=input_data[x_axis], y=input_data[y_axis], ax=ax, hue=input_data['Attrition'] if 'Attrition' in input_data.columns else None)
            ax.set_title(f"Dispersão entre {x_axis} e {y_axis}")
            st.pyplot(fig)

    # Explicação com SHAP para Random Forest
    st.write("### Explicação das Previsões com SHAP")

    # Calcular SHAP values e plotar com segurança, garantindo compatibilidade com modelos binários
    explainer = shap.TreeExplainer(rf_model)
    shap_values = explainer.shap_values(X_input)

    # Se o modelo for binário, shap_values terá duas listas (uma para cada classe). Vamos usar shap_values[1].
    if isinstance(shap_values, list) and len(shap_values) > 1:
        shap_values_to_plot = shap_values[1]  # Para problemas binários
    else:
        shap_values_to_plot = shap_values     # Para problemas multiclasses ou shap_values de uma única matriz

    # Criar figura para o gráfico SHAP
    fig, ax = plt.subplots()
    shap.summary_plot(shap_values_to_plot, X_input, feature_names=model_columns, plot_type="bar", show=False)
    st.pyplot(fig)


    # Estatísticas Descritivas
    st.write("### Análise Estatística dos Dados")
    st.write(filtered_data.describe())

    # Exportar Dados Filtrados com Previsões
    st.sidebar.write("### Exportar Dados")
    if st.sidebar.button("Baixar Dados Filtrados"):
        st.download_button(
            label="Baixar Previsões e Dados Filtrados",
            data=filtered_data.to_csv(index=False),
            file_name="filtered_employee_attrition_predictions.csv",
            mime="text/csv"
        )
else:
    st.info("Por favor, faça o upload de um arquivo CSV para previsão.")
