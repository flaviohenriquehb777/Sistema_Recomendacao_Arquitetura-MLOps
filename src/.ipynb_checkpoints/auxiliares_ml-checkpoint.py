import pandas as pd
import numpy as np
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier  # Redes Neurais Artificiais (ANN)
from xgboost import XGBClassifier  # XGBoost
from sklearn.metrics import precision_recall_curve, auc








# CLASSIFICAÇÃO - QUAL ALGORITMOS DE ML UTILIZAR?

def testar_modelos_com_undersampling(df, target, test_size=0.2, random_state=42):
    """
    Testa diferentes algoritmos de ML usando apenas o RandomUnderSampler para balanceamento.
    Retorna os três melhores modelos com base na área sob a curva precisão-recall (PR AUC).

    Parâmetros:
    df (pd.DataFrame): DataFrame contendo os dados.
    target (str): Nome da coluna alvo.
    test_size (float): Proporção do conjunto de teste.
    random_state (int): Semente para reprodutibilidade.

    Retorna:
    str: Texto formatado com os três melhores modelos e suas métricas.
    """
    
    algoritmos = {
        'RandomForest': RandomForestClassifier(random_state=random_state),
        'DecisionTree': DecisionTreeClassifier(random_state=random_state),
        'LogisticRegression': LogisticRegression(max_iter=1000, random_state=random_state),
        'SVM': SVC(probability=True, random_state=random_state),
        'KNN': KNeighborsClassifier(),
        'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=random_state),
        'ANN': MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, random_state=random_state)
    }
    
    # Separando features e target
    X = df.drop(columns=[target])
    y = df[target]
    
    # Dividindo os dados
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    # Aplicando apenas RandomUnderSampler
    rus = RandomUnderSampler(random_state=random_state)
    X_res, y_res = rus.fit_resample(X_train, y_train)
    
    resultados = []
    
    for nome_alg, modelo in algoritmos.items():
        modelo.fit(X_res, y_res)
        y_scores = modelo.predict_proba(X_test)[:, 1]  # Probabilidade da classe positiva (fraude)
        
        # Calculando a AUC da Curva Precisão-Recall
        precision, recall, _ = precision_recall_curve(y_test, y_scores)
        pr_auc = auc(recall, precision)
        
        resultados.append({
            'modelo': nome_alg,
            'pr_auc': pr_auc
        })
    
    # Ordenando pelos três melhores modelos com base no PR AUC
    melhores_resultados = sorted(resultados, key=lambda x: x['pr_auc'], reverse=True)[:3]
    
    return (f"1º Lugar: {melhores_resultados[0]['modelo']} - PR AUC: {melhores_resultados[0]['pr_auc']:.4f}\n"
            f"2º Lugar: {melhores_resultados[1]['modelo']} - PR AUC: {melhores_resultados[1]['pr_auc']:.4f}\n"
            f"3º Lugar: {melhores_resultados[2]['modelo']} - PR AUC: {melhores_resultados[2]['pr_auc']:.4f}")


# COMO USAR:

# resultado = testar_modelos_com_undersampling(df, "target")
# print(resultado)





# DOWNCAST

def downcast_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Realiza o downcast dos tipos de dados numéricos de um DataFrame,
    reduzindo o consumo de memória sem introduzir valores nulos.
    
    Parâmetros:
        df (pd.DataFrame): DataFrame de entrada.
    
    Retorna:
        pd.DataFrame: DataFrame otimizado.
    """
    df_opt = df.copy()
    
    # Downcast de inteiros
    for col in df_opt.select_dtypes(include=['int', 'int64', 'int32']).columns:
        df_opt[col] = pd.to_numeric(df_opt[col], downcast='integer')
    
    # Downcast de floats
    for col in df_opt.select_dtypes(include=['float', 'float64', 'float32']).columns:
        df_opt[col] = pd.to_numeric(df_opt[col], downcast='float')
    
    return df_opt


# Aplicando a função ao DataFrame transacoes

# transacoes = downcast_dataframe(transacoes)

