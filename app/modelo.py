import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline   # SMOTE para balanceamento
import matplotlib.pyplot as plt
import seaborn as sns
import re
from joblib import dump

# Carregar os dados
candidatos_csv = pd.read_csv('app/db/candidatos_tecnologia.csv')
vagas_csv = pd.read_csv('app/db/vagas_tecnologia.csv')

# Funções de comparação de habilidades, experiência, salário e área de formação
def calcular_match_habilidades(habilidades_candidato, requisitos_vaga):
    if pd.isna(habilidades_candidato) or pd.isna(requisitos_vaga):
        return 0.0
    habi_candidato = [h.strip().lower() for h in str(habilidades_candidato).split(',')]
    reqs_vaga = [r.strip().lower() for r in str(requisitos_vaga).split(',')]
    matches = sum(1 for h in habi_candidato if any(h in r for r in reqs_vaga))
    return matches / len(reqs_vaga) if len(reqs_vaga) > 0 else 0.0

def verificar_match_experiencia(anos_experiencia, experiencia_min):
    if pd.isna(anos_experiencia) or pd.isna(experiencia_min):
        return 0
    return 1 if anos_experiencia >= experiencia_min else 0

def verificar_match_salario(pretensao, salario_min, salario_max):
    if pd.isna(pretensao) or pd.isna(salario_min) or pd.isna(salario_max):
        return 0.5
    if pretensao < salario_min:
        return 1.0
    elif pretensao > salario_max:
        return 0.0
    else:
        return 1 - ((pretensao - salario_min) / (salario_max - salario_min))

def verificar_match_area_formacao(formacao, area_vaga):
    if pd.isna(formacao) or pd.isna(area_vaga):
        return 0.5
    # Mapeamento simplificado de áreas de formação para áreas de vagas
    mapeamento = {
        "Ciência da Computação": ["Desenvolvimento Web", "Desenvolvimento Mobile", "Desenvolvimento Full-Stack", 
        "Ciência de Dados", "Machine Learning", "Inteligência Artificial"],
        "Engenharia de Software": ["Desenvolvimento Web", "Desenvolvimento Full-Stack", "Arquitetura de Software", 
        "DevOps", "QA e Testes"],
        "Sistemas de Informação": ["Desenvolvimento Web", "Business Intelligence", "Banco de Dados", 
        "Administração de Sistemas"],
        "Análise e Desenvolvimento de Sistemas": ["Desenvolvimento Web", "Desenvolvimento Mobile", 
                                                "Desenvolvimento Full-Stack"],
        "Engenharia da Computação": ["Desenvolvimento Embedded", "IoT", "Desenvolvimento Full-Stack", 
                                    "Machine Learning"],
        "Tecnologia em Redes": ["Administração de Redes", "Cloud Computing", "Segurança da Informação"],
        "Segurança da Informação": ["Segurança da Informação", "Administração de Redes"],
        "Tecnologia em Banco de Dados": ["Banco de Dados", "Engenharia de Dados", "Business Intelligence"],
        "Tecnologia em Desenvolvimento Web": ["Desenvolvimento Web", "UX/UI Design"],
        "Engenharia Elétrica": ["IoT", "Sistemas Embarcados"],
        "Matemática Computacional": ["Ciência de Dados", "Machine Learning", "Inteligência Artificial"]
    }
    formacao_lower = formacao.lower()
    area_vaga_lower = area_vaga.lower()
    for form, areas in mapeamento.items():
        if form.lower() in formacao_lower:
            for area in areas:
                if area.lower() in area_vaga_lower:
                    return 1.0
    return 0.3

# Gerar conjunto de dados simulado de candidatos e vagas
np.random.seed(42)
n_amostras = 5000
candidato_ids = np.random.choice(candidatos_csv['ID'].values, n_amostras)
vaga_ids = np.random.choice(vagas_csv['ID'].values, n_amostras)

aplicacoes = []

for i in range(n_amostras):
    candidato_id = candidato_ids[i]
    vaga_id = vaga_ids[i]
    candidato = candidatos_csv[candidatos_csv['ID'] == candidato_id].iloc[0]
    vaga = vagas_csv[vagas_csv['ID'] == vaga_id].iloc[0]
    
    match_habilidades = calcular_match_habilidades(candidato['Habilidades_Tecnicas'], vaga['Requisitos'])
    match_soft_skills = calcular_match_habilidades(candidato['Soft_Skills'], vaga['Requisitos'])
    match_experiencia = verificar_match_experiencia(candidato['Anos_Experiencia'], vaga['Experiencia_Min_Anos'])
    match_salario = verificar_match_salario(candidato['Pretensao_Salarial'], vaga['Salario_Min'], vaga['Salario_Max'])
    match_area = verificar_match_area_formacao(candidato['Formacao'], vaga['Area'])
    
    score_composto = (match_habilidades * 0.4 + 
                      match_soft_skills * 0.15 + 
                      match_experiencia * 0.2 + 
                      match_salario * 0.15 + 
                      match_area * 0.1)
    
    score_composto = score_composto * (0.85 + 0.3 * np.random.random())
    aderente = 1 if score_composto > 0.4 else 0
    
    aplicacao = {
        'CandidatoID': candidato_id,
        'VagaID': vaga_id,
        'Idade': candidato['Idade'],
        'Anos_Experiencia': candidato['Anos_Experiencia'],
        'Nivel_Formacao': candidato['Nivel_Formacao'],
        'Experiencia_Minima_Vaga': vaga['Experiencia_Min_Anos'],
        'Nivel_Vaga': vaga['Nivel'],
        'Match_Habilidades': match_habilidades,
        'Match_Soft_Skills': match_soft_skills,
        'Match_Experiencia': match_experiencia,
        'Match_Salario': match_salario,
        'Match_Area': match_area,
        'Score_Total': score_composto,
        'Aderente': aderente
    }
    aplicacoes.append(aplicacao)

aplicacoes_df = pd.DataFrame(aplicacoes)

# Prepara os dados para o modelo
X = aplicacoes_df.drop(['Aderente', 'CandidatoID', 'VagaID', 'Score_Total'], axis=1)
y = aplicacoes_df['Aderente']

# Dividir dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Preprocessamento
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), X.select_dtypes(include=['object']).columns),
        ('num', StandardScaler(), X.select_dtypes(exclude=['object']).columns)
    ])

# Criando o modelo
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('smote', SMOTE),
    ('classifier', RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42))
])

# Treinamento do modelo
model.fit(X_train, y_train)

# Predição e avaliação
y_pred = model.predict(X_test)

# Métricas
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# Exibir as métricas
print(f"Accuracy: {accuracy :.2f}")
print(f"Precision: {precision :.2f}")
print(f"Recall: {recall :.2f}")
print(f"F1 Score: {f1 :.2f}")

# Relatório detalhado
print(classification_report(y_test, y_pred))

dump({'model': model.named_steps['classifier'],
      'preprocessor': model.named_steps['preprocessor'],
      'smote': model.named_steps['smote']}, 
     'modelo_aderencia_candidatos.joblib')