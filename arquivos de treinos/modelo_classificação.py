import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import re
from sklearn.svm import SVC
import joblib

# Carregar os dados
candidatos_df = pd.read_csv('candidatos_tecnologia.csv')
vagas_df = pd.read_csv('vagas_tecnologia.csv')

# Função para comparar habilidades entre candidato e vaga
def calcular_match_habilidades(habilidades_candidato, requisitos_vaga):
    if pd.isna(habilidades_candidato) or pd.isna(requisitos_vaga):
        return 0.0
    
    # Converter strings para listas de habilidades
    habs_candidato = [h.strip().lower() for h in str(habilidades_candidato).split(',')]
    reqs_vaga = [r.strip().lower() for r in str(requisitos_vaga).split(',')]
    
    # Calcular matches
    matches = sum(1 for h in habs_candidato if any(h in r for r in reqs_vaga))
    
    # Calcular percentual de requisitos atendidos
    if len(reqs_vaga) > 0:
        return matches / len(reqs_vaga)
    else:
        return 0.0

# Função para verificar match de experiência
def verificar_experiencia(anos_experiencia, experiencia_min):
    if pd.isna(anos_experiencia) or pd.isna(experiencia_min):
        return 0
    return 1 if anos_experiencia >= experiencia_min else 0

# Função para verificar match de salário
def verificar_salario(pretensao, salario_min, salario_max):
    if pd.isna(pretensao) or pd.isna(salario_min) or pd.isna(salario_max):
        return 0.5  # Valor neutro para dados faltantes
    
    if pretensao < salario_min:
        return 1.0  # Bom para a empresa
    elif pretensao > salario_max:
        return 0.0  # Fora do orçamento
    else:
        # Normalizar dentro da faixa
        return 1 - ((pretensao - salario_min) / (salario_max - salario_min))

# Função para verificar correspondência na área de formação
def verificar_area_formacao(formacao, area_vaga):
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
    
    return 0.3  # Valor padrão baixo para formações não diretamente relacionadas

# Criar um conjunto de dados simulado de candidatos aplicando para vagas específicas
np.random.seed(42)
n_amostras = 5000  # Número de aplicações simuladas

# Selecionar IDs aleatoriamente (com repetições permitidas)
candidato_ids = np.random.choice(candidatos_df['ID'].values, n_amostras)
vaga_ids = np.random.choice(vagas_df['ID'].values, n_amostras)

aplicacoes = []

for i in range(n_amostras):
    candidato_id = candidato_ids[i]
    vaga_id = vaga_ids[i]
    
    candidato = candidatos_df[candidatos_df['ID'] == candidato_id].iloc[0]
    vaga = vagas_df[vagas_df['ID'] == vaga_id].iloc[0]
    
    # Calcular métricas de match
    match_habilidades = calcular_match_habilidades(candidato['Habilidades_Tecnicas'], vaga['Requisitos'])
    match_soft_skills = calcular_match_habilidades(candidato['Soft_Skills'], vaga['Requisitos'])
    match_experiencia = verificar_experiencia(candidato['Anos_Experiencia'], vaga['Experiencia_Min_Anos'])
    match_salario = verificar_salario(candidato['Pretensao_Salarial'], vaga['Salario_Min'], vaga['Salario_Max'])
    match_area = verificar_area_formacao(candidato['Formacao'], vaga['Area'])
    
    # Criar score composto para determinar se é aderente ou não (simulação)
    score_composto = (
        match_habilidades * 0.4 + 
        match_soft_skills * 0.15 + 
        match_experiencia * 0.2 + 
        match_salario * 0.15 + 
        match_area * 0.1
    )
    
    # Adicionar alguma aleatoriedade (fatores subjetivos)
    score_composto = score_composto * (0.85 + 0.3 * np.random.random())
    
    # Definir aderência com base no score (threshold arbitrário)
    aderente = 1 if score_composto > 0.6 else 0
    
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

# Criar DataFrame de aplicações
aplicacoes_df = pd.DataFrame(aplicacoes)

# ----- Visualização da distribuição dos dados -----
print("Distribuição de candidatos aderentes vs não aderentes:")
print(aplicacoes_df['Aderente'].value_counts())
print(f"Porcentagem de aderentes: {aplicacoes_df['Aderente'].mean() * 100:.2f}%")

# Visualizar distribuição de scores
plt.figure(figsize=(10, 6))
sns.histplot(data=aplicacoes_df, x='Score_Total', hue='Aderente', bins=30, kde=True)
plt.title('Distribuição de Scores por Classificação')
plt.savefig('distribuicao_scores.png')
plt.close()

# Visualizar importância das features para a classificação
plt.figure(figsize=(12, 8))
correlations = aplicacoes_df.corr()['Aderente'].sort_values(ascending=False)
sns.barplot(x=correlations.index, y=correlations.values)
plt.xticks(rotation=45, ha='right')
plt.title('Correlação das Features com a Aderência')
plt.tight_layout()
plt.savefig('correlacao_features.png')
plt.close()

# ----- Preparação para o modelo de classificação -----

# Dividir em conjuntos de treino e teste
X = aplicacoes_df.drop(['Aderente', 'CandidatoID', 'VagaID', 'Score_Total'], axis=1)
y = aplicacoes_df['Aderente']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Preprocessamento
numeric_features = ['Idade', 'Anos_Experiencia', 'Experiencia_Minima_Vaga', 
                    'Match_Habilidades', 'Match_Soft_Skills', 'Match_Experiencia', 
                    'Match_Salario', 'Match_Area']
categorical_features = ['Nivel_Formacao', 'Nivel_Vaga']

numeric_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Criar e treinar o modelo
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
])

# Treinar o modelo
model.fit(X_train, y_train)

# Avaliar o modelo
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]  # Probabilidade da classe positiva

# Métricas de avaliação
print("\n----- Avaliação do Modelo -----")
print(f"Acurácia: {accuracy_score(y_test, y_pred):.4f}")
print("\nRelatório de Classificação:")
print(classification_report(y_test, y_pred))

# Matriz de confusão
plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Previsto')
plt.ylabel('Real')
plt.title('Matriz de Confusão')
plt.savefig('matriz_confusao.png')
plt.close()

# Importância das features
if hasattr(model[-1], 'feature_importances_'):
    # Obter nomes das colunas após transformação
    ohe_features = []
    if hasattr(model[0], 'transformers_'):
        for name, _, column_names in model[0].transformers_:
            if name == 'cat':
                for col in column_names:
                    categories = model[0].named_transformers_[name]['onehot'].categories_[0]
                    for cat in categories:
                        ohe_features.append(f"{col}_{cat}")
            else:
                ohe_features.extend(column_names)
    
    # Usar os nomes originais se não conseguir encontrar os transformados
    if not ohe_features:
        ohe_features = X.columns
    
    # Importância das features
    feature_importances = model[-1].feature_importances_
    
    # Plotar importância das features
    plt.figure(figsize=(12, 8))
    indices = np.argsort(feature_importances)[::-1]
    top_indices = indices[:min(len(indices), 10)]  # Top 10 features
    
    plt.bar(range(len(top_indices)), feature_importances[top_indices])
    plt.xticks(range(len(top_indices)), [ohe_features[i] if i < len(ohe_features) else f"feature_{i}" for i in top_indices], rotation=45, ha='right')
    plt.title('Top 10 Features mais Importantes')
    plt.tight_layout()
    plt.savefig('feature_importance.png')
    plt.close()

# Salvar o modelo para uso posterior
joblib.dump(model, 'modelo_aderencia_candidatos.joblib')
print("\nModelo salvo como 'modelo_aderencia_candidatos.joblib'")

# ----- Criação de uma função para utilizar o modelo treinado -----
def classificar_candidato_para_vaga(candidato_dict, vaga_dict, modelo):
    """
    Classifica se um candidato é aderente ou não para uma vaga específica.
    
    Args:
        candidato_dict: Dicionário com dados do candidato
        vaga_dict: Dicionário com dados da vaga
        modelo: Modelo treinado para classificação
        
    Returns:
        dict: Resultado da classificação com score e probabilidade
    """
    # Calcular métricas de match
    match_habilidades = calcular_match_habilidades(candidato_dict.get('Habilidades_Tecnicas', ''), 
                                                 vaga_dict.get('Requisitos', ''))
    match_soft_skills = calcular_match_habilidades(candidato_dict.get('Soft_Skills', ''), 
                                                 vaga_dict.get('Requisitos', ''))
    match_experiencia = verificar_experiencia(candidato_dict.get('Anos_Experiencia', 0), 
                                            vaga_dict.get('Experiencia_Min_Anos', 0))
    match_salario = verificar_salario(candidato_dict.get('Pretensao_Salarial', 0), 
                                    vaga_dict.get('Salario_Min', 0), 
                                    vaga_dict.get('Salario_Max', 0))
    match_area = verificar_area_formacao(candidato_dict.get('Formacao', ''), 
                                        vaga_dict.get('Area', ''))
    
    # Criar DataFrame para previsão
    dados_previsao = pd.DataFrame([{
        'Idade': candidato_dict.get('Idade', 0),
        'Anos_Experiencia': candidato_dict.get('Anos_Experiencia', 0),
        'Nivel_Formacao': candidato_dict.get('Nivel_Formacao', 'Graduação'),
        'Experiencia_Minima_Vaga': vaga_dict.get('Experiencia_Min_Anos', 0),
        'Nivel_Vaga': vaga_dict.get('Nivel', 'Júnior'),
        'Match_Habilidades': match_habilidades,
        'Match_Soft_Skills': match_soft_skills,
        'Match_Experiencia': match_experiencia,
        'Match_Salario': match_salario,
        'Match_Area': match_area
    }])
    
    # Fazer previsão
    aderente = modelo.predict(dados_previsao)[0]
    probabilidade = modelo.predict_proba(dados_previsao)[0][1]  # Probabilidade da classe positiva
    
    # Calcular score composto (similar ao usado no treinamento)
    score_composto = (
        match_habilidades * 0.4 + 
        match_soft_skills * 0.15 + 
        match_experiencia * 0.2 + 
        match_salario * 0.15 + 
        match_area * 0.1
    )
    
    # Calcular métricas detalhadas para explicabilidade
    detalhes_match = {
        'Match_Habilidades_Tecnicas': {
            'score': match_habilidades,
            'peso': 0.4,
            'contribuicao': match_habilidades * 0.4
        },
        'Match_Soft_Skills': {
            'score': match_soft_skills,
            'peso': 0.15,
            'contribuicao': match_soft_skills * 0.15
        },
        'Match_Experiencia': {
            'score': match_experiencia,
            'peso': 0.2,
            'contribuicao': match_experiencia * 0.2
        },
        'Match_Salario': {
            'score': match_salario,
            'peso': 0.15,
            'contribuicao': match_salario * 0.15
        },
        'Match_Area_Formacao': {
            'score': match_area,
            'peso': 0.1,
            'contribuicao': match_area * 0.1
        }
    }
    
    return {
        'aderente': bool(aderente),
        'probabilidade': float(probabilidade),
        'score_composto': float(score_composto),
        'detalhes_match': detalhes_match
    }

# ----- Exemplo de uso do modelo -----
print("\n----- Exemplo de uso do modelo -----")

# Carregar modelo treinado
modelo_carregado = joblib.load('modelo_aderencia_candidatos.joblib')

# Selecionar um candidato e uma vaga aleatórios para teste
candidato_teste = candidatos_df.sample(1).iloc[0].to_dict()
vaga_teste = vagas_df.sample(1).iloc[0].to_dict()

print(f"Candidato: {candidato_teste['Nome']} - {candidato_teste['Formacao']}")
print(f"Vaga: {vaga_teste['Titulo']} - {vaga_teste['Empresa']}")

# Classificar candidato
resultado = classificar_candidato_para_vaga(candidato_teste, vaga_teste, modelo_carregado)

print(f"\nResultado da classificação:")
print(f"Aderente: {'Sim' if resultado['aderente'] else 'Não'}")
print(f"Probabilidade de aderência: {resultado['probabilidade']:.2%}")
print(f"Score composto: {resultado['score_composto']:.2f}")

print("\nDetalhes do match:")
for criterio, dados in resultado['detalhes_match'].items():
    print(f"  {criterio}: {dados['score']:.2f} (peso: {dados['peso']:.2f}, contribuição: {dados['contribuicao']:.2f})")

# ----- Função para recomendação de vagas para um candidato -----
def recomendar_vagas_para_candidato(candidato_dict, todas_vagas_df, modelo, top_n=5):
    """
    Recomenda as melhores vagas para um candidato específico.
    
    Args:
        candidato_dict: Dicionário com dados do candidato
        todas_vagas_df: DataFrame com todas as vagas disponíveis
        modelo: Modelo treinado para classificação
        top_n: Número de vagas a recomendar
        
    Returns:
        DataFrame: Top N vagas recomendadas com scores
    """
    resultados = []
    
    for _, vaga in todas_vagas_df.iterrows():
        vaga_dict = vaga.to_dict()
        resultado = classificar_candidato_para_vaga(candidato_dict, vaga_dict, modelo)
        
        resultados.append({
            'VagaID': vaga_dict['ID'],
            'Titulo': vaga_dict['Titulo'],
            'Empresa': vaga_dict['Empresa'],
            'Probabilidade': resultado['probabilidade'],
            'Score': resultado['score_composto'],
            'Aderente': resultado['aderente']
        })
    
    # Criar DataFrame de resultados e ordenar pelo score
    resultados_df = pd.DataFrame(resultados)
    resultados_df = resultados_df.sort_values(by='Probabilidade', ascending=False)
    
    return resultados_df.head(top_n)

# Exemplo de recomendação de vagas
print("\n----- Exemplo de recomendação de vagas -----")
candidato_recomendacao = candidatos_df.sample(1).iloc[0].to_dict()
print(f"Recomendações para: {candidato_recomendacao['Nome']}")
print(f"Formação: {candidato_recomendacao['Formacao']}")
print(f"Experiência: {candidato_recomendacao['Anos_Experiencia']} anos")
print(f"Habilidades: {candidato_recomendacao['Habilidades_Tecnicas']}")

# Usar apenas um subconjunto de vagas para economizar tempo de processamento
amostra_vagas = vagas_df.sample(100)
recomendacoes = recomendar_vagas_para_candidato(candidato_recomendacao, amostra_vagas, modelo_carregado, top_n=5)

print("\nTop 5 vagas recomendadas:")
for i, (_, rec) in enumerate(recomendacoes.iterrows(), 1):
    print(f"{i}. {rec['Titulo']} - {rec['Empresa']}")
    print(f"   Probabilidade de aderência: {rec['Probabilidade']:.2%}")
    print(f"   Score: {rec['Score']:.2f}")
    print(f"   Aderente: {'Sim' if rec['Aderente'] else 'Não'}")
    print()

# ----- Função para recomendação de candidatos para uma vaga -----
def recomendar_candidatos_para_vaga(vaga_dict, todos_candidatos_df, modelo, top_n=5):
    """
    Recomenda os melhores candidatos para uma vaga específica.
    
    Args:
        vaga_dict: Dicionário com dados da vaga
        todos_candidatos_df: DataFrame com todos os candidatos disponíveis
        modelo: Modelo treinado para classificação
        top_n: Número de candidatos a recomendar
        
    Returns:
        DataFrame: Top N candidatos recomendados com scores
    """
    resultados = []
    
    for _, candidato in todos_candidatos_df.iterrows():
        candidato_dict = candidato.to_dict()
        resultado = classificar_candidato_para_vaga(candidato_dict, vaga_dict, modelo)
        
        resultados.append({
            'CandidatoID': candidato_dict['ID'],
            'Nome': candidato_dict['Nome'],
            'Formacao': candidato_dict['Formacao'],
            'Anos_Experiencia': candidato_dict['Anos_Experiencia'],
            'Probabilidade': resultado['probabilidade'],
            'Score': resultado['score_composto'],
            'Aderente': resultado['aderente']
        })
    
    # Criar DataFrame de resultados e ordenar pelo score
    resultados_df = pd.DataFrame(resultados)
    resultados_df = resultados_df.sort_values(by='Probabilidade', ascending=False)
    
    return resultados_df.head(top_n)

# Exemplo de recomendação de candidatos
print("\n----- Exemplo de recomendação de candidatos -----")
vaga_recomendacao = vagas_df.sample(1).iloc[0].to_dict()
print(f"Recomendações para vaga: {vaga_recomendacao['Titulo']} - {vaga_recomendacao['Empresa']}")
print(f"Área: {vaga_recomendacao['Area']}")
print(f"Requisitos: {vaga_recomendacao['Requisitos']}")

# Usar apenas um subconjunto de candidatos para economizar tempo de processamento
amostra_candidatos = candidatos_df.sample(100)
recomendacoes = recomendar_candidatos_para_vaga(vaga_recomendacao, amostra_candidatos, modelo_carregado, top_n=5)

print("\nTop 5 candidatos recomendados:")
for i, (_, rec) in enumerate(recomendacoes.iterrows(), 1):
    print(f"{i}. {rec['Nome']} - {rec['Formacao']}")
    print(f"   Experiência: {rec['Anos_Experiencia']} anos")
    print(f"   Probabilidade de aderência: {rec['Probabilidade']:.2%}")
    print(f"   Score: {rec['Score']:.2f}")
    print(f"   Aderente: {'Sim' if rec['Aderente'] else 'Não'}")
    print()

print("\n----- Sistema de classificação completo -----")
print("Os arquivos gerados são:")
print("1. candidatos_tecnologia.csv - Base de dados com 1.000 candidatos")
print("2. vagas_tecnologia.csv - Base de dados com 2.000 vagas")
print("3. modelo_aderencia_candidatos.joblib - Modelo de classificação treinado")
print("4. distribuicao_scores.png - Gráfico da distribuição de scores")
print("5. correlacao_features.png - Gráfico de correlação das features")
print("6. matriz_confusao.png - Matriz de confusão do modelo")
print("7. feature_importance.png - Importância das features para o modelo")