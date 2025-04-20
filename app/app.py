import pandas as pd
import numpy as np
import joblib
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from pdfminer.high_level import extract_text
import re
import spacy
from spacy.matcher import Matcher
from pathlib import Path
from io import BytesIO
from docx import Document
from pathlib import Path
from joblib import load


try:
    modelo = joblib.load("app/modelo_aderencia_candidatos.joblib")
except Exception as e:
    st.error(f"Erro ao carregar modelo: {str(e)}")
    st.stop()

VAGAS_PATH = "app/db/vagas_tecnologia.csv"
# Carregar modelo de NLP
nlp = spacy.load("pt_core_news_sm")

# Funções de processamento de arquivos
def extract_text_from_pdf(file):
    try:
        # Garantir que o arquivo está em bytes
        file_bytes = file.getvalue()
        return extract_text(BytesIO(file_bytes))
    except Exception as e:
        st.error(f"Erro ao ler PDF: {str(e)}")
        return ""

def extract_text_from_docx(file):
    doc = Document(BytesIO(file.read()))
    return '\n'.join([paragraph.text for paragraph in doc.paragraphs])

def parse_resume(text):
    text = text.replace('\n', ' ').strip()
    doc = nlp(text.lower())
    
    # Extrair informações
    info = {
        'Nome': 'Candidato sem nome',
        'Idade': 30,
        'Formacao': '',
        'Nivel_Formacao': 'Graduação',
        'Anos_Experiencia': 0,
        'Habilidades_Tecnicas': '',
        'Soft_Skills': '',
        'Pretensao_Salarial': 0
    }

     # Padrões para detecção de nome
    matcher = Matcher(nlp.vocab)
    patterns = [
        [{"LOWER": "nome"}, {"LOWER": ":"}, {"POS": "PROPN", "OP": "+"}],
        [{"LOWER": "nome"}, {"LOWER": "completo"}, {"LOWER": ":"}, {"POS": "PROPN", "OP": "+"}],
        [{"LOWER": "currículo"}, {"LOWER": "vitae"}, {"LOWER": "de"}, {"POS": "PROPN", "OP": "+"}]
    ]
    matcher.add("NOME", patterns)
    

    matches = matcher(doc)
    for match_id, start, end in matches:
        span = doc[start:end]
        if span.text.lower() not in ["nome", "nome completo"]:
            info['Nome'] = span.text.title()
            break

    
    # Fallback 1: Procurar a primeira entidade PERSON
    if info['Nome'] == 'Candidato sem nome':
        for ent in doc.ents:
            if ent.label_ == "PER":
                info['Nome'] = ent.text.title()
                break
    
    # Fallback 2: Usar primeira linha do texto
    if info['Nome'] == 'Candidato sem nome':
        first_line = text.split('.')[0].strip()
        if len(first_line.split()) > 1:
            info['Nome'] = first_line.title()


    # Extrair nome
    for ent in doc.ents:
        if ent.label_ == "PER":
            info['Nome'] = ent.text.title()
            break
    
    # Extrair experiência
    exp_matches = re.findall(r'(\d+)\+?\s*(anos?|years?)', text, re.IGNORECASE)
    if exp_matches:
        info['Anos_Experiencia'] = max([int(m[0]) for m in exp_matches])
    
    # Extrair habilidades
    tech_skills = []
    skills_keywords = ['python', 'java', 'sql', 'machine learning', 'aws', 'docker','spark', 'tensorflow', 'pytorch', 'flask', 'django', 'git']
    for token in doc:
        if token.text in skills_keywords:
            tech_skills.append(token.text.title())
    info['Habilidades_Tecnicas'] = ', '.join(list(set(tech_skills)))
    
    # Extrair formação
    education = []
    for sent in doc.sents:
        if any(word in sent.text for word in ['bacharel', 'graduação', 'mestrado', 'doutorado',
                                            'bachelor', 'master', 'phd']):
            education.append(sent.text.capitalize())
    info['Formacao'] = ', '.join(education[:3])
    
    return info
# Carregar as funções definidas anteriormente
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

def verificar_experiencia(anos_experiencia, experiencia_min):
    if pd.isna(anos_experiencia) or pd.isna(experiencia_min):
        return 0
    return 1 if anos_experiencia >= experiencia_min else 0

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

def classificar_candidato_para_vaga(candidato_dict, vaga_dict, modelo):
    """
    Classifica se um candidato é aderente ou não para uma vaga específica.
    
    Args:
        candidato_dict: Dicionário com dados do candidato
        vaga_dict: Dicionário com dados da vaga
        modelo: Dicionário com componentes do modelo treinado
        
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
    
    # Aplicar pré-processamento
    processed_data = modelo['preprocessor'].transform(dados_previsao)
    
    # Fazer previsão
    classifier = modelo['model']
    aderente = classifier.predict(processed_data)[0]
    probabilidade = classifier.predict_proba(processed_data)[0][1]
    
    # Calcular score composto
    score_composto = (
        match_habilidades * 0.4 + 
        match_soft_skills * 0.15 + 
        match_experiencia * 0.2 + 
        match_salario * 0.15 + 
        match_area * 0.1
    )
    
    # Detalhes para explicabilidade
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
# Interface Streamlit modificada
st.set_page_config(page_title="Sistema de Classificação de Candidatos", layout="wide")

st.title("Analisador Automático de Currículos")

st.write("""
Sistema inteligente para análise de currículos contra vagas pré-cadastradas.
Carregue os currículos (PDF ou DOCX) para ver a compatibilidade com nossas vagas.
""")

# Sidebar para carregar arquivos
st.sidebar.header("Carregamento de Dados")

uploaded_candidatos = st.sidebar.file_uploader(
    "Carregar Candidatos (CSV, PDF, DOCX)",
    type=["csv", "pdf", "docx"],
    accept_multiple_files=True
)

vagas_df = pd.read_csv(VAGAS_PATH)

# VERIFICAÇÃO DOS ARQUIVOS ESSENCIAIS
if not Path("app/modelo_aderencia_candidatos.joblib").exists():
    st.error("Modelo não encontrado! Certifique-se que o arquivo '/home/guilherme/Documentos/GitHub/Classificador_RH/app/modelo_aderencia_candidatos.joblib' está na pasta do projeto.")
    st.stop()

if not Path(VAGAS_PATH).exists():
    st.error("Arquivo de vagas não encontrado! Certifique-se que o arquivo 'vagas.csv' está na pasta do projeto.")
    st.stop()


# Processar arquivos carregados
candidatos_df = pd.DataFrame()


if uploaded_candidatos:
    try:
            # Carregar vagas primeiro
        VAGAS_PATH = "app/db/vagas_tecnologia.csv"
       
        # Processar candidatos
        csv_files = [f for f in uploaded_candidatos if f.name.endswith('.csv')]
        if csv_files:
            candidatos_df = pd.read_csv(csv_files[0])
        else:
            candidatos = []
            for file in uploaded_candidatos:
                try:
                    if file.name.endswith('.pdf'):
                        text = extract_text_from_pdf(file)
                    elif file.name.endswith('.docx'):
                        text = extract_text_from_docx(file)
                    else:
                        continue
                    if not text:
                        continue
                    parsed = parse_resume(text)
                    parsed['Nome'] = parsed['Nome'] or file.name.replace('.pdf', '').replace('.docx', '').title()
                    parsed['ID'] = len(candidatos) + 1
                    candidatos.append(parsed)
                except Exception as e:
                    st.error(f"Erro ao processar o arquivo {file.name}: {str(e)}")
                    continue
            candidatos_df = pd.DataFrame(candidatos)
                    
            # Garantir colunas obrigatórias
            required_columns = ['Idade', 'Nivel_Formacao', 'Soft_Skills', 'Pretensao_Salarial',
                              'Habilidades_Tecnicas', 'Formacao', 'Anos_Experiencia', 'Nome']
            
            for col in required_columns:
                if col not in candidatos_df.columns:
                    if col == 'Idade': 
                        candidatos_df[col] = 30
                    elif col == 'Nivel_Formacao': 
                        candidatos_df[col] = 'Não Especificado'
                    elif col == 'Soft_Skills': 
                        candidatos_df[col] = 'Comunicação, Trabalho em equipe'
                    elif col == 'Pretensao_Salarial': 
                        candidatos_df[col] = 0
                    else: 
                        candidatos_df[col] = ''

        st.sidebar.success("Dados carregados com sucesso! Modelo já incorporado no sistema.")
    
    except Exception as e:
        st.error(f"Erro no processamento: {str(e)}")
        st.stop()  # Parar execução se ocorrer erro



    # Exibir estatísticas básicas
    st.header("Estatísticas Básicas")
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Candidatos")
        st.write(f"Total de candidatos: {len(candidatos_df)}")
        st.write(f"Média de idade: {candidatos_df['Idade'].mean():.1f} anos")
        st.write(f"Média de experiência: {candidatos_df['Anos_Experiencia'].mean():.1f} anos")
    
    with col2:
        st.subheader("Vagas")
        st.write(f"Total de vagas: {len(vagas_df)}")
        st.write(f"Média de salário mínimo: R$ {vagas_df['Salario_Min'].mean():.2f}")
        st.write(f"Média de salário máximo: R$ {vagas_df['Salario_Max'].mean():.2f}")
    
    # Tabs para diferentes funcionalidades
    tab1, tab2, tab3 = st.tabs(["Classificação Individual", "Recomendar Vagas", "Recomendar Candidatos"])
    
    with tab1:
        st.header("Classificação Individual")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Selecione um Candidato")
            candidato_id = st.selectbox(
                "ID do Candidato",
                options=candidatos_df['ID'].tolist(),
                format_func=lambda x: f"{x} - {candidatos_df[candidatos_df['ID'] == x]['Nome'].values[0]}"
            )
            
            # Mostrar detalhes do candidato
            candidato = candidatos_df[candidatos_df['ID'] == candidato_id].iloc[0]
            st.write(f"**Nome:** {candidato['Nome']}")
            st.write(f"**Idade:** {candidato['Idade']} anos")
            st.write(f"**Formação:** {candidato['Formacao']} ({candidato['Nivel_Formacao']})")
            st.write(f"**Experiência:** {candidato['Anos_Experiencia']} anos")
            st.write(f"**Pretensão Salarial:** R$ {candidato['Pretensao_Salarial']}")
            st.write("**Habilidades Técnicas:**")
            st.write(candidato['Habilidades_Tecnicas'])
            st.write("**Soft Skills:**")
            st.write(candidato['Soft_Skills'])
        
        with col2:
            st.subheader("Selecione uma Vaga")
            vaga_id = st.selectbox(
                "ID da Vaga",
                options=vagas_df['ID'].tolist(),
                format_func=lambda x: f"{x} - {vagas_df[vagas_df['ID'] == x]['Titulo'].values[0]}"
            )
            
            # Mostrar detalhes da vaga
            vaga = vagas_df[vagas_df['ID'] == vaga_id].iloc[0]
            st.write(f"**Título:** {vaga['Titulo']}")
            st.write(f"**Empresa:** {vaga['Empresa']}")
            st.write(f"**Área:** {vaga['Area']}")
            st.write(f"**Nível:** {vaga['Nivel']}")
            st.write(f"**Experiência Mínima:** {vaga['Experiencia_Min_Anos']} anos")
            st.write(f"**Faixa Salarial:** R$ {vaga['Salario_Min']} - R$ {vaga['Salario_Max']}")
            st.write("**Requisitos:**")
            st.write(vaga['Requisitos'])
        
        if st.button("Classificar Aderência"):
            candidato_dict = candidato.to_dict()
            vaga_dict = vaga.to_dict()
            
            resultado = classificar_candidato_para_vaga(candidato_dict, vaga_dict, modelo)
            
            # Exibir resultado
            st.header("Resultado da Classificação")
            
            # Exibir resultado principal
            status = "Aderente" if resultado['aderente'] else "Não Aderente"
            cor = "green" if resultado['aderente'] else "red"
            
            st.markdown(f"<h3 style='color:{cor}'>Status: {status}</h3>", unsafe_allow_html=True)
            st.write(f"Probabilidade de Aderência: {resultado['probabilidade']:.2%}")
            st.write(f"Score Composto: {resultado['score_composto']:.2f}")
            
            # Criar gráfico de barras para os detalhes
            fig, ax = plt.subplots(figsize=(10, 6))
            
            criterios = list(resultado['detalhes_match'].keys())
            scores = [detalhe['score'] for detalhe in resultado['detalhes_match'].values()]
            contribuicoes = [detalhe['contribuicao'] for detalhe in resultado['detalhes_match'].values()]
            # Continuação do código Streamlit para classificação de candidatos

            # Criar visualização dos critérios de match
            fig, ax = plt.subplots(figsize=(10, 6))
            
            criterios = list(resultado['detalhes_match'].keys())
            scores = [detalhe['score'] for detalhe in resultado['detalhes_match'].values()]
            contribuicoes = [detalhe['contribuicao'] for detalhe in resultado['detalhes_match'].values()]
            
            # Melhorar os nomes para exibição
            criterios_display = [c.replace('Match_', '').replace('_', ' ') for c in criterios]
            
            # Criar gráfico de barras com as contribuições
            bars = ax.bar(criterios_display, contribuicoes, color='skyblue')
            
            # Adicionar valor de score original em cada barra
            for i, bar in enumerate(bars):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'Score: {scores[i]:.2f}', ha='center', va='bottom', rotation=0)
            
            ax.set_ylim(0, 0.5)  # Ajustar limite para visualização adequada
            ax.set_ylabel('Contribuição para o Score Final')
            ax.set_title('Análise Detalhada de Aderência por Critério')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            
            st.pyplot(fig)
            
            # Feedback textual - pontos fortes e lacunas
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Pontos Fortes")
                
                # Identificar pontos fortes (critérios com scores altos)
                pontos_fortes = []
                
                # Verificar habilidades técnicas
                if resultado['detalhes_match']['Match_Habilidades_Tecnicas']['score'] > 0.7:
                    pontos_fortes.append("✅ Forte correspondência em habilidades técnicas")
                
                # Verificar soft skills
                if resultado['detalhes_match']['Match_Soft_Skills']['score'] > 0.6:
                    pontos_fortes.append("✅ Bom conjunto de soft skills para a posição")
                
                # Verificar experiência
                if resultado['detalhes_match']['Match_Experiencia']['score'] == 1:
                    pontos_fortes.append("✅ Experiência profissional adequada")
                
                # Verificar salário
                if resultado['detalhes_match']['Match_Salario']['score'] > 0.7:
                    pontos_fortes.append("✅ Pretensão salarial compatível com a vaga")
                
                # Verificar área de formação
                if resultado['detalhes_match']['Match_Area_Formacao']['score'] > 0.7:
                    pontos_fortes.append("✅ Formação acadêmica alinhada com a área da vaga")
                
                if not pontos_fortes:
                    st.write("Nenhum ponto forte significativo identificado.")
                else:
                    for ponto in pontos_fortes:
                        st.write(ponto)
            
            with col2:
                st.subheader("Lacunas Identificadas")
                
                # Identificar lacunas (critérios com scores baixos)
                lacunas = []
                
                # Verificar habilidades técnicas
                if resultado['detalhes_match']['Match_Habilidades_Tecnicas']['score'] < 0.5:
                    # Identificar habilidades faltantes específicas
                    habs_candidato = set([h.strip().lower() for h in str(candidato['Habilidades_Tecnicas']).split(',')])
                    reqs_vaga = set([r.strip().lower() for r in str(vaga['Requisitos']).split(',')])
                    habilidades_faltantes = [req for req in reqs_vaga if not any(hab in req for hab in habs_candidato)]
                    
                    if habilidades_faltantes:
                        top_5_faltantes = ', '.join(list(habilidades_faltantes)[:5])
                        lacunas.append(f"❌ Habilidades técnicas faltantes: {top_5_faltantes}")
                    else:
                        lacunas.append("❌ Correspondência insuficiente de habilidades técnicas")
                
                # Verificar soft skills
                if resultado['detalhes_match']['Match_Soft_Skills']['score'] < 0.4:
                    lacunas.append("❌ Soft skills não atendem completamente os requisitos da vaga")
                
                # Verificar experiência
                if resultado['detalhes_match']['Match_Experiencia']['score'] == 0:
                    lacunas.append(f"❌ Experiência abaixo do requisito mínimo de {vaga['Experiencia_Min_Anos']} anos")
                
                # Verificar salário
                if resultado['detalhes_match']['Match_Salario']['score'] < 0.3:
                    lacunas.append("❌ Pretensão salarial acima da faixa ofertada")
                
                # Verificar área de formação
                if resultado['detalhes_match']['Match_Area_Formacao']['score'] < 0.4:
                    lacunas.append("❌ Formação não diretamente relacionada à área da vaga")
                
                if not lacunas:
                    st.write("Nenhuma lacuna significativa identificada.")
                else:
                    for lacuna in lacunas:
                        st.write(lacuna)
            
            # Recomendação final
            st.subheader("Recomendação")
            if resultado['probabilidade'] > 0.75:
                st.success("✅ Candidato altamente recomendado para esta vaga.")
            elif resultado['probabilidade'] > 0.5:
                st.warning("⚠️ Candidato potencialmente adequado, recomenda-se avaliação adicional.")
            else:
                st.error("❌ Candidato não recomendado para esta vaga.")

    with tab2:
        st.header("Recomendar Vagas para um Candidato")
        
        # Selecionar candidato para recomendar vagas
        candidato_id = st.selectbox(
            "Selecione um candidato para receber recomendações de vagas",
            options=candidatos_df['ID'].tolist(),
            format_func=lambda x: f"{x} - {candidatos_df[candidatos_df['ID'] == x]['Nome'].values[0]}",
            key="tab2_candidato"
        )
        
        num_recomendacoes = st.slider(
            "Número de vagas a recomendar", 
            min_value=1, 
            max_value=10, 
            value=5
        )
        
        if st.button("Buscar Vagas Recomendadas"):
            candidato = candidatos_df[candidatos_df['ID'] == candidato_id].iloc[0].to_dict()
            
            # Exibir detalhes do candidato
            st.subheader(f"Recomendações para {candidato['Nome']}")
            
            # Calcular score para todas as vagas
            resultados_vagas = []            
            
            # Em um cenário real, poderíamos otimizar ou paginar os resultados
            amostra_vagas = vagas_df.sample(min(100, len(vagas_df)))
            
            with st.spinner("Analisando vagas compatíveis..."):
                for _, vaga in amostra_vagas.iterrows():
                    vaga_dict = vaga.to_dict()
                    resultado = classificar_candidato_para_vaga(candidato, vaga_dict, modelo)
                    
                    resultados_vagas.append({
                        'vaga_id': vaga['ID'],
                        'titulo': vaga['Titulo'],
                        'empresa': vaga['Empresa'],
                        'nivel': vaga['Nivel'],
                        'cidade': vaga['Cidade'],
                        'salario_min': vaga['Salario_Min'],
                        'salario_max': vaga['Salario_Max'],
                        'probabilidade': resultado['probabilidade'],
                        'score': resultado['score_composto']
                    })
            
            # Ordenar por probabilidade de aderência
            resultados_vagas = sorted(resultados_vagas, key=lambda x: x['probabilidade'], reverse=True)
            
            # Mostrar as top N recomendações
            top_recomendacoes = resultados_vagas[:num_recomendacoes]
            
            for i, rec in enumerate(top_recomendacoes):
                st.write(f"### {i+1}. {rec['titulo']} - {rec['empresa']}")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.write(f"**Nível:** {rec['nivel']}")
                    st.write(f"**Local:** {rec['cidade']}")
                with col2:
                    st.write(f"**Faixa Salarial:** R$: {rec['salario_min']} - R$:{rec['salario_max']}")
                with col3:
                    st.write(f"**Match:** {rec['probabilidade']:.2%}")
                    
                    # Cor baseada no score
                    cor = "green" if rec['probabilidade'] > 0.7 else ("orange" if rec['probabilidade'] > 0.5 else "red")
                    st.markdown(f"<div style='background-color:{cor}; height:10px; width:{int(rec['probabilidade']*100)}%; border-radius:5px;'></div>", unsafe_allow_html=True)
                
                # Link para mais detalhes
                if st.button(f"Ver Detalhes da Vaga {rec['vaga_id']}", key=f"vaga_detalhe_{rec['vaga_id']}"):
                    st.session_state.selected_vaga = rec['vaga_id']
                    st.session_state.selected_candidato = candidato_id
                    st.session_state.current_tab = "Classificação Individual"
                
                st.write("---")
    
    with tab3:
        st.header("Recomendar Candidatos para uma Vaga")
        
        # Selecionar vaga para recomendar candidatos
        vaga_id = st.selectbox(
            "Selecione uma vaga para receber recomendações de candidatos",
            options=vagas_df['ID'].tolist(),
            format_func=lambda x: f"{x} - {vagas_df[vagas_df['ID'] == x]['Titulo'].values[0]}",
            key="tab3_vaga"
        )
        
        num_recomendacoes = st.slider(
            "Número de candidatos a recomendar", 
            min_value=1, 
            max_value=15, 
            value=10,
            key="tab3_slider"
        )
        
        if st.button("Buscar Candidatos Recomendados"):
            vaga = vagas_df[vagas_df['ID'] == vaga_id].iloc[0].to_dict()
            
            # Exibir detalhes da vaga
            st.subheader(f"Candidatos recomendados para: {vaga['Titulo']} - {vaga['Empresa']}")
            
            # Calcular score para todos os candidatos
            resultados_candidatos = []
            
            # Para demonstração, limitamos a 200 candidatos para evitar processamento excessivo
            amostra_candidatos = candidatos_df.sample(min(200, len(candidatos_df)))
            
            with st.spinner("Analisando candidatos compatíveis..."):
                for _, candidato in amostra_candidatos.iterrows():
                    candidato_dict = candidato.to_dict()
                    resultado = classificar_candidato_para_vaga(candidato_dict, vaga, modelo)
                    
                    resultados_candidatos.append({
                        'candidato_id': candidato['ID'],
                        'nome': candidato['Nome'],
                        'idade': candidato['Idade'],
                        'experiencia': candidato['Anos_Experiencia'],
                        'formacao': candidato['Formacao'],
                        'nivel_formacao': candidato['Nivel_Formacao'],
                        'pretensao': candidato['Pretensao_Salarial'],
                        'probabilidade': resultado['probabilidade'],
                        'score': resultado['score_composto']
                    })
            
            # Ordenar por probabilidade de aderência
            resultados_candidatos = sorted(resultados_candidatos, key=lambda x: x['probabilidade'], reverse=True)
            
            # Mostrar os top N candidatos
            top_candidatos = resultados_candidatos[:num_recomendacoes]
            
            # Criar uma tabela para visualizar os candidatos
            tabela_dados = {
                'ID': [c['candidato_id'] for c in top_candidatos],
                'Nome': [c['nome'] for c in top_candidatos],
                'Idade': [c['idade'] for c in top_candidatos],
                'Experiência': [f"{c['experiencia']} anos" for c in top_candidatos],
                'Formação': [f"{c['nivel_formacao']} em {c['formacao']}" for c in top_candidatos],
                'Pretensão': [f"R$ {c['pretensao']}" for c in top_candidatos],
                'Match': [f"{c['probabilidade']:.2%}" for c in top_candidatos]
            }
            
            tabela_df = pd.DataFrame(tabela_dados)
            st.dataframe(tabela_df, use_container_width=True)
            
            # Gráfico de barras comparando os scores
            st.subheader("Comparação de Match entre Candidatos")
            fig, ax = plt.subplots(figsize=(10, 6))
            
            nomes = [f"{c['nome'][:15]}..." if len(c['nome']) > 15 else c['nome'] for c in top_candidatos]
            scores = [c['probabilidade'] for c in top_candidatos]
            
            # Definir cores baseadas no score
            cores = ['green' if s > 0.7 else ('orange' if s > 0.5 else 'red') for s in scores]
            
            barras = ax.bar(nomes, scores, color=cores)
            
            # Adicionar valor percentual em cada barra
            for i, barra in enumerate(barras):
                height = barra.get_height()
                ax.text(barra.get_x() + barra.get_width()/2., height + 0.02,
                        f"{scores[i]:.1%}", ha='center', va='bottom', rotation=0)
            
            ax.set_ylim(0, 1.1)
            ax.set_ylabel('Score de Match')
            ax.set_title('Comparação entre Candidatos')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            
            st.pyplot(fig)
            
            # Permitir selecionar candidatos para análise detalhada
            st.subheader("Visualizar Detalhes do Candidato")
            candidato_para_detalhar = st.selectbox(
                "Selecione um candidato para visualizar detalhes:",
                options=[c['candidato_id'] for c in top_candidatos],
                format_func=lambda x: f"{x} - {[c['nome'] for c in top_candidatos if c['candidato_id'] == x][0]}"
            )
            
            if st.button("Ver Análise Detalhada"):
                st.session_state.selected_vaga = vaga_id
                st.session_state.selected_candidato = candidato_para_detalhar
                st.session_state.current_tab = "Classificação Individual"
                st.rerun()

else:
    st.warning("Por favor, carregue seu currículo individual em pdf ou um banco de dados em csv de candidatos.")
    
    # Mostrar informações sobre o formato esperado dos arquivos
    with st.expander("Informações sobre o formato dos arquivos"):
        st.markdown("""
        ### Formatos Aceitos para Candidatos:
        - **CSV**: Com as colunas especificadas anteriormente
        - **PDF/DOCX**: Currículos em formato padrão com informações profissionais
        """)
        
        st.info("O sistema utiliza um modelo treinado (formato .joblib) para realizar as classificações.")

# Adicionar filtros de busca e outras funcionalidades avançadas
st.sidebar.header("Funcionalidades Adicionais")

# Estatísticas e visualizações
if st.sidebar.checkbox("Mostrar Estatísticas e Insights"):
    st.subheader("Estatísticas e Insights do Mercado")
    if 'candidatos_df' in locals() and 'vagas_df' in locals():
        # Criar abas para diferentes visualizações
        viz_tab1, viz_tab2 = st.tabs(["Estatísticas de Candidatos", "Estatísticas de Vagas"])
        
        with viz_tab1:
            st.write("### Distribuição de Candidatos")
            
            # Distribuição por formação
            fig, ax = plt.subplots(figsize=(10, 6))
            candidatos_df['Formacao'].value_counts().head(10).plot(kind='bar', ax=ax)
            plt.title('Top 10 Áreas de Formação')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            st.pyplot(fig)
            
            # Distribuição por faixa de experiência
            fig, ax = plt.subplots(figsize=(10, 6))
            candidatos_df['Anos_Experiencia'].hist(bins=10, ax=ax)
            plt.title('Distribuição de Anos de Experiência')
            plt.xlabel('Anos de Experiência')
            plt.ylabel('Número de Candidatos')
            st.pyplot(fig)
            
            # Nuvem de palavras das habilidades mais comuns
            st.write("### Habilidades Técnicas mais Comuns")
            st.info("Funcionalidade de nuvem de palavras seria implementada aqui")
        
        with viz_tab2:
            st.write("### Distribuição de Vagas")
            
            # Distribuição por área
            fig, ax = plt.subplots(figsize=(10, 6))
            vagas_df['Area'].value_counts().plot(kind='pie', autopct='%1.1f%%', ax=ax)
            plt.title('Distribuição de Vagas por Área')
            plt.axis('equal')
            st.pyplot(fig)
            
            # Distribuição de salários
            fig, ax = plt.subplots(figsize=(10, 6))
            vagas_df['Salario_Min'].plot(kind='kde', ax=ax, label='Salário Mínimo')
            vagas_df['Salario_Max'].plot(kind='kde', ax=ax, label='Salário Máximo')
            plt.title('Distribuição de Faixas Salariais')
            plt.xlabel('Salário (R$)')
            plt.legend()
            st.pyplot(fig)
            
            # Demanda por nível de experiência
            fig, ax = plt.subplots(figsize=(10, 6))
            vagas_df['Nivel'].value_counts().plot(kind='bar', ax=ax)
            plt.title('Demanda por Nível de Senioridade')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            st.pyplot(fig)
    else:
        st.warning("Carregue os dados para visualizar estatísticas e insights.")

# Sobre o sistema
st.sidebar.markdown("---")
with st.sidebar.expander("Sobre o Sistema"):
    st.write("""
    ### Sistema de Classificação de Candidatos
    
    Este sistema utiliza algoritmos de machine learning para analisar a compatibilidade entre candidatos e vagas na área de tecnologia.
    
    **Principais funcionalidades:**
    - Classificação de aderência candidato-vaga
    - Recomendação de vagas para candidatos
    - Recomendação de candidatos para vagas
    - Análises estatísticas do mercado
    
    **Métricas analisadas:**
    - Match de habilidades técnicas
    - Match de soft skills
    - Experiência profissional
    - Compatibilidade salarial
    - Alinhamento da formação acadêmica
    """)