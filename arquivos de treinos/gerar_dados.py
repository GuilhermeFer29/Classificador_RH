import pandas as pd
import numpy as np
import random
from faker import Faker
import csv
from random import randint, uniform, choice, sample

# Configurando o Faker para gerar dados em português brasileiro
fake = Faker('pt_BR')
Faker.seed(42)  # Para reprodutibilidade

# Listas de possíveis valores para as categorias
formacoes = [
    "Ciência da Computação", "Engenharia de Software", "Sistemas de Informação", 
    "Análise e Desenvolvimento de Sistemas", "Engenharia da Computação", 
    "Tecnologia em Redes", "Segurança da Informação", "Tecnologia em Banco de Dados",
    "Tecnologia em Desenvolvimento Web", "Engenharia Elétrica", "Matemática Computacional"
]

niveis_formacao = ["Técnico", "Graduação", "Pós-graduação", "Mestrado", "Doutorado"]

certificacoes = [
    "AWS Certified Solutions Architect", "Microsoft Certified: Azure Developer", 
    "CompTIA A+", "Cisco CCNA", "Oracle Certified Professional", "Scrum Master",
    "PMI-PMP", "ITIL Foundation", "Google Cloud Professional", "Certified Ethical Hacker",
    "MongoDB Certified Developer", "Kubernetes Administrator (CKA)", "Terraform Associate",
    "Docker Certified Associate", "Red Hat Certified Engineer", "Salesforce Developer",
    "Python Institute PCEP", "ISTQB Certified Tester", "IBM Data Science Professional"
]

habilidades_tecnicas = [
    "Python", "Java", "JavaScript", "C#", "C++", "PHP", "Ruby", "Swift", "Kotlin", 
    "Go", "Rust", "SQL", "MongoDB", "Redis", "PostgreSQL", "MySQL", "Oracle", 
    "React", "Angular", "Vue.js", "Node.js", "Django", "Flask", "Spring", ".NET", 
    "AWS", "Azure", "GCP", "Docker", "Kubernetes", "Jenkins", "Git", "TensorFlow", 
    "PyTorch", "Hadoop", "Spark", "Power BI", "Tableau", "Linux", "Windows Server",
    "Network Security", "Cybersecurity", "DevOps", "CI/CD", "Agile", "Scrum"
]

soft_skills = [
    "Comunicação", "Trabalho em equipe", "Resolução de problemas", "Adaptabilidade",
    "Gestão de tempo", "Liderança", "Pensamento crítico", "Criatividade", "Empatia",
    "Inteligência emocional", "Negociação", "Gerenciamento de conflitos", 
    "Feedback construtivo", "Tomada de decisão", "Organização", "Ética de trabalho",
    "Resiliência", "Persuasão", "Mentalidade de crescimento", "Networking"
]

cargos_anteriores = [
    "Desenvolvedor Júnior", "Desenvolvedor Pleno", "Desenvolvedor Sênior", 
    "Arquiteto de Software", "DevOps Engineer", "SRE", "DBA", "QA Analyst", 
    "Tech Lead", "Scrum Master", "Product Owner", "UX Designer", "UI Designer", 
    "Data Scientist", "Data Engineer", "Business Intelligence Analyst", 
    "Security Engineer", "Network Administrator", "System Administrator",
    "Cloud Architect", "Mobile Developer", "Frontend Developer", "Backend Developer",
    "Fullstack Developer", "Machine Learning Engineer", "Gerente de Projetos"
]

# Função para gerar data aleatória no formato brasileiro (DD/MM/AAAA)
def data_aleatoria(start_year=2000, end_year=2023):
    dia = random.randint(1, 28)  # Limitado a 28 para evitar problemas com fevereiro
    mes = random.randint(1, 12)
    ano = random.randint(start_year, end_year)
    return f"{dia:02d}/{mes:02d}/{ano}"

# Função para gerar conjuntos aleatórios de habilidades técnicas
def gerar_habilidades_tecnicas():
    num_habilidades = random.randint(3, 15)
    return sample(habilidades_tecnicas, num_habilidades)

# Função para gerar conjuntos aleatórios de soft skills
def gerar_soft_skills():
    num_skills = random.randint(2, 8)
    return sample(soft_skills, num_skills)

# Função para gerar conjuntos aleatórios de certificações
def gerar_certificacoes():
    num_cert = random.randint(0, 5)
    return sample(certificacoes, num_cert) if num_cert > 0 else []

# Função para gerar experiências anteriores
def gerar_experiencias():
    num_exp = random.randint(0, 4)
    experiencias = []
    
    for _ in range(num_exp):
        cargo = random.choice(cargos_anteriores)
        empresa = fake.company()
        duracao = random.randint(6, 60)  # meses
        experiencias.append(f"{cargo} na {empresa}, {duracao} meses")
    
    return experiencias

# Criar dados para 1000 candidatos
candidatos = []

for i in range(10000):
    candidato = {
        'ID': i + 1,
        'Nome': fake.name(),
        'Email': fake.email(),
        'Telefone': fake.phone_number(),
        'Idade': random.randint(18, 60),
        'Cidade': fake.city(),
        'Estado': fake.state_abbr(),
        'Formacao': random.choice(formacoes),
        'Nivel_Formacao': random.choice(niveis_formacao),
        'Ano_Formacao': random.randint(1990, 2023),
        'Anos_Experiencia': random.randint(0, 25),
        'Ultimo_Salario': random.randint(2000, 25000),
        'Habilidades_Tecnicas': ', '.join(gerar_habilidades_tecnicas()),
        'Soft_Skills': ', '.join(gerar_soft_skills()),
        'Certificacoes': ', '.join(gerar_certificacoes()),
        'Experiencias_Anteriores': '; '.join(gerar_experiencias()),
        'Disponibilidade_Imediata': random.choice(['Sim', 'Não']),
        'Disponivel_Mudanca': random.choice(['Sim', 'Não']),
        'Pretensao_Salarial': random.randint(3000, 30000),
        'Data_Cadastro': data_aleatoria(2022, 2024)
    }
    candidatos.append(candidato)

# Criar DataFrame e salvar como CSV
df_candidatos = pd.DataFrame(candidatos)
df_candidatos.to_csv('candidatos_tecnologia.csv', index=False, encoding='utf-8-sig')

print("Banco de dados de candidatos criado com sucesso!")

# Agora vamos criar o banco de dados de 2000 vagas de tecnologia
areas_tecnologia = [
    "Desenvolvimento Web", "Desenvolvimento Mobile", "Desenvolvimento Full-Stack",
    "DevOps", "Ciência de Dados", "Engenharia de Dados", "Machine Learning",
    "Segurança da Informação", "Suporte Técnico", "Administração de Redes",
    "Administração de Sistemas", "Banco de Dados", "Cloud Computing",
    "Inteligência Artificial", "Arquitetura de Software", "QA e Testes",
    "UX/UI Design", "Business Intelligence", "Blockchain", "IoT"
]

nomes_empresas = [
    "TechSolutions", "InnovaTech", "DataCorp", "CloudMind", "ByteWise", "CodeNation",
    "InfoTrust", "SystemGo", "CyberShield", "DevMatrix", "SmartByte", "FutureTech",
    "EcoSystems", "QuantumCode", "NexusLink", "DigitalWave", "SecureNet", "InfinitySoft",
    "GlobalTech", "MetaLogic", "BrainWare", "AlgoTech", "VisionSoft", "CoreDev",
    "PrimeTech", "FusionSystems", "EliteTech", "BluePrint", "RedCode", "PurpleWave",
    "GreenData", "OrangeStack", "SilverStream", "GoldenByte", "CrystalNet", "PearlSystems",
    "RubyTech", "SapphireCode", "TopazData", "EmeraldSoft", "JadeWare", "ObsidianSystems"
]

# Expandir para mais empresas ficticias
for i in range(len(nomes_empresas), 100):
    sufixos = ["Technologies", "Software", "IT", "Systems", "Digital", "Compute", "Networks", "Cloud", "Data", "Tech"]
    prefixos = ["Neo", "Cyber", "Future", "Smart", "Logic", "Inno", "Next", "Dynamic", "Prime", "Omni", "Hyper", "Ultra"]
    nomes_empresas.append(f"{random.choice(prefixos)}{random.choice(sufixos)}")

cidades_grandes_br = [
    "São Paulo", "Rio de Janeiro", "Brasília", "Salvador", "Fortaleza", 
    "Belo Horizonte", "Manaus", "Curitiba", "Recife", "Porto Alegre",
    "Belém", "Goiânia", "Guarulhos", "Campinas", "São Luís", "São Gonçalo",
    "Maceió", "Duque de Caxias", "Natal", "Teresina", "Florianópolis"
]

estados_br = [
    "SP", "RJ", "DF", "BA", "CE", "MG", "AM", "PR", "PE", "RS",
    "PA", "GO", "SP", "SP", "MA", "RJ", "AL", "RJ", "RN", "PI", "SC"
]

modalidades = ["Presencial", "Remoto", "Híbrido"]
tipos_contrato = ["CLT", "PJ", "Temporário", "Estágio", "Trainee"]
niveis_senioridade = ["Júnior", "Pleno", "Sênior", "Especialista", "Gerente", "Diretor"]

def gerar_requisitos_vaga(area):
    req_habilidades = []
    
    # Adicionar habilidades gerais com base na área
    habs_por_area = {
        "Desenvolvimento Web": ["HTML", "CSS", "JavaScript", "React", "Angular", "Vue.js", "Node.js", "PHP"],
        "Desenvolvimento Mobile": ["Swift", "Kotlin", "Flutter", "React Native", "Android Studio", "iOS"],
        "Desenvolvimento Full-Stack": ["JavaScript", "TypeScript", "Node.js", "React", "MongoDB", "SQL", "REST API"],
        "DevOps": ["Docker", "Kubernetes", "Jenkins", "AWS", "Azure", "GCP", "Terraform", "CI/CD"],
        "Ciência de Dados": ["Python", "R", "SQL", "Pandas", "NumPy", "Scikit-Learn", "TensorFlow", "Estatística"],
        "Engenharia de Dados": ["Python", "Hadoop", "Spark", "Kafka", "Airflow", "SQL", "NoSQL", "ETL"],
        "Machine Learning": ["Python", "TensorFlow", "PyTorch", "Scikit-Learn", "Algoritmos de ML", "NLP"],
        "Segurança da Informação": ["Pentest", "Segurança de Redes", "Criptografia", "Análise de Vulnerabilidades"],
        "Suporte Técnico": ["Windows", "Linux", "Redes", "Suporte ao Usuário", "Troubleshooting"],
        "Administração de Redes": ["TCP/IP", "Cisco", "Roteadores", "Firewalls", "VPN", "VLAN"],
        "Administração de Sistemas": ["Linux", "Windows Server", "Active Directory", "Shell Script", "PowerShell"],
        "Banco de Dados": ["SQL", "Oracle", "MySQL", "PostgreSQL", "MongoDB", "DBA", "Modelagem de Dados"],
        "Cloud Computing": ["AWS", "Azure", "GCP", "IaaS", "PaaS", "SaaS", "Serverless"],
        "Inteligência Artificial": ["Python", "Algoritmos de IA", "Redes Neurais", "NLP", "Computer Vision"],
        "Arquitetura de Software": ["Design Patterns", "Microserviços", "SOA", "API Design", "UML"],
        "QA e Testes": ["Testes Unitários", "Selenium", "Testes de Integração", "JUnit", "Automação de Testes"],
        "UX/UI Design": ["Figma", "Adobe XD", "Sketch", "Usabilidade", "Wireframing", "Prototipagem"],
        "Business Intelligence": ["Power BI", "Tableau", "SQL", "ETL", "Data Warehouse", "Reporting"],
        "Blockchain": ["Solidity", "Ethereum", "Smart Contracts", "DApps", "Criptografia"],
        "IoT": ["Sensores", "Arduino", "Raspberry Pi", "MQTT", "Protocolos IoT", "Sistemas Embarcados"]
    }
    
    # Adicionar habilidades específicas da área
    especificas = habs_por_area.get(area, ["Programação"])
    num_esp = random.randint(2, min(5, len(especificas)))
    req_habilidades.extend(sample(especificas, num_esp))
    
    # Adicionar algumas habilidades gerais/técnicas
    gerais = sample(habilidades_tecnicas, random.randint(2, 5))
    req_habilidades.extend(gerais)
    
    # Adicionar soft skills
    softs = sample(soft_skills, random.randint(2, 4))
    req_habilidades.extend(softs)
    
    return list(set(req_habilidades))  # Remover duplicatas

def gerar_titulo_vaga(area, nivel):
    titulos = {
        "Desenvolvimento Web": ["Desenvolvedor Web", "Desenvolvedor Front-end", "Desenvolvedor Back-end"],
        "Desenvolvimento Mobile": ["Desenvolvedor Mobile", "Desenvolvedor Android", "Desenvolvedor iOS"],
        "Desenvolvimento Full-Stack": ["Desenvolvedor Full-Stack", "Engenheiro Full-Stack"],
        "DevOps": ["Engenheiro DevOps", "Especialista DevOps", "SRE"],
        "Ciência de Dados": ["Cientista de Dados", "Analista de Dados", "Especialista em Analytics"],
        "Engenharia de Dados": ["Engenheiro de Dados", "Arquiteto de Dados"],
        "Machine Learning": ["Engenheiro de Machine Learning", "Especialista em ML"],
        "Segurança da Informação": ["Analista de Segurança", "Especialista em Segurança da Informação", "Pentester"],
        "Suporte Técnico": ["Analista de Suporte", "Técnico de Suporte", "Help Desk"],
        "Administração de Redes": ["Administrador de Redes", "Analista de Redes", "Especialista em Redes"],
        "Administração de Sistemas": ["Administrador de Sistemas", "SysAdmin"],
        "Banco de Dados": ["DBA", "Administrador de Banco de Dados", "Desenvolvedor de Banco de Dados"],
        "Cloud Computing": ["Arquiteto de Cloud", "Especialista em Cloud", "Engenheiro de Cloud"],
        "Inteligência Artificial": ["Engenheiro de IA", "Especialista em Inteligência Artificial"],
        "Arquitetura de Software": ["Arquiteto de Software", "Arquiteto de Soluções"],
        "QA e Testes": ["Analista de Testes", "QA Engineer", "Especialista em Qualidade"],
        "UX/UI Design": ["Designer UX/UI", "UX Designer", "UI Designer"],
        "Business Intelligence": ["Analista de BI", "Especialista em Business Intelligence"],
        "Blockchain": ["Desenvolvedor Blockchain", "Especialista em Blockchain"],
        "IoT": ["Engenheiro de IoT", "Especialista em Internet das Coisas"]
    }
    
    titulo_base = random.choice(titulos.get(area, ["Especialista em Tecnologia"]))
    return f"{titulo_base} {nivel}"

# Criar dados para 2000 vagas
vagas = []

for i in range(5000):
    area = random.choice(areas_tecnologia)
    nivel = random.choice(niveis_senioridade)
    cidade_idx = random.randint(0, len(cidades_grandes_br) - 1)
    
    empresa = random.choice(nomes_empresas)
    suffix = random.choice(["S.A.", "Ltda.", "Inc.", "Tech", "Soluções", "Tecnologia", "Systems"])
    if not any(s in empresa for s in ["Technologies", "Software", "IT", "Systems", "Digital", "Tech"]):
        empresa = f"{empresa} {suffix}"
    
    requisitos = gerar_requisitos_vaga(area)
    salario_min = random.randint(1500, 20000)
    salario_max = salario_min + random.randint(1000, 10000)
    
    experiencia_min = 0 if nivel in ["Estágio", "Trainee", "Júnior"] else random.randint(1, 10)
    
    vaga = {
        'ID': i + 1,
        'Titulo': gerar_titulo_vaga(area, nivel),
        'Empresa': empresa,
        'Area': area,
        'Nivel': nivel,
        'Cidade': cidades_grandes_br[cidade_idx],
        'Estado': estados_br[cidade_idx],
        'Modalidade': random.choice(modalidades),
        'Tipo_Contrato': random.choice(tipos_contrato),
        'Salario_Min': salario_min,
        'Salario_Max': salario_max,
        'Experiencia_Min_Anos': experiencia_min,
        'Requisitos': ', '.join(requisitos),
        'Beneficios': 'Vale-Refeição, Vale-Transporte, Plano de Saúde' + (', Seguro de Vida, Bônus Anual' if random.random() > 0.5 else ''),
        'Descricao': fake.text(max_nb_chars=500),
        'Data_Publicacao': data_aleatoria(2023, 2024),
        'Vagas_Disponiveis': random.randint(1, 5),
        'Link_Inscricao': f"https://carreiras.{empresa.lower().replace(' ', '').replace(',', '').replace('.', '')}.com.br/vaga/{i+1}"
    }
    vagas.append(vaga)

# Criar DataFrame e salvar como CSV
df_vagas = pd.DataFrame(vagas)
df_vagas.to_csv('vagas_tecnologia.csv', index=False, encoding='utf-8-sig')

print("Banco de dados de vagas criado com sucesso!")