# Medical Triage — Etapas 1 e 2

API de classificação de textos médicos construída como baseline para um pipeline incremental de MLOps.

> **Escopo atual:** este README documenta funcionalidades já implementadas e validadas nas **Etapas 1 e 2**. A Etapa 1 cobre baseline de NLP, FastAPI, Docker e medição de latência. A Etapa 2 adiciona testes de integração, pre-commit, CI com GitHub Actions e pipeline de treinamento com Apache Airflow. Prometheus, Grafana e otimizações de inferência serão adicionados somente quando forem implementados.

---

## 1. Objetivo

O projeto demonstra, de forma incremental e reproduzível, como construir e evoluir um serviço de Machine Learning para classificação de textos médicos.

Já foram implementados:

- download automático do dataset;
- treinamento de um classificador baseline;
- persistência de modelo e métricas;
- API REST com FastAPI;
- validação de entrada;
- Docker multi-stage para inferência;
- testes unitários e de integração;
- Ruff, mypy e pre-commit;
- CI com GitHub Actions;
- pipeline de treinamento com Apache Airflow;
- validação do dataset antes do treinamento;
- validação dos artefatos após o treinamento;
- medição de latência local e em Docker;
- documentação de reprodução para terceiros.

---

## 2. Importante: escopo clínico do dataset

O projeto utiliza o **Medical Abstracts TC Corpus**.

O dataset possui cinco categorias:

1. neoplasms;
2. digestive system diseases;
3. nervous system diseases;
4. cardiovascular diseases;
5. general pathological conditions.

**O dataset não possui rótulos de urgência clínica.**

Portanto, o projeto demonstra infraestrutura de classificação de textos médicos e serving de modelos, mas **não deve ser interpretado como um sistema clínico real de triagem `normal / attention / urgent`**.

Não é feita conversão artificial entre categoria de doença e nível de urgência.

---

## 3. Arquitetura da aplicação

A aplicação utiliza organização inspirada em Clean Architecture e princípios SOLID.

```text
Client
  │
  ▼
FastAPI
Presentation Layer
  │
  ▼
Use Case
Application Layer
  │
  ▼
ClassifierPort
  │
  ▼
SklearnClassifier
Infrastructure
  │
  ▼
classifier.joblib
TF-IDF + Logistic Regression
```

O treinamento baseline segue:

```text
Medical Abstracts TC Corpus
          │
          ▼
     DatasetLoader
          │
          ▼
    Training Dataset
          │
          ▼
Stratified Train/Validation Split
          │
          ▼
        TF-IDF
          │
          ▼
 Logistic Regression
          │
       ┌──┴──────────────┐
       ▼                 ▼
classifier.joblib    metrics.json
```

Na Etapa 2, o treinamento também pode ser orquestrado pelo Airflow:

```text
prepare_dataset_task
        │
        ▼
validate_dataset_task
        │
        ▼
train_model_task
        │
        ▼
validate_artifacts_task
```

---

## 4. Inferência em tempo real

A API utiliza inferência **real-time**:

```text
texto médico
    │
    ▼
POST /predict
    │
    ▼
inferência
    │
    ▼
resposta imediata
```

Uma abordagem batch seria mais adequada para processamento periódico em grande volume, mas não para uma aplicação interativa que precisa responder a uma requisição individual.

---

## 5. Arquitetura de cloud selecionada

A arquitetura planejada para deploy é:

```text
Developer / CI
      │
      ▼
Docker Image
      │
      ▼
AWS ECR
      │
      ▼
AWS EC2
      │
      ▼
FastAPI + Model
      │
      ▼
HTTP /predict
```

O Amazon ECR foi selecionado como registro de imagem e uma instância EC2 como ambiente de execução contínua da API.

> A arquitetura de cloud está definida, mas o deploy automatizado em AWS ainda não faz parte do estado atual do repositório.

---

## 6. Tecnologias

### Aplicação e Machine Learning

- Python 3.12
- uv
- FastAPI
- Uvicorn
- Scikit-learn
- Pandas
- Joblib

### Qualidade

- Pytest
- Ruff
- mypy
- pre-commit

### MLOps e infraestrutura

- Docker
- Docker Compose
- GitHub Actions
- Apache Airflow 3.3.1

Modelo baseline:

```text
TfidfVectorizer
      +
LogisticRegression
```

---

## 7. Estrutura do projeto

```text
medical-triage/
├── .github/
│   └── workflows/
│       └── ci.yml
├── airflow/
│   ├── dags/
│   │   └── medical_triage_training.py
│   ├── Dockerfile
│   ├── docker-compose.yml
│   └── requirements.txt
├── data/
│   ├── processed/
│   └── raw/
├── models/
│   ├── classifier.joblib
│   └── metrics.json
├── scripts/
│   └── measure_latency.py
├── src/
│   └── medical_triage/
│       ├── application/
│       ├── data/
│       ├── domain/
│       ├── infrastructure/
│       ├── observability/
│       ├── presentation/
│       └── training/
├── tests/
│   ├── integration/
│   │   └── test_api.py
│   └── units/
│       ├── test_classify_use_case.py
│       ├── test_dataset_loader.py
│       └── test_training_workflow.py
├── .dockerignore
├── .gitignore
├── .pre-commit-config.yaml
├── .python-version
├── Dockerfile
├── pyproject.toml
├── README.md
└── uv.lock
```

Os diretórios abaixo contêm artefatos locais ou gerados e não precisam ser versionados:

```text
data/raw/
data/processed/
models/
airflow/logs/
```

Isso permite que terceiros reconstruam o ambiente sem depender dos arquivos gerados originalmente pelo autor.

---

## 8. Repositório e clone por terceiros

Repositório:

```text
https://github.com/RafaExMachina/medical-triage
```

Para terceiros, prefira o clone por HTTPS, pois não exige configuração prévia de chave SSH:

```bash
git clone https://github.com/RafaExMachina/medical-triage.git
cd medical-triage
```

Quem já possui SSH configurado pode utilizar:

```bash
git clone git@github.com:RafaExMachina/medical-triage.git
cd medical-triage
```

---

## 9. Pré-requisitos

- Linux, macOS ou Windows com WSL;
- Git;
- Python compatível com o projeto;
- `uv`;
- Docker Engine;
- Docker Compose;
- acesso à internet na primeira execução do treinamento.

Verifique:

```bash
git --version
uv --version
python3 --version
docker --version
docker compose version
```

Em Linux, se necessário:

```bash
sudo systemctl start docker
docker info
```

---

## 10. Instalação do ambiente Python

Depois do clone:

```bash
uv sync --locked
```

O `uv.lock` é versionado para tornar a instalação determinística.

Não é necessário ativar manualmente a `.venv`; utilize:

```bash
uv run ...
```

---

## 11. Pre-commit

Instale o hook local:

```bash
uv run pre-commit install
```

Valide todo o repositório:

```bash
uv run pre-commit run --all-files
```

Quality gates locais:

```text
Ruff format check
Ruff lint
mypy
```

---

## 12. Dataset

O projeto não exige download manual dos CSVs.

O componente:

```text
src/medical_triage/data/dataset_loader.py
```

verifica `data/raw/` e utiliza:

```text
medical_tc_train.csv
medical_tc_test.csv
medical_tc_labels.csv
```

Se um arquivo já existir e não estiver vazio, ele é reutilizado. Se estiver ausente, o loader realiza o download.

Execução opcional do loader:

```bash
uv run python -m medical_triage.data.dataset_loader
```

O treinamento standalone já executa a preparação automaticamente.

---

## 13. Treinamento standalone

Execute:

```bash
uv run python -m medical_triage.training.train
```

Fluxo:

```text
preparação do dataset
        ↓
validação dos dados
        ↓
split treino/validação
        ↓
TF-IDF
        ↓
Logistic Regression
        ↓
avaliação
        ↓
persistência
```

O split utiliza:

```text
90% training
10% validation
random_state = 42
```

O arquivo oficial `medical_tc_test.csv` permanece reservado e não é utilizado para ajustar ou selecionar o baseline atual.

Artefatos gerados:

```text
models/
├── classifier.joblib
└── metrics.json
```

Versão atual do modelo:

```text
tfidf-logreg-v1
```

---

## 14. Resultado do baseline

```text
Training samples:    10,395
Validation samples:   1,155
Classes:                  5
```

| Métrica | Resultado |
|---|---:|
| Accuracy | 0.5931 |
| Macro F1 | 0.5908 |

Esses valores representam o baseline atual, não uma otimização final do modelo.

---

## 15. Executando a API localmente

Se o modelo ainda não existir:

```bash
uv run python -m medical_triage.training.train
```

Depois:

```bash
uv run uvicorn \
    medical_triage.presentation.api.main:app \
    --host 0.0.0.0 \
    --port 8000
```

API:

```text
http://localhost:8000
```

Swagger/OpenAPI:

```text
http://localhost:8000/docs
```

---

## 16. Health check

```bash
curl http://localhost:8000/health
```

Resposta esperada:

```json
{
  "status": "healthy"
}
```

---

## 17. Classificação

Endpoint:

```text
POST /predict
```

Exemplo:

```bash
curl \
    -X POST \
    http://localhost:8000/predict \
    -H "Content-Type: application/json" \
    -d '{
        "text": "The patient presented with acute myocardial infarction and severe coronary artery disease."
    }'
```

Exemplo de resposta:

```json
{
  "label_id": 4,
  "label_name": "cardiovascular diseases",
  "confidence": 0.8846411668151615,
  "model_version": "tfidf-logreg-v1",
  "inference_ms": 10.36887200007186
}
```

`inference_ms` varia entre requisições.

---

## 18. Validação de entrada

O campo `text` deve possuir pelo menos 20 caracteres.

Exemplo inválido:

```bash
curl \
    -X POST \
    http://localhost:8000/predict \
    -H "Content-Type: application/json" \
    -d '{"text": "short"}'
```

A API responde com HTTP `422`.

---

## 19. Logs

A aplicação utiliza logging estruturado em saída padrão.

Exemplo:

```text
2026-08-26T10:42:35-0300 | INFO | __main__ | Starting training workflow | model_version=tfidf-logreg-v1
```

Os logs podem conter versão do modelo, classe prevista, latência, quantidade de amostras e estado de inicialização.

**O texto médico recebido pela API não deve ser registrado nos logs.**

---

## 20. Qualidade de código e testes

```bash
uv run ruff format --check .
uv run ruff check .
uv run mypy src
uv run pytest -v
```

Estado validado ao final da Etapa 2:

```text
Ruff:       PASS
mypy:       PASS
Pytest:     12 passed
pre-commit: PASS
```

Os testes cobrem:

- classificação válida;
- texto vazio;
- texto curto;
- preservação e preparação do dataset;
- download simulado;
- criação de diretório;
- `GET /health`;
- `POST /predict`;
- erro HTTP 422;
- delegação de preparação para `DatasetLoader`;
- orquestração reutilizável do treinamento.

Os testes usam mocks/fakes quando apropriado para não depender de internet nem executar treinamento completo durante a suíte.

---

## 21. CI com GitHub Actions

Workflow:

```text
.github/workflows/ci.yml
```

Gatilhos:

```text
push
pull_request → master
```

Jobs:

```text
CI
├── Quality
│   ├── ruff format --check
│   ├── ruff check
│   └── mypy src
│
└── Tests
    └── pytest
```

As dependências são instaladas no runner com:

```bash
uv sync --locked
```

Execuções validadas:

```text
Quality (push)          PASS
Tests (push)            PASS
Quality (pull_request)  PASS
Tests (pull_request)    PASS
```

---

## 22. Docker da API

O Dockerfile da raiz é responsável pelo serviço de inferência FastAPI.

O modelo deve existir antes do build:

```bash
uv run python -m medical_triage.training.train
```

Build:

```bash
docker build \
    -t medical-triage-api:0.1.0 \
    .
```

Execução:

```bash
docker run \
    --rm \
    --name medical-triage-api \
    -p 8000:8000 \
    medical-triage-api:0.1.0
```

Health check:

```bash
curl http://localhost:8000/health
```

---

## 23. Baseline de latência

Com a API em execução:

```bash
uv run python scripts/measure_latency.py \
    --runs 200 \
    --warmup 10
```

| Métrica | Local | Docker |
|---|---:|---:|
| Mean | 4.913 ms | 7.909 ms |
| Median | 4.640 ms | 7.509 ms |
| Minimum | 3.070 ms | 4.637 ms |
| Maximum | 19.437 ms | 30.680 ms |
| P95 | 6.852 ms | 11.037 ms |
| P99 | 9.664 ms | 16.849 ms |

Esses números serão usados como referência para futuras otimizações de inferência.

---

## 24. Apache Airflow

O Airflow é executado em ambiente Docker separado do runtime da API.

Isso evita adicionar a árvore de dependências do Airflow ao container enxuto de inferência.

```text
airflow/
├── dags/
│   └── medical_triage_training.py
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

Versões:

```text
Apache Airflow 3.3.1
Python 3.12
```

Interface web:

```text
http://localhost:8081
```

Mapeamento:

```text
host 8081 → container 8080
```

---

## 25. Subindo o Airflow

Valide o Compose:

```bash
AIRFLOW_UID=$(id -u) docker compose \
    -f airflow/docker-compose.yml \
    config
```

Build:

```bash
AIRFLOW_UID=$(id -u) docker compose \
    -f airflow/docker-compose.yml \
    build
```

Suba:

```bash
AIRFLOW_UID=$(id -u) docker compose \
    -f airflow/docker-compose.yml \
    up -d
```

Confira:

```bash
AIRFLOW_UID=$(id -u) docker compose \
    -f airflow/docker-compose.yml \
    ps
```

Logs:

```bash
AIRFLOW_UID=$(id -u) docker compose \
    -f airflow/docker-compose.yml \
    logs -f airflow
```

---

## 26. Validando e executando a DAG

Liste as DAGs:

```bash
AIRFLOW_UID=$(id -u) docker compose \
    -f airflow/docker-compose.yml \
    exec airflow \
    airflow dags list --local
```

A DAG esperada é:

```text
medical_triage_training
```

Confira erros de importação:

```bash
AIRFLOW_UID=$(id -u) docker compose \
    -f airflow/docker-compose.yml \
    exec airflow \
    airflow dags list-import-errors --local
```

Resultado esperado:

```text
No data found
```

Liste as tasks:

```bash
AIRFLOW_UID=$(id -u) docker compose \
    -f airflow/docker-compose.yml \
    exec airflow \
    airflow tasks list medical_triage_training
```

Tasks:

```text
prepare_dataset_task
validate_dataset_task
train_model_task
validate_artifacts_task
```

A DAG possui `schedule=None`, portanto é disparada manualmente nesta etapa.

Remova o pause:

```bash
AIRFLOW_UID=$(id -u) docker compose \
    -f airflow/docker-compose.yml \
    exec airflow \
    airflow dags unpause medical_triage_training
```

Dispare:

```bash
AIRFLOW_UID=$(id -u) docker compose \
    -f airflow/docker-compose.yml \
    exec airflow \
    airflow dags trigger medical_triage_training
```

Acompanhe:

```bash
AIRFLOW_UID=$(id -u) docker compose \
    -f airflow/docker-compose.yml \
    exec airflow \
    airflow dags list-runs medical_triage_training
```

Execução validada:

```text
prepare_dataset_task      SUCCESS
validate_dataset_task     SUCCESS
train_model_task          SUCCESS
validate_artifacts_task   SUCCESS
DagRun                     SUCCESS
```

A execução reproduziu:

```text
Accuracy: 0.5931
Macro F1: 0.5908
```

---

## 27. Responsabilidade das tasks Airflow

### `prepare_dataset_task`

Garante que os arquivos necessários estejam disponíveis. Possui retry para falhas transitórias de rede.

### `validate_dataset_task`

Valida:

- dataset não vazio;
- classes disponíveis;
- consistência entre labels e dados.

### `train_model_task`

Executa a função reutilizável:

```python
run_training()
```

Somente métricas pequenas são retornadas via XCom. DataFrames, dataset completo e modelo serializado não são enviados via XCom.

### `validate_artifacts_task`

Valida:

- existência de `classifier.joblib`;
- existência de `metrics.json`;
- artefatos não vazios;
- métricas obrigatórias;
- métricas entre `0.0` e `1.0`;
- consistência entre métricas retornadas e persistidas.

---

## 28. Reprodução completa por terceiros

A sequência abaixo permite validar o projeto a partir de um clone novo.

### 1. Clonar

```bash
git clone https://github.com/RafaExMachina/medical-triage.git
cd medical-triage
```

### 2. Instalar dependências

```bash
uv sync --locked
```

### 3. Configurar pre-commit

```bash
uv run pre-commit install
```

### 4. Validar qualidade e testes

```bash
uv run pre-commit run --all-files
uv run ruff format --check .
uv run ruff check .
uv run mypy src
uv run pytest -v
```

Resultado esperado atualmente:

```text
12 passed
```

### 5. Baixar dados e treinar

```bash
uv run python -m medical_triage.training.train
```

Esse comando baixa automaticamente arquivos ausentes e gera:

```text
models/classifier.joblib
models/metrics.json
```

### 6. Executar a API local

```bash
uv run uvicorn \
    medical_triage.presentation.api.main:app \
    --host 0.0.0.0 \
    --port 8000
```

### 7. Validar a API

```bash
curl http://localhost:8000/health
```

### 8. Fazer uma inferência

```bash
curl \
    -X POST \
    http://localhost:8000/predict \
    -H "Content-Type: application/json" \
    -d '{
        "text": "The patient presented with acute myocardial infarction and severe coronary artery disease."
    }'
```

### 9. Encerrar a API local

```text
Ctrl+C
```

### 10. Construir a API Docker

```bash
docker build \
    -t medical-triage-api:0.1.0 \
    .
```

### 11. Executar a API Docker

```bash
docker run \
    --rm \
    --name medical-triage-api \
    -p 8000:8000 \
    medical-triage-api:0.1.0
```

### 12. Validar o container

```bash
curl http://localhost:8000/health
```

### 13. Encerrar a API Docker

```text
Ctrl+C
```

### 14. Construir o Airflow

```bash
AIRFLOW_UID=$(id -u) docker compose \
    -f airflow/docker-compose.yml \
    build
```

### 15. Subir o Airflow

```bash
AIRFLOW_UID=$(id -u) docker compose \
    -f airflow/docker-compose.yml \
    up -d
```

### 16. Verificar a DAG

```bash
AIRFLOW_UID=$(id -u) docker compose \
    -f airflow/docker-compose.yml \
    exec airflow \
    airflow dags list --local
```

```bash
AIRFLOW_UID=$(id -u) docker compose \
    -f airflow/docker-compose.yml \
    exec airflow \
    airflow dags list-import-errors --local
```

### 17. Executar a DAG

```bash
AIRFLOW_UID=$(id -u) docker compose \
    -f airflow/docker-compose.yml \
    exec airflow \
    airflow dags unpause medical_triage_training
```

```bash
AIRFLOW_UID=$(id -u) docker compose \
    -f airflow/docker-compose.yml \
    exec airflow \
    airflow dags trigger medical_triage_training
```

### 18. Verificar a execução

```bash
AIRFLOW_UID=$(id -u) docker compose \
    -f airflow/docker-compose.yml \
    exec airflow \
    airflow dags list-runs medical_triage_training
```

### 19. Verificar artefatos

```bash
ls -lh models/
cat models/metrics.json
```

### 20. Encerrar o Airflow

```bash
AIRFLOW_UID=$(id -u) docker compose \
    -f airflow/docker-compose.yml \
    down
```

Se todos os passos forem concluídos, um terceiro conseguiu reconstruir e validar as Etapas 1 e 2 sem depender dos arquivos gerados originalmente pelo autor.

---

## 29. Fluxo recomendado para contribuidores

Antes de abrir um Pull Request:

```bash
uv sync --locked
uv run pre-commit run --all-files
uv run pytest -v
```

Fluxo recomendado:

```text
fork ou clone
     │
     ▼
feature branch
     │
     ▼
desenvolvimento
     │
     ▼
pre-commit
     │
     ▼
git push
     │
     ▼
Pull Request → master
     │
     ▼
GitHub Actions
     │
     ├── Quality
     └── Tests
```

Exemplo:

```bash
git switch -c feat/minha-feature
```

Dados e artefatos de modelo não devem ser adicionados ao commit.

---

## 30. Critérios concluídos até a Etapa 2

```text
[OK] DatasetLoader
[OK] Download automático do dataset
[OK] Split treino/validação estratificado
[OK] TF-IDF + Logistic Regression
[OK] Modelo persistido
[OK] Métricas persistidas
[OK] Accuracy baseline = 0.5931
[OK] Macro F1 baseline = 0.5908

[OK] FastAPI
[OK] GET /health
[OK] POST /predict
[OK] Validação de entrada
[OK] Logging sem conteúdo médico

[OK] Docker multi-stage da API
[OK] API executada em container
[OK] Baseline local medido
[OK] Baseline Docker medido

[OK] Ruff
[OK] mypy
[OK] pre-commit
[OK] 12 testes
[OK] Testes de integração da API

[OK] GitHub Actions
[OK] CI em push
[OK] CI em pull_request
[OK] Job Quality
[OK] Job Tests

[OK] Apache Airflow 3.3.1
[OK] Airflow isolado em Docker
[OK] DAG medical_triage_training
[OK] prepare_dataset_task
[OK] validate_dataset_task
[OK] train_model_task
[OK] validate_artifacts_task
[OK] DAG sem import errors
[OK] DagRun executada com sucesso

[OK] Arquitetura AWS ECR + EC2 definida
[OK] Reprodução por terceiros documentada
```

---

## 31. Próximas etapas

Ainda não fazem parte da implementação atual:

```text
Prometheus
Grafana
Docker Compose de observabilidade
dashboards de latência e erros
retraining automático por agenda ou drift
deploy automático em AWS
ONNX
quantização
pruning
outras otimizações de inferência
```

Essas funcionalidades serão documentadas somente quando forem implementadas e validadas.

---

## 32. Aviso

Este projeto possui finalidade acadêmica e demonstra conceitos de engenharia de software, Machine Learning e MLOps.

O modelo atual não é um dispositivo médico e não deve ser utilizado para diagnóstico, priorização clínica ou tomada de decisão sobre pacientes.
