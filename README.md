# Medical Triage — Etapas 1, 2, 3 e 4

**Versão atual do pacote: `0.4.0`**  
**Escopo documentado: Etapas 1, 2, 3 e 4**

API de classificação de textos médicos desenvolvida como base para um pipeline incremental de MLOps.

> **Escopo atual:** a Etapa 1 implementa o baseline de NLP, FastAPI, Docker e medição de latência. A Etapa 2 adiciona testes de integração, pre-commit, CI com GitHub Actions e pipeline de treinamento com Apache Airflow. A Etapa 3 adiciona instrumentação Prometheus, Docker Compose de observabilidade e dashboard Grafana provisionado automaticamente.

---

## 1. Objetivo

O projeto demonstra, de forma incremental e reproduzível, como construir e evoluir um serviço de Machine Learning para classificação de textos médicos.

Funcionalidades implementadas:

- download automático do dataset;
- treinamento de um classificador baseline;
- persistência de modelo e métricas;
- API REST com FastAPI;
- validação de entrada;
- Docker multi-stage para inferência;
- medição de latência local e em Docker;
- testes unitários e de integração;
- Ruff, mypy e pre-commit;
- CI com GitHub Actions;
- pipeline de treinamento com Apache Airflow;
- validação do dataset antes do treinamento;
- validação dos artefatos após o treinamento;
- endpoint `/metrics`;
- métricas Prometheus de tráfego, latência e predições;
- monitoramento do processo Python;
- Prometheus em Docker Compose;
- Grafana em Docker Compose;
- datasource Prometheus provisionado automaticamente;
- dashboard Grafana provisionado automaticamente;
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

## 3. Evolução por etapas

### Etapa 1 — baseline, API e Docker

```text
Medical Abstracts TC Corpus
          │
          ▼
     DatasetLoader
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
     ┌────┴─────────────┐
     ▼                  ▼
classifier.joblib   metrics.json
          │
          ▼
       FastAPI
          │
          ▼
        Docker
```

Principais entregas:

- baseline de NLP;
- API `/health` e `/predict`;
- validação Pydantic;
- imagem Docker multi-stage;
- benchmark de latência;
- logging sem registrar o texto médico.

### Etapa 2 — qualidade, CI/CD e Airflow

```text
feature branch
     │
     ▼
pre-commit
     │
     ▼
GitHub Actions
 ┌───┴────┐
 ▼        ▼
Quality  Tests
```

Treinamento orquestrado:

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

Principais entregas:

- testes de integração;
- Ruff;
- mypy;
- pre-commit;
- GitHub Actions;
- Airflow 3.3.1;
- DAG de treinamento;
- validações antes e depois do treinamento.

### Etapa 3 — observabilidade

```text
                  Docker Compose

┌──────────────────────────────────────────────┐
│                                              │
│        FastAPI :8000                         │
│        /health                               │
│        /predict                              │
│        /metrics                              │
│            │                                 │
│            │ scrape                          │
│            ▼                                 │
│        Prometheus :9090                      │
│            │                                 │
│            │ PromQL                          │
│            ▼                                 │
│        Grafana :3000                         │
│                                              │
└──────────────────────────────────────────────┘
```

Principais entregas:

- instrumentação com `prometheus-client`;
- Counter de requisições HTTP;
- Histogram de latência HTTP;
- Counter de predições por classe;
- métricas do processo Python;
- Prometheus;
- Grafana;
- dashboard com seis painéis;
- provisionamento automático de datasource e dashboard.

---

## 4. Arquitetura da aplicação

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

Estrutura lógica:

```text
presentation
     │
     ▼
application
     │
     ▼
domain
     ▲
     │
infrastructure
```

A camada `observability` concentra logging e instrumentação de métricas.

---

## 5. Inferência em tempo real

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

## 6. Arquitetura de cloud selecionada

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

## 7. Tecnologias

### Aplicação e Machine Learning

- Python 3.12;
- uv;
- FastAPI;
- Uvicorn;
- Scikit-learn;
- Pandas;
- Joblib.

Modelo baseline:

```text
TfidfVectorizer
      +
LogisticRegression
```

### Qualidade

- Pytest;
- Ruff;
- mypy;
- pre-commit.

### CI/CD e orquestração

- Git;
- GitHub;
- GitHub Actions;
- Apache Airflow 3.3.1.

### Observabilidade

- `prometheus-client`;
- Prometheus;
- PromQL;
- Grafana.

### Infraestrutura

- Docker;
- Docker Compose.

---

## 8. Estrutura do projeto

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
├── docs/
│   └── observability-plan.md
├── models/
│   ├── classifier.joblib
│   └── metrics.json
├── monitoring/
│   ├── grafana/
│   │   ├── dashboards/
│   │   │   └── medical-triage.json
│   │   └── provisioning/
│   │       ├── dashboards/
│   │       │   └── dashboards.yml
│   │       └── datasources/
│   │           └── prometheus.yml
│   └── prometheus/
│       └── prometheus.yml
├── scripts/
│   └── measure_latency.py
├── src/
│   └── medical_triage/
│       ├── application/
│       ├── data/
│       ├── domain/
│       ├── infrastructure/
│       ├── observability/
│       │   ├── logging.py
│       │   ├── metrics.py
│       │   └── middleware.py
│       ├── presentation/
│       │   └── api/
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
├── docker-compose.yml
├── Dockerfile
├── pyproject.toml
├── README.md
└── uv.lock
```

Os diretórios abaixo contêm dados ou artefatos gerados localmente e não precisam ser versionados:

```text
data/raw/
data/processed/
models/
airflow/logs/
```

O modelo precisa ser gerado localmente antes do build da imagem da API.

---

## 9. Repositório e clone

Repositório:

```text
https://github.com/RafaExMachina/medical-triage
```

Clone por HTTPS:

```bash
git clone https://github.com/RafaExMachina/medical-triage.git
cd medical-triage
```

Clone por SSH:

```bash
git clone git@github.com:RafaExMachina/medical-triage.git
cd medical-triage
```

---

## 10. Pré-requisitos

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

## 11. Instalação do ambiente Python

Depois do clone:

```bash
uv sync --locked
```

O `uv.lock` é versionado para tornar a instalação determinística.

Não é necessário ativar manualmente a `.venv`; utilize:

```bash
uv run ...
```

Instale o hook local:

```bash
uv run pre-commit install
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

Execução opcional:

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
| --- | ---: |
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

## 16. Endpoints

### `GET /health`

```bash
curl http://localhost:8000/health
```

Resposta:

```json
{
  "status": "healthy"
}
```

### `POST /predict`

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

### `GET /metrics`

O endpoint expõe métricas no formato Prometheus:

```bash
curl http://localhost:8000/metrics
```

Ele não aparece no schema OpenAPI da aplicação.

---

## 17. Validação de entrada

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

## 18. Logs e privacidade

A aplicação utiliza logging estruturado em saída padrão.

Exemplo:

```text
2026-08-26T10:42:35-0300 | INFO | __main__ | Starting training workflow | model_version=tfidf-logreg-v1
```

Os logs podem conter versão do modelo, classe prevista, latência, quantidade de amostras e estado de inicialização.

**O texto médico recebido pela API não deve ser registrado nos logs.**

A mesma regra é aplicada às métricas: o texto recebido não é usado como label Prometheus.

---

## 19. Qualidade de código e testes

Quality gates:

```bash
uv run ruff format --check .
uv run ruff check .
uv run mypy src
uv run pytest -v
```

Ou, de forma consolidada:

```bash
uv run pre-commit run --all-files
```

Estado validado ao final da Etapa 3:

```text
Ruff:       PASS
mypy:       PASS
Pytest:     13 passed
pre-commit: PASS
```

Os testes cobrem, entre outros:

- classificação válida;
- texto vazio;
- texto curto;
- preservação e preparação do dataset;
- download simulado;
- criação de diretório;
- `GET /health`;
- `POST /predict`;
- erro HTTP `422`;
- `GET /metrics`;
- exposição das métricas Prometheus;
- delegação de preparação para `DatasetLoader`;
- orquestração reutilizável do treinamento.

Os testes usam mocks/fakes quando apropriado para não depender de internet nem executar treinamento completo durante a suíte.

> A suíte pode exibir um `StarletteDeprecationWarning` relacionado ao `TestClient/httpx`. O warning não impede a execução dos testes.

---

## 20. CI com GitHub Actions

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

Fluxo esperado:

```text
push / pull request
        │
        ├── Quality
        └── Tests
```

---

## 21. Docker da API

O `Dockerfile` da raiz é responsável pelo serviço de inferência FastAPI.

Como `models/` contém artefatos gerados e não é versionado, o modelo deve existir antes do build:

```bash
uv run python -m medical_triage.training.train
```

Build standalone:

```bash
docker build \
    -t medical-triage-api:0.4.0 \
    .
```

Execução standalone:

```bash
docker run \
    --rm \
    --name medical-triage-api \
    -p 8000:8000 \
    medical-triage-api:0.4.0
```

Health check:

```bash
curl http://localhost:8000/health
```

A imagem usa build multi-stage e copia `src/` e `models/` para o runtime.

---

## 22. Baseline de latência

Com a API em execução:

```bash
uv run python scripts/measure_latency.py \
    --runs 200 \
    --warmup 10
```

| Métrica | Local | Docker |
| --- | ---: | ---: |
| Mean | 4.913 ms | 7.909 ms |
| Median | 4.640 ms | 7.509 ms |
| Minimum | 3.070 ms | 4.637 ms |
| Maximum | 19.437 ms | 30.680 ms |
| P95 | 6.852 ms | 11.037 ms |
| P99 | 9.664 ms | 16.849 ms |

Esses números são referência para futuras otimizações de inferência.

---

## 23. Apache Airflow

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

Versões validadas:

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

### Subir o Airflow

Validar:

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

Subir:

```bash
AIRFLOW_UID=$(id -u) docker compose \
    -f airflow/docker-compose.yml \
    up -d
```

### Validar a DAG

```bash
AIRFLOW_UID=$(id -u) docker compose \
    -f airflow/docker-compose.yml \
    exec airflow \
    airflow dags list --local
```

DAG:

```text
medical_triage_training
```

Import errors:

```bash
AIRFLOW_UID=$(id -u) docker compose \
    -f airflow/docker-compose.yml \
    exec airflow \
    airflow dags list-import-errors --local
```

Tasks:

```text
prepare_dataset_task
validate_dataset_task
train_model_task
validate_artifacts_task
```

A DAG possui `schedule=None`, portanto é disparada manualmente nesta etapa.

### Executar

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

Acompanhar:

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
DagRun                    SUCCESS
```

Métricas reproduzidas:

```text
Accuracy: 0.5931
Macro F1: 0.5908
```

Encerrar:

```bash
AIRFLOW_UID=$(id -u) docker compose \
    -f airflow/docker-compose.yml \
    down
```

---

## 24. Responsabilidade das tasks Airflow

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

## 25. Observabilidade

A Etapa 3 implementa observabilidade da API com Prometheus e Grafana.

Documentação detalhada:

```text
docs/observability-plan.md
```

Arquitetura:

```text
FastAPI :8000
    │
    │ GET /metrics
    ▼
Prometheus :9090
    │
    │ PromQL
    ▼
Grafana :3000
```

O Prometheus utiliza o modelo `pull` e consulta:

```text
http://api:8000/metrics
```

dentro da rede do Docker Compose.

---

## 26. Métricas Prometheus

### Requisições HTTP

```text
medical_triage_http_requests_total
```

Tipo: `Counter`

Labels controladas:

```text
method
endpoint
status
```

Exemplo:

```promql
medical_triage_http_requests_total{
  endpoint="/predict",
  method="POST",
  status="200"
}
```

### Latência HTTP

```text
medical_triage_http_request_duration_seconds
```

Tipo: `Histogram`

Labels:

```text
method
endpoint
```

P95 da inferência:

```promql
histogram_quantile(
  0.95,
  sum by (le) (
    rate(
      medical_triage_http_request_duration_seconds_bucket{
        endpoint="/predict",
        method="POST"
      }[5m]
    )
  )
)
```

### Predições

```text
medical_triage_predictions_total
```

Tipo: `Counter`

Label:

```text
label_name
```

Distribuição:

```promql
sum by (label_name) (
  medical_triage_predictions_total
)
```

### Memória da API

A biblioteca cliente também expõe métricas do processo Python.

O dashboard utiliza:

```promql
process_resident_memory_bytes{
  job="medical-triage-api"
}
```

### Disponibilidade

```promql
up{job="medical-triage-api"}
```

Interpretação:

```text
1 = target disponível
0 = target indisponível
```

---

## 27. Consultas PromQL do dashboard

### Throughput

```promql
sum(
  rate(
    medical_triage_http_requests_total{
      endpoint="/predict",
      method="POST",
      status=~"2.."
    }[1m]
  )
)
or vector(0)
```

### P95

```promql
histogram_quantile(
  0.95,
  sum by (le) (
    rate(
      medical_triage_http_request_duration_seconds_bucket{
        endpoint="/predict",
        method="POST"
      }[5m]
    )
  )
)
```

### Error rate

```promql
(
  sum(
    rate(
      medical_triage_http_requests_total{
        endpoint="/predict",
        method="POST",
        status=~"4..|5.."
      }[5m]
    )
  )
  /
  clamp_min(
    sum(
      rate(
        medical_triage_http_requests_total{
          endpoint="/predict",
          method="POST"
        }[5m]
      )
    ),
    0.000001
  )
)
or vector(0)
```

A unidade do painel Grafana é `percentunit`; portanto, um valor PromQL `0.2` é exibido como `20%`.

### Total de predições

```promql
sum(medical_triage_predictions_total) or vector(0)
```

### Distribuição

```promql
sum by (label_name) (
  medical_triage_predictions_total
)
```

---

## 28. Dashboard Grafana

Dashboard:

```text
Medical Triage - Observability
```

Arquivo versionado:

```text
monitoring/grafana/dashboards/medical-triage.json
```

Datasource:

```text
Prometheus
```

URL utilizada dentro da rede Docker:

```text
http://prometheus:9090
```

O dashboard possui seis painéis:

| Painel | Objetivo |
| --- | --- |
| Total Predictions | total acumulado de predições |
| Inference Throughput | requisições de inferência por segundo |
| P95 Inference HTTP Latency | percentil 95 da latência HTTP de `/predict` |
| Prediction Error Rate | proporção de respostas 4xx e 5xx |
| Prediction Distribution | distribuição das classes previstas |
| API Memory Usage | memória residente do processo da API |

O datasource e o dashboard são provisionados automaticamente por arquivos versionados no repositório.

---

## 29. Docker Compose de observabilidade

O `docker-compose.yml` da raiz executa:

```text
api
prometheus
grafana
```

Portas:

| Serviço | Porta |
| --- | ---: |
| FastAPI | `8000` |
| Prometheus | `9090` |
| Grafana | `3000` |

O Airflow continua usando um Compose separado:

```text
airflow/docker-compose.yml
```

### Pré-condição

O modelo deve existir antes do build da API:

```bash
uv run python -m medical_triage.training.train
```

### Validar a configuração

```bash
docker compose config
docker compose config --services
```

Serviços esperados:

```text
api
prometheus
grafana
```

### Subir a stack

```bash
docker compose up -d --build
```

### Conferir

```bash
docker compose ps
```

### Validar serviços

FastAPI:

```bash
curl http://localhost:8000/health
```

Prometheus:

```bash
curl http://localhost:9090/-/healthy
```

Grafana:

```bash
curl http://localhost:3000/api/health
```

Prometheus target:

```bash
curl -sG \
  --data-urlencode 'query=up{job="medical-triage-api"}' \
  http://localhost:9090/api/v1/query \
  | python -m json.tool
```

Resultado esperado:

```text
instance="api:8000"
job="medical-triage-api"
value="1"
```

### Grafana

Abra:

```text
http://localhost:3000
```

Credenciais locais configuradas para o ambiente acadêmico:

```text
user:     admin
password: admin
```

> Essas credenciais são apenas para desenvolvimento local. Em produção, use secrets e uma política de autenticação apropriada.

### Encerrar

```bash
docker compose down
```

O comando acima remove containers e rede, mas preserva os volumes. Para remover também os volumes:

```bash
docker compose down -v
```

---

## 30. Conflito com Prometheus instalado no host

Se existir um Prometheus instalado diretamente no Ubuntu, ele pode ocupar a porta `9090` antes do container.

Diagnóstico:

```bash
sudo ss -ltnp | grep ':9090'
```

Se aparecer um processo `prometheus` do `systemd`, pare-o antes de iniciar a stack:

```bash
sudo systemctl stop prometheus
```

Se quiser impedir o início automático durante o desenvolvimento:

```bash
sudo systemctl disable --now prometheus
```

Para reativar depois:

```bash
sudo systemctl enable --now prometheus
```

Após iniciar o Compose, é normal a porta aparecer associada ao Docker:

```bash
sudo ss -ltnp | grep ':9090'
```

---

## 31. Segurança e cardinalidade das métricas

O projeto não utiliza o texto médico como label Prometheus.

Labels controladas:

```text
method
endpoint
status
label_name
```

Isso reduz:

- risco de exposição de conteúdo clínico;
- risco de incluir dados pessoais nas métricas;
- alta cardinalidade;
- crescimento desnecessário das séries temporais.

O endpoint `/metrics` também é excluído da contabilização das métricas HTTP da aplicação para que o próprio scrape do Prometheus não polua o tráfego monitorado.

---

## 32. Thresholds sugeridos

Os valores abaixo são referências iniciais para o projeto acadêmico e precisam ser calibrados para um ambiente real.

| Indicador | Atenção | Crítico |
| --- | ---: | ---: |
| Disponibilidade | `up == 0` | imediato |
| P95 da latência | > 100 ms | > 250 ms |
| Error rate | > 5% | > 10% |
| Memória residente | > 512 MiB | > 1 GiB |

Em produção, os limites devem ser definidos a partir de SLOs, volume real de tráfego, capacidade da infraestrutura e comportamento esperado do modelo.

---

## 33. Reprodução completa por terceiros

A sequência abaixo permite reconstruir e validar as Etapas 1, 2 e 3 a partir de um clone novo.

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

Resultado validado:

```text
13 passed
```

### 5. Baixar dados e treinar

```bash
uv run python -m medical_triage.training.train
```

Esse comando gera:

```text
models/classifier.joblib
models/metrics.json
```

### 6. Validar a API localmente

```bash
uv run uvicorn \
    medical_triage.presentation.api.main:app \
    --host 0.0.0.0 \
    --port 8000
```

Em outro terminal:

```bash
curl http://localhost:8000/health
```

Encerrar:

```text
Ctrl+C
```

### 7. Subir observabilidade completa

Garanta que as portas `8000`, `9090` e `3000` estejam disponíveis.

```bash
docker compose config
docker compose up -d --build
docker compose ps
```

### 8. Validar FastAPI, Prometheus e Grafana

```bash
curl http://localhost:8000/health
curl http://localhost:9090/-/healthy
curl http://localhost:3000/api/health
```

### 9. Gerar tráfego

```bash
for i in {1..10}; do
  curl -s \
    -X POST \
    http://localhost:8000/predict \
    -H "Content-Type: application/json" \
    -d '{
      "text": "The patient presented with acute myocardial infarction and severe coronary artery disease."
    }' \
    > /dev/null
done
```

### 10. Validar o Prometheus

```bash
curl -sG \
  --data-urlencode 'query=medical_triage_predictions_total' \
  http://localhost:9090/api/v1/query \
  | python -m json.tool
```

Target:

```bash
curl -sG \
  --data-urlencode 'query=up{job="medical-triage-api"}' \
  http://localhost:9090/api/v1/query \
  | python -m json.tool
```

### 11. Validar Grafana

Abra:

```text
http://localhost:3000
```

Dashboard esperado:

```text
Dashboards
└── Medical Triage
    └── Medical Triage - Observability
```

### 12. Encerrar observabilidade

```bash
docker compose down
```

### 13. Validar Airflow

Build:

```bash
AIRFLOW_UID=$(id -u) docker compose \
    -f airflow/docker-compose.yml \
    build
```

Subir:

```bash
AIRFLOW_UID=$(id -u) docker compose \
    -f airflow/docker-compose.yml \
    up -d
```

Validar DAG:

```bash
AIRFLOW_UID=$(id -u) docker compose \
    -f airflow/docker-compose.yml \
    exec airflow \
    airflow dags list --local
```

Encerrar:

```bash
AIRFLOW_UID=$(id -u) docker compose \
    -f airflow/docker-compose.yml \
    down
```

Se todos os passos forem concluídos, um terceiro conseguiu reconstruir e validar as Etapas 1, 2 e 3 sem depender dos artefatos originalmente gerados pelo autor.

---

## 34. Fluxo recomendado para contribuidores

Antes de abrir um Pull Request:

```bash
uv sync --locked
uv run pre-commit run --all-files
uv run pytest -v
```

Fluxo:

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

## 35. Critérios concluídos até a Etapa 3

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
[OK] GET /metrics
[OK] Validação de entrada
[OK] Logging sem conteúdo médico
[OK] Métricas sem texto médico
[OK] Docker multi-stage da API
[OK] API executada em container
[OK] Baseline local medido
[OK] Baseline Docker medido

[OK] Ruff
[OK] mypy
[OK] pre-commit
[OK] 13 testes
[OK] Testes de integração da API
[OK] Teste do endpoint /metrics

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

[OK] prometheus-client
[OK] Counter de requisições HTTP
[OK] Histogram de latência
[OK] Counter de predições
[OK] Prometheus
[OK] Target medical-triage-api UP
[OK] PromQL
[OK] Docker Compose de observabilidade
[OK] Grafana
[OK] Datasource provisionado
[OK] Dashboard provisionado
[OK] Total Predictions
[OK] Inference Throughput
[OK] P95 Inference HTTP Latency
[OK] Prediction Error Rate
[OK] Prediction Distribution
[OK] API Memory Usage

[OK] Arquitetura AWS ECR + EC2 definida
[OK] Reprodução por terceiros documentada
```

---

## 36. Próximas etapas

Ainda não fazem parte da implementação atual:

```text
otimização de inferência
ONNX
quantização
pruning
comparação de latência antes/depois
comparação de tamanho dos artefatos
retraining automático por agenda ou drift
alertas operacionais persistidos
deploy automático em AWS
publicação de imagem em registry
```

As próximas funcionalidades devem ser documentadas somente após implementação e validação.

---

## 37. Aviso

Este projeto possui finalidade acadêmica e demonstra conceitos de engenharia de software, Machine Learning e MLOps.

O modelo atual não é um dispositivo médico e não deve ser utilizado para diagnóstico, priorização clínica ou tomada de decisão sobre pacientes.
