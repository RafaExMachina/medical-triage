# Medical Triage — MLOps para Classificação de Textos Médicos

**Versão atual: `0.4.0`**  
**Escopo: Etapas 1, 2, 3 e 4**

API de classificação de textos médicos construída de forma incremental para demonstrar um pipeline de MLOps reproduzível: treinamento, serving, testes, CI, orquestração, observabilidade e otimização de inferência com ONNX Runtime.

> **Importante:** este projeto tem finalidade acadêmica. O dataset não possui rótulos de urgência clínica e o sistema **não é um dispositivo médico**. Ele não deve ser usado para diagnóstico, priorização clínica ou tomada de decisão sobre pacientes.

---

## Sumário

- [1. Visão geral](#1-visão-geral)
- [2. Quick start com uv](#2-quick-start-com-uv)
- [3. Quick start com Docker Compose](#3-quick-start-com-docker-compose)
- [4. Escopo clínico do dataset](#4-escopo-clínico-do-dataset)
- [5. Evolução por etapas](#5-evolução-por-etapas)
- [6. Arquitetura atual](#6-arquitetura-atual)
- [7. Dataset e modelo](#7-dataset-e-modelo)
- [8. Instalação e ambiente Python](#8-instalação-e-ambiente-python)
- [9. Executando a API localmente](#9-executando-a-api-localmente)
- [10. Endpoints](#10-endpoints)
- [11. Backends de inferência](#11-backends-de-inferência)
- [12. Treinamento e geração de artefatos](#12-treinamento-e-geração-de-artefatos)
- [13. Otimização com ONNX Runtime](#13-otimização-com-onnx-runtime)
- [14. Benchmarks](#14-benchmarks)
- [15. Qualidade, testes e pre-commit](#15-qualidade-testes-e-pre-commit)
- [16. Docker](#16-docker)
- [17. Observabilidade](#17-observabilidade)
- [18. Apache Airflow](#18-apache-airflow)
- [19. CI com GitHub Actions](#19-ci-com-github-actions)
- [20. Estrutura do projeto](#20-estrutura-do-projeto)
- [21. Reprodução completa por terceiros](#21-reprodução-completa-por-terceiros)
- [22. Variáveis de ambiente](#22-variáveis-de-ambiente)
- [23. Troubleshooting](#23-troubleshooting)
- [24. Fluxo para contribuidores](#24-fluxo-para-contribuidores)
- [25. Documentação complementar](#25-documentação-complementar)
- [26. Estado atual](#26-estado-atual)
- [27. Vídeo STAR](#27-vídeo-star)
- [28. Próximos passos](#28-próximos-passos)
- [29. Aviso](#29-aviso)

---

## 1. Visão geral

O projeto demonstra como evoluir um serviço de Machine Learning desde um baseline local até um runtime otimizado e observável.

Principais capacidades já implementadas:

- download automático do dataset;
- treinamento com TF-IDF + Logistic Regression;
- persistência de modelo e métricas;
- API REST com FastAPI;
- validação de entrada com Pydantic;
- backend de inferência selecionável;
- ONNX Runtime como backend padrão;
- Docker multi-stage;
- Docker Compose para API, Prometheus e Grafana;
- testes unitários e de integração;
- Ruff, mypy e pre-commit;
- CI com GitHub Actions;
- pipeline de treinamento com Apache Airflow;
- métricas Prometheus;
- dashboard Grafana provisionado automaticamente;
- benchmarks de latência;
- validação de equivalência sklearn × ONNX;
- documentação de reprodução para terceiros;
- decisão arquitetural de deploy real-time em AWS (ECR + EC2).

Fluxo atual de serving:

```text
Client
  |
  v
FastAPI
  |
  v
ClassifyMedicalTextUseCase
  |
  v
ClassifierPort
  |
  +----------------------------+
  |                            |
  v                            v
OnnxClassifierAdapter     SklearnClassifierAdapter
  |                            |
  v                            v
ONNX Runtime              scikit-learn
  |                            |
  v                            v
classifier.onnx           classifier.joblib
```

O backend padrão é **ONNX**.

---

## 2. Quick start com uv

Este projeto foi projetado para usar **uv** como gerenciador de Python, ambiente virtual, dependências e execução de comandos.

### 2.1 Clonar

HTTPS:

```bash
git clone https://github.com/RafaExMachina/medical-triage.git
cd medical-triage
```

SSH:

```bash
git clone git@github.com:RafaExMachina/medical-triage.git
cd medical-triage
```

### 2.2 Instalar as dependências

```bash
uv sync --locked
```

O `uv.lock` é versionado. Isso permite recriar o ambiente de forma determinística.

> Não é necessário ativar manualmente `.venv`. Prefira executar os comandos com `uv run`.

### 2.3 Validar o projeto

```bash
uv run pytest
```

Estado validado na versão `0.4.0`:

```text
21 passed
```

A suíte pode exibir um `StarletteDeprecationWarning` relacionado a `TestClient/httpx`; o warning conhecido não impede a execução dos testes.

### 2.4 Subir a API

```bash
uv run uvicorn \
  medical_triage.presentation.api.main:app \
  --host 127.0.0.1 \
  --port 8000
```

Em outro terminal:

```bash
curl http://127.0.0.1:8000/health
```

Resposta esperada:

```json
{"status":"healthy"}
```

Swagger/OpenAPI:

```text
http://127.0.0.1:8000/docs
```

A execução local usa ONNX por padrão.

---

## 3. Quick start com Docker Compose

Para executar API, Prometheus e Grafana:

```bash
docker compose up -d --build
```

Confira:

```bash
docker compose ps
```

Serviços:

| Serviço | URL |
|---|---|
| FastAPI | `http://localhost:8000` |
| Swagger | `http://localhost:8000/docs` |
| Prometheus | `http://localhost:9090` |
| Grafana | `http://localhost:3000` |

Health check da API:

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

Credenciais locais do Grafana:

```text
user:     admin
password: admin
```

> Essas credenciais são apenas para desenvolvimento local.

Para encerrar:

```bash
docker compose down
```

Para remover também os volumes:

```bash
docker compose down -v
```

A imagem de produção usa o backend ONNX e inclui somente os artefatos ONNX necessários ao serving.

---

## 4. Escopo clínico do dataset

O projeto utiliza o **Medical Abstracts TC Corpus**.

Categorias:

1. `neoplasms`;
2. `digestive system diseases`;
3. `nervous system diseases`;
4. `cardiovascular diseases`;
5. `general pathological conditions`.

**O dataset não possui rótulos de urgência clínica.**

Portanto, o projeto demonstra classificação de textos médicos e infraestrutura de MLOps, mas não implementa uma triagem clínica real do tipo:

```text
normal
attention
urgent
```

Não é feita conversão artificial entre categoria de doença e nível de urgência.

---

## 5. Evolução por etapas

### Etapa 1 — baseline, API e Docker

```text
Medical Abstracts TC Corpus
          |
          v
     DatasetLoader
          |
          v
Stratified Train/Validation Split
          |
          v
        TF-IDF
          |
          v
 Logistic Regression
          |
     +----+-----------+
     |                |
     v                v
classifier.joblib   metrics.json
          |
          v
       FastAPI
          |
          v
        Docker
```

Entregas:

- baseline de NLP;
- API `/health` e `/predict`;
- validação Pydantic;
- imagem Docker multi-stage;
- benchmark inicial de latência;
- logging sem registrar o texto médico.

#### Decisão arquitetural de deploy em nuvem

A estratégia escolhida para o serving é **real-time**. Cada requisição HTTP contém
um texto médico individual e espera uma classificação imediata, portanto o fluxo
principal é síncrono:

```text
Cliente
   |
   v
POST /predict
   |
   v
FastAPI
   |
   v
ONNX Runtime
   |
   v
Resposta imediata
```

Uma abordagem batch seria mais apropriada para cenários em que grandes volumes
de documentos fossem acumulados e processados periodicamente. Esse não é o fluxo
principal deste projeto.

Para um deploy em nuvem, a arquitetura selecionada é baseada em **AWS**:

```text
Developer / GitHub Actions
          |
          v
      Docker Image
          |
          v
        AWS ECR
          |
          v
        AWS EC2
          |
          v
 FastAPI + ONNX Runtime
          |
          v
      POST /predict
```

O **Amazon ECR** foi escolhido como registry para armazenar a imagem Docker e uma
instância **Amazon EC2** como ambiente de execução contínua da API.

A escolha prioriza simplicidade, compatibilidade com o serving real-time e baixo
acoplamento operacional para o escopo acadêmico do projeto. Uma plataforma
Kubernetes seria possível, mas adicionaria complexidade desnecessária ao cenário
atual.

> O deploy automatizado em AWS ainda não faz parte da implementação atual. A
> arquitetura de cloud está definida e documentada como decisão de projeto.

### Etapa 2 — qualidade, CI e Airflow

```text
feature branch
     |
     v
pre-commit
     |
     v
GitHub Actions
  +--+---+
  |      |
  v      v
Quality Tests
```

Treinamento orquestrado:

```text
prepare_dataset_task
        |
        v
validate_dataset_task
        |
        v
train_model_task
        |
        v
validate_artifacts_task
```

Entregas:

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
FastAPI :8000
    |
    | GET /metrics
    v
Prometheus :9090
    |
    | PromQL
    v
Grafana :3000
```

Entregas:

- `prometheus-client`;
- métricas HTTP;
- métricas de latência;
- métricas de predições;
- métricas do processo Python;
- Prometheus;
- Grafana;
- dashboard com seis painéis;
- provisionamento automático.

### Etapa 4 — otimização ONNX

```text
scikit-learn pipeline
        |
        | export
        v
classifier.onnx
        |
        v
ONNX Runtime
        |
        v
OnnxClassifierAdapter
        |
        v
FastAPI
```

Entregas:

- export do pipeline para ONNX;
- validação de equivalência;
- diagnóstico das divergências no TF-IDF convertido;
- benchmark sklearn × ONNX;
- backend ONNX integrado à aplicação;
- seleção por `MODEL_BACKEND`;
- ONNX como backend padrão;
- imagem Docker de produção sem `classifier.joblib`;
- benchmark HTTP end-to-end;
- redução de latência e tamanho do artefato.

---

## 6. Arquitetura atual

A aplicação utiliza uma organização inspirada em Clean Architecture e princípios SOLID.

```text
Presentation
FastAPI / routes
      |
      v
Application
ClassifyMedicalTextUseCase
      |
      v
Domain
ClassifierPort
      ^
      |
      +-----------------------------+
      |                             |
Infrastructure                 Infrastructure
OnnxClassifierAdapter          SklearnClassifierAdapter
```

A dependência do classificador é resolvida em:

```text
src/medical_triage/presentation/api/dependencies.py
```

A escolha do backend é feita por configuração, sem alterar o caso de uso.

---

## 7. Dataset e modelo

### Baseline

Pipeline:

```text
TfidfVectorizer
      +
LogisticRegression
```

Split de desenvolvimento:

```text
90% training
10% validation
random_state = 42
```

O arquivo oficial `medical_tc_test.csv` fica reservado para avaliação e comparação.

Resultados do baseline na validação:

| Métrica | Resultado |
|---|---:|
| Accuracy | `0.5931` |
| Macro F1 | `0.5908` |

Versão lógica do modelo:

```text
tfidf-logreg-v1
```

### Dataset

O componente:

```text
src/medical_triage/data/dataset_loader.py
```

trabalha com:

```text
medical_tc_train.csv
medical_tc_test.csv
medical_tc_labels.csv
```

Se um arquivo necessário não existir, o loader realiza o download.

---

## 8. Instalação e ambiente Python

Pré-requisitos principais:

- Git;
- Python 3.12 compatível com o projeto;
- `uv`;
- Docker Engine e Docker Compose para execução containerizada;
- internet na primeira preparação do dataset.

Verifique:

```bash
git --version
uv --version
docker --version
docker compose version
```

Sincronize o ambiente:

```bash
uv sync --locked
```

Executar comandos:

```bash
uv run <comando>
```

Exemplos:

```bash
uv run pytest
uv run ruff check .
uv run python -m medical_triage.training.train
```

Para instalar os hooks locais:

```bash
uv run pre-commit install
```

---

## 9. Executando a API localmente

### Backend ONNX — padrão

Depois de:

```bash
uv sync --locked
```

execute:

```bash
uv run uvicorn \
  medical_triage.presentation.api.main:app \
  --host 0.0.0.0 \
  --port 8000
```

Ou explicitamente:

```bash
MODEL_BACKEND=onnx \
uv run uvicorn \
  medical_triage.presentation.api.main:app \
  --host 0.0.0.0 \
  --port 8000
```

### Backend sklearn — opcional

O backend sklearn requer `models/classifier.joblib`.

Gere o artefato:

```bash
uv run python -m medical_triage.training.train
```

Depois:

```bash
MODEL_BACKEND=sklearn \
uv run uvicorn \
  medical_triage.presentation.api.main:app \
  --host 0.0.0.0 \
  --port 8000
```

---

## 10. Endpoints

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

Exemplo de resposta com ONNX:

```json
{
  "label_id": 4,
  "label_name": "cardiovascular diseases",
  "confidence": 0.884641170501709,
  "model_version": "tfidf-logreg-v1",
  "inference_ms": 4.19323400092253
}
```

`inference_ms` varia entre requisições e ambientes.

### `GET /metrics`

```bash
curl http://localhost:8000/metrics
```

O endpoint retorna métricas no formato Prometheus e não é incluído no schema OpenAPI.

### Validação de entrada

O campo `text` deve ter pelo menos 20 caracteres.

Exemplo inválido:

```bash
curl \
  -X POST \
  http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text":"short"}'
```

Resposta esperada: HTTP `422`.

---

## 11. Backends de inferência

Backends suportados:

| Backend | Uso |
|---|---|
| `onnx` | padrão e recomendado para serving |
| `sklearn` | baseline e comparação |

Selecionar:

```bash
MODEL_BACKEND=onnx
```

ou:

```bash
MODEL_BACKEND=sklearn
```

Arquivos padrão:

```text
ONNX_MODEL_PATH=models/classifier.onnx
ONNX_METADATA_PATH=models/classifier_onnx_metadata.json
MODEL_PATH=models/classifier.joblib
```

O caso de uso depende de `ClassifierPort`, portanto a troca de backend não altera as regras da aplicação.

---

## 12. Treinamento e geração de artefatos

Treinamento standalone:

```bash
uv run python -m medical_triage.training.train
```

Fluxo:

```text
preparação do dataset
        |
        v
validação
        |
        v
split treino/validação
        |
        v
TF-IDF
        |
        v
Logistic Regression
        |
        v
avaliação
        |
        v
persistência
```

Artefatos gerados pelo treinamento:

```text
models/classifier.joblib
models/metrics.json
```

Export ONNX:

```bash
uv run python scripts/export_onnx.py
```

Artefatos ONNX utilizados no serving:

```text
models/classifier.onnx
models/classifier_onnx_metadata.json
```

---

## 13. Otimização com ONNX Runtime

A Etapa 4 converteu o pipeline completo para ONNX.

### Equivalência no conjunto oficial de teste

Total:

```text
2888 exemplos
```

Resultado:

```text
Predições iguais:       2859
Predições diferentes:     29
Agreement:             98.995845%
```

Qualidade:

| Métrica | sklearn | Full ONNX |
|---|---:|---:|
| Accuracy | `0.583102` | `0.585873` |
| Macro F1 | `0.585593` | `0.587332` |

A pequena diferença **não é interpretada como melhoria de qualidade**. Ela decorre da conversão do pré-processamento textual.

### Diagnóstico

Um experimento híbrido manteve o TF-IDF do sklearn e converteu somente o `LogisticRegression`.

Resultado:

```text
Agreement:               100%
Accuracy sklearn:     0.583102
Accuracy híbrido:     0.583102
Macro F1 sklearn:     0.585593
Macro F1 híbrido:     0.585593
Max probability diff:  ~1.96e-07
Mean probability diff: ~1.51e-08
```

Isso indica que as divergências do Full ONNX estão no estágio de TF-IDF convertido, e não no classificador Logistic Regression.

Documentação detalhada:

```text
docs/stage4-onnx-optimization.md
```

---

## 14. Benchmarks

Os números abaixo foram medidos no ambiente de desenvolvimento do projeto e devem ser interpretados como comparação relativa entre os backends, não como garantia de desempenho em outras máquinas.

### 14.1 Tamanho do artefato

| Artefato | Tamanho |
|---|---:|
| `classifier.joblib` | `~3.877 MiB` |
| `classifier.onnx` | `~2.656 MiB` |

Redução:

```text
~31.49%
```

### 14.2 Benchmark isolado do modelo

Resultados consolidados de três execuções controladas:

| Métrica | sklearn | Full ONNX |
|---|---:|---:|
| Mean | `0.9965 ms` | `0.4640 ms` |
| P50 | `0.9555 ms` | `0.4487 ms` |
| P95 | `1.4067 ms` | `0.6135 ms` |
| P99 | `1.7523 ms` | `0.7388 ms` |

Resumo:

```text
Speedup médio:         ~2.15x
Redução média:         ~53.44%
```

### 14.3 Benchmark HTTP end-to-end

Metodologia:

```text
3 runs por backend
1000 requests medidos por run
50 warm-ups por run
single-text
conexão HTTP persistente
```

Latência observada pelo cliente:

| Métrica | sklearn | ONNX | Speedup | Redução |
|---|---:|---:|---:|---:|
| Mean | `10.9768 ms` | `7.1991 ms` | `1.52x` | `34.42%` |
| P50 | `9.4146 ms` | `5.9003 ms` | `1.60x` | `37.33%` |
| P95 | `21.2380 ms` | `14.5180 ms` | `1.46x` | `31.64%` |
| P99 | `31.9599 ms` | `22.6826 ms` | `1.41x` | `29.03%` |

Inferência medida dentro da API:

| Métrica | sklearn | ONNX | Speedup | Redução |
|---|---:|---:|---:|---:|
| Mean | `5.2783 ms` | `1.0792 ms` | `4.89x` | `79.55%` |
| P50 | `4.6485 ms` | `0.9451 ms` | `4.92x` | `79.67%` |
| P95 | `9.4335 ms` | `1.8705 ms` | `5.04x` | `80.17%` |
| P99 | `16.0695 ms` | `3.7322 ms` | `4.31x` | `76.77%` |

Relatórios:

```text
reports/onnx_equivalence.json
reports/onnx_hybrid_equivalence.json
reports/inference_benchmark_summary.json
reports/http_benchmark_summary.json
```

Scripts de benchmark:

```text
scripts/benchmark_inference_backends.py
scripts/benchmark_http_api.py
scripts/summarize_http_benchmarks.py
```

---

## 15. Qualidade, testes e pre-commit

Quality gates:

```bash
uv run ruff format --check .
uv run ruff check .
uv run mypy src
uv run pytest
```

Execução consolidada:

```bash
uv run pre-commit run --all-files
```

Estado validado na versão `0.4.0`:

```text
Ruff:       PASS
mypy:       PASS
Pytest:     21 passed
pre-commit: PASS
```

Os testes incluem:

- casos de uso;
- dataset loader;
- configuração de backend;
- dependency injection;
- adapter ONNX;
- workflow de treinamento;
- `GET /health`;
- `POST /predict`;
- validação HTTP;
- métricas.

---

## 16. Docker

O `Dockerfile` da raiz usa build multi-stage.

Build:

```bash
docker build \
  -t medical-triage-api:0.4.0 \
  .
```

Execução:

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

### Artefatos na imagem

A imagem de produção inclui somente:

```text
classifier.onnx
classifier_onnx_metadata.json
```

O `.dockerignore` exclui os demais arquivos de `models/`.

### Locale requerido pelo ONNX Runtime

O pipeline Full ONNX contém o operador `StringNormalizer`.

A imagem `python:3.12-slim` não fornece `en_US.UTF-8` por padrão, portanto o `Dockerfile` configura esse locale explicitamente.

Isso evita a falha de inicialização do ONNX Runtime:

```text
Failed to construct locale with name: en_US.UTF-8
```

Essa configuração já faz parte da imagem; usuários do projeto não precisam configurá-la manualmente no host.

---

## 17. Observabilidade

A stack de observabilidade usa:

```text
FastAPI
  |
  | /metrics
  v
Prometheus
  |
  | PromQL
  v
Grafana
```

Métricas principais:

```text
medical_triage_http_requests_total
medical_triage_http_request_duration_seconds
medical_triage_predictions_total
process_resident_memory_bytes
up
```

O texto médico recebido pela API **não é usado em logs nem como label Prometheus**.

Dashboard:

```text
Medical Triage - Observability
```

O dashboard é versionado em JSON no repositório:

```text
monitoring/grafana/dashboards/medical-triage.json
```

O datasource Prometheus e o dashboard são provisionados automaticamente pelo
Docker Compose a partir dos arquivos versionados.

Painéis:

| Painel | Objetivo |
|---|---|
| Total Predictions | total acumulado |
| Inference Throughput | tráfego de inferência |
| P95 Inference HTTP Latency | P95 de `/predict` |
| Prediction Error Rate | respostas 4xx/5xx |
| Prediction Distribution | distribuição das classes |
| API Memory Usage | memória residente |

Documentação detalhada:

```text
docs/observability-plan.md
```

---

## 18. Apache Airflow

O Airflow é executado em ambiente Docker separado da API.

Estrutura:

```text
airflow/
├── dags/
│   └── medical_triage_training.py
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

Versão validada:

```text
Apache Airflow 3.3.1
Python 3.12
```

Subir:

```bash
AIRFLOW_UID=$(id -u) docker compose \
  -f airflow/docker-compose.yml \
  up -d --build
```

Listar DAGs:

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

Tasks:

```text
prepare_dataset_task
validate_dataset_task
train_model_task
validate_artifacts_task
```

Disparar:

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

Encerrar:

```bash
AIRFLOW_UID=$(id -u) docker compose \
  -f airflow/docker-compose.yml \
  down
```

---

## 19. CI com GitHub Actions

Workflow:

```text
.github/workflows/ci.yml
```

Gatilhos:

```text
push
pull_request -> master
```

Pipeline:

```text
CI
├── Quality
│   ├── ruff format --check
│   ├── ruff check
│   └── mypy src
└── Tests
    └── pytest
```

As dependências são instaladas com:

```bash
uv sync --locked
```

---

## 20. Estrutura do projeto

Visão simplificada:

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
│   ├── observability-plan.md
│   └── stage4-onnx-optimization.md
├── models/
│   ├── classifier.onnx
│   └── classifier_onnx_metadata.json
├── monitoring/
│   ├── grafana/
│   └── prometheus/
├── reports/
│   ├── inference_benchmark_summary.json
│   ├── http_benchmark_summary.json
│   ├── onnx_equivalence.json
│   └── onnx_hybrid_equivalence.json
├── scripts/
│   ├── benchmark_http_api.py
│   ├── benchmark_inference_backends.py
│   ├── export_onnx.py
│   ├── measure_latency.py
│   └── summarize_http_benchmarks.py
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
│   └── units/
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

Arquivos locais como dataset baixado, `classifier.joblib` e `metrics.json` podem ser regenerados quando necessário.

---

## 21. Reprodução completa por terceiros

Esta é a sequência recomendada para validar o projeto a partir de um clone novo.

### 1. Clonar

```bash
git clone https://github.com/RafaExMachina/medical-triage.git
cd medical-triage
```

### 2. Criar/sincronizar o ambiente

```bash
uv sync --locked
```

### 3. Executar os testes

```bash
uv run pytest
```

Esperado:

```text
21 passed
```

### 4. Validar os quality gates

```bash
uv run pre-commit run --all-files
```

### 5. Validar a API ONNX local

```bash
uv run uvicorn \
  medical_triage.presentation.api.main:app \
  --host 127.0.0.1 \
  --port 8000
```

Em outro terminal:

```bash
curl http://127.0.0.1:8000/health
```

Teste uma predição:

```bash
curl \
  -X POST \
  http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "text": "The patient presented with acute myocardial infarction and severe coronary artery disease."
  }'
```

Encerre o Uvicorn com `Ctrl+C`.

### 6. Treinar o baseline sklearn, se quiser reproduzir o treinamento

```bash
uv run python -m medical_triage.training.train
```

Isso gera localmente:

```text
models/classifier.joblib
models/metrics.json
```

### 7. Subir a stack de observabilidade

Garanta que as portas `8000`, `9090` e `3000` estejam livres:

```bash
docker compose up -d --build
docker compose ps
```

### 8. Validar os serviços

```bash
curl http://localhost:8000/health
curl http://localhost:9090/-/healthy
curl http://localhost:3000/api/health
```

### 9. Validar o target do Prometheus

```bash
curl -sG \
  --data-urlencode 'query=up{job="medical-triage-api"}' \
  http://localhost:9090/api/v1/query \
  | python -m json.tool
```

O target esperado é:

```text
instance="api:8000"
job="medical-triage-api"
value="1"
```

### 10. Abrir o Grafana

```text
http://localhost:3000
```

Dashboard:

```text
Medical Triage - Observability
```

### 11. Encerrar

```bash
docker compose down
```

Se esses passos forem concluídos, o serving ONNX, a API, os testes e a stack de observabilidade foram reproduzidos sem depender do ambiente original do autor.

---

## 22. Variáveis de ambiente

| Variável | Padrão | Descrição |
|---|---|---|
| `MODEL_BACKEND` | `onnx` | backend de inferência |
| `MODEL_PATH` | `models/classifier.joblib` | modelo sklearn |
| `ONNX_MODEL_PATH` | `models/classifier.onnx` | modelo ONNX |
| `ONNX_METADATA_PATH` | `models/classifier_onnx_metadata.json` | metadados ONNX |
| `APP_NAME` | `Medical Triage API` | nome da aplicação |
| `APP_VERSION` | `0.4.0` | versão exposta pela aplicação |

Exemplo:

```bash
MODEL_BACKEND=onnx \
APP_VERSION=0.4.0 \
uv run uvicorn \
  medical_triage.presentation.api.main:app \
  --host 127.0.0.1 \
  --port 8000
```

---

## 23. Troubleshooting

### Docker não está iniciado

Ubuntu:

```bash
sudo systemctl start docker
```

Verifique:

```bash
docker info
```

### Porta 8000 ocupada

```bash
ss -ltnp | grep ':8000'
```

### Porta 9090 ocupada

Se houver um Prometheus instalado diretamente no host:

```bash
sudo ss -ltnp | grep ':9090'
```

Pare temporariamente:

```bash
sudo systemctl stop prometheus
```

Ou desative durante o desenvolvimento:

```bash
sudo systemctl disable --now prometheus
```

Para reativar:

```bash
sudo systemctl enable --now prometheus
```

### ONNX e locale no Docker

O runtime Docker já configura `en_US.UTF-8`.

Se estiver diagnosticando a imagem:

```bash
docker run --rm \
  --entrypoint sh \
  medical-triage-api \
  -c 'locale -a'
```

### Confirmar o backend no container

```bash
docker compose exec -T api python - <<'PY'
from medical_triage.config import get_settings
from medical_triage.presentation.api.dependencies import get_classifier

settings = get_settings()
classifier = get_classifier()

print(f"backend: {settings.model_backend}")
print(f"adapter: {type(classifier).__name__}")
PY
```

Esperado na configuração de produção:

```text
backend: onnx
adapter: OnnxClassifierAdapter
```

---

## 24. Fluxo para contribuidores

Crie uma branch:

```bash
git switch -c feat/minha-feature
```

Sincronize:

```bash
uv sync --locked
```

Antes do commit:

```bash
uv run pre-commit run --all-files
uv run pytest
```

Fluxo recomendado:

```text
clone / fork
    |
    v
feature branch
    |
    v
desenvolvimento
    |
    v
pre-commit + pytest
    |
    v
git push
    |
    v
Pull Request -> master
    |
    v
GitHub Actions
```

Não versione datasets baixados nem artefatos locais que estejam ignorados pelo Git.

---

## 25. Documentação complementar

Observabilidade:

```text
docs/observability-plan.md
```

Otimização ONNX:

```text
docs/stage4-onnx-optimization.md
```

Relatórios:

```text
reports/
```

---

## 26. Estado atual

```text
[OK] DatasetLoader
[OK] Download automático do dataset
[OK] Split treino/validação estratificado
[OK] TF-IDF + Logistic Regression
[OK] Baseline sklearn
[OK] FastAPI
[OK] GET /health
[OK] POST /predict
[OK] GET /metrics
[OK] Validação de entrada
[OK] Logging sem conteúdo médico
[OK] Métricas sem texto médico
[OK] Docker multi-stage
[OK] API em container
[OK] Ruff
[OK] mypy
[OK] pre-commit
[OK] 21 testes
[OK] GitHub Actions
[OK] Apache Airflow 3.3.1
[OK] DAG medical_triage_training
[OK] Prometheus
[OK] Grafana
[OK] Dashboard provisionado
[OK] ONNX Runtime
[OK] Export Full ONNX
[OK] Validação de equivalência
[OK] Backend ONNX integrado
[OK] Backend configurável
[OK] ONNX padrão de produção
[OK] Docker com artefatos ONNX
[OK] Benchmark sklearn x ONNX
[OK] Benchmark HTTP end-to-end
[OK] Versão 0.4.0
```

---

## 27. Vídeo STAR

A demonstração final do projeto deve seguir a metodologia STAR
(**Situação, Tarefa, Ação e Resultado**) e ter duração máxima de 5 minutos.

- **Situação:** necessidade de disponibilizar e operar um classificador de textos
  médicos com práticas de MLOps;
- **Tarefa:** construir serving, automação, observabilidade e otimização de
  inferência;
- **Ação:** FastAPI, Docker, GitHub Actions, Airflow, Prometheus, Grafana e
  conversão do pipeline para ONNX Runtime;
- **Resultado:** API reproduzível e observável, com redução aproximada de
  **79,55%** no tempo médio de inferência dentro da API e **34,42%** na latência
  HTTP média end-to-end.

**Link do vídeo:** _adicionar após a publicação_

> O link acima deve ser substituído pelo endereço público ou compartilhável do
> vídeo antes da entrega final.

---

## 28. Próximos passos

Itens que podem ser evoluídos em versões futuras:

- deploy automatizado em cloud;
- publicação automática da imagem em registry;
- alertas operacionais;
- retraining por agenda ou drift;
- avaliação de quantização quando aplicável;
- testes de carga concorrente;
- SLOs e alertas baseados em métricas reais.

---

## 29. Aviso

Este projeto possui finalidade acadêmica e demonstra conceitos de engenharia de software, Machine Learning e MLOps.

O modelo atual:

- não é um dispositivo médico;
- não deve ser usado para diagnóstico;
- não deve ser usado para priorização clínica;
- não deve ser usado para tomada de decisão sobre pacientes.

O objetivo do repositório é demonstrar uma arquitetura reproduzível de treinamento, serving, observabilidade e otimização de modelos.
