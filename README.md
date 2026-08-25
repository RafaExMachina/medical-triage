# Medical Triage — Etapa 1

API de classificação de textos médicos construída como baseline para um pipeline incremental de MLOps.

> **Escopo atual:** este README documenta somente o que está implementado e validado na **Etapa 1**. As próximas etapas do projeto — CI/CD, Airflow, Prometheus, Grafana e otimização de inferência — serão adicionadas incrementalmente quando forem implementadas.

---

## 1. Objetivo da Etapa 1

A Etapa 1 estabelece uma base reproduzível para servir um modelo de NLP em tempo real.

Nesta etapa, o projeto implementa:

- carregamento automático do dataset;
- treinamento de um classificador baseline;
- persistência do modelo e das métricas;
- API REST com FastAPI;
- validação de entrada;
- container Docker multi-stage;
- testes unitários;
- lint e análise estática de tipos;
- medição de latência local e em Docker;
- definição da arquitetura de deploy em nuvem;
- documentação para que um terceiro consiga reproduzir o fluxo.

---

## 2. Importante: escopo clínico do dataset

O projeto utiliza o **Medical Abstracts TC Corpus**.

O dataset possui cinco categorias de condições médicas:

1. neoplasms;
2. digestive system diseases;
3. nervous system diseases;
4. cardiovascular diseases;
5. general pathological conditions.

**O dataset não possui rótulos de urgência clínica.**

Portanto, nesta Etapa 1, o projeto demonstra a infraestrutura de classificação de textos médicos e de serving de modelos, mas **não deve ser interpretado como um sistema clínico real de triagem `normal / attention / urgent`**.

Não é feita nenhuma conversão artificial entre categoria de doença e nível de urgência.

---

## 3. Arquitetura da Etapa 1

A aplicação utiliza uma organização inspirada em Clean Architecture e princípios SOLID.

```text
                         ┌─────────────────────┐
                         │       Client        │
                         └──────────┬──────────┘
                                    │
                              HTTP / JSON
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │       FastAPI       │
                         │ Presentation Layer  │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │      Use Case       │
                         │ Application Layer   │
                         └──────────┬──────────┘
                                    │
                             ClassifierPort
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │ SklearnClassifier   │
                         │ Infrastructure      │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │ classifier.joblib   │
                         │ TF-IDF + LogReg     │
                         └─────────────────────┘
```

O treinamento segue:

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
      ┌───┴────────────┐
      ▼                ▼
classifier.joblib   metrics.json
```

---

## 4. Decisão: inferência em tempo real

A API utiliza inferência **real-time**.

A escolha é adequada porque o fluxo esperado é:

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

Uma abordagem batch seria mais adequada para grandes volumes processados de forma periódica, mas não para uma aplicação interativa que precisa responder a uma requisição individual.

---

## 5. Arquitetura de cloud selecionada

A arquitetura selecionada para deploy é:

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

### AWS ECR

O Amazon Elastic Container Registry será utilizado como registro da imagem Docker.

### AWS EC2

A instância EC2 executará continuamente o container da API, característica adequada ao cenário de inferência em tempo real e à necessidade de controle sobre o ambiente de execução.

> Nesta etapa, a arquitetura de cloud está definida e documentada. A automação de entrega será adicionada incrementalmente nas próximas etapas.

---

## 6. Tecnologias

- Python 3.12
- uv
- FastAPI
- Uvicorn
- Scikit-learn
- Pandas
- Joblib
- Pytest
- Ruff
- mypy
- Docker

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
│       │   ├── __init__.py
│       │   └── use_cases.py
│       ├── data/
│       │   ├── __init__.py
│       │   └── dataset_loader.py
│       ├── domain/
│       │   ├── __init__.py
│       │   ├── entities.py
│       │   └── ports.py
│       ├── infrastructure/
│       │   ├── __init__.py
│       │   └── sklearn_classifier.py
│       ├── observability/
│       │   ├── __init__.py
│       │   └── logging.py
│       ├── presentation/
│       │   ├── __init__.py
│       │   └── api/
│       │       ├── __init__.py
│       │       ├── dependencies.py
│       │       ├── main.py
│       │       ├── routes.py
│       │       └── schemas.py
│       ├── training/
│       │   ├── __init__.py
│       │   └── train.py
│       ├── __init__.py
│       └── config.py
├── tests/
│   └── units/
│       ├── test_classify_use_case.py
│       └── test_dataset_loader.py
├── Dockerfile
├── pyproject.toml
├── README.md
└── uv.lock
```

---

## 8. Pré-requisitos

Para executar o projeto em uma máquina nova:

- Linux, macOS ou Windows com WSL;
- Python compatível com o projeto;
- `uv`;
- Docker Engine para os testes de container;
- acesso à internet na primeira execução do treinamento, para download do dataset.

Verifique:

```bash
uv --version
python3 --version
docker --version
```

---

## 9. Clone e instalação

Clone o repositório e entre no diretório do projeto:

```bash
git clone git@github.com:RafaExMachina/medical-triage-uv.git
cd medical-triage
```

Instale exatamente as dependências registradas no lockfile:

```bash
uv sync --locked
```

Não é necessário ativar manualmente o ambiente virtual para executar os comandos descritos neste README.

Os comandos são executados com:

```bash
uv run ...
```

---

## 10. Dataset

O projeto não exige que os CSVs sejam baixados manualmente.

O componente:

```text
src/medical_triage/data/dataset_loader.py
```

verifica `data/raw/`.

Os arquivos necessários são:

```text
medical_tc_train.csv
medical_tc_test.csv
medical_tc_labels.csv
```

Se um arquivo já existir e não estiver vazio, ele é reutilizado.

Se estiver ausente, o loader realiza o download.

O fluxo é:

```text
DatasetLoader.prepare()
        │
        ├── arquivo existente ──► reutiliza
        │
        └── arquivo ausente ────► download
```

### Execução isolada do loader

Opcionalmente:

```bash
uv run python -m medical_triage.data.dataset_loader
```

Entretanto, isso não é obrigatório, pois o treinamento chama o loader automaticamente.

---

## 11. Treinamento

Execute:

```bash
uv run python -m medical_triage.training.train
```

Em uma instalação nova, o processo deverá:

```text
download do dataset
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

O split utilizado na Etapa 1 é:

```text
90% training
10% validation
```

com estratificação por classe e:

```text
random_state = 42
```

O arquivo oficial `medical_tc_test.csv` é baixado, mas permanece reservado e não é utilizado para selecionar ou ajustar o baseline nesta etapa.

### Artefatos gerados

Após o treinamento:

```text
models/
├── classifier.joblib
└── metrics.json
```

O `classifier.joblib` contém:

- pipeline treinado;
- mapeamento dos labels;
- versão do modelo.

Versão atual:

```text
tfidf-logreg-v1
```

---

## 12. Resultado do baseline de classificação

Treinamento validado com:

```text
Training samples:   10,395
Validation samples:  1,155
Classes:                 5
```

Resultado atual:

| Métrica | Resultado |
|---|---:|
| Accuracy | 0.5931 |
| Macro F1 | 0.5908 |

Esses valores representam o **baseline da Etapa 1**, não uma otimização final do modelo.

---

## 13. Executando a API localmente

O modelo precisa ter sido treinado previamente:

```bash
uv run python -m medical_triage.training.train
```

Depois execute:

```bash
uv run uvicorn \
    medical_triage.presentation.api.main:app \
    --host 0.0.0.0 \
    --port 8000
```

A API estará disponível em:

```text
http://localhost:8000
```

A documentação interativa do FastAPI estará em:

```text
http://localhost:8000/docs
```

---

## 14. Health check

Execute:

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

## 15. Classificação

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

Exemplo de resposta obtida:

```json
{
  "label_id": 4,
  "label_name": "cardiovascular diseases",
  "confidence": 0.8846411668151615,
  "model_version": "tfidf-logreg-v1",
  "inference_ms": 10.36887200007186
}
```

O valor de `inference_ms` varia entre requisições.

---

## 16. Validação de entrada

O campo `text` deve possuir pelo menos 20 caracteres.

Exemplo inválido:

```bash
curl \
    -X POST \
    http://localhost:8000/predict \
    -H "Content-Type: application/json" \
    -d '{
        "text": "short"
    }'
```

A API responde com HTTP `422`.

Exemplo:

```json
{
  "detail": [
    {
      "type": "string_too_short",
      "loc": [
        "body",
        "text"
      ],
      "msg": "String should have at least 20 characters",
      "input": "short",
      "ctx": {
        "min_length": 20
      }
    }
  ]
}
```

---

## 17. Logs

A aplicação utiliza logging estruturado em saída padrão.

Exemplo:

```text
2026-08-25T17:20:00-0300 | INFO | __main__ | Starting training workflow | model_version=tfidf-logreg-v1
```

Os logs operacionais podem conter:

- versão do modelo;
- classe prevista;
- latência;
- quantidade de amostras;
- estado de inicialização.

**O texto médico recebido pela API não deve ser registrado nos logs.**

Essa decisão evita expor conteúdo potencialmente sensível.

---

## 18. Qualidade de código

### Formatação

```bash
uv run ruff format --check .
```

### Lint

```bash
uv run ruff check .
```

### Tipagem

```bash
uv run mypy src
```

### Testes

```bash
uv run pytest -v
```

Estado validado da Etapa 1:

```text
Ruff:   PASS
mypy:   PASS
Pytest: 7 passed
```

Os testes atuais cobrem:

- classificação válida;
- texto vazio;
- texto curto;
- preservação de dataset existente;
- download simulado de arquivos ausentes;
- download apenas dos arquivos necessários;
- criação automática do diretório de destino.

Os testes do `DatasetLoader` não dependem de internet: o download é simulado com `monkeypatch`.

---

## 19. Baseline de latência local

Com a API executando localmente:

```bash
uv run python scripts/measure_latency.py \
    --runs 200 \
    --warmup 10
```

Configuração utilizada:

```text
Warm-up requests: 10
Measured requests: 200
```

Resultado:

| Métrica | Local |
|---|---:|
| Mean | 4.913 ms |
| Median | 4.640 ms |
| Minimum | 3.070 ms |
| Maximum | 19.437 ms |
| P95 | 6.852 ms |
| P99 | 9.664 ms |

Esse resultado é o baseline local da Etapa 1.

---

## 20. Docker

O projeto utiliza um Dockerfile **multi-stage**.

A primeira etapa instala as dependências de produção com `uv`.

A imagem final contém somente o ambiente necessário para executar:

- Python;
- dependências de runtime;
- código da aplicação;
- modelo treinado.

Ferramentas de desenvolvimento e dados brutos não precisam estar presentes na imagem final.

### Build

O modelo deve existir antes do build:

```bash
uv run python -m medical_triage.training.train
```

Depois:

```bash
docker build \
    -t medical-triage-api:0.1.0 \
    .
```

### Execução

```bash
docker run \
    --rm \
    --name medical-triage-api \
    -p 8000:8000 \
    medical-triage-api:0.1.0
```

### Health check do container

Em outro terminal:

```bash
curl http://localhost:8000/health
```

Resposta esperada:

```json
{
  "status": "healthy"
}
```

### Teste de inferência do container

```bash
curl \
    -X POST \
    http://localhost:8000/predict \
    -H "Content-Type: application/json" \
    -d '{
        "text": "The patient presented with acute myocardial infarction and severe coronary artery disease."
    }'
```

---

## 21. Baseline de latência em Docker

Com o container executando:

```bash
uv run python scripts/measure_latency.py \
    --runs 200 \
    --warmup 10
```

Resultado:

| Métrica | Local | Docker |
|---|---:|---:|
| Mean | 4.913 ms | 7.909 ms |
| Median | 4.640 ms | 7.509 ms |
| Minimum | 3.070 ms | 4.637 ms |
| Maximum | 19.437 ms | 30.680 ms |
| P95 | 6.852 ms | 11.037 ms |
| P99 | 9.664 ms | 16.849 ms |

As duas medições foram realizadas utilizando:

- a mesma máquina;
- o mesmo modelo;
- o mesmo endpoint;
- o mesmo payload;
- 10 requisições de warm-up;
- 200 requisições medidas.

Esses números formam a referência inicial para as otimizações de inferência que serão realizadas em etapas posteriores.

---

## 22. Reprodução completa da Etapa 1

Para uma terceira pessoa reproduzir o projeto a partir de um clone novo:

### 1. Instalar dependências

```bash
uv sync --locked
```

### 2. Verificar qualidade

```bash
uv run ruff format --check .
uv run ruff check .
uv run mypy src
uv run pytest -v
```

### 3. Baixar dados e treinar

```bash
uv run python -m medical_triage.training.train
```

### 4. Executar localmente

```bash
uv run uvicorn \
    medical_triage.presentation.api.main:app \
    --host 0.0.0.0 \
    --port 8000
```

### 5. Verificar

```bash
curl http://localhost:8000/health
```

### 6. Fazer uma inferência

```bash
curl \
    -X POST \
    http://localhost:8000/predict \
    -H "Content-Type: application/json" \
    -d '{
        "text": "The patient presented with acute myocardial infarction and severe coronary artery disease."
    }'
```

### 7. Medir latência local

```bash
uv run python scripts/measure_latency.py \
    --runs 200 \
    --warmup 10
```

### 8. Encerrar a API local

Use:

```text
Ctrl+C
```

### 9. Criar a imagem

```bash
docker build \
    -t medical-triage-api:0.1.0 \
    .
```

### 10. Executar o container

```bash
docker run \
    --rm \
    --name medical-triage-api \
    -p 8000:8000 \
    medical-triage-api:0.1.0
```

### 11. Testar o container

```bash
curl http://localhost:8000/health
```

### 12. Medir latência do container

```bash
uv run python scripts/measure_latency.py \
    --runs 200 \
    --warmup 10
```

Se todos esses passos forem concluídos com sucesso, a implementação da Etapa 1 está reproduzida.

---

## 23. Critérios de conclusão da Etapa 1

Estado atual validado:

```text
[OK] DatasetLoader implementado
[OK] Download automático do dataset
[OK] Dataset de treino carregado
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
[OK] Modelo carregado uma única vez
[OK] Logging sem conteúdo médico

[OK] Ruff
[OK] mypy
[OK] 7 testes unitários

[OK] Docker multi-stage
[OK] API executada em container
[OK] Inferência executada em container

[OK] Baseline local medido
[OK] Baseline Docker medido

[OK] Inferência real-time definida
[OK] Arquitetura AWS ECR + EC2 definida
```

---

## 24. Próximas etapas

O desenvolvimento do projeto é incremental.

Itens como os abaixo **não fazem parte da implementação da Etapa 1** e serão documentados somente após serem adicionados ao projeto:

```text
GitHub Actions
CI/CD
Apache Airflow
Prometheus
Grafana
Docker Compose de observabilidade
retraining automático
ONNX
quantização
outras otimizações de inferência
```

Isso evita que o README descreva funcionalidades que ainda não existem.

---

## 25. Aviso

Este projeto possui finalidade acadêmica e demonstra conceitos de engenharia de software, machine learning e MLOps.

O modelo atual não é um dispositivo médico e não deve ser utilizado para diagnóstico, priorização clínica ou tomada de decisão sobre pacientes.
