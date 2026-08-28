# Etapa 1 — Decisão Arquitetural de Deploy em Nuvem

## 1. Objetivo

A Etapa 1 define a estratégia de execução e deploy do Medical Triage, estabelece
a arquitetura inicial da aplicação e entrega uma API funcional empacotada em
Docker.

O foco desta etapa é responder às seguintes decisões:

- o serviço deve operar em modo batch ou real-time?
- qual arquitetura de cloud é adequada ao cenário?
- como expor o classificador por uma API simples?
- como empacotar o serviço de forma reproduzível?
- qual é o baseline inicial de latência?

O projeto foi estruturado para que as etapas seguintes pudessem evoluir sobre a
mesma base de aplicação.

---

## 2. Contexto do projeto

O Medical Triage utiliza o **Medical Abstracts TC Corpus** para classificar
abstracts médicos em cinco categorias:

1. neoplasms;
2. digestive system diseases;
3. nervous system diseases;
4. cardiovascular diseases;
5. general pathological conditions.

O objetivo do projeto é demonstrar uma arquitetura de Machine Learning e MLOps
para classificação de textos médicos.

> O dataset não possui rótulos de urgência clínica. Portanto, o sistema não deve
> ser interpretado como uma solução clínica real de triagem, diagnóstico ou
> priorização de pacientes.

---

## 3. Decisão: batch ou real-time

A estratégia escolhida para o serving é **real-time**.

Cada requisição contém um único texto médico e espera uma classificação imediata.

Fluxo:

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
Classificador
   |
   v
Resposta imediata
```

Essa arquitetura é adequada para uma aplicação interativa porque o tempo de
resposta faz parte da experiência do usuário.

### 3.1 Por que não batch?

Uma arquitetura batch seria mais adequada para cenários como:

- processamento periódico de milhares de documentos;
- classificação offline de grandes coleções;
- geração de relatórios em horários definidos;
- pipelines em que não existe necessidade de resposta imediata.

Esse não é o fluxo principal do Medical Triage.

Por isso, a aplicação foi projetada para atender requisições individuais por
HTTP.

---

## 4. Arquitetura de cloud selecionada

A arquitetura de cloud selecionada para o cenário é baseada em **AWS**.

```text
Developer
   |
   v
GitHub / CI
   |
   v
Docker Image
   |
   v
Amazon ECR
   |
   v
Amazon EC2
   |
   v
FastAPI + Model
   |
   v
POST /predict
```

Os principais componentes planejados são:

| Componente | Responsabilidade |
|---|---|
| GitHub | versionamento do código |
| GitHub Actions | automação de qualidade e CI |
| Docker | empacotamento da aplicação |
| Amazon ECR | armazenamento da imagem Docker |
| Amazon EC2 | execução contínua da API |
| FastAPI | serving HTTP |
| Modelo de ML | classificação dos textos médicos |

---

## 5. Justificativa da escolha da AWS

A AWS foi escolhida por permitir uma arquitetura simples e compatível com o
escopo do projeto.

### Amazon ECR

O Amazon Elastic Container Registry pode armazenar a imagem Docker versionada da
API.

Vantagens:

- integração direta com Docker;
- suporte a imagens versionadas;
- controle de acesso;
- integração natural com outros serviços AWS;
- separação entre código-fonte e artefato executável.

### Amazon EC2

Uma instância EC2 é suficiente para executar continuamente a API em um cenário
acadêmico ou de pequena escala.

Vantagens:

- controle direto do runtime;
- compatibilidade com Docker;
- implantação simples;
- acesso a logs e processos do host;
- possibilidade de evolução futura para arquiteturas mais robustas.

A opção evita introduzir complexidade adicional com Kubernetes antes que ela
seja necessária.

---

## 6. Alternativas consideradas

Outras alternativas possíveis seriam Azure e Google Cloud Platform.

### Azure

Uma arquitetura equivalente poderia utilizar:

```text
Azure Container Registry
        |
        v
Azure Virtual Machine
```

### Google Cloud

Uma alternativa equivalente poderia utilizar:

```text
Artifact Registry
        |
        v
Compute Engine
```

As três opções são tecnicamente viáveis.

A AWS foi adotada como referência arquitetural por simplicidade e por se encaixar
diretamente no modelo:

```text
Docker Registry
      |
      v
Virtual Machine
      |
      v
FastAPI
```

---

## 7. Arquitetura inicial da aplicação

A aplicação utiliza separação em camadas inspirada em Clean Architecture.

```text
Client
  |
  v
FastAPI
Presentation Layer
  |
  v
ClassifyMedicalTextUseCase
Application Layer
  |
  v
ClassifierPort
Domain
  |
  v
Classifier Adapter
Infrastructure
  |
  v
Modelo de Machine Learning
```

Essa organização permite evoluir a implementação do classificador sem alterar o
caso de uso principal.

Na Etapa 1, o backend de referência é o pipeline scikit-learn:

```text
TF-IDF
  +
Logistic Regression
```

---

## 8. API inicial

A API é implementada com FastAPI.

Endpoints principais:

```text
GET  /health
POST /predict
```

### 8.1 Health check

Exemplo:

```bash
curl http://localhost:8000/health
```

Resposta esperada:

```json
{
  "status": "healthy"
}
```

### 8.2 Classificação

Exemplo:

```bash
curl   -X POST   http://localhost:8000/predict   -H "Content-Type: application/json"   -d '{
    "text": "The patient presented with acute myocardial infarction and severe coronary artery disease."
  }'
```

A resposta contém a classe prevista, a confiança, a versão lógica do modelo e o
tempo de inferência.

---

## 9. Ambiente Python com uv

O projeto foi desenvolvido com **uv**.

Sincronização do ambiente:

```bash
uv sync --locked
```

Execução de comandos:

```bash
uv run <comando>
```

Exemplo:

```bash
uv run pytest
```

O `uv.lock` é versionado para tornar a instalação reproduzível.

Não é necessário ativar manualmente a `.venv`.

---

## 10. Treinamento do baseline

O baseline utiliza:

```text
TfidfVectorizer
      +
LogisticRegression
```

Treinamento:

```bash
uv run python -m medical_triage.training.train
```

Fluxo:

```text
Dataset
   |
   v
Validação
   |
   v
Train / Validation Split
   |
   v
TF-IDF
   |
   v
Logistic Regression
   |
   v
Avaliação
   |
   v
Persistência
```

Artefatos gerados localmente:

```text
models/classifier.joblib
models/metrics.json
```

Resultados de referência do baseline de validação:

| Métrica | Resultado |
|---|---:|
| Accuracy | 0.5931 |
| Macro F1 | 0.5908 |

Esses valores representam o baseline inicial do projeto.

---

## 11. Dockerização da API

A API é empacotada em uma imagem Docker multi-stage.

Build:

```bash
docker build   -t medical-triage-api:0.4.0   .
```

Execução:

```bash
docker run   --rm   --name medical-triage-api   -p 8000:8000   medical-triage-api:0.4.0
```

Validação:

```bash
curl http://localhost:8000/health
```

A utilização de Docker permite executar a aplicação de forma consistente em
diferentes máquinas.

---

## 12. Baseline inicial de latência

A Etapa 1 mede a latência inicial antes das otimizações posteriores.

Script:

```text
scripts/measure_latency.py
```

Execução:

```bash
uv run python scripts/measure_latency.py   --runs 200   --warmup 10
```

Resultados iniciais registrados:

| Métrica | Local | Docker |
|---|---:|---:|
| Mean | 4.913 ms | 7.909 ms |
| Median | 4.640 ms | 7.509 ms |
| Minimum | 3.070 ms | 4.637 ms |
| Maximum | 19.437 ms | 30.680 ms |
| P95 | 6.852 ms | 11.037 ms |
| P99 | 9.664 ms | 16.849 ms |

Esses valores servem como baseline histórico da API antes das otimizações
realizadas na Etapa 4.

> Os resultados de benchmark dependem do hardware, sistema operacional, carga e
> ambiente de execução. Eles devem ser usados como referência comparativa, não
> como garantia absoluta de desempenho.

---

## 13. Entregáveis da Etapa 1

A Etapa 1 atende aos seguintes requisitos:

```text
[OK] decisão batch vs real-time
[OK] estratégia real-time documentada
[OK] arquitetura de cloud documentada
[OK] AWS selecionada como referência
[OK] Amazon ECR definido como registry
[OK] Amazon EC2 definido como runtime
[OK] API FastAPI funcional
[OK] GET /health
[OK] POST /predict
[OK] classificador baseline
[OK] Docker multi-stage
[OK] API executável em container
[OK] baseline de latência medido
[OK] instruções reproduzíveis com uv
```

---

## 14. O que não faz parte da Etapa 1

Os seguintes itens são tratados em etapas posteriores:

```text
Etapa 2
  GitHub Actions
  Airflow
  quality gates

Etapa 3
  Prometheus
  Grafana
  observabilidade

Etapa 4
  ONNX Runtime
  otimização de latência
  comparação sklearn x ONNX
```

O deploy automatizado efetivo em AWS também permanece como evolução futura.

A Etapa 1 define e documenta a arquitetura de cloud, mas não exige que toda a
infraestrutura AWS esteja provisionada no estado atual do repositório.

---

## 15. Relação com o estado atual do projeto

A arquitetura definida nesta etapa foi preservada nas evoluções seguintes.

O fluxo lógico continua sendo:

```text
Cliente
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
   v
Backend de inferência
```

Na versão `0.4.0`, o backend padrão passou a ser ONNX Runtime, mas a decisão
arquitetural de serving real-time permanece a mesma.

A evolução do backend é documentada em:

```text
docs/stage4-onnx-optimization.md
```

---

## 16. Arquivos relacionados

Aplicação:

```text
src/medical_triage/
```

Docker:

```text
Dockerfile
docker-compose.yml
```

Treinamento:

```text
src/medical_triage/training/
```

Scripts:

```text
scripts/measure_latency.py
```

Documentação complementar:

```text
README.md
docs/observability-plan.md
docs/stage4-onnx-optimization.md
```

---

## 17. Resultado

A Etapa 1 estabelece a base arquitetural do Medical Triage:

- serving real-time;
- API FastAPI;
- baseline de classificação;
- execução reproduzível com uv;
- empacotamento Docker;
- arquitetura de cloud AWS ECR + EC2;
- baseline inicial de latência.

Essa base permite que as etapas seguintes adicionem automação, observabilidade e
otimização sem substituir a arquitetura principal da aplicação.
