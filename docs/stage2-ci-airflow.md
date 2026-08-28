# Etapa 2 — CI/CD e Pipeline Automatizado com Apache Airflow

## 1. Objetivo

A Etapa 2 adiciona automação de qualidade, testes e treinamento ao projeto
Medical Triage.

Os objetivos principais são:

- executar verificações automáticas de qualidade a cada alteração;
- validar o código com GitHub Actions;
- garantir que o ambiente Python seja reproduzível com `uv`;
- organizar o treinamento em uma DAG do Apache Airflow;
- validar dados antes do treinamento;
- validar artefatos após o treinamento;
- manter o pipeline simples, auditável e reproduzível.

A etapa foi construída sobre a base criada na Etapa 1.

---

## 2. Visão geral

O fluxo principal da Etapa 2 é dividido em dois blocos:

```text
Código
  |
  v
pre-commit
  |
  v
GitHub Actions
  |
  +-------------------+
  |                   |
  v                   v
Quality             Tests
```

e:

```text
Apache Airflow
     |
     v
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

---

## 3. Tecnologias utilizadas

A automação desta etapa utiliza:

- Git;
- GitHub;
- GitHub Actions;
- `uv`;
- Pytest;
- Ruff;
- mypy;
- pre-commit;
- Apache Airflow 3.3.1;
- Docker;
- Docker Compose.

---

## 4. Ambiente Python com uv

O projeto foi desenvolvido com **uv**.

Sincronização:

```bash
uv sync --locked
```

O arquivo:

```text
uv.lock
```

é versionado para tornar o ambiente reproduzível.

Não é necessário ativar manualmente a `.venv`.

Os comandos são executados com:

```bash
uv run <comando>
```

Exemplos:

```bash
uv run pytest
uv run ruff check .
uv run mypy src
```

---

## 5. Qualidade de código

Os principais quality gates são:

```bash
uv run ruff format --check .
uv run ruff check .
uv run mypy src
uv run pytest
```

Também é possível executar as verificações locais de forma consolidada com:

```bash
uv run pre-commit run --all-files
```

Ao final da versão `0.4.0`, o estado validado é:

```text
Ruff:       PASS
mypy:       PASS
pre-commit: PASS
Pytest:     21 passed
```

---

## 6. pre-commit

O projeto utiliza pre-commit para executar verificações antes do commit.

Arquivo:

```text
.pre-commit-config.yaml
```

Instalação:

```bash
uv run pre-commit install
```

Execução manual:

```bash
uv run pre-commit run --all-files
```

O objetivo é impedir que código com problemas de formatação, lint ou tipagem
seja enviado ao repositório.

---

## 7. GitHub Actions

O workflow principal está em:

```text
.github/workflows/ci.yml
```

Os principais gatilhos são:

```text
push
pull_request -> master
```

A estrutura lógica é:

```text
push / pull request
        |
        v
   GitHub Actions
        |
        +------------------+
        |                  |
        v                  v
     Quality             Tests
        |                  |
        v                  v
ruff / mypy            pytest
```

---

## 8. Job de qualidade

O job de qualidade executa verificações como:

```text
ruff format --check
ruff check
mypy src
```

O objetivo é validar:

- formatação;
- estilo;
- problemas de lint;
- tipagem estática;
- consistência do código antes do merge.

As dependências são instaladas com:

```bash
uv sync --locked
```

---

## 9. Job de testes

O job de testes executa:

```bash
uv run pytest
```

A suíte cobre, entre outros:

- casos de uso;
- configuração da aplicação;
- dataset loader;
- dependency injection;
- backend sklearn;
- backend ONNX;
- treinamento;
- API;
- validação HTTP;
- métricas.

Na versão `0.4.0`:

```text
21 passed
```

---

## 10. Fluxo de desenvolvimento

O fluxo recomendado é:

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
pre-commit
    |
    v
git push
    |
    v
Pull Request -> master
    |
    v
GitHub Actions
    |
    +---------+
    |         |
    v         v
 Quality    Tests
```

Exemplo:

```bash
git switch -c feat/minha-feature
```

Antes do push:

```bash
uv run pre-commit run --all-files
uv run pytest
```

---

## 11. Apache Airflow

O Apache Airflow é utilizado para orquestrar o fluxo de treinamento do modelo.

A execução do Airflow é isolada do runtime da API.

Isso evita adicionar a árvore de dependências do Airflow ao container de
inferência.

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

---

## 12. DAG de treinamento

A DAG principal é:

```text
medical_triage_training
```

Arquivo:

```text
airflow/dags/medical_triage_training.py
```

A sequência de tasks é:

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

A DAG possui:

```text
schedule=None
```

Portanto, é disparada manualmente nesta etapa.

---

## 13. prepare_dataset_task

Responsabilidade:

- verificar se os arquivos necessários do dataset estão disponíveis;
- reutilizar arquivos existentes quando válidos;
- realizar download quando necessário;
- preparar a entrada para as próximas tasks.

Fluxo:

```text
data/raw/
   |
   v
DatasetLoader
   |
   v
arquivos disponíveis
```

O componente principal reutilizado é:

```text
src/medical_triage/data/dataset_loader.py
```

---

## 14. validate_dataset_task

Responsabilidade:

- validar se o dataset não está vazio;
- validar a existência das classes esperadas;
- validar a consistência dos dados;
- interromper o pipeline antes do treinamento em caso de entrada inválida.

Exemplos de validações:

```text
dataset não vazio
classes disponíveis
labels consistentes
arquivos necessários disponíveis
```

A validação antes do treinamento reduz o risco de produzir um modelo inválido a
partir de dados incompletos.

---

## 15. train_model_task

A task de treinamento reutiliza a lógica de treinamento existente no projeto.

A função reutilizável é:

```python
run_training()
```

O fluxo é:

```text
dataset
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

Os artefatos principais são:

```text
models/classifier.joblib
models/metrics.json
```

---

## 16. validate_artifacts_task

A última task valida os artefatos produzidos pelo treinamento.

As validações incluem:

- existência de `classifier.joblib`;
- existência de `metrics.json`;
- arquivos não vazios;
- presença das métricas obrigatórias;
- métricas dentro do intervalo esperado;
- consistência entre métricas retornadas e persistidas.

Fluxo:

```text
train_model_task
      |
      v
models/
      |
      +------------------+
      |                  |
      v                  v
classifier.joblib    metrics.json
      |                  |
      +--------+---------+
               |
               v
validate_artifacts_task
```

---

## 17. Uso de XCom

Somente dados pequenos são retornados via XCom.

Exemplos apropriados:

```text
accuracy
macro_f1
status
paths pequenos
```

Não são enviados por XCom:

- DataFrames completos;
- dataset inteiro;
- modelo serializado;
- arquivos grandes.

Essa decisão evita uso inadequado do banco de metadados do Airflow.

---

## 18. Subindo o Airflow

Validar a configuração:

```bash
AIRFLOW_UID=$(id -u) docker compose   -f airflow/docker-compose.yml   config
```

Build:

```bash
AIRFLOW_UID=$(id -u) docker compose   -f airflow/docker-compose.yml   build
```

Subir:

```bash
AIRFLOW_UID=$(id -u) docker compose   -f airflow/docker-compose.yml   up -d
```

Interface web:

```text
http://localhost:8081
```

Mapeamento:

```text
host 8081 -> container 8080
```

---

## 19. Validando a DAG

Listar DAGs:

```bash
AIRFLOW_UID=$(id -u) docker compose   -f airflow/docker-compose.yml   exec airflow   airflow dags list --local
```

A DAG esperada é:

```text
medical_triage_training
```

Verificar erros de importação:

```bash
AIRFLOW_UID=$(id -u) docker compose   -f airflow/docker-compose.yml   exec airflow   airflow dags list-import-errors --local
```

O resultado esperado é ausência de erros de importação.

---

## 20. Executando a DAG

Despausar:

```bash
AIRFLOW_UID=$(id -u) docker compose   -f airflow/docker-compose.yml   exec airflow   airflow dags unpause medical_triage_training
```

Disparar:

```bash
AIRFLOW_UID=$(id -u) docker compose   -f airflow/docker-compose.yml   exec airflow   airflow dags trigger medical_triage_training
```

Acompanhar execuções:

```bash
AIRFLOW_UID=$(id -u) docker compose   -f airflow/docker-compose.yml   exec airflow   airflow dags list-runs medical_triage_training
```

---

## 21. Resultado esperado da DAG

Uma execução bem-sucedida apresenta:

```text
prepare_dataset_task      SUCCESS
validate_dataset_task     SUCCESS
train_model_task          SUCCESS
validate_artifacts_task   SUCCESS
DagRun                    SUCCESS
```

Métricas reproduzidas pelo baseline:

```text
Accuracy: 0.5931
Macro F1: 0.5908
```

---

## 22. Encerrando o Airflow

```bash
AIRFLOW_UID=$(id -u) docker compose   -f airflow/docker-compose.yml   down
```

O ambiente Airflow permanece separado da stack de observabilidade da API.

---

## 23. Relação entre CI e Airflow

GitHub Actions e Airflow possuem responsabilidades diferentes.

```text
GitHub Actions
    |
    +--> qualidade do código
    +--> testes
    +--> validação antes do merge
```

```text
Airflow
    |
    +--> preparação de dados
    +--> validação do dataset
    +--> treinamento
    +--> validação dos artefatos
```

O GitHub Actions valida o repositório.

O Airflow orquestra o processo de Machine Learning.

---

## 24. Separação de responsabilidades

A arquitetura evita acoplar o treinamento diretamente à API.

```text
API
 |
 +--> serving
```

```text
Airflow
 |
 +--> treinamento
```

Isso permite:

- manter o runtime da API menor;
- evoluir o pipeline de treinamento separadamente;
- reutilizar a mesma lógica de domínio;
- evitar dependências desnecessárias no container de serving.

---

## 25. Reprodutibilidade

O projeto utiliza:

```text
uv.lock
Dockerfile
docker-compose.yml
airflow/docker-compose.yml
GitHub Actions
```

para tornar o ambiente reproduzível.

O fluxo mínimo para terceiros é:

```bash
git clone https://github.com/RafaExMachina/medical-triage.git
cd medical-triage
uv sync --locked
uv run pytest
uv run pre-commit run --all-files
```

Para Airflow:

```bash
AIRFLOW_UID=$(id -u) docker compose   -f airflow/docker-compose.yml   up -d --build
```

---

## 26. Entregáveis da Etapa 2

A Etapa 2 atende aos seguintes requisitos:

```text
[OK] GitHub Actions
[OK] CI em push
[OK] CI em pull_request
[OK] job de qualidade
[OK] Ruff
[OK] mypy
[OK] pytest
[OK] pre-commit
[OK] ambiente reproduzível com uv
[OK] Apache Airflow 3.3.1
[OK] Airflow isolado em Docker
[OK] DAG medical_triage_training
[OK] prepare_dataset_task
[OK] validate_dataset_task
[OK] train_model_task
[OK] validate_artifacts_task
[OK] DAG sem import errors
[OK] DagRun executada com sucesso
[OK] artefatos de treinamento validados
```

---

## 27. O que não faz parte da Etapa 2

Os seguintes itens pertencem às etapas posteriores:

```text
Etapa 3
  Prometheus
  Grafana
  dashboard
  métricas operacionais

Etapa 4
  ONNX Runtime
  otimização de latência
  comparação sklearn x ONNX
```

Esses componentes foram adicionados posteriormente sem alterar a estrutura
principal de CI e treinamento definida nesta etapa.

---

## 28. Relação com o estado atual do projeto

Na versão `0.4.0`, a Etapa 2 continua sendo responsável pela base de automação.

O fluxo geral do projeto é:

```text
Código
  |
  v
GitHub Actions
  |
  v
master
```

e:

```text
Dados
  |
  v
Airflow
  |
  v
Treinamento
  |
  v
Artefatos
```

A Etapa 4 adicionou ONNX ao serving, mas o baseline sklearn continua sendo o
modelo de referência do treinamento.

---

## 29. Arquivos relacionados

CI:

```text
.github/workflows/ci.yml
```

pre-commit:

```text
.pre-commit-config.yaml
```

Configuração Python:

```text
pyproject.toml
uv.lock
```

Airflow:

```text
airflow/
├── dags/
│   └── medical_triage_training.py
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

Treinamento:

```text
src/medical_triage/training/
```

Dados:

```text
src/medical_triage/data/
```

Documentação relacionada:

```text
README.md
docs/stage1-cloud-architecture.md
docs/observability-plan.md
docs/stage4-onnx-optimization.md
```

---

## 30. Resultado

A Etapa 2 estabelece uma base de automação para o Medical Triage:

- quality gates locais;
- pre-commit;
- CI com GitHub Actions;
- testes automatizados;
- ambiente reproduzível com `uv`;
- treinamento orquestrado com Apache Airflow;
- validação de dados antes do treinamento;
- validação de artefatos depois do treinamento.

Essa estrutura permite que alterações no código sejam verificadas
automaticamente e que o processo de treinamento seja executado de forma
controlada, reproduzível e auditável.
