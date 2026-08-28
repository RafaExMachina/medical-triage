# Etapa 4 — Otimização de Inferência com ONNX Runtime

## Objetivo

A Etapa 4 otimiza a inferência do classificador de textos médicos sem alterar
seu objetivo original.

O sistema continua classificando abstracts médicos nas cinco categorias do
Medical Abstracts TC Corpus:

1. neoplasms
2. digestive system diseases
3. nervous system diseases
4. cardiovascular diseases
5. general pathological conditions

A estratégia escolhida foi converter o pipeline
`TF-IDF + LogisticRegression` do scikit-learn para ONNX e executar a
inferência com ONNX Runtime.

---

## Arquitetura

### Antes da otimização

```text
HTTP
 |
FastAPI
 |
ClassifyMedicalTextUseCase
 |
SklearnClassifierAdapter
 |
TF-IDF + LogisticRegression
 |
classifier.joblib
```

### Depois da otimização

```text
HTTP
 |
FastAPI
 |
ClassifyMedicalTextUseCase
 |
ClassifierPort
 |
 +-----------------------------+
 |                             |
 v                             v
OnnxClassifierAdapter     SklearnClassifierAdapter
 |                             |
 v                             v
ONNX Runtime               scikit-learn
 |                             |
 v                             v
classifier.onnx            classifier.joblib
```

O backend é selecionado por configuração:

```bash
MODEL_BACKEND=onnx
```

ou:

```bash
MODEL_BACKEND=sklearn
```

O backend padrão da aplicação é ONNX.

---

## Artefatos

Os principais artefatos comparados são:

| Artefato | Tamanho aproximado |
|---|---:|
| `classifier.joblib` | 3.877 MiB |
| `classifier.onnx` | 2.656 MiB |

A conversão reduziu o tamanho do artefato em aproximadamente **31,49%**.

A imagem Docker de produção utiliza apenas:

```text
models/classifier.onnx
models/classifier_onnx_metadata.json
```

O arquivo `classifier.joblib` permanece disponível localmente para comparação
e execução do backend sklearn.

---

## Export do pipeline para ONNX

O pipeline completo é exportado pelo script:

```text
scripts/export_onnx.py
```

Execução:

```bash
uv run python scripts/export_onnx.py
```

A exportação gera:

```text
models/classifier.onnx
models/classifier_onnx_metadata.json
```

O modelo ONNX é validado com o checker do próprio ONNX antes de ser utilizado
pela aplicação.

---

## Validação de equivalência

A equivalência entre o pipeline sklearn original e o pipeline Full ONNX foi
avaliada com os **2.888 exemplos** do conjunto oficial de teste.

Resultado:

```text
Predições iguais:       2859
Predições diferentes:     29
Agreement:             98.995845%
```

Métricas:

| Métrica | sklearn | Full ONNX |
|---|---:|---:|
| Accuracy | 0.583102 | 0.585873 |
| Macro F1 | 0.585593 | 0.587332 |

A pequena diferença observada não é interpretada como melhoria de qualidade do
modelo. Ela decorre de diferenças introduzidas durante a conversão do
pré-processamento textual para ONNX.

Relatório:

```text
reports/onnx_equivalence.json
```

---

## Diagnóstico das divergências

Para identificar a origem das 29 divergências, foi criado um experimento
híbrido:

```text
TF-IDF sklearn
      |
      v
LogisticRegression ONNX
```

Nesse cenário, o TF-IDF permanece no scikit-learn e somente o
`LogisticRegression` é executado via ONNX Runtime.

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

Isso indica que o `LogisticRegression` convertido para ONNX preserva o
comportamento do modelo original.

As pequenas divergências do pipeline Full ONNX são provenientes da conversão
do estágio TF-IDF.

Relatório:

```text
reports/onnx_hybrid_equivalence.json
```

---

## Benchmark isolado de inferência

Foi executado um benchmark controlado comparando:

- pipeline sklearn;
- pipeline híbrido;
- pipeline Full ONNX.

Configuração principal:

```text
3 execuções
1000 inferências medidas por execução
50 warm-ups
batch size = 1
CPUExecutionProvider
intra_op_num_threads = 1
inter_op_num_threads = 1
ORT_SEQUENTIAL
```

Resultados consolidados do sklearn e do Full ONNX:

| Métrica | sklearn | Full ONNX |
|---|---:|---:|
| Mean | 0.9965 ms | 0.4640 ms |
| P50 | 0.9555 ms | 0.4487 ms |
| P95 | 1.4067 ms | 0.6135 ms |
| P99 | 1.7523 ms | 0.7388 ms |

Resultado aproximado:

```text
Speedup médio:         ~2.15x
Redução média:         ~53.44%
```

Relatórios:

```text
reports/inference_benchmark_controlled_1.json
reports/inference_benchmark_controlled_2.json
reports/inference_benchmark_controlled_3.json
reports/inference_benchmark_summary.json
```

---

## Benchmark HTTP end-to-end

Também foi realizado um benchmark através do endpoint real:

```text
POST /predict
```

Metodologia:

```text
Backends:                  sklearn e ONNX
Runs por backend:          3
Requests medidos por run:  1000
Warm-ups por run:          50
Modo:                      single-text
Conexão HTTP:              persistente
Dataset:                   medical_tc_test.csv
```

### Latência observada pelo cliente

| Métrica | sklearn | ONNX | Speedup | Redução |
|---|---:|---:|---:|---:|
| Mean | 10.9768 ms | 7.1991 ms | 1.52x | 34.42% |
| P50 | 9.4146 ms | 5.9003 ms | 1.60x | 37.33% |
| P95 | 21.2380 ms | 14.5180 ms | 1.46x | 31.64% |
| P99 | 31.9599 ms | 22.6826 ms | 1.41x | 29.03% |

### Inferência medida dentro da API

| Métrica | sklearn | ONNX | Speedup | Redução |
|---|---:|---:|---:|---:|
| Mean | 5.2783 ms | 1.0792 ms | 4.89x | 79.55% |
| P50 | 4.6485 ms | 0.9451 ms | 4.92x | 79.67% |
| P95 | 9.4335 ms | 1.8705 ms | 5.04x | 80.17% |
| P99 | 16.0695 ms | 3.7322 ms | 4.31x | 76.77% |

O ganho observado dentro da API é maior do que no benchmark isolado porque o
adapter sklearn obtém a classe e a confiança através de chamadas separadas a
`predict()` e `predict_proba()`.

O backend ONNX obtém label e probabilidades em uma única execução da sessão.

Já a latência HTTP end-to-end inclui outras etapas não otimizadas pelo ONNX,
como:

- FastAPI;
- validação Pydantic;
- serialização;
- métricas;
- logging;
- comunicação HTTP;
- overhead do sistema operacional.

Por isso, o ganho end-to-end é menor do que o ganho medido somente na
inferência.

Relatórios:

```text
reports/http_benchmark_sklearn_1.json
reports/http_benchmark_sklearn_2.json
reports/http_benchmark_sklearn_3.json
reports/http_benchmark_onnx_1.json
reports/http_benchmark_onnx_2.json
reports/http_benchmark_onnx_3.json
reports/http_benchmark_summary.json
```

---

## Integração com a aplicação

A aplicação utiliza a interface:

```text
ClassifierPort
```

e dois adapters:

```text
SklearnClassifierAdapter
OnnxClassifierAdapter
```

A resolução da implementação é feita em:

```text
src/medical_triage/presentation/api/dependencies.py
```

Isso permite alterar o backend sem modificar o caso de uso.

Exemplo:

```bash
MODEL_BACKEND=onnx uv run uvicorn   medical_triage.presentation.api.main:app   --host 0.0.0.0   --port 8000
```

---

## Docker

A imagem de produção foi atualizada para utilizar somente os artefatos ONNX.

Arquivos incluídos:

```text
models/classifier.onnx
models/classifier_onnx_metadata.json
```

O pipeline ONNX contém o operador `StringNormalizer`.

Durante a validação da imagem Docker foi identificado que a imagem
`python:3.12-slim` não disponibiliza `en_US.UTF-8` por padrão.

Sem esse locale, o ONNX Runtime falhava ao criar a sessão de inferência.

O Dockerfile passou a configurar explicitamente:

```text
en_US.UTF-8
```

Após a correção, o container foi validado com:

```text
backend: onnx
adapter: OnnxClassifierAdapter
```

e a API permaneceu `healthy`.

---

## Scripts da Etapa 4

Os principais scripts adicionados são:

```text
scripts/export_onnx.py
scripts/export_onnx_classifier.py
scripts/validate_onnx_equivalence.py
scripts/validate_onnx_hybrid.py
scripts/benchmark_inference_backends.py
scripts/benchmark_http_api.py
scripts/summarize_http_benchmarks.py
```

---

## Testes e qualidade

Ao final da Etapa 4:

```text
Ruff:       PASS
mypy:       PASS
pre-commit: PASS
Pytest:     21 passed
```

Os testes incluem validação de:

- configuração de backend;
- dependency injection;
- `OnnxClassifierAdapter`;
- API;
- treinamento;
- comportamento do caso de uso.

---

## Resultado final

A estratégia Full ONNX foi adotada como backend padrão de produção.

Resumo:

```text
Redução do tamanho do artefato:        ~31.49%

Speedup isolado do modelo:              ~2.15x
Redução da latência isolada:           ~53.44%

Speedup de inferência na API:           ~4.89x
Redução da inferência na API:          ~79.55%

Speedup HTTP end-to-end:                ~1.52x
Redução HTTP end-to-end:               ~34.42%
```

A otimização preservou a qualidade do classificador para o objetivo do projeto,
reduziu o tamanho do artefato e diminuiu de forma significativa a latência de
inferência.

---

## Referências internas

Documentação principal:

```text
README.md
```

Relatórios:

```text
reports/onnx_equivalence.json
reports/onnx_hybrid_equivalence.json
reports/inference_benchmark_summary.json
reports/http_benchmark_summary.json
```

Artefatos de produção:

```text
models/classifier.onnx
models/classifier_onnx_metadata.json
```
