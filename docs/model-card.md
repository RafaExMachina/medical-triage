# Model Card — Medical Triage

## 1. Identificação

**Projeto:** Medical Triage  
**Versão do projeto:** `0.4.0`  
**Modelo lógico:** `tfidf-logreg-v1`  
**Backend padrão de produção:** ONNX Runtime  
**Backend de referência:** scikit-learn  
**Tipo de tarefa:** classificação multiclasse de textos médicos  
**Finalidade:** acadêmica e demonstrativa de práticas de Machine Learning e MLOps

---

## 2. Resumo

O Medical Triage utiliza um pipeline de Processamento de Linguagem Natural para
classificar abstracts médicos em uma de cinco categorias do
**Medical Abstracts TC Corpus**.

O pipeline original é composto por:

```text
TfidfVectorizer
      +
LogisticRegression
```

Na versão `0.4.0`, o pipeline também foi exportado para ONNX e passou a ser
executado com ONNX Runtime como backend padrão de produção.

A versão sklearn permanece disponível como baseline, referência de comparação e
fallback local.

---

## 3. Uso pretendido

O modelo foi desenvolvido para demonstrar:

- classificação supervisionada de textos médicos;
- serving de modelos com FastAPI;
- empacotamento com Docker;
- automação com GitHub Actions;
- orquestração de treinamento com Apache Airflow;
- observabilidade com Prometheus e Grafana;
- otimização de inferência com ONNX Runtime;
- comparação de latência e tamanho de artefatos.

O modelo pode ser utilizado em:

- experimentos acadêmicos;
- demonstrações de MLOps;
- estudos de serving de modelos;
- benchmarks de inferência;
- testes de arquitetura de APIs de Machine Learning.

---

## 4. Uso não pretendido

O modelo **não deve ser utilizado** para:

- diagnóstico médico;
- triagem clínica real;
- priorização de pacientes;
- recomendação de tratamento;
- tomada de decisão clínica;
- estimativa de risco individual;
- substituição de avaliação médica profissional;
- qualquer cenário de saúde em que uma classificação incorreta possa impactar
  diretamente um paciente.

> Este projeto não é um dispositivo médico.

---

## 5. Escopo clínico

O nome do projeto contém o termo "Triage", mas o dataset utilizado não possui
rótulos de urgência clínica.

As classes disponíveis são categorias de doenças e condições, não níveis de
prioridade médica.

Portanto, o sistema **não implementa** uma classificação do tipo:

```text
normal
attention
urgent
```

Também não é realizada qualquer conversão artificial entre categorias de doença
e níveis de urgência.

---

## 6. Dataset

O projeto utiliza o:

```text
Medical Abstracts TC Corpus
```

Arquivos utilizados:

```text
medical_tc_train.csv
medical_tc_test.csv
medical_tc_labels.csv
```

O carregamento e preparação são realizados por:

```text
src/medical_triage/data/dataset_loader.py
```

O loader reutiliza arquivos existentes quando válidos e pode realizar download
quando necessário.

---

## 7. Classes

O problema possui cinco classes:

| ID | Classe |
|---:|---|
| 1 | neoplasms |
| 2 | digestive system diseases |
| 3 | nervous system diseases |
| 4 | cardiovascular diseases |
| 5 | general pathological conditions |

---

## 8. Pipeline de Machine Learning

O baseline utiliza:

```text
Texto
  |
  v
TfidfVectorizer
  |
  v
LogisticRegression
  |
  v
Classe prevista
```

O TF-IDF utiliza representação esparsa de texto e o classificador final é uma
regressão logística multiclasse.

---

## 9. Divisão dos dados

O conjunto de desenvolvimento utiliza:

```text
90% training
10% validation
random_state = 42
```

O arquivo oficial:

```text
medical_tc_test.csv
```

é mantido separado para avaliação e comparação entre backends.

---

## 10. Métricas do baseline

Resultados registrados no conjunto de validação:

| Métrica | Resultado |
|---|---:|
| Accuracy | `0.5931` |
| Macro F1 | `0.5908` |

Essas métricas representam o baseline do projeto e não devem ser interpretadas
como desempenho clínico.

---

## 11. Avaliação no conjunto oficial de teste

A comparação da Etapa 4 utilizou:

```text
2888 exemplos
```

Resultados do backend sklearn:

| Métrica | sklearn |
|---|---:|
| Accuracy | `0.583102` |
| Macro F1 | `0.585593` |

Resultados do pipeline Full ONNX:

| Métrica | Full ONNX |
|---|---:|
| Accuracy | `0.585873` |
| Macro F1 | `0.587332` |

A pequena diferença entre os resultados **não é interpretada como melhoria de
qualidade**.

Ela decorre de diferenças introduzidas durante a conversão do
pré-processamento textual para ONNX.

---

## 12. Equivalência sklearn × ONNX

A validação de equivalência apresentou:

```text
Predições iguais:       2859
Predições diferentes:     29
Agreement:             98.995845%
```

Relatório:

```text
reports/onnx_equivalence.json
```

---

## 13. Diagnóstico das divergências

Para investigar as 29 divergências, foi avaliado um pipeline híbrido:

```text
TF-IDF sklearn
      |
      v
LogisticRegression ONNX
```

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

Esse experimento indica que o `LogisticRegression` convertido preserva o
comportamento do classificador original.

As pequenas diferenças do pipeline Full ONNX estão associadas à conversão do
estágio TF-IDF.

Relatório:

```text
reports/onnx_hybrid_equivalence.json
```

---

## 14. Backends de inferência

O projeto suporta dois backends:

| Backend | Uso |
|---|---|
| `onnx` | padrão para serving |
| `sklearn` | baseline e comparação |

Seleção:

```bash
MODEL_BACKEND=onnx
```

ou:

```bash
MODEL_BACKEND=sklearn
```

---

## 15. Artefatos

### sklearn

```text
models/classifier.joblib
models/metrics.json
```

### ONNX

```text
models/classifier.onnx
models/classifier_onnx_metadata.json
```

A imagem Docker de produção utiliza apenas os artefatos ONNX necessários ao
serving.

---

## 16. Tamanho dos artefatos

| Artefato | Tamanho aproximado |
|---|---:|
| `classifier.joblib` | `3.877 MiB` |
| `classifier.onnx` | `2.656 MiB` |

Redução aproximada:

```text
31.49%
```

---

## 17. Benchmark isolado de inferência

Resultados consolidados de três execuções controladas:

| Métrica | sklearn | Full ONNX |
|---|---:|---:|
| Mean | `0.9965 ms` | `0.4640 ms` |
| P50 | `0.9555 ms` | `0.4487 ms` |
| P95 | `1.4067 ms` | `0.6135 ms` |
| P99 | `1.7523 ms` | `0.7388 ms` |

Resumo:

```text
Speedup médio:  ~2.15x
Redução média:  ~53.44%
```

Relatório:

```text
reports/inference_benchmark_summary.json
```

---

## 18. Benchmark HTTP end-to-end

Metodologia:

```text
3 runs por backend
1000 requests medidos por run
50 warm-ups por run
single-text
conexão HTTP persistente
```

Resultados:

| Métrica | sklearn | ONNX | Redução |
|---|---:|---:|---:|
| Mean | `10.9768 ms` | `7.1991 ms` | `34.42%` |
| P50 | `9.4146 ms` | `5.9003 ms` | `37.33%` |
| P95 | `21.2380 ms` | `14.5180 ms` | `31.64%` |
| P99 | `31.9599 ms` | `22.6826 ms` | `29.03%` |

Relatório:

```text
reports/http_benchmark_summary.json
```

---

## 19. Inferência medida dentro da API

| Métrica | sklearn | ONNX | Speedup |
|---|---:|---:|---:|
| Mean | `5.2783 ms` | `1.0792 ms` | `4.89x` |
| P50 | `4.6485 ms` | `0.9451 ms` | `4.92x` |
| P95 | `9.4335 ms` | `1.8705 ms` | `5.04x` |
| P99 | `16.0695 ms` | `3.7322 ms` | `4.31x` |

O ganho interno da API é maior que o benchmark isolado porque o adapter sklearn
executa `predict()` e `predict_proba()` separadamente, enquanto o ONNX obtém
label e probabilidades em uma única execução da sessão.

---

## 20. Serving

A API utiliza FastAPI.

Endpoints principais:

```text
GET  /health
POST /predict
GET  /metrics
```

Exemplo de requisição:

```bash
curl   -X POST   http://localhost:8000/predict   -H "Content-Type: application/json"   -d '{
    "text": "The patient presented with acute myocardial infarction and severe coronary artery disease."
  }'
```

Exemplo de resposta:

```json
{
  "label_id": 4,
  "label_name": "cardiovascular diseases",
  "confidence": 0.884641170501709,
  "model_version": "tfidf-logreg-v1",
  "inference_ms": 4.19323400092253
}
```

Os valores de latência variam entre execuções e ambientes.

---

## 21. Arquitetura de software

O caso de uso depende de:

```text
ClassifierPort
```

Implementações:

```text
OnnxClassifierAdapter
SklearnClassifierAdapter
```

Arquitetura:

```text
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
```

A escolha do backend não altera o caso de uso principal.

---

## 22. Docker

A aplicação é empacotada com Docker multi-stage.

A imagem de produção utiliza:

```text
models/classifier.onnx
models/classifier_onnx_metadata.json
```

O pipeline ONNX contém o operador `StringNormalizer`.

Por isso, o runtime Docker configura explicitamente:

```text
en_US.UTF-8
```

A configuração evita falhas na criação da sessão do ONNX Runtime.

---

## 23. Observabilidade

A aplicação expõe métricas Prometheus.

Principais métricas:

```text
medical_triage_http_requests_total
medical_triage_http_request_duration_seconds
medical_triage_predictions_total
process_resident_memory_bytes
up
```

A distribuição das classes previstas pode ser acompanhada por:

```promql
sum by (label_name) (
  medical_triage_predictions_total
)
```

> A distribuição das predições não representa qualidade do modelo.

Documentação:

```text
docs/observability-plan.md
```

---

## 24. Privacidade

O texto recebido pela API:

- não deve ser registrado nos logs;
- não é utilizado como label Prometheus;
- não deve ser persistido pela camada de observabilidade.

Essa decisão reduz:

- risco de exposição de conteúdo sensível;
- risco de inclusão de dados pessoais;
- alta cardinalidade das métricas.

---

## 25. Limitações

As principais limitações conhecidas são:

- desempenho preditivo moderado;
- apenas cinco classes;
- ausência de rótulos de urgência clínica;
- ausência de avaliação clínica;
- ausência de validação por profissionais de saúde;
- ausência de análise formal de fairness;
- ausência de detecção automática de drift;
- ausência de ground truth em produção;
- pequenas divergências entre o pipeline sklearn e o Full ONNX;
- benchmarks dependentes do ambiente de execução.

---

## 26. Riscos

Possíveis riscos de interpretação inadequada incluem:

- tratar a classificação como diagnóstico;
- interpretar confidence como certeza clínica;
- converter classes de doença em níveis de urgência;
- utilizar o modelo fora do domínio do dataset;
- assumir que o desempenho medido é suficiente para uso médico.

Esses usos estão fora do escopo do projeto.

---

## 27. Considerações sobre confiança

O campo:

```text
confidence
```

representa a confiança produzida pelo classificador para a predição.

Ele não deve ser interpretado como:

- probabilidade clínica real;
- probabilidade de diagnóstico;
- grau de urgência;
- risco médico;
- garantia de correção.

---

## 28. Considerações sobre qualidade

Accuracy e Macro F1 são utilizadas para avaliar o comportamento do classificador
em dados rotulados.

Elas não medem:

- segurança clínica;
- impacto médico;
- fairness;
- robustez fora do domínio;
- capacidade diagnóstica;
- risco individual.

---

## 29. Reprodutibilidade

O projeto utiliza `uv`.

Instalação:

```bash
uv sync --locked
```

Testes:

```bash
uv run pytest
```

Quality gates:

```bash
uv run pre-commit run --all-files
```

Treinamento:

```bash
uv run python -m medical_triage.training.train
```

Export ONNX:

```bash
uv run python scripts/export_onnx.py
```

---

## 30. Qualidade de código

Na versão `0.4.0`:

```text
Ruff:       PASS
mypy:       PASS
pre-commit: PASS
Pytest:     21 passed
```

---

## 31. Monitoramento recomendado

Durante o serving, recomenda-se acompanhar:

- disponibilidade;
- throughput;
- P95 de latência;
- error rate;
- memória do processo;
- distribuição das classes previstas.

Esses indicadores fazem parte do dashboard:

```text
Medical Triage - Observability
```

---

## 32. Atualização do modelo

O treinamento atual é orquestrado por Apache Airflow.

DAG:

```text
medical_triage_training
```

Fluxo:

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

O projeto ainda não implementa retraining automático baseado em drift ou
mudança de qualidade.

---

## 33. Versionamento

Versão atual do projeto:

```text
0.4.0
```

Tag correspondente:

```text
v0.4.0
```

Modelo lógico:

```text
tfidf-logreg-v1
```

O versionamento do projeto e o versionamento lógico do modelo possuem funções
diferentes.

---

## 34. Responsabilidade do usuário

Qualquer pessoa que reutilize este projeto deve:

- preservar o aviso de finalidade acadêmica;
- não apresentar o sistema como ferramenta clínica;
- validar novamente o modelo ao alterar dataset ou pipeline;
- repetir benchmarks ao alterar runtime ou hardware;
- revisar limitações antes de qualquer reutilização.

---

## 35. Arquivos relacionados

Documentação:

```text
README.md
docs/stage1-cloud-architecture.md
docs/stage2-ci-airflow.md
docs/observability-plan.md
docs/stage4-onnx-optimization.md
```

Artefatos:

```text
models/classifier.onnx
models/classifier_onnx_metadata.json
```

Relatórios:

```text
reports/onnx_equivalence.json
reports/onnx_hybrid_equivalence.json
reports/inference_benchmark_summary.json
reports/http_benchmark_summary.json
```

---

## 36. Resumo de desempenho

```text
Baseline validation accuracy:          0.5931
Baseline validation Macro F1:          0.5908

sklearn test accuracy:                 0.583102
sklearn test Macro F1:                 0.585593

Full ONNX test accuracy:               0.585873
Full ONNX test Macro F1:               0.587332

Agreement sklearn x Full ONNX:         98.995845%
Agreement híbrido:                     100%

Redução do artefato:                   ~31.49%
Speedup isolado:                       ~2.15x
Redução da latência isolada:           ~53.44%
Speedup médio da inferência na API:    ~4.89x
Redução média da inferência na API:    ~79.55%
Redução HTTP end-to-end:               ~34.42%
```

---

## 37. Conclusão

O Medical Triage demonstra um pipeline completo de Machine Learning e MLOps para
classificação de textos médicos.

O modelo apresenta valor acadêmico para estudo de:

- NLP;
- serving;
- CI;
- orquestração;
- observabilidade;
- otimização de inferência;
- arquitetura de software.

A versão ONNX reduz latência e tamanho do artefato, mantendo comportamento muito
próximo ao pipeline sklearn de referência.

Apesar disso, o sistema permanece estritamente acadêmico e não deve ser
utilizado para diagnóstico, triagem real ou tomada de decisão clínica.
