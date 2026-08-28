# Plano de Observabilidade — Medical Triage

## 1. Objetivo

A observabilidade do Medical Triage tem como objetivo acompanhar o
comportamento operacional da API e do processo de inferência em produção.

Os principais pontos monitorados são:

- disponibilidade da API;
- volume de tráfego;
- latência das requisições;
- ocorrência de erros HTTP;
- consumo de recursos do processo Python;
- quantidade de predições realizadas;
- distribuição das classes previstas pelo modelo.

A solução foi implementada utilizando:

- FastAPI para o serviço de inferência;
- ONNX Runtime como backend padrão de inferência;
- `prometheus-client` para instrumentação da aplicação;
- Prometheus para coleta e armazenamento das métricas;
- PromQL para consulta das séries temporais;
- Grafana para visualização;
- Docker Compose para execução reproduzível da stack.

> A observabilidade implementada acompanha o comportamento operacional da API
> e a distribuição das predições. Ela não mede diretamente Accuracy, Macro F1
> ou qualidade clínica em produção, pois essas métricas exigem rótulos reais de
> referência.

---

## 2. Arquitetura

A stack de observabilidade é executada com Docker Compose.

```text
                     Docker Compose
┌───────────────────────────────────────────────────┐
│                                                   │
│  ┌─────────────────────────────────────────────┐  │
│  │              FastAPI :8000                 │  │
│  │                                             │  │
│  │  GET  /health                              │  │
│  │  POST /predict                             │  │
│  │  GET  /metrics                             │  │
│  │                                             │  │
│  │  ClassifyMedicalTextUseCase                │  │
│  │              │                              │  │
│  │              ▼                              │  │
│  │     OnnxClassifierAdapter                   │  │
│  │              │                              │  │
│  │              ▼                              │  │
│  │         ONNX Runtime                        │  │
│  └──────────────────┬──────────────────────────┘  │
│                     │                             │
│                     │ scrape /metrics             │
│                     ▼                             │
│  ┌─────────────────────────────────────────────┐  │
│  │           Prometheus :9090                 │  │
│  │                                             │  │
│  │  coleta e armazenamento de métricas        │  │
│  │  consultas PromQL                          │  │
│  └──────────────────┬──────────────────────────┘  │
│                     │                             │
│                     │ PromQL                      │
│                     ▼                             │
│  ┌─────────────────────────────────────────────┐  │
│  │             Grafana :3000                  │  │
│  │                                             │  │
│  │  Dashboard Medical Triage                  │  │
│  └─────────────────────────────────────────────┘  │
│                                                   │
└───────────────────────────────────────────────────┘
```

O Prometheus utiliza o modelo `pull` e consulta periodicamente o endpoint:

```text
http://api:8000/metrics
```

dentro da rede criada pelo Docker Compose.

O Grafana utiliza o Prometheus como datasource e consulta as séries temporais
por meio de PromQL.

---

## 3. Serviços monitorados

A stack principal executa três serviços:

| Serviço | Porta | Responsabilidade |
|---|---:|---|
| FastAPI | `8000` | API de classificação e exposição de métricas |
| Prometheus | `9090` | coleta e armazenamento das métricas |
| Grafana | `3000` | dashboards e visualização |

O Airflow permanece isolado em um Docker Compose separado e não faz parte da
stack de observabilidade da API.

---

## 4. Endpoint de métricas

A aplicação expõe métricas no endpoint:

```text
GET /metrics
```

Exemplo:

```bash
curl http://localhost:8000/metrics
```

O endpoint retorna dados no formato compatível com Prometheus.

O endpoint `/metrics` é excluído da contabilização das métricas HTTP da própria
aplicação para evitar que os scrapes do Prometheus poluam o tráfego monitorado.

---

## 5. Métricas principais

### 5.1 Requisições HTTP

Métrica:

```text
medical_triage_http_requests_total
```

Tipo:

```text
Counter
```

Labels controladas:

```text
method
endpoint
status
```

Objetivo:

- contar requisições recebidas;
- acompanhar volume de tráfego;
- separar respostas por código HTTP;
- permitir cálculo de throughput e error rate.

Exemplo:

```promql
medical_triage_http_requests_total{
  endpoint="/predict",
  method="POST",
  status="200"
}
```

---

### 5.2 Latência HTTP

Métrica:

```text
medical_triage_http_request_duration_seconds
```

Tipo:

```text
Histogram
```

Labels:

```text
method
endpoint
```

Objetivo:

- medir a duração das requisições HTTP;
- calcular percentis como P50, P95 e P99;
- acompanhar degradações de performance.

Exemplo de P95 para `/predict`:

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

---

### 5.3 Predições

Métrica:

```text
medical_triage_predictions_total
```

Tipo:

```text
Counter
```

Label:

```text
label_name
```

Objetivo:

- contar predições realizadas;
- acompanhar a distribuição das classes previstas;
- observar alterações inesperadas na distribuição de saída do modelo.

Distribuição:

```promql
sum by (label_name) (
  medical_triage_predictions_total
)
```

> Essa métrica representa a distribuição das classes previstas. Ela não mede
> qualidade do modelo nem substitui métricas de avaliação com ground truth.

---

### 5.4 Memória do processo da API

A biblioteca cliente do Prometheus também expõe métricas padrão do processo
Python.

O dashboard utiliza:

```promql
process_resident_memory_bytes{
  job="medical-triage-api"
}
```

Objetivo:

- acompanhar o consumo de memória residente;
- detectar crescimento anormal;
- apoiar investigação de vazamentos ou aumento de carga.

---

### 5.5 Disponibilidade

O Prometheus expõe automaticamente a métrica:

```promql
up{job="medical-triage-api"}
```

Interpretação:

```text
1 = target disponível
0 = target indisponível
```

Essa métrica permite detectar indisponibilidade da API ou falha no scrape.

---

## 6. Consultas PromQL do dashboard

### 6.1 Throughput de inferência

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

Objetivo:

- medir requisições de inferência bem-sucedidas por segundo.

---

### 6.2 P95 da latência HTTP

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

Objetivo:

- acompanhar a latência de cauda da API.

---

### 6.3 Error rate

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

A unidade do painel Grafana é `percentunit`.

Exemplo:

```text
0.2 -> 20%
```

---

### 6.4 Total de predições

```promql
sum(medical_triage_predictions_total) or vector(0)
```

Objetivo:

- exibir a quantidade acumulada de predições.

---

### 6.5 Distribuição das predições

```promql
sum by (label_name) (
  medical_triage_predictions_total
)
```

Objetivo:

- comparar a frequência das classes previstas.

---

### 6.6 Uso de memória

```promql
process_resident_memory_bytes{
  job="medical-triage-api"
}
```

Objetivo:

- acompanhar o consumo de memória residente do processo da API.

---

## 7. Dashboard Grafana

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

URL do datasource dentro da rede Docker:

```text
http://prometheus:9090
```

O dashboard possui seis painéis:

| Painel | Objetivo |
|---|---|
| Total Predictions | total acumulado de predições |
| Inference Throughput | requisições de inferência por segundo |
| P95 Inference HTTP Latency | P95 da latência HTTP de `/predict` |
| Prediction Error Rate | proporção de respostas 4xx e 5xx |
| Prediction Distribution | distribuição das classes previstas |
| API Memory Usage | memória residente do processo da API |

O datasource e o dashboard são provisionados automaticamente por arquivos
versionados no repositório.

---

## 8. Provisionamento do Grafana

Estrutura:

```text
monitoring/grafana/
├── dashboards/
│   └── medical-triage.json
└── provisioning/
    ├── dashboards/
    │   └── dashboards.yml
    └── datasources/
        └── prometheus.yml
```

O provisionamento automático evita configuração manual do datasource e do
dashboard após cada inicialização da stack.

---

## 9. Configuração do Prometheus

Arquivo:

```text
monitoring/prometheus/prometheus.yml
```

O Prometheus consulta a API através do nome do serviço Docker:

```text
api:8000
```

Target esperado:

```text
instance="api:8000"
job="medical-triage-api"
```

Validação:

```bash
curl -sG   --data-urlencode 'query=up{job="medical-triage-api"}'   http://localhost:9090/api/v1/query   | python -m json.tool
```

Resultado esperado:

```text
value="1"
```

---

## 10. Execução com Docker Compose

### 10.1 Validar a configuração

```bash
docker compose config
```

Serviços esperados:

```bash
docker compose config --services
```

Resultado:

```text
api
prometheus
grafana
```

### 10.2 Subir a stack

```bash
docker compose up -d --build
```

### 10.3 Conferir containers

```bash
docker compose ps
```

### 10.4 Validar FastAPI

```bash
curl http://localhost:8000/health
```

### 10.5 Validar Prometheus

```bash
curl http://localhost:9090/-/healthy
```

### 10.6 Validar Grafana

```bash
curl http://localhost:3000/api/health
```

### 10.7 Encerrar

```bash
docker compose down
```

Para remover também os volumes:

```bash
docker compose down -v
```

---

## 11. Grafana local

URL:

```text
http://localhost:3000
```

Credenciais locais configuradas para o ambiente acadêmico:

```text
user:     admin
password: admin
```

> Essas credenciais são exclusivas para desenvolvimento local. Em produção,
> devem ser utilizadas credenciais seguras e secrets apropriados.

---

## 12. Geração de tráfego para validação

Exemplo:

```bash
for i in {1..10}; do
  curl -s     -X POST     http://localhost:8000/predict     -H "Content-Type: application/json"     -d '{
      "text": "The patient presented with acute myocardial infarction and severe coronary artery disease."
    }'     > /dev/null
done
```

Depois, consulte:

```bash
curl -sG   --data-urlencode 'query=medical_triage_predictions_total'   http://localhost:9090/api/v1/query   | python -m json.tool
```

---

## 13. Segurança, privacidade e cardinalidade

O texto médico recebido pela API não é registrado como label Prometheus.

Labels utilizadas:

```text
method
endpoint
status
label_name
```

Isso reduz:

- risco de exposição de conteúdo clínico;
- risco de inclusão de dados pessoais;
- alta cardinalidade;
- crescimento desnecessário das séries temporais.

A aplicação também evita registrar o conteúdo textual recebido nos logs.

---

## 14. Thresholds sugeridos

Os valores abaixo são referências iniciais para o projeto acadêmico.

| Indicador | Atenção | Crítico |
|---|---:|---:|
| Disponibilidade | `up == 0` | imediato |
| P95 da latência | > 100 ms | > 250 ms |
| Error rate | > 5% | > 10% |
| Memória residente | > 512 MiB | > 1 GiB |

Esses limites não devem ser interpretados como SLOs definitivos.

Em produção, os thresholds devem ser calibrados com base em:

- volume real de tráfego;
- infraestrutura utilizada;
- comportamento histórico da aplicação;
- requisitos de negócio;
- SLOs definidos para o serviço.

---

## 15. Relação com a otimização ONNX

A partir da versão `0.4.0`, o backend padrão de inferência é ONNX Runtime.

O objetivo da observabilidade permanece o mesmo, mas a arquitetura de serving
passa a utilizar:

```text
FastAPI
  |
  v
ClassifyMedicalTextUseCase
  |
  v
OnnxClassifierAdapter
  |
  v
ONNX Runtime
```

Os benchmarks da Etapa 4 mostraram redução de latência em relação ao backend
sklearn.

A observabilidade permite acompanhar se esse comportamento continua estável
durante a execução da aplicação.

A documentação detalhada da otimização está em:

```text
docs/stage4-onnx-optimization.md
```

---

## 16. Limitações atuais

A solução atual não implementa automaticamente:

- alertas persistidos;
- Alertmanager;
- tracing distribuído;
- correlação automática entre logs, métricas e traces;
- métricas de drift;
- métricas de qualidade do modelo com ground truth;
- SLOs formais;
- retenção de longo prazo fora do Prometheus local.

Esses itens podem ser adicionados em versões futuras.

---

## 17. Critérios atendidos da Etapa 3

```text
[OK] Instrumentação com prometheus-client
[OK] Endpoint /metrics
[OK] Contagem de requisições HTTP
[OK] Histograma de latência HTTP
[OK] Contagem de predições
[OK] Métricas de processo
[OK] Prometheus
[OK] Target medical-triage-api UP
[OK] PromQL
[OK] Docker Compose
[OK] FastAPI + Prometheus + Grafana
[OK] Grafana
[OK] Datasource provisionado
[OK] Dashboard provisionado
[OK] Dashboard JSON versionado
[OK] Total Predictions
[OK] Inference Throughput
[OK] P95 Inference HTTP Latency
[OK] Prediction Error Rate
[OK] Prediction Distribution
[OK] API Memory Usage
```

---

## 18. Arquivos relacionados

```text
docker-compose.yml

monitoring/
├── prometheus/
│   └── prometheus.yml
└── grafana/
    ├── dashboards/
    │   └── medical-triage.json
    └── provisioning/
        ├── dashboards/
        │   └── dashboards.yml
        └── datasources/
            └── prometheus.yml
```

Código de instrumentação:

```text
src/medical_triage/observability/
├── logging.py
├── metrics.py
└── middleware.py
```

Documentação relacionada:

```text
README.md
docs/stage4-onnx-optimization.md
```

---

## 19. Resultado

A stack de observabilidade permite:

- verificar disponibilidade da API;
- acompanhar volume de requisições;
- observar latência;
- monitorar erros;
- acompanhar consumo de memória;
- analisar a distribuição das predições;
- validar o comportamento operacional do backend ONNX;
- reproduzir a configuração em outro ambiente utilizando Docker Compose.

Essa implementação atende ao escopo de Monitoramento e Observabilidade definido
para a Etapa 3 do Tech Challenge.
