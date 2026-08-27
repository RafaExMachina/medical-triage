# Plano de Observabilidade — Medical Triage

## 1. Objetivo

A observabilidade do Medical Triage tem como objetivo acompanhar a
disponibilidade, o volume de tráfego, a latência, a ocorrência de erros,
o consumo de recursos da API e o comportamento das predições realizadas
pelo modelo de Machine Learning.

A solução foi implementada utilizando:

- FastAPI para o serviço de inferência;
- `prometheus-client` para instrumentação da aplicação;
- Prometheus para coleta e armazenamento das métricas;
- Grafana para visualização;
- Docker Compose para execução reproduzível da stack.

---

## 2. Arquitetura

```text
                        Docker Compose

        ┌──────────────────────────────┐
        │                              │
        │        FastAPI :8000         │
        │                              │
        │  GET  /health                │
        │  POST /predict               │
        │  GET  /metrics               │
        │         │                    │
        └─────────┼────────────────────┘
                  │
                  │ scrape /metrics
                  ▼
        ┌──────────────────────────────┐
        │      Prometheus :9090        │
        │                              │
        │   armazenamento TSDB         │
        │   consultas PromQL           │
        └──────────────┬───────────────┘
                       │
                       │ PromQL
                       ▼
        ┌──────────────────────────────┐
        │        Grafana :3000         │
        │                              │
        │ Dashboard Medical Triage     │
        └──────────────────────────────┘