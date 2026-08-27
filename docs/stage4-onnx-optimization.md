# Etapa 4 — Otimização de Inferência com ONNX Runtime

## Objetivo

Esta etapa otimiza a inferência do classificador de textos médicos sem alterar
o objetivo original do modelo.

O sistema continua classificando abstracts médicos nas cinco categorias do
Medical Abstracts TC Corpus:

1. neoplasms
2. digestive system diseases
3. nervous system diseases
4. cardiovascular diseases
5. general pathological conditions

A otimização escolhida foi a conversão do pipeline scikit-learn para ONNX e a
execução com ONNX Runtime.

---

## Arquitetura

Antes da otimização:

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
