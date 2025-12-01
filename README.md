# GemmaGuard

This repository houses the code necessary for training GemmaGuard, alongside a logistic regression classifier and applies them to LLM responses to classify them regarding their refusal and harmfulness rates.

## Structure

0. The **Data** directory contains scripts for data download and preparation.
1. Inside of **Classifier_Development** are scripts to train GemmaGuard and LogReg and analysis scripts regarding the classifiers.
2. The LLM response generation, classification and analysis is inside **Cross_Lingual_Analysis**

Copy .env.example to .env and fill it with your keys and tokens.

## Installation

Recommended, but not required: install and manage via the uv package manager. https://docs.astral.sh/uv/

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv
uv sync
```
