# ECD-NEXUS 3.0

An Agentic Multimodal AI Framework for Infant Cry, Respiratory, Cardiac, and Motor Analysis.

## Overview

ECD-NEXUS 3.0 is a multimodal AI system that analyzes infant cry audio and movement video to generate explainable health-related insights. The project combines machine learning, pose estimation, clinical reasoning, and LLM-based report generation into a unified pipeline.

The system performs:
- Infant cry analysis
- Respiratory sound analysis
- Cardiac sound analysis
- Infant movement analysis
- Explainable AI-based reasoning
- Clinical-style report generation

## Technologies Used

- Python
- XGBoost
- Librosa
- MediaPipe
- LangChain
- Llama-3 via Groq API
- Streamlit
- Pandas / NumPy / Scikit-learn

## Datasets Used

- Donate-A-Cry Corpus
- ICBHI 2017 Respiratory Sound Dataset
- Pascal Heart Sound Dataset
- Infant Pose Estimation Dataset

## System Workflow

Input Audio/Video  
→ Feature Extraction  
→ XGBoost Predictions  
→ Motor Analysis  
→ Clinical Atlas Reasoning  
→ Agentic Fusion  
→ LLM Report Generation  
→ Streamlit Output

## Models Used

| Model | Algorithm |
| Cry Classification | XGBoost |
| Respiratory Analysis | XGBoost |
| Cardiac Analysis | XGBoost |
| Movement Analysis | MediaPipe Pose |
| Report Generation | Llama-3 |

## Main Files

- `extract_features.py` – Audio feature extraction
- `train_all_models.py` – Model training
- `synthetic_atlas.py` – Clinical reasoning rules
- `inference_v3.py` – Prediction pipeline
- `agents.py` – Multi-agent orchestration
- `app_v3.py` – Streamlit frontend

## Installation

```bash
pip install pandas numpy scikit-learn xgboost librosa
pip install streamlit mediapipe matplotlib
pip install langchain langchain-groq groq
````

## Run Project
```bash
python extract_features.py
python extract_features.py
streamlit run app_v3.py
```
## Groq API
Groq API is used to access the Llama-3 model for explainable report generation.
Get API key from:
[https://groq.com/](https://groq.com/)

## Features
* Multimodal AI pipeline
* Explainable AI using clinical atlas
* LangChain-based agentic architecture
* Decision-level fusion
* Real-time Streamlit interface
* 
## License
Developed for educational and research purposes.
