# 📘 Code Readability & Bug Analyzer

A FastAPI web app to analyze Java code readability, time/space complexity, and locate common bugs.

## 🚀 Features
- Predicts readability using a CNN model
- Estimates time and space complexity using static analysis
- Detects bugs via `javac` and suggests fixes using a fallback rule-based system or ML model

## 📁 Project Structure
- `main.py`: FastAPI backend
- `index.html`: Frontend UI in `templates/`
- `bug_analyzer.py`: Bug detection logic
- `bug_model.pkl`: Bug suggestion model
- `coderead_no_indent.h5`: Readability prediction model
- `tokenizer.json`, `label_encoder.pkl`: Model assets
- `requirements.txt`: All Python dependencies

## 🛠️ Setup

```bash
git clone https://github.com/yourusername/code-readability-analyzer.git
cd code-readability-analyzer
python -m venv venv
venv\Scripts\activate  # or source venv/bin/activate on Linux/Mac
pip install -r requirements.txt
