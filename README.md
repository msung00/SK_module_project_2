# 🔒 SME Threat Watch  
**최신 보안 뉴스 기반 위험 분석 & 대응 플레이북 자동화 시스템**

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)
![Gemini](https://img.shields.io/badge/Gemini-API-orange.svg)
![KoELECTRA](https://img.shields.io/badge/KoELECTRA-NLP-green.svg)

---

## 🎯 프로젝트 개요

> **중소기업 맞춤형 보안 위험 분석 플랫폼**  
> 최신 보안 뉴스를 자동으로 수집·분석하고, 머신러닝/딥러닝 기반으로 **위험 키워드 추출** 후  
> Google **Gemini API**가 대응 권고사항을 생성하여 **PDF 플레이북**으로 제공합니다.  

---

## ✨ 주요 특징

- 🔍 **뉴스 자동 수집**: RSS/웹 스크래핑 기반 보안 뉴스 크롤링
- 🧹 **데이터 전처리**: HTML/특수문자 제거, 형태소 분석
- 🧠 **위험 키워드 분석**: KoELECTRA or KeyBERT 기반
- 🤖 **Gemini LLM 권고 생성**: 기업 환경 맞춤 대응책 자동 작성
- 📑 **리포트 출력**: PDF / Markdown 형태 보고서 다운로드
- 🌐 **Streamlit UI**: 원클릭 실행 및 대시보드 제공

---

## 🛠️ 기술 스택

### Backend & Data
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)

### AI & ML
![Gemini](https://img.shields.io/badge/Google%20Gemini-4285F4?style=for-the-badge&logo=google&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![KoELECTRA](https://img.shields.io/badge/KoELECTRA-NLP-green?style=for-the-badge)

### Frontend
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![HTML5](https://img.shields.io/badge/HTML5-E34F26?style=for-the-badge&logo=html5&logoColor=white)
![CSS3](https://img.shields.io/badge/CSS3-1572B6?style=for-the-badge&logo=css3&logoColor=white)

### Tools & Collaboration
![Git](https://img.shields.io/badge/Git-F05032?style=for-the-badge&logo=git&logoColor=white)
![GitHub](https://img.shields.io/badge/GitHub-181717?style=for-the-badge&logo=github&logoColor=white)
![Notion](https://img.shields.io/badge/Notion-000000?style=for-the-badge&logo=notion&logoColor=white)

---
## 📋 시스템 요구사항

Python 3.10+

Google Gemini API Key

## 실행 방법
pip install streamlit pandas plotly requests beautifulsoup4 fpdf konlpy torch transformers google-generativeai python-dotenv

## 📖 사용 흐름

1️⃣ **뉴스 수집** → 최신 보안 기사 가져오기  
2️⃣ **전처리** → HTML/특수문자 제거 & 형태소 분석  
3️⃣ **위험 키워드 분석** → ML/DL 기반 키워드 추출  
4️⃣ **Gemini API** → 대응 권고사항 생성  
5️⃣ **PDF/Markdown 출력** → 플레이북 다운로드  

---

<div align="center">

⚡ **SME Threat Watch — 중소기업 보안 대응을 더 빠르고 더 스마트하게** ⚡  
Made by SK Shieldus Rookies 26기 AI 8조 팔색조

</div>

