import os
from dotenv import load_dotenv

# 환경 변수 로드
load_dotenv()

# LLM 설정
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GENERATION_CONFIG = {
    "temperature": 0.5,
    "max_output_tokens": 4096,
}

# NER 모델 설정
KOELECTRA_NER_PATH = os.getenv("KOELECTRA_NER_PATH", "").strip()

# 페이지 설정
PAGE_CONFIG = {
    "page_title": "중소기업 보안 관심/위험 분석 시스템",
    "page_icon": "🛡️",
    "layout": "wide",
    "initial_sidebar_state": "expanded"
}

# CSS 스타일
CSS_STYLES = """
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        color: white;
        text-align: center;
    }
    .recommendation-box {
        background: linear-gradient(135deg, #f8f9ff 0%, #e8f2ff 100%);
        padding: 2rem;
        border-radius: 10px;
        margin: 1rem 0;
        border: 1px solid #d4e6ff;
    }
    .sidebar-logo {
        text-align: center;
        padding: 1rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
        margin-bottom: 1rem;
        color: white;
    }
    .stSelectbox > div > div { background-color: #f8f9fa; }
    .stButton > button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 5px;
        padding: 0.5rem 1rem;
        font-weight: 600;
        width: 100%;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 4px 4px 12px rgba(102, 126, 234, 0.4);
    }
    .news-item {
        background: white;
        padding: 1.5rem;
        border-radius: 8px;
        margin-bottom: 1rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        border-left: 3px solid #3498db;
    }
    .risk-high { border-left-color: #e74c3c !important; }
    .risk-medium { border-left-color: #f39c12 !important; }
    .risk-low { border-left-color: #27ae60 !important; }
    .welcome-box {
        background: linear-gradient(45deg, #f0f8ff, #e6e6fa);
        padding: 2rem;
        border-radius: 15px;
        margin: 2rem auto;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.1);
        border: 2px solid #6a5acd;
        text-align: center;
        animation: fade-in 1s ease-in-out;
    }
    .welcome-box h3, .welcome-box h4 {
        color: #483d8b;
        text-shadow: 1px 1px 2px rgba(106, 90, 205, 0.5);
    }
    .summary-box {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 2rem;
        border-radius: 15px;
        margin: 2rem auto;
        border-left: 5px solid #6a5acd;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.08);
    }
    .summary-box h4 {
        color: #483d8b;
        margin-bottom: 1rem;
    }
    /* 이 부분이 추가되었습니다 */
    .summary-box p {
        font-size: 1.15em;
        line-height: 1.6;
    }
    @keyframes fade-in {
        from { opacity: 0; transform: translateY(-20px); }
        to   { opacity: 1; transform: translateY(0); }
    }
</style>
"""

# 기업 규모 옵션
COMPANY_SIZE_OPTIONS = ["소규모 (10-50명)", "중소규모 (50-200명)", "중규모 (200-500명)"]

# 업종 옵션
INDUSTRY_OPTIONS = ["IT/소프트웨어", "제조업", "금융업", "의료업", "교육업", "기타"]

# 인프라 옵션
INFRASTRUCTURE_OPTIONS = ["AWS", "Azure", "GCP", "On-premise", "Hybrid"]

# 페이지네이션 설정
PAGE_SIZE = 10