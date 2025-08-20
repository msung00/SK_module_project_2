# 인터페이스.py
# ------------------------------------------------------------
# 중소기업 보안 관심/위험 분석 시스템 (RSS 전용, 프롬프트 보강판)
# - 보안뉴스 RSS 수집
# - KoELECTRA NER 기반 키워드 추출 + 업종별 가중치 분석
# - CISA KEV feed 반영
# - Gemini 기반 요약/플레이북 생성 (message.txt 의도 반영 프롬프트)
# - PDF 보고서, 대시보드/뉴스/플레이북 탭
# - LLM 전달 키워드 및 Prompt/Response 로그 노출 (제거)
# ------------------------------------------------------------

import os
import re
import time
import json
import requests
import pandas as pd
import streamlit as st
import feedparser
import torch
import textwrap

from bs4 import BeautifulSoup
from datetime import datetime
from fpdf import FPDF
from dotenv import load_dotenv
from transformers import ElectraTokenizerFast, ElectraForTokenClassification
import google.generativeai as genai

# ============================================================
# 0) 공통 설정
# ============================================================
st.set_page_config(
    page_title="중소기업 보안 관심/위험 분석 시스템",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 공용 스타일
st.markdown("""
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
</style>
""", unsafe_allow_html=True)

# ============================================================
# 1) 환경 변수/LLM 설정
# ============================================================
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    st.error("Gemini API 키가 설정되지 않았습니다. .env 파일을 확인해주세요.")
    st.stop()
genai.configure(api_key=GEMINI_API_KEY)

GENERATION_CONFIG = {
    "temperature": 0.5,
    "max_output_tokens": 1400,
}
GEMINI = genai.GenerativeModel('gemini-1.5-flash', generation_config=GENERATION_CONFIG)

# ============================================================
# 2) 뉴스 수집 모듈 (RSS 전용)
# ============================================================
def scrape_article(url: str):
    """보안뉴스 기사 상세 스크래핑 (타이틀/본문/일자)"""
    try:
        res = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=7)
        if res.status_code != 200:
            return None
        soup = BeautifulSoup(res.text, "html.parser")

        # 다양한 템플릿 대응
        title = soup.select_one("#news_title02") or soup.select_one("h4.tit")
        body = soup.select_one("#news_content") or soup.select_one("div.view_txt")
        date = soup.select_one("#news_util01") or soup.select_one("span.date")

        return {
            "url": url,
            "title": title.get_text(strip=True) if title else "제목 없음",
            "date": date.get_text(strip=True) if date else datetime.now().strftime("%Y-%m-%d"),
            "content": body.get_text("\n", strip=True) if body else "내용 없음",
            "source": "보안뉴스"
        }
    except Exception:
        return None

def fetch_latest_news_by_rss():
    """보안뉴스 RSS 여러 피드에서 최신 기사 수집"""
    rss_list = [
        ("SECURITY", "http://www.boannews.com/media/news_rss.xml?mkind=1"),
        ("IT", "http://www.boannews.com/media/news_rss.xml?mkind=2"),
        ("SAFETY", "http://www.boannews.com/media/news_rss.xml?mkind=4"),
        ("사건ㆍ사고", "http://www.boannews.com/media/news_rss.xml?kind=1"),
        ("공공ㆍ정책", "http://www.boannews.com/media/news_rss.xml?kind=2"),
        ("비즈니스", "http://www.boannews.com/media/news_rss.xml?kind=3"),
        ("국제", "http://www.boannews.com/media/news_rss.xml?kind=4"),
        ("테크", "http://www.boannews.com/media/news_rss.xml?kind=5"),
    ]
    seen_urls, seen_titles, all_articles = set(), set(), []
    for feed_name, rss_url in rss_list:
        try:
            feed = feedparser.parse(rss_url)
        except Exception:
            continue
        for entry in getattr(feed, "entries", []):
            url = getattr(entry, "link", None)
            title = getattr(entry, "title", "").strip()
            if not url or url in seen_urls or title in seen_titles:
                continue
            seen_urls.add(url)
            seen_titles.add(title)
            data = scrape_article(url)
            if data and data['title'] != '제목 없음':
                data["source"] = feed_name
                all_articles.append(data)
            time.sleep(0.1)
    return all_articles

# ============================================================
# 3) NER 모델 로딩 (가능시) 및 분석 폴백
# ============================================================
@st.cache_resource(show_spinner=False)
def load_ner_model():
    """
    KoELECTRA NER 모델 로딩.
    로컬 경로에 학습된 모델이 없거나 로드 실패 시 (tokenizer/model) None 반환.
    """
    try:
        NER_MODEL_PATH = os.getenv("KOELECTRA_NER_PATH", "").strip()
        if not NER_MODEL_PATH:
            return None, None, None
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        tokenizer = ElectraTokenizerFast.from_pretrained(NER_MODEL_PATH)
        model = ElectraForTokenClassification.from_pretrained(NER_MODEL_PATH).to(device).eval()
        id2label = model.config.id2label
        return tokenizer, model, (device, id2label)
    except Exception:
        return None, None, None

ner_tokenizer, ner_model, ner_ctx = load_ner_model()

def ner_inference(sentence: str):
    """NER 기반 토큰→워드 재구성 후 라벨 O 제외 토큰 반환"""
    if not (ner_tokenizer and ner_model and ner_ctx):
        return []
    device, id2label = ner_ctx
    enc = ner_tokenizer(sentence, return_offsets_mapping=True,
                         return_tensors="pt", truncation=True)
    word_ids = enc.word_ids()
    with torch.no_grad():
        outputs = ner_model(input_ids=enc["input_ids"].to(device),
                            attention_mask=enc["attention_mask"].to(device))
        pred_ids = torch.argmax(outputs.logits, dim=-1)[0].cpu().tolist()

    results, cur_word, cur_label, prev_wid = [], "", None, None
    for idx, wid in enumerate(word_ids):
        if wid is None:
            continue
        tok_id = int(enc["input_ids"][0, idx].item())
        piece = ner_tokenizer.convert_ids_to_tokens(tok_id)
        label = ner_ctx[1][int(pred_ids[idx])]
        if wid != prev_wid:
            if cur_word:
                results.append((cur_word, cur_label))
            cur_word, cur_label = piece.replace("##", ""), label
        else:
            cur_word += piece.replace("##", "")
        prev_wid = wid
    if cur_word:
        results.append((cur_word, cur_label))
    return [w for w, l in results if l != "O"]

industry_risk_map = {
    "IT/소프트웨어": {
        "랜섬웨어":1.0,"제로데이":1.0,"취약점":1.0,"API":0.9,"클라우드":1.0,"SaaS":0.9,"DevOps":0.9,"GitHub":0.9,
        "오픈소스":0.9,"CVE":1.0,"패치":1.0,"익스플로잇":1.0,"공급망 공격":1.0,"소프트웨어 업데이트":0.9,"악성코드":1.0,
        "백도어":1.0,"웹쉘":0.9,"SQL 인젝션":1.0,"XSS":1.0,"CSRF":0.9,"SSRF":0.9,"XXE":0.9,"IDOR":0.9,
        "버퍼 오버플로우":1.0,"메모리 누수":0.8,"권한 상승":0.9,"세션 하이재킹":0.9,"세션 고정":0.8,"취약한 암호화":1.0,
        "하드코딩된 키":0.9,"평문 전송":0.9,"API 키 노출":1.0,"크리덴셜 스터핑":1.0,"브루트포스":0.9,"딕셔너리 공격":0.9,
        "피싱":0.8,"스피어피싱":0.9,"워터링홀":0.8,"APT":1.0,"사회공학":0.9,"악성 스크립트":0.8,"웹 취약점":0.9,
        "코드 서명 위조":0.9,"취약한 라이브러리":0.9,"npm 패키지 공격":0.9,"PyPI 공격":0.9,"맬웨어":1.0,
        "RAT":0.9,"루트킷":0.9,"로직밤":0.8,"봇넷":0.9,"웜":0.9,"바이러스":0.9,
        "CI/CD 공격":0.9,"컨테이너 탈출":0.9,"쿠버네티스 공격":0.9,"도커 허브 악성 이미지":0.8,
        "IaC 보안":0.8,"취약한 설정":0.9,"잘못된 권한":0.9,"IAM 오용":0.9,"S3 버킷 노출":1.0,
        "RaaS":0.8,"서비스 거부":0.8,"DoS":0.8,"DDoS":1.0,"클라우드 권한 상승":0.9,"MITM":0.9,"DNS 스푸핑":0.9,
        "패킷 스니핑":0.9,"VPN 공격":0.9,"토큰 탈취":0.9,"세션 탈취":0.9,"이메일 계정 탈취":0.9,
        "APT 공격":1.0,"북한 해킹":1.0,"중국 해킹":1.0,"라자루스":1.0,"김수키":1.0,"APT37":0.9,"APT28":0.9,
        "보안 설정 미흡":0.9,"권한 오남용":0.9,"암호화 미적용":1.0,"데이터 유출":1.0,"소스코드 유출":1.0,
        "DevSecOps 미흡":0.9,"취약한 테스트 코드":0.8,"보안 자동화 부재":0.8,"취약점 스캐닝 누락":0.9,
        "보안 로그 미수집":0.9,"SIEM 부재":0.9,"EDR 미적용":0.9,"MFA 미적용":1.0,"약한 비밀번호":1.0,
        "OAuth 취약점":0.9,"SSO 우회":0.9,"JWT 변조":0.9,"GraphQL 공격":0.9,"NoSQL 인젝션":0.9,
        "API 게이트웨이 우회":0.9,"클라우드 네이티브 공격":0.9,"IaC 스캔 누락":0.8,"CSP 설정 오류":0.8,
        "안전하지 않은 리다이렉트":0.8,"세션 토큰 재사용":0.9,"쿠키 탈취":0.9,"브라우저 익스플로잇":0.9,
        "제로트러스트 미적용":0.9,"망 분리 우회":0.9,"보안 모니터링 부재":0.9,
        "Prompt Injection":1.0,"LLM Jailbreak":1.0,"데이터 포이즈닝":1.0,"AI 모델 도용":1.0,
        "AI 환각":0.9,"모델 역추적":1.0,"시스템 프롬프트 노출":1.0
    },
    "제조업": {
        "산업제어시스템":1.0,"SCADA":1.0,"ICS":1.0,"PLC":1.0,"스마트팩토리":1.0,"로봇":0.9,"CNC":0.9,"HMI":1.0,
        "산업용 IoT":0.9,"OT 보안":1.0,"제조 라인 공격":1.0,"공급망 공격":1.0,"악성 USB":0.8,
        "랜섬웨어":1.0,"트로이목마":0.9,"워터링홀":0.8,"APT":1.0,"스피어피싱":0.8,"사회공학":0.8,
        "데이터 유출":1.0,"생산 차질":1.0,"로봇 해킹":0.9,"CVE":1.0,"제로데이":1.0,"악성코드":1.0,
        "VPN 공격":0.9,"MITM":0.8,"DoS":0.8,"DDoS":0.9,"버퍼 오버플로우":0.9,"메모리 취약점":0.9,
        "권한 상승":0.9,"세션 하이재킹":0.8,"백도어":0.9,"봇넷":0.8,"웜":0.8,"루트킷":0.8,"RAT":0.8,
        "스마트센서":0.9,"IoT 보안":0.9,"펌웨어 해킹":0.9,"취약한 암호화":0.9,"하드코딩된 키":0.8,"평문 통신":0.8,
        "원격 코드 실행":0.9,"SQL 인젝션":0.7,"XSS":0.7,"SSRF":0.7,"CSRF":0.7,"웹 취약점":0.7,
        "공장 자동화 공격":1.0,"제조 데이터 위조":1.0,"스파이웨어":0.8,"산업 스파이":1.0,"설비 파괴":1.0,
        "위조 부품":1.0,"불량품 주입":1.0,"생산 중단":1.0,"국가 지원 해킹":1.0,"라자루스":1.0,
        "APT41":1.0,"기계 제어 취약점":1.0,"산업 네트워크 침투":1.0,"보안 설정 미흡":0.9,
        "접근 통제 실패":0.9,"데이터 무결성 공격":1.0,"위조 인증서":0.9,"인증 우회":0.9,"악성 펌웨어":1.0,
        "Modbus 공격":1.0,"DNP3 공격":1.0,"HMI 위조":0.9,"산업 로봇 제어권 탈취":1.0,
        "에너지 관리시스템 공격":1.0,"PLC 로직 주입":1.0,"산업 네트워크 스니핑":0.9,"망분리 우회":0.9,
        "스마트 그리드 공격":1.0,"산업용 무선 침투":0.8,"디지털 트윈 해킹":0.9,
        "AI 기반 제조 공격":0.7,"AI 모델 위조":0.7,"프롬프트 인젝션":0.6
    },
    "금융업": {
        "피싱":1.0,"스피어피싱":1.0,"이메일 계정 탈취":1.0,"계정정보 유출":1.0,"크리덴셜 스터핑":1.0,
        "브루트포스":1.0,"딕셔너리 공격":1.0,"계좌 탈취":1.0,"은행":1.0,"카드사":1.0,"결제정보 유출":1.0,
        "암호화폐":1.0,"거래소 해킹":1.0,"DeFi 공격":1.0,"핀테크":0.9,"오픈뱅킹":1.0,"API 키 노출":1.0,
        "랜섬웨어":0.9,"트로이목마":0.9,"악성코드":0.9,"봇넷":0.9,"RAT":0.9,"루트킷":0.8,
        "APT":1.0,"사회공학":1.0,"BEC":1.0,"가짜 앱":0.9,"모바일 피싱":1.0,"스미싱":1.0,"QR 피싱":1.0,
        "DDoS":1.0,"서비스 거부":0.9,"MITM":1.0,"DNS 스푸핑":0.9,"패킷 스니핑":0.9,
        "악성 결제 모듈":1.0,"백도어":0.9,"정보 탈취":1.0,"데이터 유출":1.0,
        "고객정보 유출":1.0,"금융사기":1.0,"보이스피싱":1.0,"가짜 투자":1.0,"라자루스":1.0,"김수키":1.0,
        "APT38":1.0,"국가 지원 해킹":1.0,"SWIFT 공격":1.0,"ATM 해킹":0.9,"POS 공격":0.9,
        "핀테크 API 취약점":0.9,"암호화 미적용":1.0,"약한 비밀번호":1.0,"2FA 미적용":1.0,
        "세션 탈취":0.9,"토큰 탈취":0.9,"불법 송금":1.0,"악성 봇":0.9,"딥페이크 사기":1.0,
        "대출 사기":1.0,"가짜 보험":0.9,"모바일 뱅킹 악성앱":1.0,"핀테크 SDK 취약점":0.9,
        "암호화폐 탈취":1.0,"피싱 웹사이트":1.0,"가짜 인증서":0.9,"MFA 피싱":1.0,
        "CBDC 위협":0.9,"암호화폐 거래소 내부자 공격":1.0,
        "AI 금융 사기":1.0,"프롬프트 인젝션":0.9,"AI 챗봇 피싱":1.0,"AI 딥페이크":1.0
    },
    "의료업": {
        "환자정보":1.0,"의료기기":1.0,"IoMT":1.0,"의료 데이터 유출":1.0,"병원 해킹":1.0,"EMR":1.0,"EHR":1.0,
        "원격진료":1.0,"진단장비":0.9,"의료영상":0.9,"보건의료정보":1.0,"제약사 해킹":0.9,
        "연구데이터 유출":0.9,"임상시험 데이터":0.9,"DNA 데이터":0.9,"바이오해킹":0.9,
        "악성코드":1.0,"랜섬웨어":1.0,"트로이목마":0.9,"RAT":0.9,"루트킷":0.9,
        "피싱":1.0,"스피어피싱":1.0,"사회공학":0.9,"QR 피싱":0.8,"스미싱":0.8,
        "제로데이":1.0,"취약점":1.0,"CVE":1.0,"SQL 인젝션":0.8,"XSS":0.8,"CSRF":0.8,
        "SSRF":0.8,"서비스 거부":0.9,"DDoS":0.9,"MITM":0.9,"VPN 공격":0.9,
        "의료데이터 위조":1.0,"환자 모니터링 조작":1.0,"의료기기 오작동":1.0,
        "불법 의료 데이터 거래":1.0,"다크웹 유출":1.0,"악성 앱":0.9,"위조 처방전":1.0,
        "보안 설정 미흡":0.9,"암호화 미적용":1.0,"약한 비밀번호":1.0,"2FA 미적용":1.0,
        "의료 AI 위조":1.0,"헬스케어 IoT 공격":1.0,"환자 계정 탈취":1.0,"의료보험 사기":0.9,
        "의료 디지털 트윈 해킹":0.9,"원격 수술 해킹":1.0,
        "AI 진단 조작":1.0,"프롬프트 인젝션":0.8,"AI 의료데이터 조작":1.0
    },
    "교육업": {
        "온라인수업":1.0,"LMS":1.0,"학생정보":1.0,"교직원 계정":1.0,"학교 네트워크":0.9,
        "연구데이터":0.9,"대학 해킹":1.0,"고등학교 해킹":0.9,"입시 데이터 유출":1.0,"성적 조작":1.0,
        "피싱":1.0,"스피어피싱":1.0,"스미싱":0.9,"QR 피싱":0.9,"사회공학":0.9,
        "랜섬웨어":0.9,"악성코드":0.9,"트로이목마":0.9,"웜":0.9,"바이러스":0.9,"RAT":0.9,
        "제로데이":0.9,"취약점":0.9,"SQL 인젝션":0.9,"XSS":0.9,"CSRF":0.9,
        "SSRF":0.9,"서비스 거부":0.9,"DDoS":0.9,"MITM":0.9,"VPN 공격":0.9,
        "데이터 유출":1.0,"개인정보 유출":1.0,"출석 조작":0.9,"시험 문제 유출":1.0,
        "해킹 동아리":0.7,"다크웹 공유":0.9,"크리덴셜 스터핑":1.0,"브루트포스":0.9,"약한 암호":1.0,
        "원격 수업 툴 공격":1.0,"교수 계정 탈취":1.0,"학생 계정 도용":1.0,"교육 클라우드 취약점":0.9,
        "온라인 시험 부정행위 툴":0.9,
        "AI 숙제 자동화":0.8,"프롬프트 인젝션":0.7,"AI 커닝 툴":0.8
    },
    "기타": {
        "APT":1.0,"라자루스":1.0,"김수키":1.0,"샌드웜":1.0,"APT28":1.0,"APT29":1.0,
        "국가 지원 해킹":1.0,"사이버전":1.0,"사이버 스파이":1.0,"스파이웨어":1.0,
        "사회공학":1.0,"정치 선전 해킹":1.0,"정부기관 공격":1.0,"군사 해킹":1.0,
        "DDoS":1.0,"서비스 거부":1.0,"데이터 유출":1.0,"기밀 문서 유출":1.0,
        "랜섬웨어":1.0,"제로데이":1.0,"취약점":1.0,"악성코드":1.0,"백도어":1.0,
        "스피어피싱":1.0,"BEC":1.0,"공급망 공격":1.0,"소셜미디어 해킹":0.9,"디도스":1.0,
        "선거 해킹":1.0,"언론 조작":1.0,"인프라 공격":1.0,"전력망 공격":1.0,"수도시설 공격":1.0,
        "교통망 해킹":1.0,"위성통신 해킹":1.0,"GPS 교란":1.0,"IoT 공격":0.9,
        "딥페이크":1.0,"AI 기반 공격":1.0,"악성 드론":0.9,"사이버 테러":1.0,"핵심인프라 파괴":1.0,
        "MITRE ATT&CK TTP":1.0,"사회 혼란 조장":1.0,"사이버 첩보":1.0,
        "우크라이나 전쟁 해킹":1.0,"중동 사이버전":1.0,"사이버 용병":0.9,"정보전":1.0,
        "AI 심리전":1.0,"AI 선전 조작":1.0,"AI 기반 여론 조작":1.0,"프롬프트 인젝션":0.9
    }
}

def ner_inference(sentence: str):
    """NER 기반 토큰→워드 재구성 후 라벨 O 제외 토큰 반환"""
    if not (ner_tokenizer and ner_model and ner_ctx):
        return []
    device, id2label = ner_ctx
    enc = ner_tokenizer(sentence, return_offsets_mapping=True,
                         return_tensors="pt", truncation=True)
    word_ids = enc.word_ids()
    with torch.no_grad():
        outputs = ner_model(input_ids=enc["input_ids"].to(device),
                            attention_mask=enc["attention_mask"].to(device))
        pred_ids = torch.argmax(outputs.logits, dim=-1)[0].cpu().tolist()

    results, cur_word, cur_label, prev_wid = [], "", None, None
    for idx, wid in enumerate(word_ids):
        if wid is None:
            continue
        tok_id = int(enc["input_ids"][0, idx].item())
        piece = ner_tokenizer.convert_ids_to_tokens(tok_id)
        label = ner_ctx[1][int(pred_ids[idx])]
        if wid != prev_wid:
            if cur_word:
                results.append((cur_word, cur_label))
            cur_word, cur_label = piece.replace("##", ""), label
        else:
            cur_word += piece.replace("##", "")
        prev_wid = wid
    if cur_word:
        results.append((cur_word, cur_label))
    return [w for w, l in results if l != "O"]

def classify_cve_industry(description: str) -> str:
    desc = description.lower()
    if any(x in desc for x in ["ics","scada","plc","ot","industrial","factory","hmi"]):
        return "제조업"
    elif any(x in desc for x in ["bank","finance","payment","atm","credential","card"]):
        return "금융업"
    elif any(x in desc for x in ["medical","healthcare","hospital","ehr","emr","patient"]):
        return "의료업"
    elif any(x in desc for x in ["school","student","education","lms","university"]):
        return "교육업"
    else:
        return "IT/소프트웨어"

def update_keywords_from_cisa(industry_map: dict):
    try:
        kev_url = "https://www.cisa.gov/sites/default/files/feeds/known_exploited_vulnerabilities.json"
        kev_data = requests.get(kev_url, timeout=10).json()
        for vuln in kev_data.get("vulnerabilities", []):
            cve_id = vuln.get("cveID")
            desc = vuln.get("shortDescription", "")
            industry = classify_cve_industry(desc)
            if cve_id and cve_id not in industry_map[industry]:
                industry_map[industry][cve_id] = 1.0
    except Exception as e:
        st.warning(f"CISA KEV 업데이트 실패: {e}")

# 앱 최초 1회 KEV 반영
update_keywords_from_cisa(industry_risk_map)

def analyze_risk_with_model(text: str, industry_type: str):
    """
    1) 가능하면 NER로 엔터티 추출
    2) 업종별 키워드 가중치 합산으로 점수/레벨 산정
    3) NER 실패 시, 단순 키워드 매칭 폴백
    """
    risk_dict = industry_risk_map.get(industry_type, {})
    extracted = ner_inference(text)

    # 폴백: 관심 맵 키 중 텍스트 포함되는 것 추가
    if not extracted:
        for kw in risk_dict.keys():
            try:
                if re.search(r'\b' + re.escape(kw) + r'\b', text, flags=re.IGNORECASE):
                    extracted.append(kw)
            except re.error:
                # 정규식 이스케이프 문제 대비
                if kw.lower() in text.lower():
                    extracted.append(kw)

    extracted = list(set(extracted))
    total_score = sum(risk_dict.get(kw, 0.0) for kw in extracted)
    if total_score >= 2.0: level = "높음"
    elif total_score >= 0.8: level = "중간"
    else: level = "낮음"
    return level, extracted, total_score

# ============================================================
# 4) LLM 요약/플레이북/키워드 생성 (프롬프트 보강)
# ============================================================
@st.cache_data(show_spinner=False)
def generate_article_summary(title: str, content: str, severity_label: str, company_info: dict, infrastructure: str):
    """
    message.txt 의도 반영:
    - 3~5문장 요약
    - 마지막 줄: '왜 우리에게 중요한가' 1문장
    - 맥락 힌트: 심각도/업종/인프라
    """
    prompt = f"""
다음 한국어 보안 뉴스를 3~5문장으로 간결하게 요약하세요.
마지막 줄에 '왜 우리에게 중요한가'를 1문장으로 설명하세요.
과장 없이 사실만, 가능하면 수치/기술 요소를 구체적으로.

[기사 제목]
{title}

[기사 본문(발췌)]
{content[:2800]}

[맥락 힌트]
- 심각도: {severity_label}
- 업종: {company_info.get('industry')}
- 인프라: {infrastructure}

출력 형식: 문단 3~5개 + 마지막 1문장(왜 중요한가).
""".strip()
    try:
        res = GEMINI.generate_content(prompt)
        return res.text
    except Exception:
        return "요약 생성 실패."

def generate_playbook_with_llm(keywords, company_info, infrastructure, constraints, news_briefs=None):
    """
    - message.txt 의도 반영 통합 플레이북:
      즉시/7일/30일 구간 + 탐지룰 + 커뮤니케이션 + 체크리스트
    - LLM 인풋 및 결과 로그 저장
    - 중요 키워드 JSON 재요청
    """
    # 0) 상위 뉴스 1줄 요약 목록
    news_briefs = news_briefs or []
    company_info_str = json.dumps(company_info, ensure_ascii=False)

    mode_line = "가능한 저예산/간소화 모드를 우선 고려" if (constraints and any(x in constraints.lower() for x in ["저예산","budget","비용","한정"])) else "표준 모드로 실행"
    
    # 1) 본문 플레이북 생성 프롬프트
    prompt = f"""
당신은 중소기업 보안 전문가입니다. 아래 정보를 바탕으로 **통합 장문 대응 플레이북**을 작성하세요.
- 중복되는 조치는 통합/정리
- **즉시(오늘~48h)/7일/30일** 구간으로 나눌 것
- {mode_line}
- **구체 설정/서비스명**(예: AWS/GCP/Azure, EDR/SIEM/MFA 등)을 포함
- 각 조치는 **검증 기준(어떻게 확인할지)**을 명시

[회사 프로필(JSON)]
{company_info_str}

[인프라]
{infrastructure}

[최신 보안 키워드 후보]
{", ".join(keywords[:40])}

[상위 뉴스 요약(각 1줄)]
{chr(10).join(f"- {line}" for line in news_briefs[:8]) if news_briefs else "- (없음)"}

[제약]
{constraints or "없음"}

출력 형식: Markdown 섹션
1) 상황요약
2) 즉시(오늘~48h)
3) 7일
4) 30일
5) 탐지룰/모니터링(로그 소스, 룰 또는 쿼리 개요)
6) 커뮤니케이션(임직원 공지/훈련/외부 보고)
7) 체크리스트(측정 가능한 완료 조건)
""".strip()

    try:
        resp = GEMINI.generate_content(prompt)
        playbook = resp.text or ""
    except Exception as e:
        playbook = f"플레이북 생성 실패: {e}"

    # 2) LLM이 중요하다고 판단한 키워드만 JSON으로 재요청
    kw_prompt = f"""
다음 키워드 후보에서 중소기업 환경에 가장 관련 높은 상위 12개를 고르세요.
JSON 배열만 출력하세요.

후보: {', '.join(keywords)}

스키마:
[
  {{"keyword": "문자열", "rationale": "간단 근거(10자~30자)"}}
]
다른 텍스트는 절대 포함하지 마세요.
""".strip()
    llm_selected_keywords = []
    try:
        kw_resp = GEMINI.generate_content(kw_prompt)
        raw = (kw_resp.text or "").strip()
        # JSON만 출력하도록 요청했지만 방어적으로 파싱
        json_str = re.search(r'\[.*\]', raw, flags=re.S)
        if json_str:
            llm_selected_keywords = json.loads(json_str.group(0))
        else:
            raise ValueError("JSON 파싱 실패")
    except Exception:
        # 실패 시 상위 12개 키워드 단순 절단
        llm_selected_keywords = [{"keyword": k, "rationale": "자동 대체(파싱 실패)"} for k in keywords[:12]]

    st.session_state.llm_selected_keywords = llm_selected_keywords
    return playbook

# ============================================================
# 5) PDF 보고서 (폰트 경로 유연화 + NanumGothic 적용 + 안전 래핑)
# ============================================================
import textwrap
import re
from fpdf import FPDF

def _try_add_font(pdf: FPDF):
    # 가능한 경로들: 로컬/상대경로 모두 시도
    candidates = [
        "font/NanumGothic.ttf",
        "font/NanumGothicBold.ttf",
        "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
        "/usr/share/fonts/truetype/nanum/NanumGothicBold.ttf",
        "C:/Windows/Fonts/NanumGothic.ttf",
        "C:/Windows/Fonts/NanumGothicBold.ttf",
    ]
    for path in candidates:
        try:
            # 동일 패밀리명으로 하나만 등록해도 본문 사용에는 지장 없음
            pdf.add_font("Nanum", "", path, uni=True)
            return True
        except Exception:
            continue
    return False

def _usable_width(pdf: FPDF) -> float:
    # 현재 페이지에서 좌/우 마진을 제외한 사용 가능 폭
    return pdf.w - pdf.l_margin - pdf.r_margin

def _normalize_long_tokens(s: str) -> str:
    """
    FPDF가 줄바꿈하지 못하는 긴 토큰(URL, CVE, 경로 등)을 안전하게 끊기 위해
    분리자 뒤에 여백을 추가해 가시적 끊김 포인트를 만든다.
    """
    # 분리자 뒤에 공백 추가
    s = re.sub(r'([/@:_\-\.\|\+\=])', r'\1 ', s)
    # 다중 공백 축소
    s = re.sub(r'\s{2,}', ' ', s)
    return s

def _safe_multicell(pdf: FPDF, text: str, line_height: float = 7.0, width: float = None, wrap_chars: int = 100):
    """
    - 폭을 명시적으로 지정해 남은 폭이 0에 가까워지는 상황 방지
    - 긴 토큰을 정규화하고, 실패 시 글자수 기준 하드 래핑으로 폴백
    """
    if width is None:
        width = _usable_width(pdf)

    # 항상 좌측 마진으로 위치 초기화
    pdf.set_x(pdf.l_margin)

    # 1차: 정상 출력 시도
    try:
        norm = _normalize_long_tokens(text)
        # textwrap으로 1차 래핑(한 줄 최대 글자수 기준)
        wrapped = textwrap.fill(norm, width=wrap_chars)
        pdf.multi_cell(width, line_height, wrapped)
        return
    except Exception:
        pass

    # 2차: 폰트 크기 1pt 낮춰 재시도
    cur_family, cur_style, cur_size = pdf.font_family, pdf.font_style, pdf.font_size_pt
    try:
        if cur_size > 8:
            pdf.set_font(cur_family, cur_style, cur_size - 1)
        pdf.set_x(pdf.l_margin)
        wrapped = textwrap.fill(_normalize_long_tokens(text), width=max(60, wrap_chars - 20))
        pdf.multi_cell(width, line_height, wrapped)
        return
    except Exception:
        pass
    finally:
        # 폰트 복구
        pdf.set_font(cur_family, cur_style, cur_size)

    # 3차: 하드 슬라이스(강제 청크 분할)
    for i in range(0, len(text), 100):
        pdf.set_x(pdf.l_margin)
        chunk = text[i:i+100]
        pdf.multi_cell(width, line_height, chunk)

def create_pdf_report(report_data, company_name="중소기업"):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    # 폰트 설정
    if _try_add_font(pdf):
        base_font = "Nanum"
        title_size = 20
        h1_size = 14
        body_size = 12
    else:
        # fallback (영문 전용)
        base_font = "Arial"
        title_size = 16
        h1_size = 13
        body_size = 11

    # 제목
    pdf.set_font(base_font, "", title_size)
    pdf.set_x(pdf.l_margin)
    pdf.cell(_usable_width(pdf), 10, f"{company_name} 보안 분석 보고서", 0, 1, "C")
    pdf.ln(6)

    # 1. 요약
    pdf.set_font(base_font, "", h1_size)
    pdf.set_x(pdf.l_margin)
    pdf.cell(_usable_width(pdf), 10, "1. 요약 정보", 0, 1)
    pdf.set_font(base_font, "", body_size)
    _safe_multicell(pdf, report_data.get("summary", ""), line_height=7.0, width=_usable_width(pdf), wrap_chars=100)

    # 2. 키워드
    pdf.ln(5)
    pdf.set_font(base_font, "", h1_size)
    pdf.set_x(pdf.l_margin)
    pdf.cell(_usable_width(pdf), 10, "2. 주요 키워드", 0, 1)
    pdf.set_font(base_font, "", body_size)
    for kw in report_data.get("keywords", []):
        keyword = str(kw.get("keyword", ""))
        level = kw.get("risk_level") or kw.get("interest_level") or ""
        freq  = kw.get("frequency", "")
        line = f"- {keyword} | 레벨: {level} | 빈도: {freq}"
        _safe_multicell(pdf, line, line_height=7.0, width=_usable_width(pdf), wrap_chars=80)

    # 3. 대응 플레이북
    pdf.ln(5)
    pdf.set_font(base_font, "", h1_size)
    pdf.set_x(pdf.l_margin)
    pdf.cell(_usable_width(pdf), 10, "3. AI 생성 대응 플레이북", 0, 1)
    pdf.set_font(base_font, "", body_size)
    _safe_multicell(pdf, report_data.get("playbook", ""), line_height=7.0, width=_usable_width(pdf), wrap_chars=100)

    # 바이트 반환 (Streamlit download_button에 바로 사용 가능)
    out = pdf.output(dest="S")
    if isinstance(out, str):
        out = out.encode("latin1", errors="ignore")
    return bytes(out)


# ============================================================
# 6) 사이드바 (환경 설정 / RSS 전용)
# ============================================================
with st.sidebar:
    st.markdown("""
    <div class="sidebar-logo">
        <h2>🛡️ SecureWatch</h2>
        <p>중소기업 보안 관심/위험 분석</p>
    </div>
    """, unsafe_allow_html=True)

    st.subheader("🏢 기업 정보 설정")
    company_size = st.selectbox("기업 규모", ["소규모 (10-50명)", "중소규모 (50-200명)", "중규모 (200-500명)"])
    industry_type = st.selectbox("업종", ["IT/소프트웨어", "제조업", "금융업", "의료업", "교육업", "기타"])

    st.subheader("🌐 인프라 및 제약사항")
    infrastructure = st.selectbox("인프라 환경", ["AWS", "Azure", "GCP", "On-premise", "Hybrid"])
    constraints = st.text_area("보안 정책/예산 등 제한사항", value="")
    user_interest = st.text_area("관심 분야 키워드(쉼표 구분)", value="")
    st.session_state.user_interest = user_interest

    st.divider()
    if st.button("🔍 분석 시작", type="primary"):
        st.session_state.analysis_started = True
        st.session_state.news_data = []
        st.session_state.risk_keywords = []
        st.session_state.playbook_content = ""
        st.session_state.report_summary = ""
        st.session_state.llm_selected_keywords = []

        with st.spinner("RSS에서 뉴스 수집 중..."):
            articles = fetch_latest_news_by_rss()

        # 분석
        with st.spinner("분석/키워드 추출 중..."):
            news_data = []
            keyword_counts = {}
            for art in articles:
                combined = f"{art['title']} {art['content']}"
                risk_level, kws, score = analyze_risk_with_model(combined, industry_type)
                for k in kws:
                    keyword_counts[k] = keyword_counts.get(k, 0) + 1
                news_data.append({
                    "title": art['title'],
                    "summary": art['content'][:250] + "..." if len(art['content']) > 250 else art['content'],
                    "full_content": art['content'],
                    "source": art.get('source', '보안뉴스'),
                    "published": art.get('date', ''),
                    "risk_level": risk_level,
                    "risk_score": score,
                    "keywords": kws,
                    "url": art['url'],
                    "summary_llm": "" # LLM 요약 결과를 저장할 필드 추가
                })
            user_interest_list = [kw.strip() for kw in user_interest.split(',') if kw.strip()]
            for uk in user_interest_list:
                keyword_counts[uk] = keyword_counts.get(uk, 0) + 1
            st.session_state.news_data = sorted(news_data, key=lambda x: x['risk_score'], reverse=True)
            st.session_state.risk_keywords = [
                {"keyword": kw, "frequency": cnt, "risk_level": analyze_risk_with_model(kw, industry_type)[0]}
                for kw, cnt in sorted(keyword_counts.items(), key=lambda x: x[1], reverse=True)
            ]

        # 상위 뉴스 1줄 요약(플레이북에 전달할 브리프)
        with st.spinner("상위 뉴스 요약 정리 중..."):
            top_for_brief = st.session_state.news_data[:6]
            news_briefs = []
            for n in top_for_brief:
                brief = f"{n['title']} | 키워드: {', '.join(n['keywords'][:3])} | 관심도 {n['risk_level']}"
                news_briefs.append(brief)

        with st.spinner("LLM 플레이북 생성 중..."):
            company_name = "중소기업"
            company_info = {"name": company_name, "size": company_size, "industry": industry_type}
            keywords_list = [k["keyword"] for k in st.session_state.risk_keywords]
            st.session_state.playbook_content = generate_playbook_with_llm(
                keywords_list, company_info, infrastructure, constraints, news_briefs=news_briefs
            )

        st.session_state.report_summary = f"총 {len(st.session_state.news_data)}개 뉴스 분석 완료."
        st.success("✅ 분석 완료! 아래 탭에서 결과를 확인하세요.")
        st.rerun()

# ============================================================
# 7) 메인 헤더
# ============================================================
st.markdown("""
<div class="main-header">
    <h1>🛡️ 중소기업 보안 관심/위험 분석 시스템</h1>
    <p>AI 기반 보안 위협 키워드 추출 및 맞춤형 대응 플레이북 생성</p>
</div>
""", unsafe_allow_html=True)

# ============================================================
# 8) 탭: 대시보드/뉴스/플레이북
# ============================================================
tab1, tab2, tab3 = st.tabs([
    "📊 대시보드", "📰 뉴스 분석", "📋 대응 플레이북"
])

# --- 대시보드
with tab1:
    st.header("📊 보안 관심/위험 대시보드")
    if 'analysis_started' not in st.session_state:
        st.info("👈 사이드바에서 기업 정보를 설정하고 '분석 시작'을 눌러주세요.")
    else:
        top_news = st.session_state.news_data[:2]
        if not top_news:
            st.info("분석된 뉴스가 없습니다.")
        else:
            for current in top_news:
                css_class = "risk-high" if current["risk_level"] == "높음" else "risk-medium" if current["risk_level"] == "중간" else "risk-low"
                
                # 대시보드에서 요약 기능 제거 (대신 뉴스 분석 탭에서 제공)
                summary_content = current['summary'] + " (더 자세한 요약은 '뉴스 분석' 탭에서 확인하세요.)"
    
                st.markdown(f"""
                <div class="news-item {css_class}">
                    <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:1rem;">
                        <h5 style="margin:0;color:#2c3e50;">
                            <a href="{current['url']}" target="_blank">{current['title']}</a>
                        </h5>
                        <span style="background:{'#e74c3c' if current['risk_level']=='높음' else '#f39c12' if current['risk_level']=='중간' else '#27ae60'};color:white;padding:0.3rem 0.8rem;border-radius:15px;font-size:0.8rem;font-weight:bold;white-space:nowrap;">
                            관심도: {current['risk_level']} ({current['risk_score']:.2f})
                        </span>
                    </div>
                    <p style="color:#555; margin-bottom:1rem; white-space: pre-wrap;">{summary_content}</p>
                    <div style="color:#888; font-size:0.9rem;">
                        <strong>키워드:</strong> {', '.join(current['keywords'])} | {current['source']} | {current['published']}
                    </div>
                </div>
                """, unsafe_allow_html=True)
                st.markdown("---") # 항목 간 구분선

# --- 뉴스 분석
with tab2:
    st.header("📰 최신 보안 뉴스 분석")
    if 'analysis_started' not in st.session_state:
        st.info("👈 사이드바에서 기업 정보를 설정하고 '분석 시작'을 눌러주세요.")
    else:
        # 관심 키워드 우선 정렬 제거, 기본 위험도 순 정렬 유지
        sorted_news = st.session_state.news_data
        
        page_size = 10
        total = len(sorted_news)
        total_pages = (total - 1) // page_size + 1 if total > 0 else 1
        if 'current_page' not in st.session_state:
            st.session_state.current_page = 1
        start = (st.session_state.current_page - 1) * page_size
        end = start + page_size
        page_data = sorted_news[start:end]
        
        for idx, news in enumerate(page_data):
            css_class = "risk-high" if news["risk_level"] == "높음" \
                else "risk-medium" if news["risk_level"] == "중간" else "risk-low"
            
            # 하나의 markdown 블록으로 전체 뉴스 항목 표시
            st.markdown(f"""
            <div class="news-item {css_class}">
                <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:1rem;">
                    <h5 style="margin:0;color:#2c3e50;">
                        <a href="{news['url']}" target="_blank">{news['title']}</a>
                    </h5>
                    <span style="background:{'#e74c3c' if news['risk_level']=='높음' else '#f39c12' if news['risk_level']=='중간' else '#27ae60'};color:white;padding:0.3rem 0.8rem;border-radius:15px;font-size:0.8rem;font-weight:bold;white-space:nowrap;">
                        관심도: {news['risk_level']} ({news['risk_score']:.2f})
                    </span>
                </div>
                <p style="color:#555; margin-bottom:1rem; white-space: pre-wrap;">{news['summary']}</p>
                <div style="color:#888; font-size:0.9rem;">
                    <strong>키워드:</strong> {', '.join(news['keywords'])} | {news['source']} | {news['published']}
                </div>
            </div>
            """, unsafe_allow_html=True)
            
        st.divider()
        c1, c2, c3, c4, c5 = st.columns(5)
        with c1:
            if st.button("<< 처음", key='p_first'): st.session_state.current_page = 1
        with c2:
            if st.button("< 이전", key='p_prev', disabled=st.session_state.current_page == 1): st.session_state.current_page -= 1
        with c4:
            if st.button("다음 >", key='p_next', disabled=st.session_state.current_page == total_pages): st.session_state.current_page += 1
        with c5:
            if st.button("마지막 >>", key='p_last'): st.session_state.current_page = total_pages
        with c3:
            st.markdown(f"<p style='text-align:center; font-weight:bold;'>{st.session_state.current_page} / {total_pages}</p>", unsafe_allow_html=True)

# --- 대응 플레이북
with tab3:
    if 'analysis_started' not in st.session_state:
        st.info("👈 사이드바에서 '분석 시작'을 눌러주세요.")
    else:
        # PDF 다운로드 버튼을 헤더 옆에 배치
        col1, col2 = st.columns([0.7, 0.3])
        with col1:
            st.markdown("### 📋 AI 생성 대응 플레이북")
        with col2:
            st.markdown('<div style="height: 1.5rem;"></div>', unsafe_allow_html=True) # 헤더와 높이 맞추기
            report_data = {
                "summary": st.session_state.report_summary,
                "keywords": st.session_state.risk_keywords,
                "playbook": st.session_state.playbook_content
            }
            pdf_output = create_pdf_report(report_data, "중소기업")
            st.download_button(
                label="📄 PDF 다운로드",
                data=pdf_output,
                file_name=f"보안_분석_보고서_중소기업.pdf",
                mime="application/pdf"
            )

        st.markdown(
            f"""<div class="recommendation-box">
            <p style="white-space: pre-wrap;">{st.session_state.playbook_content}</p></div>""",
            unsafe_allow_html=True
        )
        if st.session_state.llm_selected_keywords:
            st.subheader("🧩 LLM이 선별한 중요 키워드(Top)")
            df_llm_kw = pd.DataFrame(st.session_state.llm_selected_keywords)
            st.dataframe(df_llm_kw, use_container_width=True)

# 푸터
st.divider()
st.markdown(
    """<div style="text-align:center;color:#888;padding:1rem;">
    <p>🛡️ SecureWatch - 중소기업 보안 관심/위험 분석 시스템</p>
    <p>AI 기반 위협 키워드 추출 및 대응 플레이북 생성</p>
    </div>""",
    unsafe_allow_html=True
)