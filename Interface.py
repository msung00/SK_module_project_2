import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
from bs4 import BeautifulSoup
from konlpy.tag import Okt
from fpdf import FPDF
import time
import torch
from transformers import ElectraForSequenceClassification, ElectraTokenizer
import google.generativeai as genai
import os
from dotenv import load_dotenv
from datetime import datetime
import re
import feedparser

# Load environment variables from .env file
load_dotenv()

# --- 뉴스 기사 스크래핑 모듈 통합 ---
def scrape_article(url):
    try:
        res = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=5)
        if res.status_code != 200:
            return None
        soup = BeautifulSoup(res.text, "html.parser")
        title = soup.select_one("#news_title02") or soup.select_one("h4.tit")
        body = soup.select_one("#news_content") or soup.select_one("div.view_txt")
        date = soup.select_one("#news_util01") or soup.select_one("span.date")
        return {
            "url": url,
            "title": title.get_text(strip=True) if title else "제목 없음",
            "date": date.get_text(strip=True) if date else datetime.now().strftime("%Y-%m-%d"),
            "content": body.get_text("\n", strip=True) if body else "내용 없음",
        }
    except Exception:
        return None

def fetch_latest_news():
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
    
    seen_urls = set()
    all_articles = []
    
    for feed_name, rss_url in rss_list:
        try:
            feed = feedparser.parse(rss_url)
        except Exception:
            continue
        
        for entry in getattr(feed, "entries", []):
            url = getattr(entry, "link", None)
            if not url or url in seen_urls:
                continue
            seen_urls.add(url)

            article_data = scrape_article(url)
            if article_data and article_data['title'] != '제목 없음':
                article_data["source"] = feed_name # 출처 정보 추가
                all_articles.append(article_data)
            
            time.sleep(0.1)
    
    return all_articles

# --- 1. KoELECTRA 모델 로딩 (가상 함수) ---
@st.cache_resource
def load_koelectra_model():
    model_name = "monologg/koelectra-base-v3-discriminator"
    tokenizer = ElectraTokenizer.from_pretrained(model_name)
    model = ElectraForSequenceClassification.from_pretrained(model_name, num_labels=5)
    # 실제 학습된 모델 가중치가 필요합니다.
    # model.load_state_dict(torch.load("path/to/your/model.pt"))
    return tokenizer, model

# --- 2. Gemini API 설정 ---
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    st.error("Gemini API 키가 설정되지 않았습니다. .env 파일을 확인해주세요.")
    st.stop()
    
genai.configure(api_key=GEMINI_API_KEY)
generation_config = {
    "temperature": 0.7,
    "max_output_tokens": 1000,
}
gemini_model = genai.GenerativeModel('gemini-1.5-flash', generation_config=generation_config)

# --- 3. 데이터 전처리 모듈 (가상 함수) ---
@st.cache_data
def preprocess_text(text):
    okt = Okt()
    preprocessed = [token for token, pos in okt.pos(text, norm=True, stem=True)
                    if pos in ['Noun', 'Verb', 'Adjective']]
    return " ".join(preprocessed)

# --- 4. KoELECTRA 기반 키워드/관심도 분석 (가상 함수 -> 키워드 기반 로직으로 대체) ---
def analyze_interest_with_model(text):
    # 관심도 키워드 점수표
    interest_keywords_map = {
        "ATTACK": {
            "랜섬웨어": 1.0, "제로데이": 1.0, "피싱": 0.9, "DDoS": 0.8, "악성코드": 0.9,
            "해킹": 0.9, "공격": 0.8, "침해": 0.8, "유출": 0.8, "탈취": 0.8
        },
        "VULN": {
            "취약점": 1.0, "CVE": 1.0, "버그": 0.9, "결함": 0.9, "보안 결함": 1.0
        },
        "ORG": {
            "마이크로소프트": 0.3, "구글": 0.3, "삼성": 0.2, "애플": 0.2, "정부": 0.1,
            "기업": 0.1, "기관": 0.1, "대학교": 0.1
        },
        "STRATEGY": {
            "제로 트러스트": 0.7, "방화벽": 0.6, "패치": 0.6, "백업": 0.5, "인증": 0.4,
            "모니터링": 0.5, "정책": 0.4
        }
    }
    
    extracted_keywords = []
    total_score = 0
    
    for category, keywords in interest_keywords_map.items():
        for keyword, score in keywords.items():
            if re.search(r'\b' + re.escape(keyword) + r'\b', text, re.I):
                extracted_keywords.append(keyword)
                total_score += score
    
    extracted_keywords = list(set(extracted_keywords))
    if total_score >= 2.0:
        interest_level = "높음"
    elif total_score >= 0.8:
        interest_level = "중간"
    else:
        interest_level = "낮음"
    
    return interest_level, extracted_keywords, total_score

# --- 5. LLM 기반 대응 방안 생성 및 요약 함수 (수정) ---
def generate_playbook_with_llm(keywords, company_info, infrastructure, constraints, user_interest):
    try:
        keywords_for_prompt = keywords[:10]
        
        prompt = f"""
        당신은 중소기업을 위한 보안 전문가입니다. 다음 정보를 기반으로 맞춤형 보안 대응 플레이북을 생성해주세요.
        - 기업명: {company_info['name']}
        - 기업 규모: {company_info['size']}
        - 업종: {company_info['industry']}
        - 인프라 환경: {infrastructure}
        - 주요 위협 키워드: {', '.join(keywords_for_prompt)}
        - 사용자의 관심 분야: {user_interest}
        - 고려해야 할 제한사항: {constraints}
        - 상세도 수준: 상세형

        위협에 대한 즉시 대응 조치와 중장기 대응 방안을 포함하고, 이해하기 쉬운 체크리스트 형식으로 정리해주세요.
        """
        response = genai.GenerativeModel('gemini-1.5-flash').generate_content(prompt)
        return response.text
    except Exception as e:
        # 이 부분을 수정하여 오류 내용을 터미널에 출력합니다.
        st.error(f"Gemini API 호출 중 오류가 발생했습니다.")
        print("="*50)
        print("Gemini API 호출 오류 상세 내용:")
        print(e)
        print("="*50)
        return "플레이북 생성에 실패했습니다. 터미널의 오류 내용을 확인해주세요."

@st.cache_data(show_spinner=False)
def generate_article_summary(article_title, article_content):
    prompt = f"""
    아래 뉴스 기사를 5~6줄로 간결하게 요약해 주세요.
    제목: {article_title}
    내용: {article_content}
    """
    try:
        response = genai.GenerativeModel('gemini-1.5-flash').generate_content(prompt)
        return response.text
    except Exception as e:
        return "요약 생성에 실패했습니다."

# --- 6. PDF 보고서 생성 모듈 ---
def create_pdf_report(report_data, company_name="중소기업"):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    
    pdf.add_font('NanumGothic', '', 'font/NanumGothic.ttf', uni=True)
    pdf.add_font('NanumGothicBold', '', 'font/NanumGothicBold.ttf', uni=True)

    pdf.set_font('NanumGothicBold', '', 20)
    pdf.cell(0, 10, '중소기업 보안 관심도 분석 보고서', 0, 1, 'C')
    pdf.ln(10)

    pdf.set_font('NanumGothicBold', '', 14)
    pdf.cell(0, 10, '1. 요약 정보', 0, 1)
    pdf.set_font('NanumGothic', '', 12)
    pdf.multi_cell(0, 7, report_data['summary'])
    pdf.ln(5)

    pdf.set_font('NanumGothicBold', '', 14)
    pdf.cell(0, 10, '2. 주요 관심 키워드', 0, 1)
    pdf.set_font('NanumGothic', '', 12)
    for kw in report_data['keywords']:
        pdf.multi_cell(0, 7, f"- {kw['keyword']}: {kw['interest_level']}")
    pdf.ln(5)
    
    pdf.set_font('NanumGothicBold', '', 14)
    pdf.cell(0, 10, '3. AI 생성 대응 플레이북', 0, 1)
    pdf.set_font('NanumGothic', '', 12)
    pdf.multi_cell(0, 7, report_data['playbook'])
    
    pdf_output = pdf.output(dest='S').encode('latin1')
    return pdf_output

# --- Streamlit UI 구성 ---
st.set_page_config(
    page_title="중소기업 보안 관심도 분석 시스템",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

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
    /* 버튼 배경색상 변경 */
    .stButton > button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 5px;
        padding: 0.5rem 1rem;
        font-weight: 600;
        width: 100%;
    }
    /* 버튼 텍스트가 잘 보이도록 조정 */
    .stButton > button > div > p {
        color: white !important;
        font-weight: bold;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
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

# --- 사이드바 ---
with st.sidebar:
    st.markdown("""
    <div class="sidebar-logo">
        <h2>🛡️ SecureWatch</h2>
        <p>중소기업 보안 관심도 분석</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.subheader("🏢 기업 정보 설정")
    company_name = "중소기업"
    company_size = st.selectbox("기업 규모", ["소규모 (10-50명)", "중소규모 (50-200명)", "중규모 (200-500명)"])
    industry_type = st.selectbox("업종", ["IT/소프트웨어", "제조업", "금융업", "의료업", "교육업", "기타"])
    
    st.subheader("🌐 인프라 및 제약사항")
    infrastructure = st.selectbox("인프라 환경", ["AWS", "Azure", "GCP", "On-premise", "Hybrid"])
    constraints = st.text_area("보안 정책/예산 등 제한사항", value="")
    user_interest = st.text_area("관심 분야 (예: 클라우드 보안, 개인정보보호)", value="")
    
    st.divider()
    
    if st.button("🔍 분석 시작", type="primary"):
        st.session_state.analysis_started = True
        st.session_state.news_data = []
        st.session_state.risk_keywords = []
        st.session_state.playbook_content = ""
        st.session_state.report_summary = ""

        with st.spinner("최신 보안 뉴스를 수집 및 분석 중입니다..."):
            scraped_articles = fetch_latest_news()
            
            st.session_state.all_articles = scraped_articles
            
            st.session_state.news_data = []
            keyword_counts = {}
            for news in scraped_articles:
                combined_text = news['title'] + " " + news['content']
                interest_level, keywords, interest_score = analyze_interest_with_model(combined_text)
                
                for kw in keywords:
                    keyword_counts[kw] = keyword_counts.get(kw, 0) + 1
                
                news_item = {
                    "title": news['title'],
                    "summary": news['content'][:250] + "..." if len(news['content']) > 250 else news['content'],
                    "full_content": news['content'],
                    "source": news.get('source', '보안뉴스'),
                    "published": news['date'],
                    "interest_score": interest_score,
                    "keywords": keywords,
                    "interest_level": interest_level,
                    "url": news['url']
                }
                st.session_state.news_data.append(news_item)
            
            user_interest_list = [kw.strip() for kw in user_interest.split(',') if kw.strip()]
            for user_kw in user_interest_list:
                keyword_counts[user_kw] = keyword_counts.get(user_kw, 0) + 1

            st.session_state.risk_keywords = [
                {"keyword": kw, "frequency": count, "interest_level": analyze_interest_with_model(kw)[0]}
                for kw, count in keyword_counts.items()
            ]
            st.session_state.news_data = sorted(st.session_state.news_data, key=lambda x: x['interest_score'], reverse=True)

        with st.spinner("AI가 맞춤형 플레이북을 생성하고 있습니다..."):
            company_info = {"name": company_name, "size": company_size, "industry": industry_type}
            keywords_list = list(keyword_counts.keys())
            
            st.session_state.playbook_content = generate_playbook_with_llm(
                keywords_list, company_info, infrastructure, constraints, user_interest
            )

        st.session_state.report_summary = f"총 {len(st.session_state.news_data)}개의 뉴스를 스크래핑하고 분석했습니다."
        
        st.success("✅ 분석 완료! 아래 탭에서 결과를 확인하세요.")
        st.rerun()

# 메인 헤더
st.markdown("""
<div class="main-header">
    <h1>🛡️ 중소기업 보안 관심도 분석 시스템</h1>
    <p>AI 기반 실시간 보안 위협 분석 및 맞춤형 대응 방안 제공</p>
</div>
""", unsafe_allow_html=True)

# 메인 탭 구성
tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 대시보드", "📰 뉴스 분석", "🎯 관심 키워드", "📋 대응 플레이북", "📄 보고서 다운로드"])

# --- 대시보드 탭 ---
with tab1:
    st.header("📊 보안 관심도 대시보드")
    if 'analysis_started' not in st.session_state:
        st.info("👈 사이드바에서 기업 정보를 설정하고 '분석 시작' 버튼을 눌러주세요.")
    else:
        st.markdown("<h3 style='margin-top:2rem; font-size: 2rem;'>주요 보안 뉴스</h3>", unsafe_allow_html=True)
        
        top_news_to_display = st.session_state.news_data[:2]
        
        if not top_news_to_display:
            st.info("현재 분석된 뉴스가 없습니다. '분석 시작' 버튼을 눌러주세요.")
        else:
            if 'current_news_idx' not in st.session_state or st.session_state.current_news_idx >= len(top_news_to_display):
                st.session_state.current_news_idx = 0
            
            current_news_item = top_news_to_display[st.session_state.current_news_idx]

            article_summary = generate_article_summary(current_news_item['title'], current_news_item['full_content'])
            
            interest_class = "risk-high" if current_news_item["interest_level"] == "높음" else "risk-medium" if current_news_item["interest_level"] == "중간" else "risk-low"
            st.markdown(f"""
            <div class="news-item {interest_class}">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 1rem;">
                        <h5 style="margin: 0; color: #2c3e50;"><a href="{current_news_item['url']}" target="_blank">{current_news_item['title']}</a></h5>
                        <span style="background: {'#e74c3c' if current_news_item['interest_level'] == '높음' else '#f39c12' if current_news_item['interest_level'] == '중간' else '#27ae60'}; color: white; padding: 0.3rem 0.8rem; border-radius: 15px; font-size: 0.8rem; font-weight: bold;">
                            관심도: {current_news_item['interest_level']} ({current_news_item['interest_score']:.2f})
                        </span>
                </div>
                <p style="color: #555; margin-bottom: 1rem; white-space: pre-wrap;">{article_summary}</p>
                <div style="color: #888; font-size: 0.9rem;">
                        <strong>키워드:</strong> {', '.join(current_news_item['keywords'])} | {current_news_item['published']}
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            col_prev, col_next = st.columns([1, 1])
            with col_prev:
                if st.button("< 이전", key='tab1_prev_btn', disabled=st.session_state.current_news_idx == 0):
                    st.session_state.current_news_idx -= 1
            with col_next:
                if st.button("다음 >", key='tab1_next_btn', disabled=st.session_state.current_news_idx == len(top_news_to_display) - 1):
                    st.session_state.current_news_idx += 1

# --- 뉴스 분석 탭 ---
with tab2:
    st.header("📰 최신 보안 뉴스 분석")
    if 'analysis_started' not in st.session_state:
        st.info("👈 사이드바에서 '분석 시작' 버튼을 눌러주세요.")
    else:
        interest_keywords = [kw.strip().lower() for kw in user_interest.split(',') if kw.strip()]
        
        sorted_news_data = sorted(
            st.session_state.news_data,
            key=lambda news: (
                sum(1 for kw in interest_keywords if kw in news['title'].lower() or kw in news['full_content'].lower()),
                news['interest_score']
            ),
            reverse=True
        )

        if interest_keywords:
            st.info(f"입력하신 관심 키워드('{' '.join(interest_keywords)}')를 포함하는 뉴스가 우선적으로 정렬되었습니다.")
        else:
            st.info("현재는 전체 뉴스 기사를 관심도 점수 순으로 정렬했습니다. 관심 분야에 키워드를 입력하면 해당 뉴스가 우선적으로 표시됩니다.")

        page_size = 10
        total_articles = len(sorted_news_data)
        total_pages = (total_articles - 1) // page_size + 1 if total_articles > 0 else 1

        if 'current_page' not in st.session_state:
            st.session_state.current_page = 1
        
        start_idx = (st.session_state.current_page - 1) * page_size
        end_idx = start_idx + page_size
        paged_news_data = sorted_news_data[start_idx:end_idx]
        
        st.divider()

        if not paged_news_data:
            st.info("선택한 필터에 해당하는 뉴스가 없습니다.")
        else:
            for news in paged_news_data:
                interest_class = "risk-high" if news["interest_level"] == "높음" else "risk-medium" if news["interest_level"] == "중간" else "risk-low"
                
                st.markdown(f"""
                <div class="news-item {interest_class}">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 1rem;">
                        <h3 style="margin: 0; color: #2c3e50;"><a href="{news['url']}" target="_blank">{news['title']}</a></h3>
                        <span style="background: {'#e74c3c' if news['interest_level'] == '높음' else '#f39c12' if news['interest_level'] == '중간' else '#27ae60'}; color: white; padding: 0.3rem 0.8rem; border-radius: 15px; font-size: 0.8rem; font-weight: bold;">
                            관심도: {news['interest_level']} ({news['interest_score']:.2f})
                        </span>
                    </div>
                    <p style="color: #555; margin-bottom: 1rem;">{news['summary']}</p>
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <div>
                            <strong>키워드:</strong> {', '.join(news['keywords'])}
                        </div>
                        <div style="color: #888; font-size: 0.9rem;">
                            {news['source']} | {news['published']}
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                with st.expander("🔍 AI 분석 결과"):
                    st.markdown(f"""
                    **관심 분석:**- 이 뉴스는 **{news['interest_level']} 관심**을 나타냅니다. (키워드 기반 분석)- 주요 키워드: {', '.join(news['keywords'][:2])}- 이 분석은 키워드 점수표를 기반으로 합니다.
                    """)
        
        st.divider()
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            if st.button("<< 처음", key='tab2_first_btn'): st.session_state.current_page = 1
        with col2:
            if st.button("< 이전", key='tab2_prev_btn', disabled=st.session_state.current_page == 1): st.session_state.current_page -= 1
        with col4:
            if st.button("다음 >", key='tab2_next_btn', disabled=st.session_state.current_page == total_pages): st.session_state.current_page += 1
        with col5:
            if st.button("마지막 >>", key='tab2_last_btn'): st.session_state.current_page = total_pages
        with col3:
            st.markdown(f"<p style='text-align:center; font-weight:bold;'>{st.session_state.current_page} / {total_pages}</p>", unsafe_allow_html=True)

# --- 관심 키워드 탭 ---
with tab3:
    st.header("🎯 AI 추출 관심 키워드")
    if 'analysis_started' not in st.session_state:
        st.info("👈 사이드바에서 '분석 시작' 버튼을 눌러주세요.")
    else:
        st.subheader("📊 실시간 통계")
        col1, col2 = st.columns(2)
        high_interest_kw = sum(1 for kw in st.session_state.risk_keywords if kw['interest_level'] == '높음')
        with col1:
            st.metric("총 추출 키워드", f"{len(st.session_state.risk_keywords)}개")
        with col2:
            st.metric("높은 관심도 키워드", f"{high_interest_kw}개")
        st.divider()
        def get_interest_color(interest_level):
            colors = {"높음": "🔴", "중간": "🟡", "낮음": "🟢"}
            return colors.get(interest_level, "⚪")
        st.subheader("🔍 추출된 관심 키워드")
        for idx, kw in enumerate(st.session_state.risk_keywords):
            col1, col2, col3 = st.columns([3, 1, 1])
            with col1: st.markdown(f"**{kw['keyword']}** {get_interest_color(kw['interest_level'])}")
            with col2: st.markdown(f"빈도: **{kw['frequency']}**")
            with col3: st.markdown(f"관심도: **{kw['interest_level']}**")

# --- 대응 플레이북 탭 ---
with tab4:
    st.header("📋 AI 생성 대응 플레이북")
    if 'analysis_started' not in st.session_state:
        st.info("👈 사이드바에서 '분석 시작' 버튼을 눌러주세요.")
    else:
        st.markdown(f"""<div class="recommendation-box"><h3>🛡️ 맞춤형 대응 플레이북</h3><p style="white-space: pre-wrap;">{st.session_state.playbook_content}</p></div>""", unsafe_allow_html=True)
        
# --- 보고서 다운로드 탭 ---
with tab5:
    st.header("📄 분석 보고서 다운로드")
    if 'analysis_started' not in st.session_state:
        st.info("👈 사이드바에서 '분석 시작' 버튼을 눌러 보고서를 생성해주세요.")
    else:
        st.write("분석 결과를 PDF 보고서로 다운로드할 수 있습니다.")
        report_data = {
            "summary": st.session_state.report_summary,
            "keywords": st.session_state.risk_keywords,
            "playbook": st.session_state.playbook_content
        }
        pdf_output = create_pdf_report(report_data, company_name)
        st.download_button(label="📄 PDF 보고서 다운로드", data=pdf_output, file_name=f"보안_관심도_분석_보고서_{company_name}.pdf", mime="application/pdf")

# 하단 정보
st.divider()
st.markdown("""<div style="text-align: center; color: #888; padding: 2rem;"><p>🛡️ SecureWatch - 중소기업 보안 관심도 분석 시스템</p><p>AI 기반 실시간 보안 위협 분석 및 맞춤형 대응 방안 제공 | Made by 팔색조</p></div>""", unsafe_allow_html=True)