import streamlit as st
import pandas as pd
import json
from datetime import datetime

# 모듈 임포트
from config import *
from news_scraper import fetch_latest_news_by_rss
from ner_analyzer import load_ner_model, update_keywords_from_cisa, analyze_risk_with_model, industry_risk_map
from llm_generator import generate_playbook_with_llm, fetch_headlines_for_summary, generate_dashboard_summary
from pdf_reporter import create_pdf_report
from database import *

# =... (main, render_sidebar, start_analysis, render_tabs, render_dashboard 함수는 이전과 동일) ...

def main():
    global ner_tokenizer, ner_model, ner_ctx, gemini_model
    st.set_page_config(**PAGE_CONFIG)
    st.markdown(CSS_STYLES, unsafe_allow_html=True)
    
    if not GEMINI_API_KEY:
        st.error("Gemini API 키가 설정되지 않았습니다. .env 파일을 확인해주세요.")
        st.stop()
    
    import google.generativeai as genai
    genai.configure(api_key=GEMINI_API_KEY)
    gemini_model = genai.GenerativeModel('gemini-1.5-flash', generation_config=GENERATION_CONFIG)
    
    ner_tokenizer, ner_model, ner_ctx = load_ner_model()
    update_keywords_from_cisa(industry_risk_map)
    init_db()
    
    if 'analysis_started' not in st.session_state:
        st.session_state.analysis_started = False
        st.session_state.news_data = []
        st.session_state.risk_keywords = []
        st.session_state.playbook_content = ""
        st.session_state.report_summary = ""
        st.session_state.llm_selected_keywords = []
        st.session_state.dashboard_summary = ""
        st.session_state.company_name = "중소기업"
        st.session_state.company_size = COMPANY_SIZE_OPTIONS[0]
        st.session_state.industry_type = INDUSTRY_OPTIONS[0]
        st.session_state.infrastructure = INFRASTRUCTURE_OPTIONS[0]
        st.session_state.constraints = ""
        st.session_state.user_interest = ""
        st.session_state.current_page = 1

    query_params = st.query_params
    if "delete_news_id" in query_params:
        delete_news_from_favorites(query_params["delete_news_id"])
        del st.query_params["delete_news_id"]
        st.rerun()
    elif "delete_playbook_id" in query_params:
        delete_playbook_from_favorites(query_params["delete_playbook_id"])
        del st.query_params["delete_playbook_id"]
        st.rerun()

    render_sidebar()
    
    st.markdown("""
    <div class="main-header">
        <h1>🛡️ 중소기업 보안 관심/위험 분석 시스템</h1>
        <p>AI 기반 보안 위협 키워드 추출 및 맞춤형 대응 플레이북 생성</p>
    </div>
    """, unsafe_allow_html=True)
    
    render_tabs()

def render_sidebar():
    with st.sidebar:
        st.markdown("""
        <div class="sidebar-logo">
            <h2>🛡️ SecureWatch</h2>
            <p>중소기업 보안 관심/위험 분석</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.subheader("🏢 기업 정보 설정")
        st.session_state.company_name = st.text_input("기업명", value=st.session_state.company_name, key='sidebar_company_name')
        st.session_state.company_size = st.selectbox("기업 규모", COMPANY_SIZE_OPTIONS, index=COMPANY_SIZE_OPTIONS.index(st.session_state.company_size), key='sidebar_company_size_select')
        st.session_state.industry_type = st.selectbox("업종", INDUSTRY_OPTIONS, index=INDUSTRY_OPTIONS.index(st.session_state.industry_type), key='sidebar_industry_type_select')
        
        st.subheader("🌐 인프라 및 제약사항")
        st.session_state.infrastructure = st.selectbox("인프라 환경", INFRASTRUCTURE_OPTIONS, index=INFRASTRUCTURE_OPTIONS.index(st.session_state.infrastructure), key='sidebar_infrastructure_select')
        st.session_state.constraints = st.text_area("보안 정책/예산 등 제한사항", value=st.session_state.constraints, key='sidebar_constraints')
        st.session_state.user_interest = st.text_area("관심 분야 키워드(쉼표 구분)", value=st.session_state.user_interest, key='sidebar_user_interest')
        
        st.divider()
        if st.button("🔍 분석 시작", type="primary"):
            start_analysis()

def start_analysis():
    global ner_tokenizer, ner_model, ner_ctx, gemini_model
    
    st.session_state.analysis_started = True
    st.session_state.news_data = []
    st.session_state.risk_keywords = []
    st.session_state.playbook_content = ""
    st.session_state.report_summary = ""
    st.session_state.llm_selected_keywords = []
    st.session_state.current_page = 1
    
    with st.spinner("RSS에서 뉴스 수집 중..."):
        articles = fetch_latest_news_by_rss()
    
    with st.spinner("분석/키워드 추출 중..."):
        news_data = []
        keyword_counts = {}
        for art in articles:
            combined = f"{art['title']} {art['content']}"
            risk_level, kws, score = analyze_risk_with_model(combined, st.session_state.industry_type, ner_tokenizer, ner_model, ner_ctx)
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
                "url": art['url']
            })
        user_interest_list = [kw.strip() for kw in st.session_state.user_interest.split(',') if kw.strip()]
        for uk in user_interest_list:
            keyword_counts[uk] = keyword_counts.get(uk, 0) + 1
        st.session_state.news_data = sorted(news_data, key=lambda x: x['risk_score'], reverse=True)
        st.session_state.risk_keywords = [
            {"keyword": kw, "frequency": cnt, "risk_level": analyze_risk_with_model(kw, st.session_state.industry_type, ner_tokenizer, ner_model, ner_ctx)[0]}
            for kw, cnt in sorted(keyword_counts.items(), key=lambda x: x[1], reverse=True)
        ]

    with st.spinner("LLM 플레이북 생성 중..."):
        try:
            company_info = {"name": st.session_state.company_name, "size": st.session_state.company_size, "industry": st.session_state.industry_type}
            keywords_list = [k["keyword"] for k in st.session_state.risk_keywords]
            playbook_content, llm_selected_keywords = generate_playbook_with_llm(
                keywords_list, company_info, st.session_state.infrastructure, st.session_state.constraints, gemini_model, news_briefs=None
            )
            st.session_state.playbook_content = playbook_content
            st.session_state.llm_selected_keywords = llm_selected_keywords
        except Exception as e:
            error_msg = str(e)
            if "429" in error_msg or "quota" in error_msg.lower():
                st.error("⚠️ Gemini API 할당량이 초과되었습니다.")
                st.info("기본 템플릿으로 플레이북을 생성합니다.")
            else:
                st.error(f"플레이북 생성 중 오류가 발생했습니다: {error_msg}")
                st.session_state.playbook_content = "플레이북 생성에 실패했습니다."
                st.session_state.llm_selected_keywords = []

    with st.spinner("대시보드 요약 생성 중..."):
        dashboard_rss_url = "http://www.boannews.com/media/news_rss.xml?skind=5"
        headlines = fetch_headlines_for_summary(dashboard_rss_url)
        if headlines:
            company_info = {"name": st.session_state.company_name, "size": st.session_state.company_size, "industry": st.session_state.industry_type}
            summary_text = generate_dashboard_summary(
                headlines, company_info, st.session_state.infrastructure, st.session_state.constraints, gemini_model
            )
            st.session_state.dashboard_summary = summary_text
        else:
            st.session_state.dashboard_summary = "최신 보안 동향 요약 정보를 가져오는 데 실패했습니다."

    st.session_state.report_summary = f"총 {len(st.session_state.news_data)}개 뉴스 분석 완료."
    st.success("✅ 분석 완료! 아래 탭에서 결과를 확인하세요.")
    st.rerun()

def render_tabs():
    tab1, tab2, tab3, tab4 = st.tabs(["📊 대시보드", "📰 뉴스 분석", "📋 대응 플레이북", "⭐ 즐겨찾기"])
    with tab1: render_dashboard()
    with tab2: render_news_analysis()
    with tab3: render_playbook()
    with tab4: render_favorites()

def render_dashboard():
    if not st.session_state.analysis_started:
        st.info("👈 사이드바에서 기업 정보를 설정하고 '분석 시작'을 눌러주세요.")
    else:
        st.markdown(f"""
        <div class="welcome-box">
            <h3 style='margin: 0;'>{st.session_state.company_name}님, 환영합니다.</h3>
            <h4 style='margin: 10px 0 0;'>AI가 분석한 최신 보안 동향 및 권장 조치는 다음과 같습니다.</h4>
        </div>
        """, unsafe_allow_html=True)
        if st.session_state.dashboard_summary:
            st.markdown(f"""
            <div class="summary-box">
                <h4>📈 오늘의 보안 동향 요약 및 권장 조치</h4>
                <p>{st.session_state.dashboard_summary}</p>
            </div>
            """, unsafe_allow_html=True)

# --- 이 함수가 전체적으로 수정되었습니다 ---
def render_news_analysis():
    """뉴스 분석 탭 렌더링 (안정적인 3열 카드 레이아웃)"""
    st.header("📰 최신 보안 뉴스 분석")
    if not st.session_state.analysis_started:
        st.info("👈 사이드바에서 기업 정보를 설정하고 '분석 시작'을 눌러주세요.")
        return

    # 카드 UI를 위한 CSS 스타일 정의
    st.markdown("""
    <style>
    .news-card-wrapper {
        padding: 1rem;
        border-radius: 12px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        /* --- 이 부분이 수정되었습니다 --- */
        margin-bottom: 0.5rem; /* 버튼과의 간격을 줄임 */
        height: 300px; /* 카드 높이를 소폭 줄임 */
        display: flex;
        flex-direction: column;
        /* justify-content: space-between; <- 이 속성을 제거하여 내용을 위로 정렬 */
        border-left: 5px solid #ccc;
    }
    .news-card-content {
        flex-grow: 1; /* 내용이 남는 공간을 채우도록 설정 */
    }
    .news-card-content h5 { font-size: 1rem; margin-bottom: 0.5rem; line-height: 1.3; }
    .news-card-content h5 a { text-decoration: none; color: inherit; }
    .news-card-content p {
        font-size: 0.9rem; color: #555; line-height: 1.4;
        display: -webkit-box; -webkit-line-clamp: 4; /* p태그는 4줄로 늘림 */
        -webkit-box-orient: vertical; overflow: hidden; text-overflow: ellipsis;
        margin-top: 0.5rem;
    }
    .news-card-footer .keywords { color:#888; font-size:0.8rem; }
    
    .risk-high { 
        border-left-color: #e74c3c;
        background-color: #fff5f5;
    }
    .risk-medium { 
        border-left-color: #f39c12;
        background-color: #fff8f0;
    }
    .risk-low { 
        border-left-color: #27ae60;
        background-color: #f6fcf6;
    }
    </style>
    """, unsafe_allow_html=True)

    news_data = st.session_state.news_data
    high_news = [n for n in news_data if n["risk_level"] == "높음"][:2]
    medium_news = [n for n in news_data if n["risk_level"] == "중간"][:2]
    low_news = [n for n in news_data if n["risk_level"] == "낮음"][:2]

    col1, col2, col3 = st.columns(3)

    def display_news_in_column(column, news_list, risk_level_str, risk_level_emoji, risk_level_color, key_prefix):
        with column:
            st.subheader(f"{risk_level_emoji} {risk_level_str}")
            if not news_list:
                st.info("관련 뉴스가 없습니다.")
            for i, news in enumerate(news_list):
                with st.container():
                    risk_css_class = f"risk-{key_prefix}"
                    st.markdown(f"""
                    <div class="news-card-wrapper {risk_css_class}">
                        <div class="news-card-content">
                            <h5><a href="{news['url']}" target="_blank">{news['title']}</a></h5>
                            <span style="background:{risk_level_color};color:white;padding:0.2rem 0.6rem;border-radius:10px;font-size:0.8rem;font-weight:bold;">
                                관심도: {news['risk_level']} ({news['risk_score']:.2f})
                            </span>
                            <p>{news['summary']}</p>
                        </div>
                        <div class="news-card-footer">
                            <div class="keywords"><strong>키워드:</strong> {', '.join(news['keywords'])}</div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    if st.button("💾 즐겨찾기 추가", key=f"save_{key_prefix}_{i}", use_container_width=True):
                        success, message = save_news_to_favorites(news)
                        if success:
                            st.toast(f"✅ '{news['title'][:20]}...' 저장 완료!")
                        else:
                            st.toast(f"⚠️ {message}")

    display_news_in_column(col1, high_news, "높음", "🔴", "#e74c3c", "high")
    display_news_in_column(col2, medium_news, "중간", "🟠", "#f39c12", "medium")
    display_news_in_column(col3, low_news, "낮음", "🟢", "#27ae60", "low")

def render_playbook():
    if not st.session_state.analysis_started:
        st.info("👈 사이드바에서 '분석 시작'을 눌러주세요.")
    else:
        col1, col2 = st.columns([0.7, 0.3])
        with col1:
            st.markdown("### 📋 AI 생성 대응 플레이북")
        with col2:
            st.markdown('<div style="height: 1.5rem;"></div>', unsafe_allow_html=True)
            report_data = {
                "summary": st.session_state.report_summary,
                "keywords": st.session_state.risk_keywords,
                "playbook": st.session_state.playbook_content
            }
            pdf_output = create_pdf_report(report_data, st.session_state.company_name)
            st.download_button(
                label="📄 PDF 다운로드",
                data=pdf_output,
                file_name=f"보안_분석_보고서_{st.session_state.company_name}.pdf",
                mime="application/pdf",
                key="playbook_pdf_download"
            )
        
        if st.button("⭐ 플레이북 즐겨찾기", key="save_playbook_btn"):
            success, message = save_playbook_to_favorites(
                "AI 생성 대응 플레이북",
                st.session_state.playbook_content,
                st.session_state.report_summary,
                st.session_state.llm_selected_keywords
            )
            if success:
                st.success(message)
            else:
                st.warning(message)
            
        st.markdown(
            f"""<div class="recommendation-box">
            <p style="white-space: pre-wrap;">{st.session_state.playbook_content.replace('<br>', '\n')}</p></div>""",
            unsafe_allow_html=True
        )
        if st.session_state.llm_selected_keywords:
            st.subheader("🧩 LLM이 선별한 중요 키워드(Top)")
            df_llm_kw = pd.DataFrame(st.session_state.llm_selected_keywords)
            st.dataframe(df_llm_kw, use_container_width=True)

def render_favorites():
    st.header("⭐ 즐겨찾기")
    saved_news = get_saved_news()
    saved_playbooks = get_saved_playbooks()

    if not saved_news and not saved_playbooks:
        st.info("저장된 기사나 플레이북이 없습니다.")
    else:
        st.subheader("저장된 플레이북")
        if saved_playbooks:
            for playbook in saved_playbooks:
                pb_id, title, summary, content, kws, saved_at = playbook
                with st.expander(f"**{title}** (저장일: {saved_at.split(' ')[0]})"):
                    st.markdown(f"**요약:** {summary}")
                    st.markdown("---")
                    st.markdown(f"**내용:**\n\n{content}", unsafe_allow_html=True)
                    st.markdown("---")
                    st.markdown(f"**주요 키워드:** {', '.join(json.loads(kws))}")
                    if st.button("❌ 삭제", key=f"delete_pb_btn_{pb_id}"):
                        delete_playbook_from_favorites(pb_id)
                        st.rerun()
        else:
            st.info("저장된 플레이북이 없습니다.")

        st.markdown("---")
        st.subheader("저장된 뉴스 기사")
        if saved_news:
            for news in saved_news:
                news_id, title, url, summary, kws, risk_level, risk_score, saved_at = news
                st.markdown(f"""
                    <div class="news-item risk-{risk_level.lower() if risk_level else 'low'}">
                        <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:1rem;">
                            <h5 style="margin:0;color:#2c3e50;">
                                <a href="{url}" target="_blank">{title}</a>
                            </h5>
                            <span style="background:{'#e74c3c' if risk_level=='높음' else '#f39c12' if risk_level=='중간' else '#27ae60'};color:white;padding:0.3rem 0.8rem;border-radius:15px;font-size:0.8rem;font-weight:bold;white-space:nowrap;">
                                관심도: {risk_level} ({risk_score:.2f})
                            </span>
                        </div>
                        <p style="color:#555; margin-bottom:1rem; white-space: pre-wrap;">{summary}</p>
                        <div style="color:#888; font-size:0.9rem;">
                            <strong>키워드:</strong> {', '.join(json.loads(kws))} | 저장일: {saved_at.split(' ')[0]}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                if st.button("❌ 삭제", key=f"delete_news_fav_btn_{news_id}"):
                    delete_news_from_favorites(news_id)
                    st.rerun()
        else:
            st.info("저장된 뉴스 기사가 없습니다.")
            
def render_footer():
    st.divider()
    st.markdown(
        """<div style="text-align:center;color:#888;padding:1rem;">
        <p>🛡️ SecureWatch - 중소기업 보안 관심/위험 분석 시스템</p>
        <p>AI 기반 위협 키워드 추출 및 대응 플레이북 생성</p>
        </div>""",
        unsafe_allow_html=True
    )
    
if __name__ == "__main__":
    main()