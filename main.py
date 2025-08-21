import streamlit as st
import pandas as pd
import json
from datetime import datetime

# 모듈 임포트
from config import *
from news_scraper import fetch_latest_news_by_rss
from ner_analyzer import load_ner_model, update_keywords_from_cisa, analyze_risk_with_model
from llm_generator import generate_playbook_with_llm
from pdf_reporter import create_pdf_report
from database import *

# ============================================================
# 메인 애플리케이션
# ============================================================

# 전역 변수로 NER 모델과 Gemini 모델 선언
ner_tokenizer = None
ner_model = None
ner_ctx = None
gemini_model = None

def main():
    global ner_tokenizer, ner_model, ner_ctx, gemini_model
    # 페이지 설정
    st.set_page_config(**PAGE_CONFIG)
    
    # CSS 스타일 적용
    st.markdown(CSS_STYLES, unsafe_allow_html=True)
    
    # 환경 변수 검증
    if not GEMINI_API_KEY:
        st.error("Gemini API 키가 설정되지 않았습니다. .env 파일을 확인해주세요.")
        st.stop()
    
    # Gemini 모델 초기화
    import google.generativeai as genai
    genai.configure(api_key=GEMINI_API_KEY)
    gemini_model = genai.GenerativeModel('gemini-1.5-flash', generation_config=GENERATION_CONFIG)
    
    # NER 모델 로딩
    ner_tokenizer, ner_model, ner_ctx = load_ner_model()
    
    # CISA KEV 업데이트 (앱 최초 1회)
    from ner_analyzer import industry_risk_map
    update_keywords_from_cisa(industry_risk_map)
    
    # 데이터베이스 초기화
    init_db()
    
    # 회사명은 전역 변수로 고정
    company_name = "중소기업"
    
    # 세션 상태를 초기화
    if 'analysis_started' not in st.session_state:
        st.session_state.analysis_started = False
        st.session_state.news_data = []
        st.session_state.risk_keywords = []
        st.session_state.playbook_content = ""
        st.session_state.report_summary = ""
        st.session_state.llm_selected_keywords = []
        
        # 사이드바 위젯의 초기값을 세션 상태에 저장
        st.session_state.company_size = COMPANY_SIZE_OPTIONS[0]
        st.session_state.industry_type = INDUSTRY_OPTIONS[0]
        st.session_state.infrastructure = INFRASTRUCTURE_OPTIONS[0]
        st.session_state.constraints = ""
        st.session_state.user_interest = ""
    
    # 사이드바 렌더링
    render_sidebar()
    
    # 메인 헤더
    st.markdown("""
    <div class="main-header">
        <h1>🛡️ 중소기업 보안 관심/위험 분석 시스템</h1>
        <p>AI 기반 보안 위협 키워드 추출 및 맞춤형 대응 플레이북 생성</p>
    </div>
    """, unsafe_allow_html=True)
    
    # 탭 렌더링
    render_tabs()

def render_sidebar():
    """사이드바 렌더링"""
    with st.sidebar:
        st.markdown("""
        <div class="sidebar-logo">
            <h2>🛡️ SecureWatch</h2>
            <p>중소기업 보안 관심/위험 분석</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.subheader("🏢 기업 정보 설정")
        company_size = st.selectbox("기업 규모", COMPANY_SIZE_OPTIONS,
                                    index=COMPANY_SIZE_OPTIONS.index(st.session_state.company_size),
                                    key='sidebar_company_size')
        industry_type = st.selectbox("업종", INDUSTRY_OPTIONS,
                                     index=INDUSTRY_OPTIONS.index(st.session_state.industry_type),
                                     key='sidebar_industry_type')
        
        st.subheader("🌐 인프라 및 제약사항")
        infrastructure = st.selectbox("인프라 환경", INFRASTRUCTURE_OPTIONS,
                                      index=INFRASTRUCTURE_OPTIONS.index(st.session_state.infrastructure),
                                      key='sidebar_infrastructure')
        constraints = st.text_area("보안 정책/예산 등 제한사항", value=st.session_state.constraints, key='sidebar_constraints')
        user_interest = st.text_area("관심 분야 키워드(쉼표 구분)", value=st.session_state.user_interest, key='sidebar_user_interest')
        
        st.divider()
        if st.button("🔍 분석 시작", type="primary"):
            start_analysis(company_size, industry_type, infrastructure, constraints, user_interest)

def start_analysis(company_size, industry_type, infrastructure, constraints, user_interest):
    """분석 시작 및 실행"""
    global gemini_model
    
    st.session_state.analysis_started = True
    st.session_state.news_data = []
    st.session_state.risk_keywords = []
    st.session_state.playbook_content = ""
    st.session_state.report_summary = ""
    st.session_state.llm_selected_keywords = []
    
    # 세션 상태 업데이트
    st.session_state.company_size = company_size
    st.session_state.industry_type = industry_type
    st.session_state.infrastructure = infrastructure
    st.session_state.constraints = constraints
    st.session_state.user_interest = user_interest
    
    with st.spinner("RSS에서 뉴스 수집 중..."):
        articles = fetch_latest_news_by_rss()
    
    # 분석
    with st.spinner("분석/키워드 추출 중..."):
        news_data = []
        keyword_counts = {}
        for art in articles:
            combined = f"{art['title']} {art['content']}"
            risk_level, kws, score = analyze_risk_with_model(combined, industry_type, ner_tokenizer, ner_model, ner_ctx)
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
        user_interest_list = [kw.strip() for kw in user_interest.split(',') if kw.strip()]
        for uk in user_interest_list:
            keyword_counts[uk] = keyword_counts.get(uk, 0) + 1
        st.session_state.news_data = sorted(news_data, key=lambda x: x['risk_score'], reverse=True)
        st.session_state.risk_keywords = [
            {"keyword": kw, "frequency": cnt, "risk_level": analyze_risk_with_model(kw, industry_type, ner_tokenizer, ner_model, ner_ctx)[0]}
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
        try:
            company_info = {"name": "중소기업", "size": company_size, "industry": industry_type}
            keywords_list = [k["keyword"] for k in st.session_state.risk_keywords]
            playbook_content, llm_selected_keywords = generate_playbook_with_llm(
                keywords_list, company_info, infrastructure, constraints, gemini_model, news_briefs=news_briefs
            )
            st.session_state.playbook_content = playbook_content
            st.session_state.llm_selected_keywords = llm_selected_keywords
        except Exception as e:
            error_msg = str(e)
            if "429" in error_msg or "quota" in error_msg.lower():
                st.error("⚠️ Gemini API 할당량이 초과되었습니다. 다음 중 하나를 시도해주세요:")
                st.markdown("""
                1. **잠시 후 재시도** (약 1시간 후)
                2. **새로운 API 키 생성** (Google AI Studio에서)
                3. **유료 플랜으로 업그레이드**
                
                현재는 기본 템플릿으로 플레이북을 생성합니다.
                """)
                
                # 기본 템플릿 플레이북 생성
                basic_playbook = f"""
# {industry_type} 업종 보안 대응 플레이북

## 주요 위협 키워드
{', '.join(keywords_list[:10])}

## 기본 보안 대응 방안
1. **네트워크 보안**
   - 방화벽 설정 강화
   - VPN 접속 관리
   - 네트워크 모니터링

2. **엔드포인트 보안**
   - 안티바이러스 업데이트
   - OS 보안 패치 적용
   - USB 장치 사용 제한

3. **사용자 교육**
   - 피싱 메일 인식 교육
   - 비밀번호 정책 준수
   - 소셜 엔지니어링 방지

4. **데이터 보호**
   - 중요 데이터 암호화
   - 정기 백업 수행
   - 접근 권한 관리

## 인프라별 특화 방안
**{infrastructure}** 환경에 맞는 추가 보안 설정을 적용하세요.

## 제약사항 고려사항
{constraints if constraints else "특별한 제약사항 없음"}
                """
                
                st.session_state.playbook_content = basic_playbook
                st.session_state.llm_selected_keywords = [{"keyword": kw, "reason": "기본 템플릿"} for kw in keywords_list[:5]]
            else:
                st.error(f"플레이북 생성 중 오류가 발생했습니다: {error_msg}")
                st.session_state.playbook_content = "플레이북 생성에 실패했습니다."
                st.session_state.llm_selected_keywords = []
    
    st.session_state.report_summary = f"총 {len(st.session_state.news_data)}개 뉴스 분석 완료."
    st.success("✅ 분석 완료! 아래 탭에서 결과를 확인하세요.")
    st.rerun()

def render_tabs():
    """탭 렌더링"""
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 대시보드", "📰 뉴스 분석", "📋 대응 플레이북", "⭐ 즐겨찾기"
    ])
    
    # 대시보드 탭
    with tab1:
        render_dashboard()
    
    # 뉴스 분석 탭
    with tab2:
        render_news_analysis()
    
    # 대응 플레이북 탭
    with tab3:
        render_playbook()
    
    # 즐겨찾기 탭
    with tab4:
        render_favorites()

def render_dashboard():
    """대시보드 탭 렌더링"""
    st.header("📊 보안 관심/위험 대시보드")
    if not st.session_state.analysis_started:
        st.info("👈 사이드바에서 기업 정보를 설정하고 '분석 시작'을 눌러주세요.")
    else:
        top_news = st.session_state.news_data[:2]
        if not top_news:
            st.info("분석된 뉴스가 없습니다.")
        else:
            for current in top_news:
                css_class = "risk-high" if current["risk_level"] == "높음" else "risk-medium" if current["risk_level"] == "중간" else "risk-low"
                
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
                st.markdown("---")

def render_news_analysis():
    """뉴스 분석 탭 렌더링"""
    st.header("📰 최신 보안 뉴스 분석")
    if not st.session_state.analysis_started:
        st.info("👈 사이드바에서 기업 정보를 설정하고 '분석 시작'을 눌러주세요.")
    else:
        sorted_news = st.session_state.news_data
        
        total = len(sorted_news)
        total_pages = (total - 1) // PAGE_SIZE + 1 if total > 0 else 1
        if 'current_page' not in st.session_state:
            st.session_state.current_page = 1
        start = (st.session_state.current_page - 1) * PAGE_SIZE
        end = start + PAGE_SIZE
        page_data = sorted_news[start:end]
        
        for idx, news in enumerate(page_data):
            css_class = "risk-high" if news["risk_level"] == "높음" \
                else "risk-medium" if news["risk_level"] == "중간" else "risk-low"
            
            with st.container():
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
                
                if st.button("💾 즐겨찾기 추가", key=f"save_news_btn_{idx}_{st.session_state.current_page}"):
                    success, message = save_news_to_favorites(news)
                    if success:
                        st.success(message)
                    else:
                        st.warning(message)
            
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

def render_playbook():
    """대응 플레이북 탭 렌더링"""
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
            pdf_output = create_pdf_report(report_data, "중소기업")
            st.download_button(
                label="📄 PDF 다운로드",
                data=pdf_output,
                file_name=f"보안_분석_보고서_중소기업.pdf",
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
    """즐겨찾기 탭 렌더링"""
    st.header("⭐ 즐겨찾기")
    saved_news = get_saved_news()
    saved_playbooks = get_saved_playbooks()

    if not saved_news and not saved_playbooks:
        st.info("저장된 기사나 플레이북이 없습니다. '뉴스 분석' 탭에서 기사를, '대응 플레이북' 탭에서 플레이북을 저장할 수 있습니다.")
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
                    <div class="news-item risk-{risk_level.lower()}">
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
                
                if st.button("❌ 삭제", key=f"delete_news_btn_{news_id}"):
                    delete_news_from_favorites(news_id)
                    st.rerun()
        else:
            st.info("저장된 뉴스 기사가 없습니다.")

    # 푸터
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
