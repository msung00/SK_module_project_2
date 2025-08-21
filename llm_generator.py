import json
import re
import google.generativeai as genai

def generate_article_summary(title: str, content: str, severity_label: str, company_info: dict, infrastructure: str, gemini_model):
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
        res = gemini_model.generate_content(prompt)
        return res.text
    except Exception:
        return "요약 생성 실패."

def generate_playbook_with_llm(keywords, company_info, infrastructure, constraints, gemini_model, news_briefs=None):
    """
    - message.txt 의도 반영 통합 플레이북:
      긴급/단기/중장기 구간 + 탐지룰 + 커뮤니케이션 + 체크리스트
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
- 각 조치는 **담당자**(예: IT 담당자, 보안 담당자)를 명시하세요.
- **긴급/단기/중장기** 구간으로 나눌 것
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
2) 긴급
3) 단기
4) 중장기
5) 탐지룰/모니터링(로그 소스, 룰 또는 쿼리 개요)
6) 커뮤니케이션(임직원 공지/훈련/외부 보고)
7) 체크리스트(측정 가능한 완료 조건)
""".strip()

    try:
        resp = gemini_model.generate_content(prompt)
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
        kw_resp = gemini_model.generate_content(kw_prompt)
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

    return playbook, llm_selected_keywords
