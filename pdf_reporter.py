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

    # 바이트 반환
    out = pdf.output(dest="S")
    if isinstance(out, str):
        out = out.encode("latin1", errors="ignore")
    return bytes(out)
