# ===== final1.py =====
import pandas as pd
import re
import os

# 현재 파일이 있는 폴더에 articles.csv 파일을 두면 이처럼 파일 이름만 입력해도 됩니다.
SRC = "articles.csv"
OUT = "articles_preprocessed.csv"

# 입력 파일 존재 여부 확인
if not os.path.exists(SRC):
    print(f"오류: 입력 파일 '{SRC}'을(를) 찾을 수 없습니다.")
    print("articles.csv 파일을 final1.py 파일과 같은 폴더에 넣어주세요.")
else:
    df = pd.read_csv(SRC)

    def extract_org(text):
        if not isinstance(text, str): return []
        cands = []
        # 회사/기관/대학교 등
        cands += re.findall(r'\\b[가-힣A-Z][가-힣A-Za-z0-9]{1,15}(?:사|기업|대학교|기관|원|회)', text)
        # 영문 대문자로 시작하는 단어 (2자 이상)
        cands += re.findall(r'\\b[A-Z][A-Za-z]{2,}\\b', text)
        return list(set(cands))

    def extract_vuln(text):
        if not isinstance(text, str): return []
        cands = []
        # CVE ID
        cands += re.findall(r'CVE-\\d{4}-\\d+', text)
        # 키워드
        for kw in ["취약점", "버그", "보안 결함"]:
            if kw in text:
                cands.append(kw)
        return list(set(cands))

    def extract_attack(text):
        if not isinstance(text, str): return []
        cands = []
        keywords = ["공격", "해킹", "피싱", "랜섬웨어", "DDoS",
                    "Exploit", "SQL Injection", "XSS", "트로이목마"]
        for kw in keywords:
            if re.search(rf"\\b{kw}\\b", text, flags=re.I):
                cands.append(kw)
        return list(set(cands))

    # 새로 추가된 함수: 보안 전략 키워드 추출
    def extract_strategy(text):
        if not isinstance(text, str): return []
        cands = []
        # '제로 트러스트' 관련 키워드 추가
        keywords = ["제로 트러스트", "zero trust", "zero-trust", "zero trust architecture", "ZTNA"]
        for kw in keywords:
            if re.search(kw, text, flags=re.I):
                cands.append(kw)
        return list(set(cands))

    rows = []
    for _, r in df.iterrows():
        content = str(r.get("content", "") or "")
        if not content.strip():
            continue
        rows.append({
            "clean_text": content.strip(),
            "ORG": extract_org(content),
            "VULN": extract_vuln(content),
            "ATTACK": extract_attack(content),
            "STRATEGY": extract_strategy(content)  # 새로운 키워드 추출 함수 적용
        })

    out_df = pd.DataFrame(rows)
    out_df.to_csv(OUT, index=False, encoding="utf-8-sig")
    print(f"자동 후보 추출 완료: {len(out_df)} rows → {OUT}")