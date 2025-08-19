import re
import pandas as pd


# 파일 경로 설정
SRC = "articles.csv" # 입력 파일명
OUT = "articles_preprocessed.csv" # 출력 파일명

# --------------------------
# 엔티티 후보 추출 함수들
# --------------------------
def extract_org(text):
    if not isinstance(text, str):
        return []
    cands = []
    cands += re.findall(r'\b[가-힣A-Z][가-힣A-Za-z0-9]{1,15}(?:사|기업|대학교|기관|원|회)', text)
    cands += re.findall(r'\b[A-Z][A-Za-z]{2,}\b', text)
    return list(set(cands))

def extract_vuln(text):
    if not isinstance(text, str):
        return []
    cands = []
    cands += re.findall(r'CVE-\d{4}-\d+', text)
    for kw in ["취약점", "버그", "보안 결함"]:
        if kw in text:
            cands.append(kw)
    return list(set(cands))

def extract_attack(text):
    if not isinstance(text, str):
        return []
    keywords = ["공격", "해킹", "피싱", "랜섬웨어", "DDoS",
                "Exploit", "SQL Injection", "XSS", "트로이목마"]
    cands = []
    for kw in keywords:
        if re.search(rf"\b{kw}\b", text, flags=re.I):
            cands.append(kw)
    return list(set(cands))

# --------------------------
# 실행
# --------------------------
df = pd.read_csv(SRC)
rows = []
for _, r in df.iterrows():
    content = str(r.get("content", "") or "")
    if not content.strip():
        continue
    rows.append({
        "clean_text": content.strip(),
        "ORG": extract_org(content),
        "VULN": extract_vuln(content),
        "ATTACK": extract_attack(content)
    })

pre_df = pd.DataFrame(rows)
pre_df.to_csv(OUT, index=False, encoding="utf-8-sig")

print(f"[1단계 완료] {OUT} 저장 ({len(pre_df)}행)")
