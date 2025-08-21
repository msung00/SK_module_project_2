import os
import re
import torch
import requests
from transformers import ElectraTokenizerFast, ElectraForTokenClassification

# 업종별 위험도 맵
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
        "원격진료":0.9,"진단장비":0.9,"의료영상":0.9,"보건의료정보":1.0,"제약사 해킹":0.9,
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
        "온라인수업":1.0,"LMS":1.0,"학생정보":1.0,"교직원 계정":0.9,"학교 네트워크":0.9,
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

def ner_inference(sentence: str, ner_tokenizer, ner_model, ner_ctx):
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
        print(f"CISA KEV 업데이트 실패: {e}")

def analyze_risk_with_model(text: str, industry_type: str, ner_tokenizer=None, ner_model=None, ner_ctx=None):
    """
    1) 가능하면 NER로 엔터티 추출
    2) 업종별 가중치 합산으로 점수/레벨 산정
    3) NER 실패 시, 단순 키워드 매칭 폴백
    """
    risk_dict = industry_risk_map.get(industry_type, {})
    extracted = ner_inference(text, ner_tokenizer, ner_model, ner_ctx)

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
