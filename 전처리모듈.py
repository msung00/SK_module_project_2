import pandas as pd
import re, ast, random, os
from typing import List, Tuple
from datasets import Dataset, DatasetDict, Features, Sequence, ClassLabel, Value

# 입력 및 출력 파일/폴더 설정
# 첫 번째 단계에서 생성된 'articles_preprocessed.csv' 파일을 사용합니다.
SRC = "articles_preprocessed.csv"
OUT_DIR = "."  # 출력물을 현재 폴더에 저장
HF_DATASET_DIR = os.path.join(OUT_DIR, "ner_dataset")
random.seed(42)

# ---
# NER 태그 레이블 정의
# ---
LABELS = [
    'O',
    'B-ORG', 'I-ORG',
    'B-VULN', 'I-VULN',
    'B-ATTACK', 'I-ATTACK',
    'B-PROD', 'I-PROD',
    'B-EVT', 'I-EVT',
]

# ---
# 도우미 함수들 (기존 전처리2단계 파일의 내용)
# ---
def safe_list(x) -> list:
    """문자열 형태의 리스트를 실제 리스트로 변환하고 오류를 방지합니다."""
    if isinstance(x, list):
        return x
    if pd.isna(x):
        return []
    try:
        # ast.literal_eval로 리스트 형태의 문자열을 파싱
        v = ast.literal_eval(x)
        if isinstance(v, list):
            # 리스트 내부에 NaN이 있을 경우 제거
            return [item for item in v if pd.notna(item) and isinstance(item, str)]
        return []
    except Exception as e:
        # 파싱 오류가 발생하면 빈 리스트 반환
        print(f"DEBUG: 'safe_list' 오류 - '{x}' 파싱 실패: {e}")
        return []

def normalize_list(v: List[str]) -> List[str]:
    """리스트 내부의 문자열을 정리하고 중복을 제거합니다."""
    out = []
    for s in v:
        if not isinstance(s, str):
            continue
        s = s.strip()
        if len(s) < 2:
            continue
        out.append(s)
    out = sorted(set(out), key=lambda z: len(z), reverse=True)
    return out

def sentence_split(text: str) -> List[Tuple[str, int]]:
    """텍스트를 문장 단위로 분리합니다."""
    sents = []
    start = 0
    pattern = re.compile(r'(.+?)(?:[\.\?\!]+|\n+|다\.)\s*', flags=re.S)
    for m in pattern.finditer(text):
        seg = m.group(0)
        sents.append((seg.strip(), m.start()))
        start = m.end()
    if start < len(text):
        tail = text[start:].strip()
        if tail:
            sents.append((tail, start))
    return sents

def find_entity_spans(text: str, phrases: List[str]) -> List[Tuple[int, int]]:
    """주어진 텍스트에서 엔티티가 포함된 위치(span)를 찾습니다."""
    occupied = [False] * len(text)
    spans = []
    for p in phrases:
        if not p or len(p) < 2:
            continue
        use_boundary = bool(re.search(r'[A-Za-z0-9]', p))
        if use_boundary:
            pat = re.compile(rf'\b{re.escape(p)}\b', flags=re.I)
        else:
            pat = re.compile(re.escape(p))
        for m in pat.finditer(text):
            s, e = m.start(), m.end()
            if any(occupied[s:e]):
                continue
            spans.append((s, e))
            for i in range(s, e):
                occupied[i] = True
    return spans

def tokens_with_offsets(sent: str, sent_start: int):
    """문장을 토큰화하고 오프셋을 계산합니다."""
    for m in re.finditer(r'\w+|[^\w\s]', sent, flags=re.UNICODE):
        tok = m.group(0)
        s = sent_start + m.start()
        e = sent_start + m.end()
        yield tok, s, e

def assign_bio(tokens: List[Tuple[str, int, int]], spans_by_type: dict) -> List[Tuple[str, str]]:
    """토큰에 BIO 태그를 할당합니다. 우선순위: VULN > ATTACK > PROD > EVT > ORG"""
    prio = ["VULN", "ATTACK", "PROD", "EVT", "ORG"]
    tags = []
    for tok, s, e in tokens:
        tag = "O"
        hit_type = None
        hit_span = None
        for t in prio:
            for (ss, ee) in spans_by_type[t]:
                if not (e <= ss or s >= ee):
                    hit_type = t
                    hit_span = (ss, ee)
                    break
            if hit_type:
                break
        if hit_type:
            if s == hit_span[0]:
                tag = f"B-{hit_type}"
            else:
                tag = f"I-{hit_type}"
        tags.append((tok, tag))
    return tags

def to_dict_format(seqs: List[Tuple[List[str], List[int]]]) -> dict:
    """데이터를 HuggingFace Dataset 형식에 맞는 딕셔너리로 변환합니다."""
    tokens = [s[0] for s in seqs]
    labels = [s[1] for s in seqs]
    return {"tokens": tokens, "ner_tags": labels}

# ---
# 후처리 모듈 (새로 추가된 내용)
# ---
def postprocess_ner_results(tokens: List[str], ner_tags_as_numbers: List[int]) -> dict:
    """
    모델의 예측 결과(토큰과 숫자 태그)를 받아
    원하는 형식의 키워드 딕셔너리로 변환합니다.
    
    Args:
        tokens (List[str]): 입력 문장을 구성하는 토큰 목록.
        ner_tags_as_numbers (List[int]): 각 토큰에 대한 예측 태그(숫자) 목록.
    
    Returns:
        Dict[str, List[str]]: 분류된 키워드를 담은 딕셔너리.
                              예: {'ATTACK': ['랜섬웨어 공격'], 'VULN': ['제로데이 취약점']}
    """
    
    # 결과를 담을 딕셔너리 초기화
    results = {
        'ORG': [],
        'VULN': [],
        'ATTACK': [],
        'PROD': [],
        'EVT': []
    }
    
    current_entity = ""
    current_tag_type = ""
    
    # 예측된 태그와 토큰을 순회하며 키워드 추출
    for i, (token, tag_number) in enumerate(zip(tokens, ner_tags_as_numbers)):
        tag = LABELS[tag_number]
        
        # B- (Beginning) 태그를 만났을 때
        if tag.startswith('B-'):
            # 이전 개체명이 있으면 저장
            if current_entity and current_tag_type:
                results[current_tag_type].append(current_entity.strip())
            
            # 새로운 개체명 시작
            current_entity = token
            current_tag_type = tag.split('-')[1]
            
        # I- (Inside) 태그를 만났을 때
        elif tag.startswith('I-'):
            # 현재 개체명에 토큰 추가
            if current_tag_type == tag.split('-')[1]:
                current_entity += " " + token
            else:
                # 태그가 불일치하면 이전 개체명 저장 후 새로운 개체명으로 취급 (오류 처리)
                if current_entity and current_tag_type:
                    results[current_tag_type].append(current_entity.strip())
                current_entity = ""
                current_tag_type = ""
                
        # O (Outside) 태그를 만났을 때
        else:
            if current_entity and current_tag_type:
                results[current_tag_type].append(current_entity.strip())
            current_entity = ""
            current_tag_type = ""

    # 마지막에 남은 개체명 저장
    if current_entity and current_tag_type:
        results[current_tag_type].append(current_entity.strip())
        
    return results


# ---
# 메인 실행 코드
# ---
if __name__ == "__main__":
    print("---")
    print("🚀 BIO+Dataset 생성 프로세스 시작!")
    
    # --------------------------
    # 1. 원본 데이터 읽기
    # --------------------------
    print("1. 'articles_preprocessed.csv' 파일 읽는 중...")
    df = pd.read_csv(SRC)
    df = df.fillna('')
    
    # --------------------------
    # 2. BIO 태그 시퀀스 생성
    # --------------------------
    print("2. BIO 태그 시퀀스 생성 중...")
    
    # 이 부분은 기존 코드와 동일합니다.
    rows = []
    print(f"✅ CSV 파일 로드 완료. 총 행 개수: {len(df)}")
    
    for i, r in df.iterrows():
        text = str(r.get("clean_text", "") or "")
        orgs = normalize_list(safe_list(r.get("ORG", [])))
        vulns = normalize_list(safe_list(r.get("VULN", [])))
        attacks = normalize_list(safe_list(r.get("ATTACK", [])))
        prods = normalize_list(safe_list(r.get("PROD", [])))
        evts = normalize_list(safe_list(r.get("EVT", [])))

        if not text.strip():
            continue
        
        sents = sentence_split(text)
        for sent, s0 in sents:
            toks = list(tokens_with_offsets(sent, s0))
            if not toks:
                continue
            spans_by_type = {
                "ORG":      find_entity_spans(text, orgs),
                "VULN":     find_entity_spans(text, vulns),
                "ATTACK":   find_entity_spans(text, attacks),
                "PROD":     find_entity_spans(text, prods),
                "EVT":      find_entity_spans(text, evts)
            }
            seq = assign_bio(toks, spans_by_type)
            if any(lab != "O" for _, lab in seq):
                rows.append(seq)
    
    seqs = rows
    
    # --------------------------
    # 3. 데이터셋 분할 및 저장
    # --------------------------
    print("3. 데이터셋 분할 및 HuggingFace Dataset으로 변환 중...")
    
    features = Features({
        "tokens": Sequence(Value("string")),
        "ner_tags": Sequence(ClassLabel(names=LABELS)),
    })
    
    random.shuffle(seqs)
    n = len(seqs)
    n_train = int(n*0.8)
    n_valid = int(n*0.1)
    train = seqs[:n_train]
    valid = seqs[n_train:n_train + n_valid]
    test = seqs[n_train + n_valid:]
    
    ds = DatasetDict({
        "train": Dataset.from_dict(to_dict_format(train), features=features),
        "validation": Dataset.from_dict(to_dict_format(valid), features=features),
        "test": Dataset.from_dict(to_dict_format(test), features=features),
    })

    # 폴더가 없으면 생성
    if not os.path.exists(HF_DATASET_DIR):
        os.makedirs(HF_DATASET_DIR)
        
    ds.save_to_disk(HF_DATASET_DIR)
    
    print("---")
    print("✅ BIO+Dataset 생성 완료!")
    print(f"train={len(train)}, valid={len(valid)}, test={len(test)}")
    print(f"HuggingFace Dataset 저장 위치: {HF_DATASET_DIR}")
    print("---")
    
    # --------------------------
    # 후처리 로직 예시
    # 이 부분은 실제 모델이 예측한 결과(predicted_tags)를 처리하는 데 사용됩니다.
    # --------------------------
    
    def predict_ner_tags(tokens: List[str]) -> List[int]:
        """
        주어진 토큰에 대해 NER 태그를 예측하는 시뮬레이션 함수.
        이 함수는 실제 모델을 사용하는 로직으로 교체되어야 합니다.
        """
        
        # 예측 로직을 시뮬레이션하는 딕셔너리
        # 'B-ORG'에 해당하는 '삼성'은 1, 'O'에 해당하는 '사이버'는 0 등
        predictions_map = {
            "삼성": 1,         # B-ORG
            "사이버": 0,       # O
            "공격": 5,         # B-ATTACK
            "제로데이": 3,     # B-VULN
            "취약점": 4,       # I-VULN
            "안랩": 1,         # B-ORG
            "AhnLab": 1,       # B-ORG
        }
        
        # 예측 태그 번호 리스트를 초기화 (모두 'O' 태그)
        predicted_tags = [0] * len(tokens)
        
        # 예측 맵을 기반으로 태그 할당
        for i, token in enumerate(tokens):
            if token in predictions_map:
                predicted_tags[i] = predictions_map[token]
            
        return predicted_tags

    print("\n[후처리 로직 예시 - 모델 예측 결과 처리]")
    # 뉴스 기사에서 추출된 토큰이라고 가정
    example_tokens = ["삼성", "이", "사이버", "공격", "으로", "제로데이", "취약점", "을", "발견했다", "."]
    
    # 'predict_ner_tags' 함수를 사용하여 예측 결과를 얻습니다.
    # 실제로는 이 부분에서 모델을 로드하고 예측을 실행하게 됩니다.
    predicted_tags = predict_ner_tags(example_tokens)

    print(f"예측된 태그: {predicted_tags}")
    
    # 후처리 함수 호출
    processed_keywords = postprocess_ner_results(example_tokens, predicted_tags)
    
    print("모델 예측 결과 후처리 완료:")
    for tag_type, keywords in processed_keywords.items():
        if keywords:
            print(f"- {tag_type}: {keywords}")