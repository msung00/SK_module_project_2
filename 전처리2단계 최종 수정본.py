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

# --------------------------
# 도우미 함수들
# --------------------------
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
    """토큰에 BIO 태그를 할당합니다. 우선순위: VULN > ATTACK > STRATEGY > ORG"""
    prio = ["VULN", "ATTACK", "STRATEGY", "ORG"]
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

# --------------------------
# BIO 시퀀스 빌드
# --------------------------
def build_sequences(pre_csv: str) -> List[List[Tuple[str, str]]]:
    """전처리된 CSV 파일에서 BIO 시퀀스를 생성합니다."""
    df = pd.read_csv(pre_csv)
    rows = []
    print(f"✅ CSV 파일 로드 완료. 총 행 개수: {len(df)}")
    
    for i, r in df.iterrows():
        text = str(r.get("clean_text", "") or "")
        orgs = normalize_list(safe_list(r.get("ORG", [])))
        vulns = normalize_list(safe_list(r.get("VULN", [])))
        attacks = normalize_list(safe_list(r.get("ATTACK", [])))
        strategies = normalize_list(safe_list(r.get("STRATEGY", [])))

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
                "STRATEGY": find_entity_spans(text, strategies)
            }
            seq = assign_bio(toks, spans_by_type)
            if any(lab != "O" for _, lab in seq):
                rows.append(seq)
    return rows

# HuggingFace Dataset을 CSV로 변환하는 함수
def convert_to_csv_and_save(dataset: Dataset, split_name: str, output_dir: str):
    """
    HuggingFace Dataset 분할을 Pandas DataFrame으로 변환하고 CSV로 저장합니다.
    """
    # HuggingFace Dataset의 'tokens'와 'ner_tags'를 리스트로 추출
    tokens_list = dataset["tokens"]
    ner_tags_list = dataset["ner_tags"]
    
    # 각 행의 토큰과 태그를 문자열로 결합
    combined_data = []
    for tokens, tags in zip(tokens_list, ner_tags_list):
        # 토큰과 태그를 'token tag' 형식으로 결합
        combined_row = " ".join([f"{token} {tag}" for token, tag in zip(tokens, tags)])
        combined_data.append(combined_row)
        
    # DataFrame 생성 및 CSV로 저장
    df = pd.DataFrame(combined_data, columns=['token_tag_sequence'])
    output_path = os.path.join(output_dir, f"ner_{split_name}.csv")
    df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"✅ '{output_path}' 파일이 성공적으로 생성되었습니다.")


if not os.path.exists(SRC):
    print(f"❌ 오류: 입력 파일 '{SRC}'을(를) 찾을 수 없습니다.")
else:
    try:
        seqs = build_sequences(SRC)
        
        # --------------------------
        # HuggingFace Dataset 변환
        # --------------------------
        LABELS = ["O", "B-ORG", "I-ORG", "B-VULN", "I-VULN", "B-ATTACK", "I-ATTACK", "B-STRATEGY", "I-STRATEGY"]
        
        def to_dict_format(seqs):
            tokens = [[tok for tok, lab in seq] for seq in seqs]
            labels = [[lab for tok, lab in seq] for seq in seqs]
            return {"tokens": tokens, "ner_tags": labels}
        
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
        # CSV 파일로 변환 및 저장
        # --------------------------
        convert_to_csv_and_save(ds['train'], 'train', OUT_DIR)
        convert_to_csv_and_save(ds['validation'], 'validation', OUT_DIR)
        convert_to_csv_and_save(ds['test'], 'test', OUT_DIR)
        
    except Exception as e:
        print(f"❌ 치명적 오류 발생: {e}")