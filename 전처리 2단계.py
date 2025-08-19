import re, ast, random
import pandas as pd
from datasets import Dataset, DatasetDict, Features, Sequence, ClassLabel, Value
from google.colab import drive

drive.mount('/content/drive')

SRC = "/content/drive/MyDrive/hf_data/articles_preprocessed.csv"  # 1단계 출력
OUT = "/content/drive/MyDrive/hf_dataset"                         # HuggingFace Dataset 저장
random.seed(42)

# --------------------------
# 도우미 함수들 (safe_list, normalize_list 등)
# --------------------------
def safe_list(x):
    if isinstance(x, list):
        return x
    if pd.isna(x):
        return []
    try:
        v = ast.literal_eval(x)
        if isinstance(v, list):
            return v
        return []
    except Exception:
        return []

def normalize_list(v):
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

def sentence_split(text: str):
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

def find_entity_spans(text: str, phrases):
    occupied = [False]*len(text)
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
    for m in re.finditer(r'\w+|[^\w\s]', sent, flags=re.UNICODE):
        tok = m.group(0)
        s = sent_start + m.start()
        e = sent_start + m.end()
        yield tok, s, e

def assign_bio(tokens, spans_by_type):
    prio = ["VULN", "ATTACK", "ORG"]
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
def build_sequences(pre_csv):
    df2 = pd.read_csv(pre_csv)
    rows = []
    for _, r in df2.iterrows():
        text = str(r.get("clean_text", "") or "")
        orgs = normalize_list(safe_list(r.get("ORG", [])))
        vulns = normalize_list(safe_list(r.get("VULN", [])))
        attacks = normalize_list(safe_list(r.get("ATTACK", [])))
        if not text.strip():
            continue
        sents = sentence_split(text)
        for sent, s0 in sents:
            toks = list(tokens_with_offsets(sent, s0))
            if not toks:
                continue
            spans_by_type = {
                "ORG":    find_entity_spans(text, orgs),
                "VULN":   find_entity_spans(text, vulns),
                "ATTACK": find_entity_spans(text, attacks),
            }
            seq = assign_bio(toks, spans_by_type)
            if any(lab != "O" for _, lab in seq):
                rows.append(seq)
    return rows

seqs = build_sequences(SRC)

# --------------------------
# HuggingFace Dataset 변환
# --------------------------
LABELS = ["O","B-ORG","I-ORG","B-VULN","I-VULN","B-ATTACK","I-ATTACK"]

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
train = seqs[: int(n*0.8)]
valid = seqs[int(n*0.8): int(n*0.9)]
test  = seqs[int(n*0.9):]

ds = DatasetDict({
    "train": Dataset.from_dict(to_dict_format(train), features=features),
    "validation": Dataset.from_dict(to_dict_format(valid), features=features),
    "test": Dataset.from_dict(to_dict_format(test), features=features),
})

ds.save_to_disk(OUT)
print(f"[2단계 완료] {OUT} 저장")
print(ds)
