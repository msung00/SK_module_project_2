# ===== build_bio_dataset.py =====
import pandas as pd
import re, ast, random, os
from typing import List, Tuple
from datasets import Dataset, DatasetDict

SRC = "/content/drive/MyDrive/articles_preprocessed.csv"
OUT_DIR = "/content/drive/MyDrive"
HF_DATASET_DIR = os.path.join(OUT_DIR, "ner_dataset")
random.seed(42)

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

def normalize_list(v: List[str]) -> List[str]:
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

def find_entity_spans(text: str, phrases: List[str]) -> List[Tuple[int,int]]:
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
            spans.append((s,e))
            for i in range(s,e):
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

def write_conll(path, sequences):
    with open(path, "w", encoding="utf-8") as f:
        for seq in sequences:
            for tok, lab in seq:
                f.write(f"{tok} {lab}\n")
            f.write("\n")

# --- 메인 ---
df = pd.read_csv(SRC)
rows = []
for _, r in df.iterrows():
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

# 분할
random.shuffle(rows)
n = len(rows)
n_train = int(n*0.8)
n_valid = int(n*0.1)
train_seqs = rows[:n_train]
valid_seqs = rows[n_train:n_train+n_valid]
test_seqs  = rows[n_train+n_valid:]

# CoNLL 저장
os.makedirs(OUT_DIR, exist_ok=True)
write_conll(os.path.join(OUT_DIR, "ner_train.conll"), train_seqs)
write_conll(os.path.join(OUT_DIR, "ner_valid.conll"), valid_seqs)
write_conll(os.path.join(OUT_DIR, "ner_test.conll"),  test_seqs)

# HF DatasetDict 생성
def to_dict_format(seqs):
    tokens = [[tok for tok, lab in seq] for seq in seqs]
    labels = [[lab for tok, lab in seq] for seq in seqs]
    return {"tokens": tokens, "ner_tags": labels}

ds = DatasetDict({
    "train": Dataset.from_dict(to_dict_format(train_seqs)),
    "validation": Dataset.from_dict(to_dict_format(valid_seqs)),
    "test": Dataset.from_dict(to_dict_format(test_seqs)),
})

ds.save_to_disk(HF_DATASET_DIR)

# 미리보기 CSV
preview = []
for seq in train_seqs[:30]:
    preview.extend(seq+[("","")])
pd.DataFrame(preview, columns=["token","label"]).to_csv(
    os.path.join(OUT_DIR, "ner_preview.csv"),
    index=False, encoding="utf-8-sig"
)

print("BIO+Dataset 생성 완료:",
      f"train={len(train_seqs)}, valid={len(valid_seqs)}, test={len(test_seqs)}")
print("저장 위치(HF Dataset):", HF_DATASET_DIR)
