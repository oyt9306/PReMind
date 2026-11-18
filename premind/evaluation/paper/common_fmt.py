import sys,re,json,collections
from pathlib import Path
FC=str(Path(__file__).resolve().parents[2] / 'bv3' / 'final_construction')
sys.path[:0]=[FC,FC+"/.."]
from trainer_v2 import reward_v5 as R5, metrics_v9 as MV
R2=R5.R2
STRICT_PC=R5.parse_citations
# ---- lenient normaliser (reference -> index only if unambiguous) ----
NUM=r"\d+"
REF=r"(?:memory|memories|mem|entry|entries|item|items|record|records|profile|profiles|m|no\.?|#)"
# a reference group: optional word, then ints joined by , / and / & / ; / + / or '-'? (ranges not expanded)
GROUP=re.compile(rf"\b{REF}\s*[#:.\-]?\s*({NUM}(?:\s*(?:,|;|/|&|\+|and|or)\s*(?:{REF}\s*[#:.\-]?\s*)?{NUM})*)",re.I)
BARE=re.compile(rf"^\W*(?:\[|\()?\s*({NUM}(?:\s*(?:,|;|/|&|\+|and)\s*{NUM})*)\s*(?:\]|\))?\W*$",re.I)
HASH=re.compile(rf"#\s*({NUM})")
def lenient_pc(cu,k):
    cited,valid=STRICT_PC(cu,k)
    ids=set(cited); cu=cu or ""
    for m in GROUP.finditer(cu): ids|={int(x) for x in re.findall(NUM,m.group(1))}
    for m in HASH.finditer(cu): ids.add(int(m.group(1)))
    if not ids:
        m=BARE.match(cu)
        if m: ids|={int(x) for x in re.findall(NUM,m.group(1))}
    return ids,all(1<=n<=k for n in ids)
def classify(raw_present,cu):
    """format class of stage-2 <context_use> value"""
    if not raw_present: return "tag_missing"
    c=(cu or "").strip()
    if c=="" or re.fullmatch(r"[\[\(]?\s*(none|n/?a|no\s+memor\w*|nothing|null|empty|-|\.|\[\s*\])\s*[\]\)]?\.?",c,re.I): return "none_empty"
    if re.fullmatch(r"(\s*\[\s*\d+(?:\s*,\s*\d+)*\s*\]\s*,?)+",c): return "int_list"
    if re.search(r"\[[^\[\]]*\]",c) and re.search(r"\d",c): return "bracket_with_text"
    if re.search(r"\b(memory|memories|mem|entry|entries)\b",c,re.I) and re.search(r"\d",c): return "memory_N_text"
    if re.search(r"\d",c): return "other_numeric"
    return "prose_no_index"
def tagflags(s):
    f={}
    for t in ("decision","visual_evidence","reasoning","context_use","risk","action","confidence","answer"):
        f[t]=bool(re.search(rf"<{t}>",s))
    f["closed_all"]=all(re.search(rf"</{t}>",s) for t in ("context_use","risk","action","confidence","answer"))
    return f
