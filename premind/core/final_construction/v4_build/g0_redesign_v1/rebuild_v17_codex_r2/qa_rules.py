"""Owner D4 CPU QA rules; reviewable keys, no model or API."""
import re,unicodedata
SYNONYMS={'sitting':'sit','seated':'sit','standing':'stand','lying':'lie','laying':'lie','grey':'gray',
          'inside':'in','outside':'out','resting':'rest','mounted':'mount','hanging':'hang',
          'shelves':'shelf','leaves':'leaf','children':'child','feet':'foot','hands':'hand'}
def normalize(text):
    source=unicodedata.normalize('NFKC',str(text)).lower().replace('’',"'")
    source=re.sub(r"'s\b",'',source)
    words=re.findall(r"[a-z0-9]+",source)
    out=[]
    for word in words:
        if word in ('the','a','an'):continue
        word=SYNONYMS.get(word,word)
        if word.endswith('ies') and len(word)>4:word=word[:-3]+'y'
        elif word.endswith(('ches','shes','xes','zes','sses')):word=word[:-2]
        elif word.endswith('s') and len(word)>3 and not word.endswith(('ss','is','us')):word=word[:-1]
        out.append(word)
    return out
def matches(text,keys):
    words=normalize(text)
    for key in keys:
        phrase=normalize(key)
        for i in range(len(words)-len(phrase)+1):
            if words[i:i+len(phrase)]!=phrase:continue
            prefix=words[max(0,i-3):i]
            # Do not count negated lure mentions ("on the bed, not in the suitcase").
            if any(w in ('not','never','neither','without') for w in prefix) and not any(w in ('not','never','no') for w in phrase):continue
            yield key
def score(answer,keys):
    gold=list(matches(answer,keys['gold_keys']));lure=list(matches(answer,keys['lure_keys']))
    return {'qa_correct':bool(gold) and not lure,'qa_primed':bool(lure),'gold_hits':gold,'lure_hits':lure}
def extract(text):
    text=str(text).strip()
    text=re.sub(r'^(?:it|they)\s+(?:is|are)\s+','',text,flags=re.I)
    # Keep a complete positive location/state clause; do not create disjunctive
    # object keys for a multi-object answer ("book, blanket and notebook").
    text=re.split(r',\s*(?:not|with|near)|;|\.\s+',text,maxsplit=1,flags=re.I)[0]
    return [text.rstrip('.?! ')] if text else []
