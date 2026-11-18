"""CPU phrase scorer: six normalized-token negation scope, contrast and sentence boundaries."""
import re
from ..rebuild_v17_codex_r2.qa_rules import normalize,extract
NEG={'not','never','neither','without','no'}
PAIRS={('rather','than'),('instead','of'),('out','of')}
def _negative(tokens):
 return bool(set(tokens)&NEG) or any(tuple(tokens[i:i+2]) in PAIRS for i in range(len(tokens)-1))
def matches(text,keys):
 # Separate sentences and explicit contrast clauses; "not X but Y" denies X only.
 pieces=re.split(r'([.!?;]|\bbut\b|\bhowever\b)',str(text),flags=re.I)
 clauses=[(normalize(pieces[n]),n+1<len(pieces) and pieces[n+1].lower()=='but') for n in range(0,len(pieces),2)]
 for key in keys:
  phrase=normalize(key)
  if not phrase:continue
  for words,before_but in clauses:
   for i in range(len(words)-len(phrase)+1):
    if words[i:i+len(phrase)]!=phrase:continue
    if (_negative(words[max(0,i-6):i]) or (before_but and 'not' in words[:i])) and not _negative(phrase):continue
    yield key
    break
   else:continue
   break
def score(answer,keys):
 gold=list(matches(answer,keys['gold_keys']));lure=list(matches(answer,keys['lure_keys']))
 return {'qa_correct':bool(gold) and not lure,'qa_primed':bool(lure),'gold_hits':gold,'lure_hits':lure}
