import json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

def write_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    def clean(x):
        if isinstance(x, dict): return {str(k):clean(v) for k,v in x.items()}
        if isinstance(x, (tuple,list)): return [clean(v) for v in x]
        if isinstance(x, np.generic): return clean(x.item())
        if isinstance(x, float) and not np.isfinite(x): return None
        return x
    path.write_text(json.dumps(clean(value), indent=2, ensure_ascii=False), encoding='utf-8')

def counts(pred, target, valid):
    p=np.asarray(pred,dtype=bool); y=np.asarray(target,dtype=bool); v=np.asarray(valid,dtype=bool)
    return np.array([(p&y&v).sum(),(p&~y&v).sum(),(~p&y&v).sum(),(~p&~y&v).sum()],dtype=np.int64)

def metrics(c):
    tp,fp,fn,tn=map(int,c)
    def div(a,b): return a/b if b else float('nan')
    return dict(iou=div(tp,tp+fp+fn), f1=div(2*tp,2*tp+fp+fn), precision=div(tp,tp+fp), recall=div(tp,tp+fn), tp=tp,fp=fp,fn=fn,tn=tn)

def masked_loss(logits,target,valid):
    if not valid.any(): return None
    # Index BEFORE BCE so NaN/unknown values outside valid cannot propagate.
    z=logits[valid].float(); y=target[valid].float(); p=z.sigmoid()
    return F.binary_cross_entropy_with_logits(z,y)+1-(2*(p*y).sum()+1e-6)/(p.sum()+y.sum()+1e-6)

def assemble(patches, size=512, patch=256):
    if size%patch or len(patches)!=(size//patch)**2: raise ValueError('Incomplete tiling')
    out=np.zeros((size,size),dtype=np.asarray(patches[0]).dtype)
    for k,a in enumerate(patches):
        r,c=divmod(k,size//patch); out[r*patch:(r+1)*patch,c*patch:(c+1)*patch]=a
    return out

def validate_splits(rows):
    seen=set()
    for r in rows:
        if r['chip_id'] in seen: raise ValueError('Duplicate/leaking chip: '+r['chip_id'])
        seen.add(r['chip_id'])
        if (r['event']=='Bolivia') != (r['split']=='bolivia'): raise ValueError('Bolivia holdout leakage')
