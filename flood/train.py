import json, random, time, platform, subprocess
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset
from .models import build_model
from .data import FloodDataset
from .core import masked_loss, counts, metrics, write_json

DEFAULT=dict(model='resnet34',pretrained=True,seed=42,batch_size=8,workers=2,epochs=100,patience=15,lr=1e-4,weight_decay=1e-4)

def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True

def save_checkpoint(path,state):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix('.tmp'); torch.save(state,temp); temp.replace(path)

def environment(out):
    out=Path(out); out.mkdir(parents=True,exist_ok=True)
    import sys
    versions=subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True)
    (out/'requirements-lock.txt').write_text(versions,encoding='utf-8')
    info=dict(python=platform.python_version(),torch=torch.__version__,cuda=torch.cuda.is_available(),gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
    write_json(out/'environment.json',info); return info

def epoch(model,loader,device,optimizer=None,scaler=None):
    training=optimizer is not None; model.train(training); total=np.zeros(4,dtype=np.int64); losses=[]
    for batch in loader:
        x=batch['image'].to(device); y=batch['target'].to(device); v=batch['valid'].to(device)
        if training: optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training),torch.autocast(device_type=device.type,enabled=device.type=='cuda'):
            z=model(x); loss=masked_loss(z,y,v)
        if loss is None: continue
        if not torch.isfinite(loss): raise FloatingPointError('Non-finite loss')
        if training:
            scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
        losses.append(float(loss.detach())); total+=counts(z.detach().sigmoid().cpu().numpy()>=.5,y.cpu().numpy(),v.cpu().numpy())
    if not losses: raise ValueError('Empty valid dataset')
    return {'loss':float(np.mean(losses)),**metrics(total)}

def overfit_check(df,norm,out,steps=80,device=None):
    device=torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
    seed_all(42); dataset=FloodDataset(df,'train',norm,False)
    loader=DataLoader(Subset(dataset,list(range(min(4,len(dataset))))),batch_size=2,shuffle=False)
    model=build_model('resnet34',False).to(device); optimizer=torch.optim.AdamW(model.parameters(),lr=1e-3)
    scaler=torch.amp.GradScaler('cuda',enabled=device.type=='cuda'); history=[]; start=time.time()
    for i in range(steps):
        m=epoch(model,loader,device,optimizer,scaler); history.append(m['loss'])
    result={'first_loss':history[0],'last_loss':history[-1],'seconds':time.time()-start,'steps':steps}
    write_json(Path(out)/'overfit_check.json',result)
    if history[-1]>=history[0]*.7: raise RuntimeError('Tiny-set overfit check failed; inspect data and model')
    del model,optimizer
    if device.type=='cuda': torch.cuda.empty_cache()
    return result

def train(df,norm,out,config=None,resume=True,allow_cpu=False):
    cfg={**DEFAULT,**(config or {})}; out=Path(out); out.mkdir(parents=True,exist_ok=True)
    if not torch.cuda.is_available() and not allow_cpu: raise RuntimeError('Select GPU in Colab; full CPU training is disabled')
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); seed_all(cfg['seed'])
    trainset=FloodDataset(df,'train',norm,True); valset=FloodDataset(df,'valid',norm,False)
    generator=torch.Generator().manual_seed(cfg['seed'])
    loader=DataLoader(trainset,batch_size=cfg['batch_size'],shuffle=True,num_workers=cfg['workers'],generator=generator,pin_memory=device.type=='cuda')
    valid=DataLoader(valset,batch_size=cfg['batch_size'],num_workers=cfg['workers'])
    last=out/'last.pt'; state=None
    if resume and last.exists():
        state=torch.load(last,map_location='cpu',weights_only=False) # Only our own checkpoint.
        for key in cfg:
            if key not in ['epochs','workers'] and state['config'][key]!=cfg[key]: raise ValueError('Resume config mismatch: '+key)
        if state['normalization']!=norm: raise ValueError('Resume normalization mismatch')
        old_ids=state['split_ids']; ids={s:sorted(df[df.split==s].chip_id.tolist()) for s in df.split.unique()}
        if old_ids!=ids: raise ValueError('Resume dataset mismatch')
    model=build_model(cfg['model'],cfg['pretrained'] and state is None).to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=cfg['lr'],weight_decay=cfg['weight_decay'])
    scaler=torch.amp.GradScaler('cuda',enabled=device.type=='cuda')
    history=[]; best=-1.; stale=0; start_epoch=0
    if state:
        model.load_state_dict(state['model']); optimizer.load_state_dict(state['optimizer']); scaler.load_state_dict(state['scaler'])
        history=state['history']; best=state['best']; stale=state['stale']; start_epoch=state['epoch']+1
        random.setstate(state['python_rng']); np.random.set_state(state['numpy_rng']); torch.set_rng_state(state['torch_rng'])
        generator.set_state(state['loader_rng'])
        if device.type=='cuda' and state['cuda_rng']: torch.cuda.set_rng_state_all(state['cuda_rng'])
    environment(out); write_json(out/'config.json',cfg); write_json(out/'normalization.json',norm)
    print(f'{cfg["model"]}: {len(trainset)} train patches; device={device}; resume epoch={start_epoch}',flush=True)
    for e in range(start_epoch,cfg['epochs']):
        if stale>=cfg['patience']: break
        t=time.time(); tr=epoch(model,loader,device,optimizer,scaler); va=epoch(model,valid,device)
        if not np.isfinite(va['iou']): raise ValueError('Validation IoU undefined')
        improved=va['iou']>best
        if improved: best=va['iou']; stale=0
        else: stale+=1
        record={'epoch':e+1,'seconds':time.time()-t,**{'train_'+k:v for k,v in tr.items()},**{'valid_'+k:v for k,v in va.items()}}
        history.append(record)
        checkpoint=dict(model=model.state_dict(),optimizer=optimizer.state_dict(),scaler=scaler.state_dict(),epoch=e,best=best,stale=stale,history=history,config=cfg,normalization=norm,python_rng=random.getstate(),numpy_rng=np.random.get_state(),torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all() if device.type=='cuda' else [],loader_rng=generator.get_state(),split_ids={s:sorted(df[df.split==s].chip_id.tolist()) for s in df.split.unique()})
        if improved: save_checkpoint(out/'best.pt',checkpoint)
        save_checkpoint(last,checkpoint); pd.DataFrame(history).to_csv(out/'history.csv',index=False)
        print(f'Epoch {e+1}: train loss={tr["loss"]:.4f}, valid IoU={va["iou"]:.4f}, {record["seconds"]:.1f}s',flush=True)
    return out/'best.pt'
