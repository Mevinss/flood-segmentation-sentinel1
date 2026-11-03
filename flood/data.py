import csv, hashlib, io, json, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
import pandas as pd
import requests
import rasterio
from rasterio.windows import Window
import torch
from torch.utils.data import Dataset
from .core import validate_splits, write_json

BASE='https://storage.googleapis.com/sen1floods11/v1.1/'
HAND='data/flood_events/HandLabeled/'

def download(url,path):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists() and path.stat().st_size:
        try:
            if path.suffix=='.tif':
                with rasterio.open(path) as ds: ds.read()
            return
        except rasterio.errors.RasterioError: pass
    temp=path.with_suffix(path.suffix+'.part')
    for attempt in range(3):
        try:
            with requests.get(url,stream=True,timeout=(20,120)) as r:
                r.raise_for_status()
                with temp.open('wb') as f:
                    for chunk in r.iter_content(1024*1024): f.write(chunk)
            temp.replace(path); return
        except requests.RequestException:
            if attempt==2: raise
            time.sleep(attempt+1)

def prepare(root, include_jrc=True, limit=None):
    root=Path(root); rows=[]; jobs=[]
    for split,remote in [('train','train'),('valid','valid'),('test','test'),('bolivia','bolivia')]:
        p=root/'splits'/f'flood_{remote}_data.csv'
        download(BASE+'splits/flood_handlabeled/'+p.name,p)
        pairs=[r for r in csv.reader(io.StringIO(p.read_text())) if r]
        if limit is not None: pairs=pairs[:limit]
        for image,label in pairs:
            chip=image.removesuffix('_S1Hand.tif')
            row=dict(chip_id=chip,event=chip.split('_')[0],split=split)
            for key,folder,name in [('image','S1Hand',image),('label','LabelHand',label)]+([('jrc','JRCWaterHand',chip+'_JRCWaterHand.tif')] if include_jrc else []):
                dest=root/folder/name; row[key]=str(dest.resolve()); jobs.append((BASE+HAND+folder+'/'+name,dest))
            rows.append(row)
    validate_splits(rows)
    with ThreadPoolExecutor(max_workers=6) as pool:
        for i,_ in enumerate(pool.map(lambda a:download(*a),jobs),1):
            if i%100==0 or i==len(jobs): print(f'Downloaded/verified {i}/{len(jobs)}',flush=True)
    df=pd.DataFrame(rows); df.to_csv(root/'manifest.csv',index=False)
    return df

def read_image(row, window=None):
    with rasterio.open(row['image']) as ds:
        x=ds.read(window=window).astype('float32'); iv=(ds.read_masks(window=window)>0).all(0)
    return x,iv&np.isfinite(x).all(0)

def read_pair(row, window=None):
    x,iv=read_image(row,window)
    with rasterio.open(row['label']) as ds:
        y=ds.read(1,window=window); lv=ds.read_masks(1,window=window)>0
    valid=iv&lv&np.isfinite(x).all(0)&np.isin(y,[0,1])
    return x,(y==1).astype('float32'),valid

def audit(df,out):
    out=Path(out); out.mkdir(parents=True,exist_ok=True); records=[]; bounds=[]
    sums=np.zeros(2); squares=np.zeros(2); n=0
    for row in df.to_dict('records'):
        with rasterio.open(row['image']) as a, rasterio.open(row['label']) as b:
            if a.count!=2 or a.shape!=(512,512) or b.count!=1 or a.shape!=b.shape or a.crs!=b.crs or not a.transform.almost_equals(b.transform):
                raise ValueError('Raster geometry mismatch: '+row['chip_id'])
            values=np.unique(b.read(1))
            if not set(values).issubset({-1,0,1}): raise ValueError(f'Unexpected labels: {values}')
            bounds.append((row['chip_id'],row['split'],str(a.crs),a.bounds))
        x,y,v=read_pair(row)
        rec={k:row[k] for k in ['chip_id','event','split']}
        rec.update(valid_pixels=int(v.sum()),water_pixels=int(y[v].sum()),invalid_fraction=float(1-v.mean()))
        for key in ['image','label']:
            p=Path(row[key]); rec[key+'_bytes']=p.stat().st_size; rec[key+'_sha256']=hashlib.sha256(p.read_bytes()).hexdigest()
        records.append(rec)
        if row['split']=='train':
            _,image_valid=read_image(row)
            z=x[:,image_valid].astype('float64'); sums+=z.sum(1); squares+=(z*z).sum(1); n+=z.shape[1]
    overlaps=[]
    for i,(aid,asp,crs,a) in enumerate(bounds):
        for bid,bsp,bcrs,b in bounds[i+1:]:
            if asp!=bsp and crs==bcrs and min(a.right,b.right)-max(a.left,b.left)>1e-8 and min(a.top,b.top)-max(a.bottom,b.bottom)>1e-8:
                overlaps.append([aid,bid])
    write_json(out/'spatial_overlaps.json',overlaps)
    if overlaps: raise ValueError('Spatial overlap between splits; inspect report before training')
    if not n: raise ValueError('No valid training pixels')
    mean=sums/n; std=np.sqrt(np.maximum(squares/n-mean**2,1e-12))
    norm={'mean':mean.tolist(),'std':std.tolist(),'train_valid_pixels':n,'units':'dB'}
    write_json(out/'normalization.json',norm)
    stats=pd.DataFrame(records); stats.to_csv(out/'audit.csv',index=False)
    stats.groupby(['split','event']).agg(tiles=('chip_id','count'),valid_pixels=('valid_pixels','sum'),water_pixels=('water_pixels','sum')).to_csv(out/'split_summary.csv')
    return norm,stats

def normalize(x, valid, norm):
    z=(x-np.asarray(norm['mean'],dtype='float32')[:,None,None])/np.asarray(norm['std'],dtype='float32')[:,None,None]
    z[:,~valid]=0
    return z

class FloodDataset(Dataset):
    def __init__(self,df,split,norm,augment=False):
        self.rows=df[df.split==split].to_dict('records'); self.norm=norm; self.augment=augment; self.items=[]; self.skipped=0
        for i,r in enumerate(self.rows):
            _,_,v=read_pair(r)
            for top in [0,256]:
                for left in [0,256]:
                    if v[top:top+256,left:left+256].any(): self.items.append((i,top,left))
                    else: self.skipped+=1
    def __len__(self): return len(self.items)
    def __getitem__(self,k):
        i,top,left=self.items[k]; row=self.rows[i]
        window=Window(left,top,256,256)
        x,y,v=read_pair(row,window); _,image_valid=read_image(row,window); x=normalize(x,image_valid,self.norm)
        y=y[None]; v=v[None]
        if self.augment:
            turns=int(torch.randint(4,()).item()); flip=bool(torch.rand(())<.5)
            arrays=[np.rot90(a,turns,axes=(-2,-1)) for a in [x,y,v]]
            x,y,v=[np.flip(a,-1) if flip else a for a in arrays]
        return {'image':torch.from_numpy(x.copy()),'target':torch.from_numpy(y.copy()),'valid':torch.from_numpy(v.copy()),'chip_id':row['chip_id'],'top':top,'left':left}

def eda(df,out):
    import matplotlib.pyplot as plt
    out=Path(out); selected=df[df.split=='train'].groupby('event').head(1).head(8)
    for row in selected.to_dict('records'):
        x,y,v=read_pair(row); fig,axes=plt.subplots(1,4,figsize=(14,4))
        for ax,a,title,cmap in zip(axes,[x[0],x[1],np.where(v,y,np.nan),v],['VV (dB)','VH (dB)','Water label','Valid pixels'],['gray','gray','Blues','gray']):
            ax.imshow(a,cmap=cmap); ax.set_title(title); ax.axis('off')
        fig.suptitle(row['chip_id']); fig.tight_layout(); fig.savefig(out/(row['chip_id']+'_data.png'),dpi=180); plt.close(fig)
