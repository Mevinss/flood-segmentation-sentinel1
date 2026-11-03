from pathlib import Path
import numpy as np
import pandas as pd
import torch
import rasterio
from rasterio.warp import reproject,Resampling
from .data import read_pair,read_image,normalize
from .models import build_model
from .core import counts,metrics,assemble,write_json

def load_model(checkpoint,device=None):
    device=torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
    state=torch.load(checkpoint,map_location='cpu',weights_only=False)
    model=build_model(state['config']['model'],False).to(device)
    model.load_state_dict(state['model']); model.eval()
    return model,state['normalization'],device

@torch.inference_mode()
def predict(model,x,valid,norm,device):
    z=normalize(x,valid,norm); patches=[]
    for top in [0,256]:
        for left in [0,256]:
            a=torch.from_numpy(z[:,top:top+256,left:left+256].copy())[None].to(device)
            patches.append(model(a).sigmoid()[0,0].cpu().numpy())
    return assemble(patches)

def otsu(x,valid):
    from skimage.filters import threshold_otsu
    values=x[1][valid]
    if not len(values) or np.ptp(values)==0: return np.zeros_like(valid)
    return x[1]<threshold_otsu(values)

def choose_threshold(df,model,norm,device,out):
    thresholds=np.round(np.arange(.1,.901,.05),2); totals=np.zeros((len(thresholds),4),dtype=np.int64)
    for row in df[df.split=='valid'].to_dict('records'):
        x,y,v=read_pair(row); _,image_valid=read_image(row); prob=predict(model,x,image_valid,norm,device)
        for i,t in enumerate(thresholds): totals[i]+=counts(prob>=t,y,v)
    scores=[metrics(c)['iou'] for c in totals]
    order=sorted(range(len(thresholds)),key=lambda i:(-scores[i],abs(thresholds[i]-.5),thresholds[i]))
    threshold=float(thresholds[order[0]])
    write_json(Path(out)/'threshold.json',{'threshold':threshold,'source':'valid','curve':[{'threshold':float(t),'iou':s} for t,s in zip(thresholds,scores)]})
    return threshold

def export_raster(source,path,array,valid,probability=False):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    nodata=-9999 if probability else 255; dtype='float32' if probability else 'uint8'
    with rasterio.open(source) as ds: profile=ds.profile.copy()
    profile.update(count=1,dtype=dtype,nodata=nodata,compress='deflate')
    with rasterio.open(path,'w',**profile) as dst: dst.write(np.where(valid,array,nodata).astype(dtype),1)

def aligned_jrc(row):
    with rasterio.open(row['image']) as dst,rasterio.open(row['jrc']) as src:
        a=src.read(1); raw_valid=src.read_masks(1)>0
        # Dataset JRC masks use binary water labels. Fail closed on other semantics.
        if not set(np.unique(a[raw_valid])).issubset({-1,0,1}): raise ValueError('JRC is not binary; inspect source semantics')
        raw_valid &= np.isin(a,[0,1])
        result=np.full(dst.shape,-1,dtype='int16')
        reproject(np.where(raw_valid,a,-1).astype('int16'),result,src_transform=src.transform,src_crs=src.crs,src_nodata=-1,dst_transform=dst.transform,dst_crs=dst.crs,dst_nodata=-1,resampling=Resampling.nearest)
    return result==1,result>=0

def area_km2(mask,source):
    from pyproj import Geod,Transformer
    with rasterio.open(source) as ds:
        transform=ds.transform; crs=ds.crs
    geod=Geod(ellps='WGS84'); converter=Transformer.from_crs(crs,'EPSG:4326',always_xy=True); total=0.
    # Vectorize equal-area-within-latitude rows on north-up geographic rasters.
    if crs.to_epsg()==4326 and transform.b==0 and transform.d==0:
        for row,n in enumerate(mask.sum(1)):
            if not n: continue
            pts=[transform*(c,r) for c,r in [(0,row),(1,row),(1,row+1),(0,row+1)]]
            area,_=geod.polygon_area_perimeter(*zip(*pts)); total+=abs(area)*int(n)
    else:
        for r,c in zip(*np.where(mask)):
            pts=[converter.transform(*(transform*(cc,rr))) for cc,rr in [(c,r),(c+1,r),(c+1,r+1),(c,r+1)]]
            area,_=geod.polygon_area_perimeter(*zip(*pts)); total+=abs(area)
    return total/1e6

def panel(row,prob,threshold,path):
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    x,y,v=read_pair(row); pred=prob>=threshold
    errors=np.zeros(v.shape,dtype=int); errors[pred&(y==1)&v]=1; errors[pred&(y==0)&v]=2; errors[~pred&(y==1)&v]=3; errors[~v]=4
    fig,axes=plt.subplots(2,3,figsize=(13,9))
    views=[(x[0],'VV (dB)','gray'),(x[1],'VH (dB)','gray'),(np.where(v,y,np.nan),'Water ground truth','Blues'),(np.where(v,prob,np.nan),'Water probability','viridis'),(np.where(v,pred,np.nan),'Predicted water','Blues'),(errors,'TN black / TP green / FP red / FN blue / invalid gray',ListedColormap(['black','limegreen','red','royalblue','gray']))]
    for ax,(a,title,cmap) in zip(axes.flat,views):
        ax.imshow(a,cmap=cmap,vmin=0 if a is errors else None,vmax=4 if a is errors else None); ax.set_title(title,fontsize=9); ax.axis('off')
    fig.suptitle(row['chip_id']); fig.tight_layout(); fig.savefig(path,dpi=200); plt.close(fig)

def evaluate(df,out,checkpoint=None):
    out=Path(out); out.mkdir(parents=True,exist_ok=True)
    if checkpoint:
        model,norm,device=load_model(checkpoint); threshold=choose_threshold(df,model,norm,device,out)
    else: model=None; threshold=.5
    records=[]
    for row in df[df.split.isin(['test','bolivia'])].to_dict('records'):
        x,y,v=read_pair(row)
        _,image_valid=read_image(row)
        prob=predict(model,x,image_valid,norm,device) if model is not None else otsu(x,image_valid).astype('float32')
        c=counts(prob>=threshold,y,v)
        records.append({'chip_id':row['chip_id'],'event':row['event'],'split':row['split'],**metrics(c)})
        export_raster(row['image'],out/'probabilities'/(row['chip_id']+'.tif'),prob,image_valid,True)
    table=pd.DataFrame(records); table.to_csv(out/'per_tile.csv',index=False)
    summary={}
    for split,g in table.groupby('split'):
        summary[split]=metrics(g[['tp','fp','fn','tn']].sum().to_numpy())
        summary[split]['defined_tile_iou']=int(g.iou.notna().sum())
    write_json(out/'metrics.json',summary)
    events=[]
    for (split,event),g in table.groupby(['split','event']): events.append({'split':split,'event':event,**metrics(g[['tp','fp','fn','tn']].sum().to_numpy())})
    pd.DataFrame(events).to_csv(out/'per_event.csv',index=False)
    selection=[]
    for split,g in table.groupby('split'):
        good=g.dropna(subset=['iou']).sort_values('iou')
        if len(good): selection.extend([good.iloc[0].chip_id,good.iloc[len(good)//2].chip_id,good.iloc[-1].chip_id])
        selection.extend(g.sample(min(2,len(g)),random_state=42).chip_id)
    for chip in dict.fromkeys(selection):
        row=df[df.chip_id==chip].iloc[0].to_dict()
        with rasterio.open(out/'probabilities'/(chip+'.tif')) as src: prob=src.read(1)
        panel(row,prob,threshold,out/(chip+'_comparison.png'))
    write_json(out/'figure_selection.json',{'method':'worst, median, best IoU and 2 fixed-random tiles per split','chip_ids':list(dict.fromkeys(selection))})
    return summary

def flood_maps(df,evaluation_dir,out,threshold):
    evaluation_dir=Path(evaluation_dir); out=Path(out); out.mkdir(parents=True,exist_ok=True); records=[]
    for row in df[df.split.isin(['test','bolivia'])].to_dict('records'):
        path=evaluation_dir/'probabilities'/(row['chip_id']+'.tif')
        with rasterio.open(path) as src: prob=src.read(1); valid=src.read_masks(1)>0
        permanent,jvalid=aligned_jrc(row); valid &= jvalid; flood=(prob>=threshold)&~permanent&valid
        export_raster(row['image'],out/(row['chip_id']+'_flood_estimate.tif'),flood,valid)
        export_raster(row['image'],out/(row['chip_id']+'_water.tif'),prob>=threshold,valid)
        records.append({'chip_id':row['chip_id'],'split':row['split'],'estimated_flood_km2':area_km2(flood,row['image'])})
    pd.DataFrame(records).to_csv(out/'estimated_areas.csv',index=False)
    write_json(out/'limitations.json',{'interpretation':'Water outside JRC permanent water; not independent confirmation of flood. Seasonal water and JRC errors remain.','independent_flood_accuracy':None})
    return pd.DataFrame(records)

def learning_curves(run_dir):
    import matplotlib.pyplot as plt
    run_dir=Path(run_dir); h=pd.read_csv(run_dir/'history.csv'); fig,axes=plt.subplots(1,2,figsize=(10,4))
    axes[0].plot(h.epoch,h.train_loss,label='train'); axes[0].plot(h.epoch,h.valid_loss,label='valid'); axes[0].set_title('BCE + Dice loss'); axes[0].legend()
    axes[1].plot(h.epoch,h.valid_iou,label='IoU'); axes[1].plot(h.epoch,h.valid_f1,label='F1'); axes[1].legend(); axes[1].set_title('Validation water segmentation')
    for ax in axes: ax.set_xlabel('Epoch'); ax.grid(alpha=.2)
    fig.tight_layout(); fig.savefig(run_dir/'learning_curves.png',dpi=300); plt.close(fig)
