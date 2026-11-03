import tempfile, unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import from_origin
import torch
from flood.data import audit,FloodDataset,read_image
from flood.evaluate import export_raster,aligned_jrc,area_km2
from flood.models import build_model
from flood.train import train

class PipelineTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        self.temp=tempfile.TemporaryDirectory(); self.root=Path(self.temp.name); rows=[]
        for i,split in enumerate(['train','valid']):
            row=dict(chip_id=f'Spain_{i}',event='Spain',split=split)
            for key in ['image','label','jrc']:
                row[key]=str(self.root/f'{i}_{key}.tif')
                arr=np.ones((2 if key=='image' else 1,512,512),dtype='float32')
                if key=='image': arr*=(-10 if split=='train' else 100)
                if key=='label': arr[:,:,256:]=0; arr[:,:256,:256]=-1
                with rasterio.open(row[key],'w',driver='GTiff',height=512,width=512,count=len(arr),dtype='float32',crs='EPSG:4326',transform=from_origin(10,10,.0001,.0001)) as dst: dst.write(arr)
            rows.append(row)
        # Move validation spatially so synthetic split geometry is independent.
        for key in ['image','label','jrc']:
            with rasterio.open(rows[1][key],'r+') as dst: dst.transform=from_origin(11,10,.0001,.0001)
        self.df=pd.DataFrame(rows)
        self.norm={'mean':[-10.,-10.],'std':[1.,1.]}
    def tearDown(self): self.temp.cleanup()

    def test_normalization_train_only_and_invalid_patch(self):
        norm,_=audit(self.df,self.root/'audit')
        self.assertEqual(norm['mean'],[-10.,-10.])
        ds=FloodDataset(self.df,'train',self.norm)
        self.assertEqual(len(ds),3); self.assertEqual(ds.skipped,1)
        self.assertEqual(ds[0]['image'].shape,(2,256,256))

    def test_joint_augmentation_alignment(self):
        row=self.df.iloc[0]
        with rasterio.open(row.label) as ds: mask=ds.read(1)
        mask[256:384,:128]=0
        with rasterio.open(row.label,'r+') as ds: ds.write(mask,1)
        with rasterio.open(row.image,'r+') as ds: ds.write(np.stack([mask,mask]).astype('float32'))
        ds=FloodDataset(self.df,'train',{'mean':[0,0],'std':[1,1]},augment=True)
        for _ in range(8):
            b=ds[1]; v=b['valid'][0]
            torch.testing.assert_close(b['image'][0][v],b['target'][0][v])

    def test_annotation_nodata_does_not_change_input(self):
        before=FloodDataset(self.df,'train',self.norm)[1]['image'].clone()
        row=self.df.iloc[0]
        with rasterio.open(row.label,'r+') as ds:
            a=ds.read(1); a[300:320,40:60]=-1; ds.write(a,1)
        after=FloodDataset(self.df,'train',self.norm)[1]['image']
        torch.testing.assert_close(before,after)
        _,image_valid=read_image(row)
        self.assertTrue(image_valid.all())

    def test_export_and_jrc_alignment(self):
        row=self.df.iloc[0].to_dict(); v=np.ones((512,512),bool); v[0,0]=False
        p=self.root/'output.tif'; export_raster(row['image'],p,np.ones((512,512)),v)
        with rasterio.open(p) as ds:
            self.assertEqual(ds.nodata,255); self.assertEqual(ds.read(1)[0,0],255); self.assertEqual(ds.crs.to_epsg(),4326)
        # A coarse JRC grid covering the same extent must align correctly.
        with rasterio.open(row['jrc'],'w',driver='GTiff',height=256,width=256,count=1,dtype='int16',crs='EPSG:4326',transform=from_origin(10,10,.0002,.0002)) as ds: ds.write(np.ones((256,256),dtype='int16'),1)
        permanent,valid=aligned_jrc(row)
        self.assertTrue(permanent.all()); self.assertTrue(valid.all())
        self.assertGreater(area_km2(v,row['image']),0)

    def test_models_forward_backward(self):
        for name in ['plain','resnet34']:
            model=build_model(name,False).eval(); x=torch.randn(1,2,32,32)
            z=model(x); self.assertEqual(z.shape,(1,1,32,32)); z.mean().backward()
            del model

    def test_resume_continues_epoch(self):
        cfg={'model':'plain','pretrained':False,'epochs':1,'workers':0,'batch_size':2}
        out=self.root/'run'
        with patch('flood.train.build_model',side_effect=lambda *a:torch.nn.Conv2d(2,1,1)),patch('flood.train.environment',return_value={}):
            train(self.df,self.norm,out,cfg,allow_cpu=True)
            train(self.df,self.norm,out,{**cfg,'epochs':2},allow_cpu=True)
        state=torch.load(out/'last.pt',weights_only=False)
        self.assertEqual(state['epoch'],1); self.assertEqual(len(state['history']),2)

if __name__=='__main__': unittest.main()
