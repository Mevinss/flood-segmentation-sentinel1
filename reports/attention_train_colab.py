from pathlib import Path
import json, importlib, shutil, pandas as pd, torch
import flood.models, flood.train, flood.evaluate
from flood.data import audit
att_run=SAVE/'runs/attention_resnet34_seed42'
att_run.mkdir(exist_ok=True,parents=True)
model_source="import torch\nfrom torch import nn\nimport torch.nn.functional as F\n\ndef block(a,b):\n    return nn.Sequential(nn.Conv2d(a,b,3,padding=1,bias=False),nn.BatchNorm2d(b),nn.ReLU(inplace=True),nn.Conv2d(b,b,3,padding=1,bias=False),nn.BatchNorm2d(b),nn.ReLU(inplace=True))\n\nclass PlainUNet(nn.Module):\n    def __init__(self):\n        super().__init__(); widths=[32,64,128,256,512]\n        self.enc=nn.ModuleList([block(a,b) for a,b in zip([2]+widths[:-1],widths)])\n        self.dec=nn.ModuleList([block(a+b,b) for a,b in zip(widths[:0:-1],widths[-2::-1])])\n        self.head=nn.Conv2d(32,1,1)\n    def forward(self,x):\n        skips=[]\n        for i,layer in enumerate(self.enc):\n            if i: x=F.max_pool2d(x,2)\n            x=layer(x); skips.append(x)\n        for layer,skip in zip(self.dec,skips[-2::-1]):\n            x=layer(torch.cat([F.interpolate(x,size=skip.shape[-2:],mode='bilinear',align_corners=False),skip],1))\n        return self.head(x)\n\nclass GatedSkip(nn.Module):\n    \"\"\"Modulate a skip using a decoder-conditioned spatial attention map.\"\"\"\n    def __init__(self, decoder_channels, skip_channels):\n        super().__init__()\n        self.decoder_channels = decoder_channels\n        hidden = max(skip_channels // 2, 1)\n        self.decoder_projection = nn.Conv2d(decoder_channels, hidden, 1)\n        self.skip_projection = nn.Conv2d(skip_channels, hidden, 1)\n        self.score = nn.Conv2d(hidden, 1, 1)\n\n    def forward(self, fused):\n        decoder = fused[:, :self.decoder_channels]\n        skip = fused[:, self.decoder_channels:]\n        score = self.score(F.relu(self.decoder_projection(decoder) + self.skip_projection(skip)))\n        return torch.cat((decoder, skip * score.sigmoid()), dim=1)\n\n\ndef build_model(name='resnet34',pretrained=True):\n    if name=='plain': return PlainUNet()\n    if name not in ('resnet34', 'attention_resnet34'): raise ValueError(name)\n    import segmentation_models_pytorch as smp\n    model = smp.Unet(encoder_name='resnet34',encoder_weights='imagenet' if pretrained else None,in_channels=2,classes=1,activation=None)\n    if name == 'attention_resnet34':\n        skip_widths = reversed(model.encoder.out_channels[1:-1])\n        for stage, skip_width in zip(model.decoder.blocks, skip_widths):\n            decoder_width = stage.conv1[0].in_channels - skip_width\n            stage.attention1 = GatedSkip(decoder_width, skip_width)\n    return model\n"
Path(flood.models.__file__).write_text(model_source)
(att_run/'models.py').write_text(model_source)
importlib.reload(flood.models)
importlib.reload(flood.train)
importlib.reload(flood.evaluate)
df=pd.read_csv(WORK/'data/manifest.csv')
base_run=SAVE/'runs/resnet34_seed42'
cfg=json.loads((base_run/'config.json').read_text()); assert cfg['pretrained'] and cfg['seed']==42
cfg['model']='attention_resnet34'
norm, _=audit(df, WORK/'audit_attention')
assert norm==json.loads((base_run/'normalization.json').read_text())
basestate=torch.load(base_run/'best.pt',weights_only=False,map_location='cpu'); assert basestate['split_ids']=={s:sorted(df[df.split==s].chip_id) for s in df.split.unique()}; del basestate
att_best=flood.train.train(df,norm,att_run,cfg,resume=True)
flood.evaluate.learning_curves(att_run)
print('ATTENTION TRAINED',att_best)