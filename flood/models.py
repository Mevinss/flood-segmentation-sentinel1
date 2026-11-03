import torch
from torch import nn
import torch.nn.functional as F

def block(a,b):
    return nn.Sequential(nn.Conv2d(a,b,3,padding=1,bias=False),nn.BatchNorm2d(b),nn.ReLU(inplace=True),nn.Conv2d(b,b,3,padding=1,bias=False),nn.BatchNorm2d(b),nn.ReLU(inplace=True))

class PlainUNet(nn.Module):
    def __init__(self):
        super().__init__(); widths=[32,64,128,256,512]
        self.enc=nn.ModuleList([block(a,b) for a,b in zip([2]+widths[:-1],widths)])
        self.dec=nn.ModuleList([block(a+b,b) for a,b in zip(widths[:0:-1],widths[-2::-1])])
        self.head=nn.Conv2d(32,1,1)
    def forward(self,x):
        skips=[]
        for i,layer in enumerate(self.enc):
            if i: x=F.max_pool2d(x,2)
            x=layer(x); skips.append(x)
        for layer,skip in zip(self.dec,skips[-2::-1]):
            x=layer(torch.cat([F.interpolate(x,size=skip.shape[-2:],mode='bilinear',align_corners=False),skip],1))
        return self.head(x)

def build_model(name='resnet34',pretrained=True):
    if name=='plain': return PlainUNet()
    if name!='resnet34': raise ValueError(name)
    import segmentation_models_pytorch as smp
    return smp.Unet(encoder_name='resnet34',encoder_weights='imagenet' if pretrained else None,in_channels=2,classes=1,activation=None)
