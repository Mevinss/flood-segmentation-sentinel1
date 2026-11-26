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

class GatedSkip(nn.Module):
    """Modulate a skip using a decoder-conditioned spatial attention map."""
    def __init__(self, decoder_channels, skip_channels):
        super().__init__()
        self.decoder_channels = decoder_channels
        hidden = max(skip_channels // 2, 1)
        self.decoder_projection = nn.Conv2d(decoder_channels, hidden, 1)
        self.skip_projection = nn.Conv2d(skip_channels, hidden, 1)
        self.score = nn.Conv2d(hidden, 1, 1)

    def forward(self, fused):
        decoder = fused[:, :self.decoder_channels]
        skip = fused[:, self.decoder_channels:]
        score = self.score(F.relu(self.decoder_projection(decoder) + self.skip_projection(skip)))
        return torch.cat((decoder, skip * score.sigmoid()), dim=1)


def build_model(name='resnet34',pretrained=True):
    if name=='plain': return PlainUNet()
    if name not in ('resnet34', 'attention_resnet34'): raise ValueError(name)
    import segmentation_models_pytorch as smp
    model = smp.Unet(encoder_name='resnet34',encoder_weights='imagenet' if pretrained else None,in_channels=2,classes=1,activation=None)
    if name == 'attention_resnet34':
        skip_widths = reversed(model.encoder.out_channels[1:-1])
        for stage, skip_width in zip(model.decoder.blocks, skip_widths):
            decoder_width = stage.conv1[0].in_channels - skip_width
            stage.attention1 = GatedSkip(decoder_width, skip_width)
    return model
