import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.modules import ContextBlock2, ContextBlock4, catCtxV3, Loss

class Ctxnet(nn.Module):
    '''
        ctx16
    '''
    def __init__(self):
        super(Ctxnet, self).__init__()

        self.backbone = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 24, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(24),
            nn.ReLU(inplace=True),
            nn.Conv2d(24, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 48, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
            ContextBlock2(48, red=48),   # stride4 256x512
        )

        # stride8(128x256)
        self.B8 = nn.Sequential(
            nn.AvgPool2d(kernel_size=2, stride=2, padding=0),
            ContextBlock2(96, red=64))

        # stride16(64x128)
        self.B16 = nn.Sequential(
            nn.AvgPool2d(kernel_size=2, stride=2, padding=0),
            ContextBlock4(160, red=80))

        # stride32(32x64)
        self.B32 = nn.Sequential(
            nn.AvgPool2d(kernel_size=2, stride=2, padding=0),
            ContextBlock4(240, red=96))

        # stride64(16x32)
        self.B64 = nn.Sequential(
            nn.AvgPool2d(kernel_size=2, stride=2, padding=0),
            ContextBlock4(336, red=112))

        self.gap = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(448,448, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(448),
            nn.ReLU(inplace=True))
        
        # ch = [96, 160, 240, 336, 448, 448]
        ch = [448, 448, 336, 240, 160, 96]
        self.ctxTf = []
        for i in range(len(ch)):
            self.ctxTf.append( nn.Sequential(
                nn.Conv2d(ch[i], 48, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(48),
                nn.ReLU(inplace=True)
            ))
        self.ctxTf = nn.ModuleList(self.ctxTf)

        self.sam = catCtxV3(48,48,ch)

        self.cls = nn.Sequential(
            ContextBlock2(48, red=48),
            nn.Conv2d(96, 19, kernel_size=1, stride=1, padding=0, bias=True)
            )
        
        self.loss = Loss()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


    def forward(self,x, target):
        x_size = x.size()
        ctxVects=[]
        f4 = self.backbone(x) # 48x128x256
        ff4 = self.ctxTf[5](f4)

        f8 = self.B8(f4)  # 96x64x128
        ctxVects.append(self.ctxTf[4](f8))

        f16 = self.B16(f8)
        ctxVects.append(self.ctxTf[3](f16))

        f32 = self.B32(f16)
        ctxVects.append(self.ctxTf[2](f32))

        f64 = self.B64(f32)
        ctxVects.append(self.ctxTf[1](f64))

        glob = self.gap(f64)
        ctxVects.append(self.ctxTf[0](glob))

        out = self.sam(ff4, ctxVects)
        
        out = self.cls(out)
        out = F.interpolate(out, x_size[2:], mode='bilinear', align_corners=True)
        main_loss =  self.loss(out, target)

        return main_loss, out.max(1)[1]
    
    def init_weights(self,pretrained=''):

        # logger.info('=> init weights with pretrained weight')
        print('=> init weights with pretrained weight')
        for name, m in self.named_modules():
            if any(part in name for part in {'backbone', 'B8', 'B16', 'B32', 'B64'}):
                continue
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if os.path.isfile(pretrained):
            pretrained_dict = torch.load(pretrained, map_location={'cuda:0': 'cpu'})
            pretrained_dict = pretrained_dict['state_dict']
            # logger.info('=> loading pretrained model {}'.format(pretrained))
            print('=> loading pretrained model {}'.format(pretrained))
            model_dict = self.state_dict()

            pretrained_dict = {k[7:]: v for k, v in pretrained_dict.items()
                               if k[7:] in model_dict.keys()}

            # print(sorted(set(model_dict) - set(pretrained_dict)))
            # print(sorted(set(pretrained_dict)-set(model_dict)))
            assert not (set(pretrained_dict)-set(model_dict))
            # print(sorted(set(model_dict) - set(pretrained_dict)-set(inited)))

            model_dict.update(pretrained_dict)
            self.load_state_dict(model_dict, strict=True)

        elif pretrained:
            raise RuntimeError('No such file {}'.format(pretrained))

    # def train(self, mode=True):
    #     super(Ctxnet, self).train(mode=mode)
    #     """
    #         Override nn.Module.train() for freeze the BN parameters
    #     """
    #     print("Freezing Mean/Var and Weight/Bias of BatchNorm2D.")
    #     # exist=[]
    #     # freezed=[]
    #     for name, m in self.named_modules():
    #         # exist.append(m)
    #         if any(part in name for part in {'backbone', 'B8', 'B16', 'B32', 'B64'}):
    #             if isinstance (m, (nn.BatchNorm2d, nn.SyncBatchNorm)):
    #                 # # freezed.append(m)
    #                 # if hasattr(m, 'weight'):
    #                 #     m.weight.requires_grad_(False)
    #                 # if hasattr(m, 'bias'):
    #                 #     m.bias.requires_grad_(False)
    #                 m.eval()
    #             # print(name, m, m.training)
    #         # print(exist)
    #         # print(freezed)