import torch
import torch.nn as nn
import torch.nn.functional as F

class catCtxV3(nn.Module):
    '''
        Self Attention Module
        mid_dim = in_dim//8
    '''
    def __init__(self, in_dim, out_dim, bins):
        super(catCtxV3, self).__init__()

        self.bins = bins
        self.conv_ctx = nn.Sequential(
            nn.Conv2d(in_dim*len(bins), in_dim, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(in_dim),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, contexts):
        _,_,H,W = x.size() # B,C,H,W
        
        sum_feats=[]
        for context in contexts:
            sum_feats.append(F.interpolate(context, (H,W), mode='bilinear', align_corners=True))

        sum_feats.append(x)

        context = self.conv_ctx(torch.cat(sum_feats, dim=1))

        return context


class CBRCB(nn.Module):
    def __init__(self, inplanes, kernel_size=3, stride=1, dilation=1):
        super(CBRCB, self).__init__()
        self.module = nn.Sequential(
            nn.Conv2d(inplanes, inplanes, kernel_size=kernel_size, stride=stride, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(inplanes),
            nn.ReLU(inplace=True),
            nn.Conv2d(inplanes, inplanes, kernel_size=kernel_size, stride=stride, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(inplanes))

    def forward(self,x):
        x = self.module(x)
        return x

class ContextBlock2(nn.Module):
    def __init__(self, inplanes, red, dilation=1):
        '''
            2 residual addition, CAT 3 features. 
        '''
        super(ContextBlock2, self).__init__()

        self.block1 = CBRCB(inplanes) 
        self.block2 = CBRCB(inplanes) 

        self.relu = nn.ReLU(inplace=True)

        self.ctx_reduce = nn.Sequential(
            nn.Conv2d(3*inplanes,red, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(red),
            nn.ReLU(inplace=True))

        self.ctx_fuse = nn.Sequential(
            nn.Conv2d((inplanes+red), (inplanes+red), kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d((inplanes+red)),
            nn.ReLU(inplace=True))

    def forward(self, x):

        ctx = []
        ctx.append(x)

        identity1 = x
        x1 = self.block1(x)
        x1 = x1 + identity1
        x1 = self.relu(x1)
        ctx.append(x1)

        identity2 = x1
        x2 = self.block2(x1)
        x2 = x2 + identity2
        x2 = self.relu(x2)
        ctx.append(x2)

        ctx_feats = self.ctx_reduce(torch.cat(ctx, dim=1))
        
        out = self.ctx_fuse(torch.cat((x2, ctx_feats), dim=1))

        return out

class ContextBlock4(nn.Module):
    def __init__(self, inplanes, red, dilation=1):
        '''
            4 residual addition, CAT 5 features. 
        '''
        super(ContextBlock4, self).__init__()

        self.block1 = CBRCB(inplanes) 
        self.block2 = CBRCB(inplanes) 
        self.block3 = CBRCB(inplanes)
        self.block4 = CBRCB(inplanes)

        self.relu = nn.ReLU(inplace=True)

        self.ctx_reduce = nn.Sequential(
            nn.Conv2d(5*inplanes,red, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(red),
            nn.ReLU(inplace=True)
        )

        self.ctx_fuse = nn.Sequential(
            nn.Conv2d((inplanes+red), (inplanes+red), kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d((inplanes+red)),
            nn.ReLU(inplace=True))

    def forward(self, x):

        ctx = []
        ctx.append(x)

        identity1 = x
        x1 = self.block1(x)
        x1 = x1 + identity1
        x1 = self.relu(x1)
        ctx.append(x1)

        identity2 = x1
        x2 = self.block2(x1)
        x2 = x2 + identity2
        x2 = self.relu(x2)
        ctx.append(x2)

        identity3 = x2
        x3 = self.block3(x2)
        x3 = x3 + identity3
        x3 = self.relu(x3)
        ctx.append(x3)

        identity4 = x3
        x4 = self.block4(x3)
        x4 = x4 + identity4
        x4 = self.relu(x4)
        ctx.append(x4)

        ctx_feats = self.ctx_reduce(torch.cat(ctx, dim=1))

        out = self.ctx_fuse(torch.cat((x4, ctx_feats), dim=1))

        return out


class Loss(nn.Module):
    """
    Explain about loss function.
    Arguments:
        ignore_label: Integer, label to ignore.
        weight: Tensor, a manual rescaling weight given to each class.
    """
    def __init__(self, ignore_label=255, weight=None):
        super(Loss, self).__init__()
        self.ignore_label = ignore_label
        self.weight = torch.FloatTensor([0.8373, 0.918, 0.866, 1.0345, 1.0166, 0.9969, 
                                        0.9754, 1.0489, 0.8786, 1.0023, 0.9539, 0.9843, 
                                        1.1116, 0.9037, 1.0865, 1.0955, 1.0865, 1.1529, 1.0507])
        # self.criterion = nn.CrossEntropyLoss(ignore_index=ignore_label)
        self.criterion = nn.CrossEntropyLoss(weight=self.weight, ignore_index=ignore_label)
        self.loss_weights = [1.0]

    def _forward(self, logits, labels):
        ph, pw = logits.size(2), logits.size(3)
        h, w = labels.size(1), labels.size(2)
        if ph != h or pw != w:
            logits = F.interpolate(input=logits, size=(
                h, w), mode='bilinear', align_corners=True)

        loss = self.criterion(logits, labels)

        return loss

    def forward(self, logits, labels):

        # assert len(self.loss_weights) == len(logits)

        # pixel_losses = sum([w * self._forward(x, labels) for (w, x) in zip(self.loss_weights, logits)])

        pixel_losses = self.criterion(logits, labels)
        return pixel_losses