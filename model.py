'''
    KAU-RML ingee hong
    Model for Experiment19
 
        Model 21_1: Resnet50 -> spatial attention  ->       ->  sum -> predict

'''
import torch
import torch.nn as nn
import torch.nn.functional as F

class Bottleneck(nn.Module):
    def __init__(self, inplanes, dilation=False):
        super(Bottleneck, self).__init__()

        reduction=4
        reduced_chan = inplanes//reduction

        self.conv1 = nn.Conv2d(inplanes, reduced_chan, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(reduced_chan)
        
        if dilation:
            self.conv2 = nn.Conv2d(reduced_chan, reduced_chan, kernel_size=3, stride=1, padding=2, dilation=2, bias=False)
        else:
            self.conv2 = nn.Conv2d(reduced_chan, reduced_chan, kernel_size=3, stride=1, padding=1, dilation=1, bias=False)
        self.bn2 = nn.BatchNorm2d(reduced_chan)
        
        self.conv3 = nn.Conv2d(reduced_chan, inplanes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(inplanes)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += identity
        out = self.relu(out)

        return out
class Bottleneck_Down(nn.Module):
    def __init__(self, inplanes, dilation=False):
        super(Bottleneck_Down, self).__init__()

        reduction=2
        reduced_chan = inplanes//reduction
        self.conv1 = nn.Conv2d(inplanes, reduced_chan, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(reduced_chan)
                
        if dilation:
            self.conv2 = nn.Conv2d(reduced_chan, reduced_chan, kernel_size=3, stride=1, padding=2, dilation=2, bias=False)
        else:
            self.conv2 = nn.Conv2d(reduced_chan, reduced_chan, kernel_size=3, stride=2, padding=1, dilation=1, bias=False)
        self.bn2 = nn.BatchNorm2d(reduced_chan)
        
        self.conv3 = nn.Conv2d(reduced_chan, 2* inplanes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(2*inplanes)
        self.relu = nn.ReLU(inplace=True)

        if dilation:
            self.downsample = nn.Sequential(
                    nn.Conv2d(inplanes,2*inplanes, kernel_size=3, stride=1, padding=2, dilation=2, bias=False),
                    nn.BatchNorm2d(2*inplanes)
                )
        else:
            self.downsample = nn.Sequential(
                    nn.Conv2d(inplanes,2*inplanes, kernel_size=3, stride=2, padding=1, dilation=1, bias=False),
                    nn.BatchNorm2d(2*inplanes)
                )
        
    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        
        identity = self.downsample(identity)

        out += identity
        out = self.relu(out)

        return out
class Stem(nn.Module):
    '''
        stem
        conv(3,64,k=3,s=2) -> conv(64,64,k=3,s=1) -> conv(64,128,k=3,s=1) -> maxpool(k=3,s=2)
    '''
    def __init__(self):
        super(Stem,self).__init__()
        ## STEM
        self.stem=nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

    def forward(self, x):
        out = self.stem(x)
        return out

class Backbone(nn.Module):
    '''
        Explain about this module
    '''
    def __init__(self):
        super(Backbone, self).__init__()

        self.stem = Stem()
        self.bottle4_1_main = nn.Sequential(
                nn.Conv2d(128, 64, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 256, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(256)
        )
        self.bottle4_1_branch = nn.Sequential(
            nn.Conv2d(128,256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256))

        self.relu = nn.ReLU(inplace=True)

        self.bottle_down4_2 = Bottleneck_Down(inplanes=256)

        ## Stride 8
        self.bottle8_1 = Bottleneck(inplanes=512)
        self.bottle_down8_2 = Bottleneck_Down(inplanes=512, dilation=True)

        ## Stride 16
        self.bottle16_1 = Bottleneck(inplanes=1024,dilation=True)
        self.bottle16_2 = Bottleneck(inplanes=1024,dilation=True)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        
        ## Stem
        out = self.stem(x)
        
        ## Stride 4
        identity = out
        out = self.bottle4_1_main(out)

        identity = self.bottle4_1_branch(identity)
        out +=identity
        out = self.relu(out)
        
        out = self.bottle_down4_2(out)
        
        ## Stride 8
        out = self.bottle8_1(out)
        out = self.bottle_down8_2(out)

        ## Stride 16
        out = self.bottle16_1(out)
        out = self.bottle16_2(out)
        
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
        self.criterion = nn.CrossEntropyLoss(ignore_index=ignore_label)
        # self.criterion = nn.CrossEntropyLoss(weight=self.weight, ignore_index=ignore_label)
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

        assert len(self.loss_weights) == len(logits)

        # pixel_losses = sum([w * self._forward(x, labels) for (w, x) in zip(self.loss_weights, logits)])

        pixel_losses = self.criterion(logits[-1], labels)
        return pixel_losses


class Model(nn.Module):
    '''
        Explain about this model
    '''

    def __init__(self, cfg):
        super(Model, self).__init__()

        feats_ch=1024
        self.backbone = Backbone()

        self.cls = nn.Sequential(
            nn.Conv2d(feats_ch, 19, kernel_size=1, stride=1, padding=0, bias=True)
            )


        self.loss = Loss()

    
    def forward(self,x, target):
        pred=[]
        x_size = x.size()

        x = self.backbone(x)

        out_feats = self.cls(x) # Bx19xHxW
        pred.append(F.interpolate(out_feats, x_size[2:], mode='bilinear', align_corners=True))

        main_loss =  self.loss(pred, target)

        return main_loss, pred[-1].max(1)[1]# return return max indexes 



def build_model(cfg):

    model = Model(cfg)
    
    return model

#  if __name__=='__main__':
    
