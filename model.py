import torch.nn as nn
import torch
from torch.nn.init import xavier_uniform
from torch.autograd import Variable
import torchvision.models as models
import torch.nn.functional as F
from torchvision.models.resnet import BasicBlock

class SSD_Resnet(nn.Module):

    def __init__(self, num_anchors, num_coords=4, num_classes=2, arch='resnet34', fo=64):
        super().__init__()
        assert len(num_anchors) == 6
        self.fo = fo
        self.arch = arch
        self.base = getattr(models, arch)(pretrained=True)
        fi = 512
        self.layer5 = self.base._make_layer(BasicBlock, fi, fo, stride=2)
        self.layer6 = self.base._make_layer(BasicBlock, fo, fo, stride=2)
        self.layer7 = self.base._make_layer(BasicBlock, fo, fo, stride=2)

        self.layer5.apply(weights_init)
        self.layer6.apply(weights_init)
        self.layer7.apply(weights_init)

        if arch == 'resnet18':
            f_in = [128, 256, 512, fi, fo, fo]
        elif arch == 'resnet34':
            f_in = [128, 256, 512, fi, fo, fo]
        elif arch == 'resnet50':
            f_in = [512, 1024, 2048, fi, fo, fo]
        else:
            raise ValueError('Architecture {} not supported'.format(arch))
        self.out1b = nn.Conv2d(f_in[0], num_anchors[0] * num_coords, kernel_size=3, padding=1)
        self.out1c = nn.Conv2d(f_in[0], num_anchors[0] * num_classes, kernel_size=3, padding=1)
        
        self.out2b = nn.Conv2d(f_in[1], num_anchors[1] * num_coords, kernel_size=3, padding=1)
        self.out2c = nn.Conv2d(f_in[1], num_anchors[1] * num_classes, kernel_size=3, padding=1)
        
        self.out3b = nn.Conv2d(f_in[2], num_anchors[2] * num_coords, kernel_size=3, padding=1)
        self.out3c = nn.Conv2d(f_in[2], num_anchors[2] * num_classes, kernel_size=3, padding=1)
 
        self.out4b = nn.Conv2d(f_in[3], num_anchors[3] * num_coords, kernel_size=3, padding=1)
        self.out4c = nn.Conv2d(f_in[3], num_anchors[3] * num_classes, kernel_size=3, padding=1)
        
        self.out5b = nn.Conv2d(f_in[4], num_anchors[4] * num_coords, kernel_size=3, padding=1)
        self.out5c = nn.Conv2d(f_in[4],  num_anchors[4] * num_classes, kernel_size=3, padding=1)
        
        self.out6b = nn.Conv2d(f_in[5], num_anchors[5] * num_coords, kernel_size=3, padding=1)
        self.out6c = nn.Conv2d(f_in[5], num_anchors[5] * num_classes, kernel_size=3, padding=1)

    def forward(self, x):
        outs = []
        x = self.base.conv1(x)
        x = self.base.bn1(x)
        x = self.base.relu(x)
        x = self.base.maxpool(x)
        x = self.base.layer1(x)
        x = self.base.layer2(x)
        outs.append((self.out1b(x), self.out1c(x)))
        x = self.base.layer3(x)
        outs.append((self.out2b(x), self.out2c(x)))
        x = self.base.layer4(x)
        outs.append((self.out3b(x), self.out3c(x)))
        x = self.layer5(x)
        outs.append((self.out4b(x), self.out4c(x)))
        x = self.layer6(x)
        outs.append((self.out5b(x), self.out5c(x)))
        x = self.layer7(x)
        outs.append((self.out6b(x), self.out6c(x)))
        return outs


class SSD_VGG(nn.Module):

    def __init__(self, num_anchors, num_coords=4, num_classes=2):
        super().__init__()
        assert len(num_anchors) == 6 # 6 scales are proposed in the SSD paper
        # base = from input to conv4_3
        self.base = nn.Sequential(
            #conv1
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # conv2
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            #conv3
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # conv4
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(True),
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            #nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            #nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            #nn.BatchNorm2d(512),
            nn.ReLU(True),
        )
        self.conv6 = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6),
            #nn.BatchNorm2d(1024),
            nn.ReLU(True),
            nn.Conv2d(1024, 1024, kernel_size=1),
            #nn.BatchNorm2d(1024),
            nn.ReLU(True)
        )
        self.conv7 = nn.Sequential(
            nn.Conv2d(1024, 256, kernel_size=1),
            #nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.Conv2d(256, 512, kernel_size=3, padding=1, stride=2),
            #nn.BatchNorm2d(512),
            nn.ReLU(True)
        )
        self.conv8 = nn.Sequential(
            nn.Conv2d(512, 128, kernel_size=1),
            #nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.Conv2d(128, 256, kernel_size=3, padding=1, stride=2),
            #nn.BatchNorm2d(256),
            nn.ReLU(True)
        )
        self.conv9 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=1),
            #nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.Conv2d(128, 256, kernel_size=3, padding=1, stride=2),
            #nn.BatchNorm2d(256),
            nn.ReLU(True)
        )
        self.conv10 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=1),
            #nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.Conv2d(128, 256, kernel_size=3, padding=0, stride=1),
            #nn.BatchNorm2d(256),
            nn.ReLU(True)
        )
        self.conv11 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=1),
            #nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.Conv2d(128, 256, kernel_size=3, padding=0, stride=1),
            #nn.BatchNorm2d(256),
            nn.ReLU(True)
        )
        self.out1b = nn.Conv2d(512, num_anchors[0] * num_coords, kernel_size=3, padding=1)
        self.out1c = nn.Conv2d(512, num_anchors[0] * num_classes, kernel_size=3, padding=1)
        
        self.out2b = nn.Conv2d(512, num_anchors[1] * num_coords, kernel_size=3, padding=1)
        self.out2c = nn.Conv2d(512, num_anchors[1] * num_classes, kernel_size=3, padding=1)
        
        self.out3b = nn.Conv2d(256, num_anchors[2] * num_coords, kernel_size=3, padding=1)
        self.out3c = nn.Conv2d(256, num_anchors[2] * num_classes, kernel_size=3, padding=1)
 
        self.out4b = nn.Conv2d(256, num_anchors[3] * num_coords, kernel_size=3, padding=1)
        self.out4c = nn.Conv2d(256, num_anchors[3] * num_classes, kernel_size=3, padding=1)
        
        self.out5b = nn.Conv2d(256, num_anchors[4] * num_coords, kernel_size=3, padding=1)
        self.out5c = nn.Conv2d(256, num_anchors[4] * num_classes, kernel_size=3, padding=1)
        
        self.out6b = nn.Conv2d(256, num_anchors[5] * num_coords, kernel_size=3, padding=1)
        self.out6c = nn.Conv2d(256, num_anchors[5] * num_classes, kernel_size=3, padding=1)

        S = 20
        self.norm1b = L2Norm(num_anchors[0] * num_coords, S)
        self.norm1c = L2Norm(num_anchors[0] * num_classes, S)
        
        self.norm2b = L2Norm(num_anchors[1] * num_coords, S)
        self.norm2c = L2Norm(num_anchors[1] * num_classes, S)
        
        self.norm3b = L2Norm(num_anchors[2] * num_coords, S)
        self.norm3c = L2Norm(num_anchors[2] * num_classes, S)
 
        self.norm4b = L2Norm(num_anchors[3] * num_coords, S)
        self.norm4c = L2Norm(num_anchors[3] * num_classes, S)
 
        self.norm5b = L2Norm(num_anchors[4] * num_coords, S)
        self.norm5c = L2Norm(num_anchors[4] * num_classes, S)

        self.norm6b = L2Norm(num_anchors[5] * num_coords, S)
        self.norm6c = L2Norm(num_anchors[5] * num_classes, S)
        self.apply(weights_init)
        # pretrained weights
        vgg16 = models.vgg16(pretrained=True)
        for i in range(len(self.base)):
            if hasattr(self.base[i], 'weight'):
                self.base[i].weight = vgg16.features[i].weight
                self.base[i].bias = vgg16.features[i].bias

    def forward(self, x):
        outs = []
        x = self.base(x)
        outs.append((
            (self.norm1b(self.out1b(x))),
            (self.norm1c(self.out1c(x)))
        ))
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        outs.append((
            (self.norm2b(self.out2b(x))),
            (self.norm2c(self.out2c(x)))
        ))
        x = self.conv8(x)
        outs.append((
            (self.norm3b(self.out3b(x))),
            (self.norm3c(self.out3c(x)))
        ))
        x = self.conv9(x)
        outs.append((
            (self.norm4b(self.out4b(x))),
            (self.norm4c(self.out4c(x)))
        ))
        x = self.conv10(x)
        outs.append((
            (self.norm5b(self.out5b(x))),
            (self.norm5c(self.out5c(x)))
        ))
        x = self.conv11(x)
        outs.append((
            (self.norm6b(self.out6b(x))),
            (self.norm6c(self.out6c(x)))
        ) )
        return outs


def weights_init(m):
    if isinstance(m, nn.Conv2d):
        xavier_uniform(m.weight.data)


class L2Norm(nn.Module):
    '''L2Norm layer across all channels.'''
    def __init__(self, in_features, scale):
        super(L2Norm, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(in_features))
        self.reset_parameters(scale)

    def reset_parameters(self, scale):
        nn.init.constant(self.weight, scale)

    def forward(self, x):
        x = F.normalize(x, dim=1)
        scale = self.weight[None,:,None,None]
        return scale * x


if __name__ == '__main__':
    m = SSD_Resnet()
    x = Variable(torch.randn(1, 3, 300, 300))
    outs = m(x)
    for a, b in outs:
        print(a.size(), b.size())
