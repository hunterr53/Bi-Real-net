import os
import pandas as pd
import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F
import numpy as np

__all__ = ['birealnet18', 'birealnet34']

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class BinaryActivation(nn.Module):
    def __init__(self):
        super(BinaryActivation, self).__init__()

    def forward(self, x):
        out_forward = torch.sign(x)
        #out_e1 = (x^2 + 2*x)
        #out_e2 = (-x^2 + 2*x)
        out_e_total = 0
        mask1 = x < -1
        mask2 = x < 0
        mask3 = x < 1
        out1 = (-1) * mask1.type(torch.float32) + (x*x + 2*x) * (1-mask1.type(torch.float32))
        out2 = out1 * mask2.type(torch.float32) + (-x*x + 2*x) * (1-mask2.type(torch.float32))
        out3 = out2 * mask3.type(torch.float32) + 1 * (1- mask3.type(torch.float32))
        out = out_forward.detach() - out3.detach() + out3

        return out

class HardBinaryConv(nn.Module):
    def __init__(self, in_chn, out_chn, kernel_size=3, stride=1, padding=1):
        super(HardBinaryConv, self).__init__()
        self.stride = stride
        self.padding = padding
        self.number_of_weights = in_chn * out_chn * kernel_size * kernel_size
        self.shape = (out_chn, in_chn, kernel_size, kernel_size)
        self.weights = nn.Parameter(torch.rand((self.number_of_weights,1)) * 0.001, requires_grad=True)

    def forward(self, x):
        real_weights = self.weights.view(self.shape)
        scaling_factor = torch.mean(torch.mean(torch.mean(abs(real_weights),dim=3,keepdim=True),dim=2,keepdim=True),dim=1,keepdim=True)
        #print(scaling_factor, flush=True)
        scaling_factor = scaling_factor.detach()
        binary_weights_no_grad = scaling_factor * torch.sign(real_weights)
        cliped_weights = torch.clamp(real_weights, -1.0, 1.0)
        binary_weights = binary_weights_no_grad.detach() - cliped_weights.detach() + cliped_weights
        #print(binary_weights, flush=True)
        y = F.conv2d(x, binary_weights, stride=self.stride, padding=self.padding)

        # Done as 2nd step. "
        # we constraint the weights to -1 and 1, and set the learning rate
            # in all convolution layers to 0 and retrain the BatchNorm layer for 1 epoch to
            # absorb the scaling factor."
        actualBinaryWeights = torch.sign(real_weights)
        temp = F.conv2d(x, actualBinaryWeights, stride=self.stride, padding=self.padding)

        return y

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()

        self.binary_activation = BinaryActivation()
        self.binary_conv = HardBinaryConv(inplanes, planes, stride=stride)
        self.bn1 = nn.BatchNorm2d(planes)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        isPrint = True
        residual = x
        if isPrint: saveFeaturesCsv(x, 'PyResidual0_1')

        out = self.binary_activation(x)
        if isPrint: saveFeaturesCsv(out, 'PyBinaryAct0_1')

        out = self.binary_conv(out)
        if isPrint: saveFeaturesCsv(out, 'PyConv0_1')
        
        out = self.bn1(out)
        if isPrint: saveFeaturesCsv(out, 'PyBN0_1')

        if self.downsample is not None:
            # print(residual.shape, 'pre downsample')
            residual = self.downsample(x)

        # print('Residual:', residual.shape, '- Out', out.shape)
        out += residual

        return out

class BiRealNet(nn.Module):
    def __init__(self, block, layers, num_classes=10, zero_init_residual=False):
        super(BiRealNet, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2) 
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2) 
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.AvgPool2d(kernel_size=2, stride=stride),
                conv1x1(self.inplanes, planes * block.expansion),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x, isPrint=False):
        x = x.to(torch.float32)
        if isPrint: saveFeaturesCsv(x, 'input0')
        x = self.conv1(x)
        if isPrint: saveFeaturesCsv(x, 'conv1')
        x = self.bn1(x)
        if isPrint: saveFeaturesCsv(x, 'bn1_1')
        x = self.maxpool(x)
        if isPrint: saveFeaturesCsv(x, 'maxpool_1')

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def birealnet18(pretrained=False, **kwargs):
    """Constructs a BiRealNet-18 model. """
    model = BiRealNet(BasicBlock, [4, 4, 4, 4], **kwargs)
    return model


def birealnet34(pretrained=False, **kwargs):
    """Constructs a BiRealNet-34 model. """
    model = BiRealNet(BasicBlock, [6, 8, 12, 6], **kwargs)
    return model

def saveFeaturesCsv(x, name):
    with torch.no_grad():
        path = 'pytorch_implementation/BiReal18_34/savedWeights/' + 'features_' + name + '.csv'
        # np.savetxt(path, x.cpu().detach().numpy(), delimiter=',')
        
        try:
            os.remove(path)
            print(f"File '{path}' successfully deleted.")
        except FileNotFoundError:
            print(f"Error: File '{path}' not found.")
        except PermissionError:
            print(f"Error: Permission denied to delete '{path}'.")
        except OSError as e:
             print(f"Error: An unexpected error occurred while deleting '{path}': {e}")
        
        for i, kernel in enumerate(x[0]):
            test_df = pd.DataFrame(kernel.numpy().astype(np.float32))
            test_df.to_csv(path, index=False, mode = 'a', header=True)
            if i > 3:
                break # Only save first 3 kernels
