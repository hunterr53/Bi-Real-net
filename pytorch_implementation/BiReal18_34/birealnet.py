import os
import pandas as pd
import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F
import numpy as np

globalCounter = 0
globalImageNum = 0
maxInt = 0 # Used to find max fixed point value
minDecimal = 0.999999

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

        # return out
        return out_forward # Match C Code. Don't need piecewise function during inference

class HardBinaryConv(nn.Module):
    def __init__(self, in_chn, out_chn, kernel_size=3, stride=1, padding=1):
        super(HardBinaryConv, self).__init__()
        self.stride = stride
        self.padding = padding
        self.number_of_weights = in_chn * out_chn * kernel_size * kernel_size
        self.shape = (out_chn, in_chn, kernel_size, kernel_size)
        self.weights = nn.Parameter(torch.rand((self.number_of_weights,1)) * 0.001, requires_grad=True)
        self.requires_grad_(False) # Required for test to absorb BN

    def forward(self, x):
        real_weights = self.weights.view(self.shape)
        scaling_factor = torch.mean(torch.mean(torch.mean(abs(real_weights),dim=3,keepdim=True),dim=2,keepdim=True),dim=1,keepdim=True)
        #print(scaling_factor, flush=True)
        scaling_factor = scaling_factor.detach()
        binary_weights_no_grad = scaling_factor * torch.sign(real_weights)
        cliped_weights = torch.clamp(real_weights, -1.0, 1.0)
        binary_weights = binary_weights_no_grad.detach() - cliped_weights.detach() + cliped_weights
        #print(binary_weights, flush=True)
        y = F.conv2d(x, binary_weights, stride=self.stride, padding=self.padding) # For training

        # Done as 2nd step.
            # "we constraint the weights to -1 and 1, and set the learning rate
            # in all convolution layers to 0 and retrain the BatchNorm layer for 1 epoch to
            # absorb the scaling factor."
        actualBinaryWeights = torch.sign(real_weights)
        temp = F.conv2d(x, actualBinaryWeights, stride=self.stride, padding=self.padding) # For inference

        return temp

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
        global globalCounter, maxInt, minDecimal
        isPrint = False
        residual = x
        if isPrint: saveFeaturesCsv(residual,  str(globalCounter) + '_PyResidual')
        if np.max(np.abs(x.numpy())) > maxInt:
            maxInt = np.max(np.abs(x.numpy())) # Finding max fixed point value
        decTemp = np.modf(x.numpy())[0]
        if np.min(np.abs(decTemp)) < minDecimal and np.min(np.abs(decTemp)) != 0:
            minDecimal = np.min(np.abs(decTemp)) # Finding most precise decimal value

        out = self.binary_activation(x)
        if isPrint: saveFeaturesCsv(out, str(globalCounter) + '_PyBinaryAct')
        if np.max(np.abs(out.numpy())) > maxInt:
            maxInt = np.max(np.abs(out.numpy())) # Finding max fixed point value
        decTemp = np.modf(out.numpy())[0]
        if np.min(np.abs(decTemp)) < minDecimal and np.min(np.abs(decTemp)) != 0:
            minDecimal = np.min(np.abs(decTemp)) # Finding most precise decimal value

        out = self.binary_conv(out)
        if isPrint: saveFeaturesCsv(out, str(globalCounter) + '_PyConv')
        if np.max(np.abs(out.numpy())) > maxInt:
            maxInt = np.max(np.abs(out.numpy())) # Finding max fixed point value
        decTemp = np.modf(out.numpy())[0]
        if np.min(np.abs(decTemp)) < minDecimal and np.min(np.abs(decTemp)) != 0:
            minDecimal = np.min(np.abs(decTemp)) # Finding most precise decimal value
        
        out = self.bn1(out)
        if isPrint: saveFeaturesCsv(out, str(globalCounter) + '_PyBN')
        if np.max(np.abs(out.numpy())) > maxInt:
            maxInt = np.max(np.abs(out.numpy())) # Finding max fixed point value
        decTemp = np.modf(out.numpy())[0]
        if np.min(np.abs(decTemp)) < minDecimal and np.min(np.abs(decTemp)) != 0:
            minDecimal = np.min(np.abs(decTemp)) # Finding most precise decimal value

        if self.downsample is not None:
            # print(residual.shape, 'pre downsample')
            residual = self.downsample(x)
            if isPrint: saveFeaturesCsv(residual, str(globalCounter) + '_PyDownSample')
            if np.max(np.abs(residual.numpy())) > maxInt:
                maxInt = np.max(np.abs(residual.numpy())) # Finding max fixed point value
            if np.min(np.modf(residual.numpy())) > minDecimal:
                minDecimal = np.min(np.modf(x.numpy())) # Finding most precise decimal value

            # downSamp = nn.Sequential(nn.AvgPool2d(kernel_size=2, stride=self.stride)) # Won't work with Conv/BN layers b/c of weights
            # residualDebug = downSamp(x)
            # if isPrint: saveFeaturesCsv(residualDebug, str(globalCounter) + '_PyAvgPool')

        # print('Residual:', residual.shape, '- Out', out.shape)
        out += residual
        if isPrint: saveFeaturesCsv(out, str(globalCounter) + '_PyAdd')
        if np.max(np.abs(out.numpy())) > maxInt:
            maxInt = np.max(np.abs(out.numpy())) # Finding max fixed point value
        decTemp = np.modf(out.numpy())[0]
        if np.min(np.abs(decTemp)) < minDecimal and np.min(np.abs(decTemp)) != 0:
            minDecimal = np.min(np.abs(decTemp)) # Finding most precise decimal value

        globalCounter += 1
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
        global globalCounter
        global globalImageNum
        global maxInt, minDecimal

        x = x.to(torch.float32)
        if np.max(np.abs(x.numpy())) > maxInt:
            maxInt = np.max(np.abs(x.numpy())) # Finding max fixed point value
        decTemp = np.modf(x.numpy())[0]
        if np.min(np.abs(decTemp)) < minDecimal and np.min(np.abs(decTemp)) != 0:
            minDecimal = np.min(np.abs(decTemp)) # Finding most precise decimal value
        if isPrint: saveFeaturesCsv(x, 'input0')

        x = self.conv1(x)
        if np.max(np.abs(x.numpy())) > maxInt:
            maxInt = np.max(np.abs(x.numpy())) # Finding max fixed point value
        decTemp = np.modf(x.numpy())[0]
        if np.min(np.abs(decTemp)) < minDecimal and np.min(np.abs(decTemp)) != 0:
            minDecimal = np.min(np.abs(decTemp)) # Finding most precise decimal value
        if isPrint: saveFeaturesCsv(x, 'conv1')

        x = self.bn1(x)
        if np.max(np.abs(x.numpy())) > maxInt:
            maxInt = np.max(np.abs(x.numpy())) # Finding max fixed point value
        decTemp = np.modf(x.numpy())[0]
        if np.min(np.abs(decTemp)) < minDecimal and np.min(np.abs(decTemp)) != 0:
            minDecimal = np.min(np.abs(decTemp)) # Finding most precise decimal value
        if isPrint: saveFeaturesCsv(x, 'bn1')

        x = self.maxpool(x)
        if np.max(np.abs(x.numpy())) > maxInt:
            maxInt = np.max(np.abs(x.numpy())) # Finding max fixed point value
        decTemp = np.modf(x.numpy())[0]
        if np.min(np.abs(decTemp)) < minDecimal and np.min(np.abs(decTemp)) != 0:
            minDecimal = np.min(np.abs(decTemp)) # Finding most precise decimal value
        if isPrint: saveFeaturesCsv(x, 'maxpool1')

        x = self.layer1(x)
        if np.max(np.abs(x.numpy())) > maxInt:
            maxInt = np.max(np.abs(x.numpy())) # Finding max fixed point value
        decTemp = np.modf(x.numpy())[0]
        if np.min(np.abs(decTemp)) < minDecimal and np.min(np.abs(decTemp)) != 0:
            minDecimal = np.min(np.abs(decTemp)) # Finding most precise decimal value

        x = self.layer2(x)
        if np.max(np.abs(x.numpy())) > maxInt:
            maxInt = np.max(np.abs(x.numpy())) # Finding max fixed point value
        decTemp = np.modf(x.numpy())[0]
        if np.min(np.abs(decTemp)) < minDecimal and np.min(np.abs(decTemp)) != 0:
            minDecimal = np.min(np.abs(decTemp)) # Finding most precise decimal value

        x = self.layer3(x)
        if np.max(np.abs(x.numpy())) > maxInt:
            maxInt = np.max(np.abs(x.numpy())) # Finding max fixed point value
        decTemp = np.modf(x.numpy())[0]
        if np.min(np.abs(decTemp)) < minDecimal and np.min(np.abs(decTemp)) != 0:
            minDecimal = np.min(np.abs(decTemp)) # Finding most precise decimal value

        x = self.layer4(x)
        if np.max(np.abs(x.numpy())) > maxInt:
            maxInt = np.max(np.abs(x.numpy())) # Finding max fixed point value
        decTemp = np.modf(x.numpy())[0]
        if np.min(np.abs(decTemp)) < minDecimal and np.min(np.abs(decTemp)) != 0:
            minDecimal = np.min(np.abs(decTemp)) # Finding most precise decimal value

        x = self.avgpool(x) # Tensor([1, 512, 1, 1])
        if np.max(np.abs(x.numpy())) > maxInt:
            maxInt = np.max(np.abs(x.numpy())) # Finding max fixed point value
        decTemp = np.modf(x.numpy())[0]
        if np.min(np.abs(decTemp)) < minDecimal and np.min(np.abs(decTemp)) != 0:
            minDecimal = np.min(np.abs(decTemp)) # Finding most precise decimal value
        if isPrint: saveFeaturesCsv(x, 'avgpoolFC')

        x = x.view(x.size(0), -1)
        x = self.fc(x)
        if np.max(np.abs(x.numpy())) > maxInt:
            maxInt = np.max(np.abs(x.numpy())) # Finding max fixed point value
        decTemp = np.modf(x.numpy())[0]
        if np.min(np.abs(decTemp)) < minDecimal and np.min(np.abs(decTemp)) != 0:
            minDecimal = np.min(np.abs(decTemp)) # Finding most precise decimal value
        print(f"Max Int: {maxInt}\tMinDec: {minDecimal}")

        globalCounter = 0
        globalImageNum += 1
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
    global globalImageNum
    with torch.no_grad():
        directory_name = 'pytorch_implementation/BiReal18_34/savedWeights/image_' + str(globalImageNum)
        path = directory_name + '/features_' + name + '.csv'
        # np.savetxt(path, x.cpu().detach().numpy(), delimiter=',')
        # Create the directory
        try:
            os.mkdir(directory_name)
            # print(f"Directory '{directory_name}' created successfully.")
        except FileExistsError:
            pass
            # print(f"Directory '{directory_name}' already exists.")
        except PermissionError:
            print(f"Permission denied: Unable to create '{directory_name}'.")
        except Exception as e:
            print(f"An error occurred: {e}")
            
        try:
            os.remove(path)
            # print(f"File '{path}' successfully deleted.")
        except FileNotFoundError:
            pass
            # print(f"Error: File '{path}' not found.")
        except PermissionError:
            print(f"Error: Permission denied to delete '{path}'.")
        except OSError as e:
             print(f"Error: An unexpected error occurred while deleting '{path}': {e}")
        
        for i, kernel in enumerate(x[0]):
            if i >= 3:
                break # Only save first 3 kernels
            test_df = pd.DataFrame(kernel.numpy().astype(np.float32))
            test_df.to_csv(path, index=False, mode = 'a', header=True)
