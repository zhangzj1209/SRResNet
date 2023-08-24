import torch
import torch.nn as nn
import math

# Define the residual module
class _Residual_Block(nn.Module):
    def __init__(self):
        super(_Residual_Block, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn = nn.BatchNorm1d(64)
        self.relu = nn.PReLU(num_parameters=1, init=0.2)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        
    def forward(self, x):
        output = self.conv1(x)
        output = self.bn(output)
        output = self.relu(output)
        output = self.conv2(output)
        output = self.bn(output)
        output = torch.add(output, x)
        return output

# Define the SRResNet network architecture
class SRResNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        super(SRResNet, self).__init__()
        self.conv_input = nn.Conv1d(in_channels=in_channels, out_channels=64, kernel_size=9, stride=1, padding=4, bias=False)
        self.relu = nn.PReLU(num_parameters=1, init=0.2)
        self.residual = self.make_layer(_Residual_Block, 16)  # 16 residual modules
        self.conv_mid = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn_mid = nn.BatchNorm1d(64)
        self.conv_output = nn.Conv1d(in_channels=64, out_channels=out_channels, kernel_size=9, stride=1, padding=4, bias=False)
        
        # init the weight of conv1d
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
            #    n = m.kernel_size[0]*m.kernel_size[1]*m.out_channels
                n = m.kernel_size[0]*m.out_channels
                m.weight.data.normal_(0, math.sqrt(2./n))
                if m.bias is not None:
                    m.bias.data.zero_()
                    
    def make_layer(self, block, num_of_layer):
        layers = []
        for _ in range(num_of_layer):
            layers.append(block())
        return nn.Sequential(*layers)
    
    def forward(self, x):
        out1 = self.relu(self.conv_input(x))
        out = self.residual(out1)
        out = self.conv_mid(out)
        out = self.bn_mid(out)
        out = torch.add(out, out1)
        out = self.conv_output(out)
        return out