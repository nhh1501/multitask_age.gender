# 1 cross-stitch
import torch
import torch.nn as nn
import torch.nn.functional as F
from model_part import *
class HNet(nn.Module):
    def __init__(self):
        super(HNet, self).__init__()
        # 1 input image channel, 2 output channels, 5x5 square convolution
        # kernel
        self.conv1 = double_conv(3,64)
        self.conv2 = double_conv(64,192)
        self.conv3 = double_conv(192,384)
        ##################
        self.conv4_a = double_conv1(384, 256)
        self.fc_a = outfully(4096,4,1024)
        self.conv4_b = double_conv1(384, 256)
        self.fc_b = outfully(4096,2,1024)

        self.cross_stich1 = cross_stich(8192, 8192);  #####
        ########################################
    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = F.max_pool2d(self.conv1(x), 2);
        x = F.max_pool2d(self.conv2(x), 2);
        x = F.max_pool2d(self.conv3(x), 2);

        x1 = self.conv4_a(x)
        x2 = self.conv4_b(x)
        x1, x2 = self.cross_stich1(x1, x2);
        x1 = self.fc_a(x1)
        x2 = self.fc_b(x2)

        output1 = F.log_softmax(x1, dim=1)
        output2 = F.log_softmax(x2, dim=1)

        return output1, output2



