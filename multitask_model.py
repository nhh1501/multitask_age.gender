# 1 cross-stitch
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
# from main import batch_size
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


class cross_stich(nn.Module):

    def __init__(self, input, input1):
        super(cross_stich, self).__init__()
        # self.cross = Variable(torch.randn(input, input1).type(torch.cuda.FloatTensor), requires_grad=True)
        self.cross = Variable(torch.randn(input, input1).type(torch.FloatTensor), requires_grad=True)

    def forward(self, input1, input2):
        input1_reshaped = input1.view(input1.size(0), -1);
        input2_reshaped = input2.view(input2.size(0), -1);
        x = torch.cat((input1_reshaped, input2_reshaped), 1);
        output = torch.matmul(x,self.cross)
        output1 = output[:, :input1_reshaped.shape[1]].view(input1.shape);
        output2 = output[:, input1_reshaped.shape[1]:].view(input2.shape);
        return output1, output2

class double_conv(nn.Module):
    '''(conv => BN => ReLU) * 2'''
    def __init__(self, in_ch, out_ch):
        super(double_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1,padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x

class double_conv1(nn.Module):
    '''(conv => BN => ReLU) * 2'''
    def __init__(self, in_ch, out_ch):
        super(double_conv1, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x

class conv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x

class outfully(nn.Module):
    def __init__(self, input, output,hidden):
        super(outfully, self).__init__()
        self.fully = nn.Sequential(
            nn.Linear(input, hidden),
            nn.Linear(hidden, hidden),
            nn.Linear(hidden, output)
        )
    def forward(self, x):
        x = x.view(x.size()[0],-1);
        x = self.fully(x)
        return x