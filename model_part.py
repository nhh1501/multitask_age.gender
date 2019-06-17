import torch.nn as nn
from torch.autograd import Variable
import torch

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