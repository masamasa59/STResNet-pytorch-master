from parames import Params as param
import torch
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable
from torch import nn
class ResUnit(nn.Module):
    """docstring for ."""

    def __init__(self,  input_channels, hidden_channels, kernel_size):
        super(ResUnit, self).__init__()

        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.padding = (kernel_size - 1) // 2

        self.conv1 = torch.nn.Conv2d(self.input_channels, self.hidden_channels, self.kernel_size, 1, self.padding)
        self.conv2 = torch.nn.Conv2d(self.input_channels, self.hidden_channels, self.kernel_size, 1, self.padding)
        self.batchnorm2d_1 =torch.nn.BatchNorm2d(self.hidden_channels)
        self.batchnorm2d_2 =torch.nn.BatchNorm2d(self.hidden_channels)
    def forward(self, inputs):
        '''
        Defines a residual unit
        input -> [layernorm->relu->conv] X 2 -> reslink -> output
        '''
        # use layernorm before applying convolution
        outputs = self.batchnorm2d_1(inputs)
        # apply relu activation
        outputs = F.relu(outputs)
        # perform a 2D convolution
        outputs = self.conv1(outputs)
        # use layernorm before applying convolution
        outputs = self.batchnorm2d_2(outputs)
        # relu activation
        outputs = F.relu(outputs)
        # perform a 2D convolution
        outputs = self.conv2(outputs)
        # add a residual link
        outputs += inputs
        return outputs


class ST_ResNets(nn.Module):
    def __init__(self, hidden_channels, kernel_size):
        super(ST_ResNets, self).__init__()

        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.padding = (kernel_size - 1) // 2

        self.B, self.H, self.W, self.C, self.P, self.T= param.batch_size, param.map_height, param.map_width, param.closeness_sequence_length*param.nb_flow,param.period_sequence_length*param.nb_flow, param.trend_sequence_length*param.nb_flow
        self.O, self.F, self.U  =  param.num_of_output ,param.num_of_filters, param.num_of_residual_units,

        z_args = (self.hidden_channels,self.hidden_channels,self.kernel_size)

        self.ResInput_C = torch.nn.Conv2d(self.C, self.hidden_channels, self.kernel_size, 1, self.padding)
        self.ResNet_C = nn.ModuleList([ResUnit(*z_args) for _ in range(param.num_of_residual_units)])
        self.ResOutput_C = torch.nn.Conv2d(self.hidden_channels, 1, self.kernel_size, 1, self.padding)

        self.ResInput_T = torch.nn.Conv2d(self.T, self.hidden_channels, self.kernel_size, 1, self.padding)
        self.ResNet_T = nn.ModuleList([ResUnit(*z_args) for _ in range(param.num_of_residual_units)])
        self.ResOutput_T = torch.nn.Conv2d(self.hidden_channels, 1, self.kernel_size, 1, self.padding)

        self.ResInput_P = torch.nn.Conv2d(self.P, self.hidden_channels, self.kernel_size, 1, self.padding)
        self.ResNet_P =nn.ModuleList([ResUnit(*z_args) for _ in range(param.num_of_residual_units)])
        self.ResOutput_P =torch.nn.Conv2d(self.hidden_channels, 1, self.kernel_size, 1, self.padding)

        self.Wc = torch.nn.Parameter(torch.randn(self.W,self.W))
        self.Wp = torch.nn.Parameter(torch.randn(self.W,self.W))
        self.Wt = torch.nn.Parameter(torch.randn(self.W,self.W))

    def forward(self,x):
        # ResNet architecture for the three modules

        x = x.permute(1,0,2,3,4)#batchと時系列の３要素を入れ替え[3part,batch,sequence,H,W]
        c_inp = x[0]#closeness[batch,sequence,H,W]
        p_inp = x[1]#period[batch,sequence,H,W]
        t_inp = x[2]#trend[batch,sequence,H,W]

        # module 1: capturing closeness (recent)
        closeness_output = self.ResInput_C(c_inp)#shape=[B, C, H, W]
        for i, ResUnit in enumerate(self.ResNet_C):
            closeness_output = ResUnit(closeness_output)
        closeness_output = F.relu(closeness_output)
        closeness_output = self.ResOutput_C(closeness_output)#[B, 1, H, W]

        # module 2: capturing period (near)
        period_output = self.ResInput_P(p_inp)#shape=[B, P, H, W]
        for i, ResUnit in enumerate(self.ResNet_P):
            period_output = ResUnit(period_output)
        period_output = F.relu(period_output)
        period_output = self.ResOutput_P(period_output)#shape=[B, 1, H, W]

        # module 3: capturing trend (distant)
        trend_output = self.ResInput_T(t_inp)#shape=[B, T, H, W]
        for i, ResUnit in enumerate(self.ResNet_T):
            trend_output = ResUnit(trend_output)
        trend_output = F.relu(trend_output)
        trend_output = self.ResOutput_T(trend_output)#[B, 1, H, W]
        '''
        Combining the output from the module into one tec map
        '''

        # parameter matrix based fusion

        closeness_output = torch.matmul(closeness_output, self.Wc)
        period_output = torch.matmul(period_output, self.Wp)
        trend_output = torch.matmul(trend_output, self.Wt)
        # fusion
        outputs = torch.add(torch.add(closeness_output, period_output), trend_output)
        # adding non-linearity
        outputs = F.tanh(outputs)

        return outputs
