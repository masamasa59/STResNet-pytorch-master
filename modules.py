
import torch
import numpy as np
from params import Params as param

def ResUnit(inputs, filters, kernel_size, strides, scope, reuse=None):
    '''
    Defines a residual unit
    input -> [layernorm->relu->conv] X 2 -> reslink -> output
    '''
    # use layernorm before applying convolution
    outputs = torch.nn.LayerNorm(inputs)
    # apply relu activation
    outputs = torch.F.relu(outputs)
    # perform a 2D convolution
    outputs = torch.nn.Conv2d(outputs, filters, kernel_size, strides, padding=0)
    # use layernorm before applying convolution
    outputs = torch.nn.LayerNorm(outputs)
    # relu activation
    outputs = torch.F.relu(outputs)
    # perform a 2D convolution
    outputs = torch.nn.Conv2d(outputs, filters, kernel_size, strides, padding=0)
    # add a residual link
    outputs += inputs
    return outputs


def ResNet(inputs, filters, kernel_size, repeats, scope, reuse=None):
    '''
    Defines the ResNet architecture
    '''

    #apply repeats number of residual layers
    for layer_id in range(repeats):
        inputs = ResUnit(inputs, filters, kernel_size, (1,1))
    outputs = torch.F.relu(inputs)
    return outputs


def ResInput(inputs, filters, kernel_size, scope, reuse=None):
    '''
    Defines the first (input) layer of the ResNet architecture
    '''

    outputs = torch.nn.conv2d(inputs, filters, kernel_size, strides=(1,1), padding=0)
    return outputs


def ResOutput(inputs, filters, kernel_size, scope, reuse=None):
    '''
    Defines the last (output) layer of the ResNet architecture
    '''

    #applying the final convolution to the tec map with depth 1 (num of filters=1)
    outputs = torch.nn.conv2d(inputs, filters, kernel_size, strides=(1,1), padding=0)
    return outputs


def Fusion(closeness_output, period_output, trend_output, scope, shape):
    '''
    Combining the output from the module into one tec map
    '''
    closeness_output = torch.squeeze(closeness_output)
    period_output = torch.squeeze(period_output)
    trend_output = torch.squeeze(trend_output)
    # apply a linear transformation to each of the outputs: closeness, period, trend and then combine
    Wc = torch.get_variable("closeness_matrix", dtype=torch.float32, shape=shape, initializer=torch.contrib.layers.xavier_initializer(), trainable=True)
    Wp = torch.get_variable("period_matrix", dtype=torch.float32, shape=shape, initializer=torch.contrib.layers.xavier_initializer(), trainable=True)
    Wt = torch.get_variable("trend_matrix", dtype=torch.float32, shape=shape, initializer=torch.contrib.layers.xavier_initializer(), trainable=True)

    output = torch.reshape(closeness_output, [closeness_output.shape[0]*closeness_output.shape[1], closeness_output.shape[2]])
    output = torch.matmul(output, Wc)
    closeness_output = torch.reshape(output, [closeness_output.shape[0], closeness_output.shape[1], closeness_output.shape[2]])

    output = torch.reshape(period_output, [period_output.shape[0]*period_output.shape[1], period_output.shape[2]])
    output = torch.matmul(output, Wp)
    period_output = torch.reshape(output, [period_output.shape[0], period_output.shape[1], period_output.shape[2]])

    output = torch.reshape(trend_output, [trend_output.shape[0]*trend_output.shape[1], trend_output.shape[2]])
    output = torch.matmul(output, Wt)
    trend_output = torch.reshape(output, [trend_output.shape[0], trend_output.shape[1], trend_output.shape[2]])
    # fusion
    outputs = torch.add(torch.add(closeness_output, period_output), trend_output)
    # adding non-linearity
    outputs = torch.F.tanh(outputs)
    # converting the dimension from (B, H, W) -> (B, H, W, 1) to match ground truth labels
    outputs = torch.expand_dims(outputs, axis=3)
    return outputs
