import torch
import torch.nn as nn


class SpatialNMS(nn.Module):
    def __init__(self, kernel_size, stride, beta):
        super(SpatialNMS, self).__init__()
        self.kernel_size = kernel_size
        self.beta = beta
        self.strides = stride

    def call(self, inputs):
        P = inputs
        Q = nn.MaxPool2d(self.kernel_size, self.strides, padding=1)
        P_Q = torch.abs(P-Q)
        exp_P_Q = torch.exp(-P_Q * self.beta)
        P2 = torch.mul([P,exp_P_Q])
        return P2

class SmoothStepFunction(nn.Module):
    def __init__(self, threshold, beta):
        super(SmoothStepFunction, self).__init__()
        self.threshold = threshold
        self.beta = beta

    def call(self, inputs):
        """
        input: 4D Tensor (b_size, weight, high, f_maps)
        output: 3D Tensor (b_size, sum_of_w_AND_h, f_maps)
        """
        threshold_factor = torch.ones_like(inputs) * self.threshold
        sigmoid_input = torch.sub(inputs, threshold_factor) * self.beta
        return torch.sigmoid(sigmoid_input)


class StepFunction(nn.Module):
    def __init__(self, threshold=0.1, *args, **kwargs):
        self.threshold = threshold
        super(StepFunction, self).__init__(*args, **kwargs)

    def call(self, inputs, **kwargs):
        """
        input: 4D Tensor (b_size, weight, high, f_maps)
        output: 3D Tensor (b_size, sum_of_w_AND_h, f_maps)
        """
        threshold_factor = torch.ones_like(inputs) * self.threshold
        return torch.where(inputs <= threshold_factor, 0, 1)


class GlobalSumPooling2D(nn.Module):
    def __init__(self, *args, **kwargs):
        super(GlobalSumPooling2D, self).__init__(*args, **kwargs)

    def call(self, inputs):
        """
        input: 4D Tensor (b_size, width, height, f_maps)
        output: 3D Tensor (b_size, sum_of_w_&_h, f_maps)
        """
        return torch.sum(inputs, (1,2))