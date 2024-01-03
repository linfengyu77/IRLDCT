import torch
import torch.nn as nn


def normalize_(image, MIN_B=-1024.0, MAX_B=3072.0):
   image = (image - MIN_B) / (MAX_B - MIN_B)
   return image

def denormalize_(image, norm_range_max = 3072, norm_range_min = -1024):
   image = image * (norm_range_max - norm_range_min) + norm_range_min
   return image

def trunc(mat, trunc_min=-1024.0, trunc_max=3072.0):
   mat[mat <= trunc_min] = trunc_min
   mat[mat >= trunc_max] = trunc_max
   return mat

class TV1Loss(nn.Module):
    def __init__(self):
        super(TV1Loss, self).__init__()

    def forward(self, x):
        batch_size = x.size()[0]
        # h_x = x.size()[2]
        # w_x = x.size()[3]
        count_h = self._tensor_size(x[:, :, 1:, :])
        count_w = self._tensor_size(x[:, :, :, 1:])
        h_tv = torch.abs((x[:, :, 1:, :]-x[:, :, :-1, :])).sum()
        w_tv = torch.abs((x[:, :, :, 1:]-x[:, :, :, :-1])).sum()
        return (h_tv/count_h+w_tv/count_w)/batch_size

    def _tensor_size(self, t):
        return t.size()[1]*t.size()[2]*t.size()[3]


class TV2Loss(nn.Module):
    def __init__(self):
        super(TV2Loss, self).__init__()

    def forward(self, x):
        batch_size = x.size()[0]
        # h_x = x.size()[2]
        # w_x = x.size()[3]
        count_h = self._tensor_size(x[:, :, 1:, :])
        count_w = self._tensor_size(x[:, :, :, 1:])
        # h_tv = torch.sqrt(torch.pow((x[:, :, 1:, :]-x[:, :, :h_x-1, :]), 2)).sum()
        # w_tv = torch.sqrt(torch.pow((x[:, :, :, 1:]-x[:, :, :, :w_x-1]), 2)).sum()
        # h_1st_gra = x[:, :, 1:, :]-x[:, :, :-1, :]
        # w_1st_gra = x[:, :, :, 1:]-x[:, :, :, :-1]
        h_tv = torch.abs(x[:, :, 2:, :]-2*x[:, :, 1:-1, :]+x[:, :, :-2, :]).sum()
        w_tv = torch.abs(x[:, :, :, 2:]-2*x[:, :, :, 1:-1]+x[:, :, :, :-2]).sum()
        # hw_tv = torch.abs(x[:, :, 2:, 2:]+x[:, :, :-2, :-2]-x[:, :, :-2, 2:]-x[:, :, 2:, :-2]).sum()

        return (h_tv/count_h+w_tv/count_w)/batch_size

    def _tensor_size(self, t):
        return t.size()[1]*t.size()[2]*t.size()[3]