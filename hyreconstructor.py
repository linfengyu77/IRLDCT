import sys
sys.path.append('./dip-ct-benchmark')
from tqdm import tqdm
import torch
from torch import nn
import os
import numpy as np
from copy import deepcopy
from torch.optim import AdamW
from torch.nn import L1Loss
from odl.contrib.torch import OperatorModule
from odl.tomo import fbp_op
from dival.reconstructors import IterativeReconstructor
from dival.measure import PSNR, SSIM
from networks import CNN, ResNet
from dliplib.reconstructors.base import BaseLearnedReconstructor
from dliplib.utils.models import get_unet_model
from utils import TV1Loss, TV2Loss

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

class HybirdReconstructor(IterativeReconstructor):
    '''
    HYPER_PARAMS = {
        'gamma':
            {'default': 1e-4,
             'range': [1e-7, 1e-0],
             'grid_search_options': {'num_samples': 20}},
        'photons_per_pixel':  # 'poisson' loss function
            {'default': 4096},
        'init_filter_type':
            {'default': 'Hann'},
        'init_frequency_scaling':
            {'default': 0.1},     
        'iterations':
            {'default': 10,
             'range': [1, 50000]},
    }
    '''
    HYPER_PARAMS = deepcopy(BaseLearnedReconstructor.HYPER_PARAMS)
    HYPER_PARAMS.update({
        'scales': {
            'default': 5,
            'retrain': True
        },
        'skip_channels': {
            'default': 4,
            'retrain': True
        },
        'channels': {
            'default': (32, 32, 64, 64, 128, 128),
            'retrain': True
        },
        'filter_type': {
            'default': 'Hann',
            'retrain': True
        },
        'frequency_scaling': {
            'default': 1.0,
            'retrain': True
        },
        'use_sigmoid': {
            'default': False,
            'retrain': True
        },
        'init_bias_zero': {
            'default': True,
            'retrain': True
        },
        'lr': {
            'default': 0.001,
            'retrain': True
        },
        'scheduler': {
            'default': 'cosine',
            'choices': ['base', 'cosine'],  # 'base': inherit
            'retrain': True
        },
        'lr_min': {  # only used if 'cosine' scheduler is selected
            'default': 1e-4,
            'retrain': True
        }
    })

    def __init__(self, ray_trafo,
                 training_epoch,
                 layer_num,
                 ks,
                 callback_func=None,
                 **kwargs):
        """
        Parameters
        ----------
        ray_trafo : :class:`odl.tomo.RayTransform`
            Ray transform (the forward operator).
        allow_multiple_workers_without_random_access : bool, optional
            Whether for datasets without support for random access
            a specification of ``num_data_loader_workers > 1`` is honored.
            If `False` (the default), the value is overridden by ``1`` for
            generator-only datasets.

        Further keyword arguments are passed to ``super().__init__()``.
        """
        super().__init__(reco_space=ray_trafo.domain, observation_space=ray_trafo.range, **kwargs)
        self.callback_func = callback_func
        self.ray_trafo = ray_trafo
        self.ray_trafo_module = OperatorModule(self.ray_trafo)
        self.callback_func = callback_func
        self.training_epoch = training_epoch
        self.layers = layer_num
        self.ks = ks
        self.tv1loss = TV1Loss()
        self.tv2loss = TV2Loss()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 

    def train(self, observation, initail_inv, gt, method, save_path, network_type=None):
       
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device('cpu') 
        if network_type.lower() == 'cnn': 
            self.model = CNN(self.layers, self.ks).type(torch.FloatTensor).apply(self.weight_init).to(self.device)
        elif network_type.lower() == 'unet':
            self.model = get_unet_model(scales=self.scales,
                                    skip=self.skip_channels,
                                    channels=self.channels,
                                    use_sigmoid=self.use_sigmoid).type(torch.FloatTensor).apply(self.weight_init).to(self.device)
        elif network_type.lower() == 'resnet':
            self.model = ResNet(1, 1).type(torch.FloatTensor).apply(self.weight_init).to(self.device)
        else:
            raise NameError('Unexcepeted Network')
        
        print('This model has {:,d} trainable parameters.'.format(count_parameters(self.model)))

        gt = np.squeeze(np.asarray(gt))
        criterion = L1Loss()
        self.initial_inv = torch.tensor(initail_inv).to(self.device, dtype=torch.float)
        self.obs = torch.tensor(np.asarray(observation)).to(self.device, dtype=torch.float)
        self.optimizer = AdamW(self.model.parameters(), lr=0.001)

        self.pnsr0 = PSNR(self.initial_inv.detach().squeeze().cpu().numpy(), gt),
        self.ssim0 = SSIM(self.initial_inv.detach().squeeze().cpu().numpy(), gt),
        self.PSNRs = []
        self.SSIMs = []
        self.Loss  = []
        # best_pnsr = 0.

        for i in tqdm(range(self.training_epoch), desc='HyRecon'):
            self.optimizer.zero_grad()
            if network_type.lower() == 'cnn':
                ds_ct = self.model(self.initial_inv)
                singo = self.ray_trafo_module(ds_ct.squeeze(0))
                loss = criterion(singo.squeeze(), self.obs) + self.tv1loss(ds_ct)                           
                loss.backward()
            elif network_type.lower() == 'unet' or network_type == 'resnet':
                ds_ct = self.model(self.initial_inv.unsqueeze(0).unsqueeze(0))
                loss = criterion(self.ray_trafo_module(ds_ct.squeeze(0)), self.obs)                            
                loss.backward()                               
            self.optimizer.step()

            output = ds_ct.squeeze().detach()
            ct_pnsr = PSNR(output.cpu().numpy(), gt)
            ct_ssim = SSIM(output.cpu().numpy(), gt)

            self.PSNRs.append(ct_pnsr)
            self.SSIMs.append(ct_ssim)
            self.Loss.append(loss.item())

            # if ct_pnsr > best_pnsr:
            #     self.callback_func(
            #         iteration = i,
            #         reconstruction = output.cpu().numpy(),
            #         initial_reco = initail_inv,
            #         gt = gt,
            #         loss = loss.item(),
            #         pnsr = ct_pnsr,
            #         ssim = ct_ssim,
            #         prefix = method,
            #         save_path=save_path)
            #     best_pnsr = np.copy(ct_pnsr)
            #     self.best_reco = output.cpu().numpy()
        return self.PSNRs, self.SSIMs, self.Loss
                    
    def weight_init(self, m):
        if isinstance(m, nn.Conv2d):
            torch.nn.init.xavier_uniform_(m.weight)
            # m.bias.data.fill_(0.01)