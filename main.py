import sys
sys.path.append('./dip-ct-benchmark')
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import numpy as np
import argparse
import odl
from dival.data import DataPairs
from dival.evaluation import TaskTable
from dival.measure import PSNR, SSIM
from dival.reconstructors.odl_reconstructors import FBPReconstructor
from dival import get_standard_dataset
from dliplib.utils.helper import set_use_latex
import torch
from dival.util.plot import plot_images
from hyreconstructor import HybirdReconstructor
from dliplib.reconstructors.tv import TVReconstructor
from dliplib.reconstructors.dip import DeepImagePriorReconstructor
import matplotlib.pyplot as plt
from dliplib.utils import Params
import pydicom
from utils import normalize_
set_use_latex()

np.random.seed(9527)
torch.manual_seed(9527)
torch.cuda.manual_seed_all(9527)
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True




def get_parser():
    """Adds arguments to the command"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--pid', type=str, default='1')
    parser.add_argument('--num_angles', type=int, default=1000)
    parser.add_argument('--impl', type=str, default='astra_cuda')
    parser.add_argument('--photons', type=int, default=1000)
    parser.add_argument('--layers', type=int, default=30)
    parser.add_argument('--iters', type=int, default=2000)
    return parser



def hy_callback_func(iteration, reconstruction, initial_reco, gt, loss, pnsr, ssim, prefix, save_path):
    _, ax = plot_images([initial_reco, reconstruction, gt], xticks=[], yticks=[], 
                        vrange='individual',
                        cbar='none', fig_size=(10, 4))
    ax[0].set_title('FBP')
    ax[1].set_xlabel('loss: {:f}; PNSR: {:2f} SSIM: {:f}'.format(loss, pnsr, ssim))
    ax[1].set_title('iteration {:d}'.format(iteration+1))
    ax[2].set_title('ground truth')
    plt.tight_layout()
    plt.savefig(save_path + 'best_result_{}.pdf'.format(prefix))
    plt.close()



if __name__ == '__main__':
    options = get_parser().parse_args()
    TASK_NAME = options.dataset
    IMPL = options.impl
    NUM_ANGLES  = options.num_angles
    PHOTOS_PER_PIXEL = options.photons
    LAYERS = options.layers
    PID = options.pid
    ITERS = options.iters
    recons = {}
    MU_AIR = 0.02
    MU_WATER = 0.02
    SAMPLE = 9

    if TASK_NAME.lower() == 'lodopab':
        save_path = './layers{}/reco_{}_dose{}_ang{}_sample{}/'.format(LAYERS, TASK_NAME, PHOTOS_PER_PIXEL, NUM_ANGLES, SAMPLE)
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        dataset = get_standard_dataset('lodopab', impl=IMPL)
        _, ground_truth = dataset.get_sample(SAMPLE, 'test')
        ground_truth = np.asarray(ground_truth)
        ground_truth = normalize_(ground_truth, ground_truth.min(), ground_truth.max())
        reco_space = odl.uniform_discr(min_pt=[-90, -90], max_pt=[91, 91], shape=[362, 362], dtype='float32')
    elif TASK_NAME.lower() == 'mayo':
        save_path = './layers{}/reco_{}_PID_{}_dose{}_iters{}/'.format(LAYERS, TASK_NAME, PID, PHOTOS_PER_PIXEL, ITERS)
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        if PID == '1':
            imafile = './AAPM_Dataset/L067/full_1mm/L067_FD_1_1.CT.0001.0001.2015.12.22.18.09.40.840353.358074219.IMA'
        elif PID == '2':
            imafile = './AAPM_Dataset/L067/full_1mm/L067_FD_1_1.CT.0001.0306.2015.12.22.18.09.40.840353.358081539.IMA'
        elif PID == '3':
            imafile = './AAPM_Dataset/L067/full_1mm/L067_FD_1_1.CT.0001.0559.2015.12.22.18.09.40.840353.358089994.IMA'
        else:
            raise NameError('Selection Is Incorrect')
        
        reco_space = odl.uniform_discr(min_pt=[-128, -128], max_pt=[128, 128], shape=[512, 512], dtype='float32')
        dataset = pydicom.read_file(imafile)
        img = dataset.pixel_array.astype(np.float32)
        ground_truth = normalize_(img, img.min(), img.max())
    else:
        raise KeyError('Task Name Is Incorrect')
    
    angle_partition = odl.uniform_partition(0, 2 * np.pi, NUM_ANGLES)
    detector_partition = odl.uniform_partition(-360, 360, 1000)
    geometry = odl.tomo.FanBeamGeometry(angle_partition, detector_partition, src_radius=500, det_radius=500)
    ray_trafo = odl.tomo.RayTransform(reco_space, geometry, impl=IMPL)

    proj_data = ray_trafo(ground_truth)
    proj_data = np.exp(-proj_data * MU_WATER)
    proj_data = odl.phantom.poisson_noise(proj_data * PHOTOS_PER_PIXEL)
    proj_data = np.maximum(proj_data, 1) / PHOTOS_PER_PIXEL
    observation = np.log(proj_data) * (-1 / MU_WATER)

#####################Competitive Method########################################
    test_data = DataPairs(observation, ground_truth, name='tv+dip')
    # task table and reconstructors
    eval_tt = TaskTable()
    fbp_reconstructor = FBPReconstructor(ray_trafo, hyper_params={
                                            'filter_type': 'Hann',
                                            'frequency_scaling': 0.8})
    if TASK_NAME.lower() == 'lodopab':
        params_tv = Params.load('lodopab_tv')
        params_dip = Params.load('lodopab_dip')
        params_tv.loss_function = 'mse'
        params_dip.loss_function = 'mse'
    elif TASK_NAME.lower() == 'mayo':
        params_tv = Params.load('lodopab_tv')
        params_dip = Params.load('lodopab_dip')
        params_tv.loss_function = 'mse'
        params_dip.loss_function = 'mse'
        params_dip.iterations = 1000
    else:
        raise KeyError('Task Name Is Incorrect')

    tv_reconstructor = TVReconstructor(ray_trafo, hyper_params=params_tv.dict)
    dip_reconstructor = DeepImagePriorReconstructor(ray_trafo, hyper_params=params_dip.dict)
    reconstructors = [fbp_reconstructor, tv_reconstructor, dip_reconstructor]
    options = {'save_iterates': True}
    eval_tt.append_all_combinations(reconstructors=reconstructors,
                                    test_data=[test_data], options=options)

    # run task table
    results = eval_tt.run()
    results.apply_measures([PSNR, SSIM])
    print(results)

    ##############################Proposed Method ################################
    prefix = 'Proposed'
    initial_reco = results.results['reconstructions'][0][0][0]
    initial_reco = np.asarray(initial_reco)
    hyreconstructor = HybirdReconstructor(
        ray_trafo =ray_trafo,
        training_epoch=ITERS,
        layer_num = LAYERS,
        ks        = 64,
        callback_func=hy_callback_func
        )
    psnrs, ssims, loss = hyreconstructor.train(observation, initial_reco, ground_truth, prefix, save_path, 'cnn')

    fig, ax = plt.subplots(1, 3, figsize=(18, 4))
    ax[0].plot(psnrs, 'o-r', markersize=1)
    ax[0].axhline(y=hyreconstructor.pnsr0, color='b')
    ax[0].grid(True)
    ax[0].set_ylabel('PNSR')
    ax[0].set_xlabel('Iterations')
    ax[0].legend(['Initial', 'Our'])
    ax[0].set_title('PNSR')

    ax[1].plot(ssims, 'o-r', markersize=1)
    ax[1].axhline(y=hyreconstructor.ssim0, color='b')
    ax[1].grid(True)
    ax[1].set_ylabel('SSIM')
    ax[1].set_xlabel('Iterations')
    ax[1].legend(['Initial', 'Our'])
    ax[1].set_title('SSIM')

    ax[2].set_yscale("log")
    ax[2].plot(loss, 'o-r', markersize=1)
    ax[2].grid(True)
    ax[2].set_ylabel('Loss')
    ax[2].set_xlabel('Iterations')
    ax[2].set_title('Loss')
    plt.tight_layout()
    plt.savefig(save_path + 'loss_ssim.pdf')
    # plt.show()

    ############################Save Results ##################################
    if TASK_NAME.lower() == 'lodopab':
        np.save(save_path + 'gt.npy', ground_truth)
        recons['Ours'] = hyreconstructor.best_reco
        for i in range(len(reconstructors)):
            row = results.results.loc[i, 0]
            recons[row['reconstructor'].name] = row.at['reconstructions'][0]
    elif TASK_NAME.lower() == 'mayo':
        np.save(save_path + 'gt.npy', np.rot90(ground_truth))
        recons['Ours'] = np.rot90(hyreconstructor.best_reco)
        for i in range(len(reconstructors)):
            row = results.results.loc[i, 0]
            recons[row['reconstructor'].name] = np.rot90(row.at['reconstructions'][0])
    else:
        raise KeyError('Task Name Is Incorrect')
    
    np.save(save_path + 'obs.npy', observation)
    np.save(save_path + 'pnsr.npy', psnrs)
    np.save(save_path + 'ssim.npy', ssims)
    np.save(save_path + 'loss.npy', loss)
    np.save(save_path + 'recos.npy', recons)