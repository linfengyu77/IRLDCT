import sys
sys.path.append('/home/fengw666/CT_Recon/dip-ct-benchmark')
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "4"
import numpy as np
import odl
from dival.data import DataPairs
from dival.evaluation import TaskTable
from dival.measure import PSNR, SSIM
from dival.reconstructors.odl_reconstructors import FBPReconstructor
from dival import get_standard_dataset
from dliplib.utils.helper import load_standard_dataset
from dliplib.utils.helper import set_use_latex
from dliplib.reconstructors.tv import TVReconstructor, TVAdamReconstructor
from dliplib.reconstructors.dip import DeepImagePriorReconstructor
from dliplib.utils import Params
import pydicom
set_use_latex()
np.random.seed(9527)
rs = np.random.RandomState(3)
r = np.random.RandomState(2)

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


if __name__ == '__main__':
    # TASK_NAME = 'ellipses'
    TASK_NAME = 'lodopab'
    # TASK_NAME = 'Mayo'
    IMPL = 'astra_cuda'
    NUM_ANGLES  = 1000
    DETECTORS   = 1000
    PHOTOS_PER_PIXEL = 1e3
    MU_WATER = 0.02
    MU_AIR = 0.02
    MU_MAX = 81.35858
    SAMPLE = 9
    THICKNESS = '1mm'
    # THICKNESS = '3mm'
    # TYPER = 'D45-1'
    TYPER = 'B30'
    recons = {}

    if TASK_NAME == 'ellipses':
        save_path = './TV_DIP_Reco/reco_{}_dose{}_ang{}_det{}/'.format(TASK_NAME, PHOTOS_PER_PIXEL, NUM_ANGLES, DETECTORS)
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        ellipses_dataset = load_standard_dataset('ellipses')
        reco_space = odl.uniform_discr(min_pt=[-90, -90], max_pt=[90, 90], shape=[360, 360], dtype='float32')
        ground_truth = odl.phantom.shepp_logan(reco_space, modified=True)
        ground_truth = np.asarray(ground_truth)
        ground_truth = normalize_(ground_truth, ground_truth.min(), ground_truth.max()) 
    elif TASK_NAME == 'lodopab':
        save_path = './TV_DIP_Reco/reco_{}_dose{}_ang{}_det{}_sample{}/'.format(TASK_NAME, PHOTOS_PER_PIXEL, NUM_ANGLES, DETECTORS, SAMPLE)
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        dataset = get_standard_dataset('lodopab', impl=IMPL)
        _, ground_truth = dataset.get_sample(SAMPLE, 'test')
        ground_truth = np.asarray(ground_truth)
        ground_truth = normalize_(ground_truth, ground_truth.min(), ground_truth.max())
        reco_space = odl.uniform_discr(min_pt=[-90, -90], max_pt=[91, 91], shape=[362, 362], dtype='float32')
    elif TASK_NAME == 'Mayo':
        save_path = './TV_DIP_Reco/reco_{}{}_{}_dose{}_ang{}_det{}/'.format(TASK_NAME, THICKNESS, TYPER, PHOTOS_PER_PIXEL, NUM_ANGLES, DETECTORS)
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        if THICKNESS == '1mm' and TYPER == 'B30':
            imafile = '/home/fengw666/CT_Recon/MayoDataset/FullDose/1mm/SfotKerneB30/L067_FD_1_1.CT.0001.0001.2015.12.22.18.09.40.840353.358074219.IMA'
        elif THICKNESS == '1mm' and TYPER == 'D45-1':
            imafile = '/home/fengw666/CT_Recon/MayoDataset/FullDose/1mm/SharpKerneD45/L067_FD_1_SHARP_1.CT.0002.0001.2016.01.21.18.11.40.977560.404629015.IMA'
        else:
            raise NameError('Selection Is Incorrect')
        reco_space = odl.uniform_discr(min_pt=[-128, -128], max_pt=[128, 128], shape=[512, 512], dtype='float32')
        dataset = pydicom.read_file(imafile)
        img = dataset.pixel_array.astype(np.float32)
        # RescaleSlope = dataset.RescaleSlope
        # RescaleIntercept = dataset.RescaleIntercept
        # image_hu = img * RescaleSlope + RescaleIntercept
        # image_hu += r.uniform(0., 1., size=image_hu.shape)
        # ground_truth = image_hu * (MU_WATER - MU_AIR)/1000 + MU_WATER
        # np.clip(ground_truth/MU_MAX, 0., 1., out=ground_truth)
        ## LOW-DOSE SINOGRAM GENERATION
        # ground_truth= reco_space.element(img)/1000
        ground_truth = normalize_(img, img.min(), img.max())

    else:
        raise NameError('Task Name Is Incorrect')
    
    angle_partition = odl.uniform_partition(0, 2 * np.pi, NUM_ANGLES)
    detector_partition = odl.uniform_partition(-360, 360, DETECTORS)
    geometry = odl.tomo.FanBeamGeometry(angle_partition, detector_partition, src_radius=500, det_radius=500)
    ray_trafo = odl.tomo.RayTransform(reco_space, geometry, impl=IMPL)

    proj_data = ray_trafo(ground_truth)
    proj_data = np.exp(-proj_data * MU_WATER)
    proj_data = odl.phantom.poisson_noise(proj_data * PHOTOS_PER_PIXEL)
    proj_data = np.maximum(proj_data, 1) / PHOTOS_PER_PIXEL
    observation = np.log(proj_data) * (-1 / MU_WATER)

    # observation = observation/np.max(np.max(np.abs(observation)))
    # proj_data *= (-1)
    # np.exp(proj_data, out=proj_data)
    # proj_data *= PHOTOS_PER_PIXEL
    # data = rs.poisson(proj_data).astype(float)
    # np.maximum(0.1, data, out=data)
    # np.log(data/PHOTOS_PER_PIXEL, out=data, dtype=float)
    # data /= (-MU_MAX)

    test_data = DataPairs(observation, ground_truth, name='tv+dip')
    # task table and reconstructors
    eval_tt = TaskTable()
    fbp_reconstructor = FBPReconstructor(ray_trafo, hyper_params={
                                            'filter_type': 'Hann',
                                            'frequency_scaling': 0.8})
    if TASK_NAME == 'ellipses':
        params_tv = Params.load('ellipses_tv')
        params_dip = Params.load('ellipses_dip')
    elif TASK_NAME == 'lodopab':
        params_tv = Params.load('lodopab_tv')
        params_dip = Params.load('lodopab_dip')
        params_tv.loss_function = 'mse'
        params_dip.loss_function = 'mse'
    elif TASK_NAME == 'Mayo':
        params_tv = Params.load('lodopab_tv')
        params_dip = Params.load('lodopab_dip')
        params_tv.loss_function = 'mse'
        # params_tv.iterations = 2000
        # params_tv.gamma =  0.
        # params_tv.loss_function = 'poisson'
        # params_dip.skip_channels = [4, 4, 4, 4, 4]
        params_dip.loss_function = 'mse'
        # params_dip.scales = 5
        # params_dip.skip_channels = [0]*5
        # params_dip.gamma = 0.
        params_dip.iterations = 1000
        # params_dip.lr = 1e-5


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


    ############################ save .npy file ##################################
    if TASK_NAME == 'ellipses':
        np.save(save_path + 'gt.npy', np.flip(ground_truth))
        for i in range(len(reconstructors)):
            row = results.results.loc[i, 0]
            recons[row['reconstructor'].name] = np.flip(row.at['reconstructions'][0])
    elif TASK_NAME == 'lodopab':
        np.save(save_path + 'gt.npy', ground_truth)
        for i in range(len(reconstructors)):
            row = results.results.loc[i, 0]
            recons[row['reconstructor'].name] = row.at['reconstructions'][0]
    elif TASK_NAME == 'Mayo':
        np.save(save_path + 'gt.npy', np.rot90(ground_truth))
        for i in range(len(reconstructors)):
            row = results.results.loc[i, 0]
            recons[row['reconstructor'].name] = np.rot90(row.at['reconstructions'][0])   
    else:
        raise NameError('Task Name Is Incorrect')
    np.save(save_path + 'fbp_tv_dip_recos.npy', recons)