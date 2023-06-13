
# %%
import numpy as np
from pathlib import Path
from tabulate import tabulate
import pandas as pd
from datetime import datetime
import time
import warnings
warnings.simplefilter(action="ignore", category=FutureWarning)

from argparse import ArgumentParser
from scipy.interpolate import griddata, bisplrep, bisplev
from skimage.restoration import inpaint
from sklearn import preprocessing
from sklearn.gaussian_process import GaussianProcessRegressor as GPR
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

import torch
import torch.nn as nn
import h5py

from utils.tools import random_bbox, mask_image, get_config
from utils.tools import calc_div, calc_curl, random_bbox, mask_image
from model.networks import Generator

# Parameters
img_idx = 100
plt_scale = 0.1
rng = np.random.default_rng(0)
path_orig = Path(__file__).parent.resolve() / 'checkpoints' / 'boundary_1_256'

# methods = ['wgan', 'linear', 'spline', 'gaussian']
# methods = ['linear', 'spline']
methods = ['gaussian']
it_number = 600000

file = h5py.File('data/magfield_val_256.h5')

# Empty matrix so append errors (4 models, 5 performance tests: mse, psnr, mape, divergence, curl)
err_str = ['MAE [mT]','MSE [mT]', 'MAPE [%]', 'Div [mT/px]', 'Curl [μT/px]', 'Inference']
err_mat = np.zeros([len(err_str), len(methods) + 1])

# %%
for method in methods:
    exp_path = Path(path_orig, 'in_94_l1')
    print('Starting: '+method)
    
    # Matrices for storing errors for the samples
    mae_mat = np.zeros([img_idx])
    mse_mat = np.zeros([img_idx])
    mape_mat = np.zeros([img_idx])
    div_mat = np.zeros([img_idx])
    curl_mat = np.zeros([img_idx])
    df_eval = pd.DataFrame([])
    inference=[]
    
    for j in range(img_idx):
        field = file['field'][j,:,:,:,1]

        # Make box 
        config = get_config(Path(exp_path, 'config.yaml'))
        bboxes = random_bbox(config, rng=rng)
        x, mask, orig = mask_image(np.array([field]), bboxes, config, bnd=config['boundary'])
    
        if method == 'wgan':
            last_model_name = Path(exp_path, f'gen_00{str(it_number)}.pt')
            netG = Generator(config['netG'], config['coarse_G'], True, config['gpu_ids'])
            netG.load_state_dict(torch.load(last_model_name))
            netG = nn.parallel.DataParallel(netG, device_ids=[0])
            corrupt_t = torch.from_numpy(x[0].astype('float32')).cuda().unsqueeze(0)
            mask_t = torch.from_numpy(mask[0].astype('float32')).cuda().unsqueeze(0)

            # Inference
            start_time = time.time()
            _, out, x_fixed = netG(corrupt_t, mask_t)
            elapsed_time = time.time() - start_time
            inference.append(elapsed_time)

            out_np = out.squeeze(0).cpu().data.numpy()
            x2 = out.squeeze(0).cpu().data.numpy()
            

        elif method in ['linear', 'spline', 'gaussian']:
            # Splitting up for each spatial location
            x_post = np.zeros(x.shape)
            grid_x, grid_y = np.mgrid[0:config['mask_shape'][0]+2*config['boundary']:1,
                                        0:config['mask_shape'][1]+2*config['boundary']:1]
            if method == 'gaussian':
                kernel = C(1.0, (1e-3, 1e3)) * RBF(10, (1e-2, 1e2))

            for l in range(config['image_shape'][0]):
                points = np.concatenate(
                    (
                        np.expand_dims(np.where(mask[0,0] == 0)[0], axis=-1),
                        np.expand_dims(np.where(mask[0,0] == 0)[1], axis=-1)
                    ), axis=-1)
                values = x[0,l,points[:,0],points[:,1]]#.cpu().data.numpy()
                eval_pts = np.concatenate(
                    (
                        np.expand_dims(grid_x, axis=-1),
                        np.expand_dims(grid_y, axis=-1)
                    ), axis=-1).reshape((-1,2))

                if method == 'linear':
                    start_time = time.time()
                    x_post[0,l,:,:] = griddata(points, values, (grid_x, grid_y),
                                                method='linear')
                    elapsed_time = time.time() - start_time
                    inference.append(elapsed_time)

                elif method == 'spline':
                    start_time = time.time()
                    x_post[0,l,:,:] = griddata(points, values, (grid_x, grid_y),
                                                method='cubic')
                    elapsed_time = time.time() - start_time
                    inference.append(elapsed_time)
                                
                elif method == 'gaussian':
                    scaler = preprocessing.StandardScaler().fit(points)
                    pts_scaled = scaler.transform(points)
                    eval_pts_scaled = scaler.transform(eval_pts)
                    gpr = GPR(kernel=kernel, random_state=0).fit(pts_scaled, values)
                    start_time = time.time()
                    x_post[0,l,:,:] = gpr.predict(eval_pts_scaled).reshape(
                        config['mask_shape'][0] + 2*config['boundary'], config['mask_shape'][1] + 2*config['boundary'])
                    elapsed_time = time.time() - start_time
                    # print(f"The interpolation with Gaussian processes took {elapsed_time} seconds to execute.")
                    inference.append(elapsed_time)

            x2 = torch.from_numpy(x_post)
            x2_eval = x2 * mask + x * (1. - mask)

            out = x2_eval

        else:
            raise NotImplementedError(f'Method {method} is currently not supported')

        # Calculate performance of models
        if method == 'wgan':
            diff = np.abs(orig - out_np)
        elif method in ['linear', 'spline', 'gaussian']:
            # diff = np.abs(orig - out.cpu().detach().numpy())
            diff = np.abs(orig - x_post)

        
        mae_mat[j] = np.mean(diff)
        mse_mat[j] = np.mean(diff**2) 
        mape_mat[j] = 100 * (np.mean(diff) / np.mean(np.abs(orig)))
 
        div_mat[j] = calc_div(out, True).cpu().data.numpy()
        curl_mat[j] = calc_curl(out, True).cpu().data.numpy()

    
    err_mat[:,methods.index(method) + 1] = [np.mean(mae_mat)*1e3,np.mean(mse_mat)*1e3, np.mean(mape_mat), 
                                          np.mean(div_mat)*1e3, np.mean(curl_mat)*1e6, np.mean(inference[1:])]

    df_eval = df_eval.append(pd.DataFrame([
            [
                mae_mat*1e3,
                (mse_mat)*1e3, 
                (mape_mat), 
                (div_mat)*1e3, 
                (curl_mat)*1e6,
                inference[1:]
            ]
        ], columns=['MAE','MSE', 'MAPE', 'div', 'curl', 'inference']), ignore_index=True)
    

    # df_eval.attrs['perc'] = perc
    # df_eval.attrs['box_amount'] = config['box_amount']
    # df_eval.attrs['mask_shape'] = config['mask_shape'][0]
    timestamp = datetime.now().strftime("%y%m%d_%H%M%S")
    
    fname = method+ '_' + timestamp + '_' + str(img_idx)
    df_eval.to_pickle(f'{path_orig}/{fname}.p')

#%%
err_list = err_mat.tolist()
for i, err_n in enumerate(err_str):
    err_list[i][0] = err_n
print(tabulate(err_list, headers=['Test']+methods, tablefmt="grid", showindex=False))


# %%
