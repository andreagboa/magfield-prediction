
# %%
import numpy as np
from pathlib import Path
from tabulate import tabulate
import pandas as pd
from datetime import datetime
import time

import torch
import torch.nn as nn
import h5py

from utils.tools import random_bbox, mask_image, get_config, random_bnd, random_bnd2
from model.networks import Generator

# Parameters
img_idx = 5
plt_scale = 0.1
rng = np.random.default_rng(0)
path_orig = Path(__file__).parent.resolve() / 'checkpoints' / 'boundary_1_256'

percentages = [10,25,50,75,100]
# model = 'in_94_l1'
model = 'in_94_perc_uniform_05'
# file = h5py.File('data/bnd_256/magfield_256_large.h5')
# Maybe use the validation fields
it_number = 500000

file = h5py.File('data/magfield_val_256.h5')

# Empty matrix so append errors (4 models, 5 performance tests: mse, psnr, mape, divergence, curl)
# err_mat = np.empty([5,5])
err_str = ['MSE [mT]', 'PSNR [dB]', 'MAPE [%]', 'Div [mT/px]', 'Curl [μT/px]', 'Inference [sec]']
err_mat = np.zeros([len(err_str), len(percentages) + 1])
std_mat = np.zeros([len(err_str), len(percentages) + 1])

# %%
for perc in percentages:
    exp_path = Path(path_orig, model)

    # Matrices for storing errors for the samples
    mse_mat = np.zeros([img_idx])
    psnr_mat = np.zeros([img_idx])
    mape_mat = np.zeros([img_idx])
    div_mat = np.zeros([img_idx])
    curl_mat = np.zeros([img_idx])
    df_eval = pd.DataFrame([])
    inference =[]

    for j in range(img_idx):
        # print(file['field'][img_idx,:,:,:,1].shape)
        field = file['field'][j,:,:,:,1]
        # Plot field chosen
        # sample_check(field, v_max=plt_scale, filename = 'orig_'+ model)
        
        # Make box 
        config = get_config(Path(exp_path, 'config.yaml'))
        bboxes = random_bbox(config, rng=rng)
        # x, mask, orig = mask_image(np.array([field]), bboxes, config, bnd=config['boundary'])
        x, mask, orig = mask_image(np.array([field]), bboxes, config, bnd=config['boundary'], perc = perc)

        # print(x.shape)
        # Plot box made
        # sample_check(x[0], v_max=plt_scale, filename = 'boundary_'+model)

        # Test last generator ran
        last_model_name = Path(exp_path, f'gen_00{str(it_number)}.pt')
        netG = Generator(config['netG'], config['coarse_G'], True, [0])
        netG.load_state_dict(torch.load(last_model_name))
        netG = nn.parallel.DataParallel(netG, device_ids=[0])
        corrupt_t = torch.from_numpy(x[0].astype('float32')).cuda().unsqueeze(0)
        mask_t = torch.from_numpy(mask[0].astype('float32')).cuda().unsqueeze(0)

        # Inference
        start_time = time.time()
        _, out, x_fixed = netG(corrupt_t, mask_t)
        elapsed_time = time.time() - start_time
        print(f'Model {model} took {elapsed_time} seconds to run.')
        inference.append(elapsed_time)

        # Plot original box (input)
        # sample_check(orig[0], v_max=plt_scale, filename = 'orig_box_'+model)

        out_np = out.squeeze(0).cpu().data.numpy()
        # Plot output
        # sample_check(out_np, v_max=plt_scale, filename='wgan_'+model)

        # Calculate performance of models
        diff = np.abs(orig - out_np)
        # mse_final = np.mean(diff**2)
        mse_mat[j] = np.mean(diff**2)
        # mse_mat[j] = np.mean(diff) #For MAE instead of MSE
        # if mse_mat[j] < 1e-4:
        #     psnr_mat[j] = 0
        # else:
            # psnr = 20 * np.log10(np.max(orig) / np.sqrt(mse_final))
        
        psnr_mat[j] = 20 * np.log10(np.max(np.abs(orig)) / np.sqrt(mse_mat[j]))
        # mape = 100*(np.abs(np.mean(diff)/np.mean(orig)))
        mape_mat[j] = 100 * (np.mean(diff) / np.mean(np.abs(orig)))

        # print(f"Recon loss: {np.mean(np.abs(diff)):.4f}")
        # print(f"PSNR: {psnr:.4f} dB")
        # print(f"MAPE: {mape:.4f} %")

        # Div
        Hx_x = torch.gradient(out[0,0], dim=1, edge_order=2)[0]
        Hy_y = torch.gradient(out[0,1], dim=0, edge_order=2)[0]
        if len(out.size()[1:]) > 3 : 
            Hz_z = torch.gradient(out[0,2], dim=2, edge_order=2)[0]
            div_mag = torch.stack([Hx_x, Hy_y, Hz_z], dim=0)[:,:,:,1]
        else:
            div_mag = torch.stack([Hx_x, Hy_y], dim=0)
        # div = torch.mean(torch.abs(div_mag.sum(dim=0)))
        div_mat[j] = torch.mean(torch.abs(div_mag.sum(dim=0)))

        #Curl
        Hx_y = torch.gradient(out[0,0], dim=0, edge_order=2)[0]
        Hy_x = torch.gradient(out[0,1], dim=1, edge_order=2)[0]
        if len(out.size()[1:]) > 3 :
            Hx_z = torch.gradient(out[0,0], dim=2, edge_order=2)[0]
            Hy_z = torch.gradient(out[0,1], dim=2, edge_order=2)[0]
            Hz_x = torch.gradient(out[0,2], dim=1, edge_order=2)[0]
            Hz_y = torch.gradient(out[0,2], dim=0, edge_order=2)[0]
            curl_vec = torch.stack([Hz_y - Hy_z, Hx_z - Hz_x, Hy_x - Hx_y], dim=0)[:,:,:,1]
            curl_mag = curl_vec.square().sum(dim=0)
        else:
            curl_mag = (Hy_x - Hx_y).square()
        # curl = torch.mean(curl_mag)
        curl_mat[j] = torch.mean(curl_mag)
        # print(f"divergence: {div:.5f}")
        # print(f"curl: {curl:.5f}")
       
    err_mat[:,percentages.index(perc) + 1] = [np.mean(mse_mat)*1e3, np.mean(psnr_mat), np.mean(mape_mat), 
                                          np.mean(div_mat)*1e3, np.mean(curl_mat)*1e6, np.mean(inference)]
    # std_mat[:,percentages.index(perc) + 1] = [np.std(mse_mat)*1e3, np.std(psnr_mat), np.std(mape_mat), 
                                        #   np.std(div_mat)*1e3, np.std(curl_mat)*1e6]
    
    df_eval = df_eval.append(pd.DataFrame([
            [
                (mse_mat)*1e3, 
                (mape_mat), 
                (div_mat)*1e3, 
                (curl_mat)*1e6,
                inference
            ]
        ], columns=['MSE', 'MAPE', 'div', 'curl', 'inference']), ignore_index=True)
    

    # df_eval.attrs['perc'] = perc
    # df_eval.attrs['box_amount'] = config['box_amount']
    # df_eval.attrs['mask_shape'] = config['mask_shape'][0]
    timestamp = datetime.now().strftime("%y%m%d_%H%M%S")
    
    fname = model+ '_' + timestamp + '_' + str(perc)
    # df_eval.to_pickle(f'{path_orig}/{fname}.p')



#%%

err_list = err_mat.tolist()
std_list = std_mat.tolist()
for i, err_n in enumerate(err_str):
    err_list[i][0] = err_n
    std_list[i][0] = err_n
print(tabulate(err_list, headers=['%']+(percentages), tablefmt="grid", showindex=False))

# print(tabulate(std_list, headers=['%']+(percentages), tablefmt="grid", showindex=False))



# %%

# err_mat1 = err_mat.astype('S7')
# std_mat1 = std_mat.astype('S7')
# final = []

# for i in range(err_mat.shape[0]):
#     for j in range(err_mat.shape[1]):
#         final[i][j] = (err_mat1[i][j])+' $\pm$ '+(std_mat1[i][j])



# %%
