"""
===============================================================================
Get predictions and plots for trained TransGrow model
===============================================================================
"""

import sys, os, io, warnings, time, logging
import torch
import torch.multiprocessing
import pytorch_lightning as pl
import torchvision
import yaml
import imageio
import cv2
import random
import numpy as np
import matplotlib.pyplot as plt

from torch.autograd import Variable
from psnr_hvsm import psnr_hvsm

from configs.config_test_transgrow import cfg

from utils import utils
import pytorch_msssim
from datasets.seq_datamodule import SeqDataModule
from models.transgrow_gan_plm import TransGrowGANModel
from models.transgrow_wgan_plm import TransGrowWGANModel
from models.transgrow_wgangp_plm import TransGrowWGANGPModel

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

# # this makes lightning reports not look like errors
pl._logger.handlers = [logging.StreamHandler(sys.stdout)]

torch.multiprocessing.set_sharing_strategy('file_system')


#%% print versions stuff
print('python', sys.version, sys.executable)
print('pytorch', torch.__version__)
print('torchvision', torchvision.__version__)
print('pytorch-lightning', pl.__version__)
print('CUDA Available:', torch.cuda.is_available())
print(torch._C._cuda_getCompiledVersion(), 'cuda compiled version')
print(torch._C._nccl_version(), 'nccl')
for i in range(torch.cuda.device_count()):
    print('device %s:'%i, torch.cuda.get_device_properties(i))    

# # Evaluation score losses
loss_l1 = torch.nn.L1Loss()
loss_msssim = pytorch_msssim.MSSSIM()

#%% 
if __name__ == '__main__':
    #%% write cfg.yaml to pred_dir
    with io.open(os.path.join(cfg['pred_dir'], 'cfg_pred.yaml'), 'w', encoding='utf8') as outfile:
        yaml.dump(cfg, outfile, default_flow_style=False, allow_unicode=True)
    
    #%% get dataModule
    dataModule = SeqDataModule(cfg['img_size'], cfg['batch_size'], cfg['nworkers'], cfg['img_dir'], cfg['img_ext'], cfg['n_imgs'], cfg['data_name'], cfg['data_time'], cfg['time_unit'], cfg['sample_type'], cfg['rem_dup'], cfg['img_path_dist'], cfg['img_path_skip'], cfg['sample_factor'], cfg['sample_range'], transform_train=cfg['transform_train'], transform_test=cfg['transform_test'], val_test_shuffle=cfg['val_test_shuffle'])
    
    # setup dataModule
    dataModule.prepare_data()
    dataModule.setup()
    
    # show dim and len of different data subsets
    print('---Some Training Stats---')
    print('Input dims:', dataModule.data_dims)
    print('#Traindata:', len(dataModule.train_dataloader().dataset))
    print('#Valdata:', len(dataModule.val_dataloader().dataset))
    print('#Testdata:', len(dataModule.test_dataloader().dataset))
    
    if cfg['train_results']:
        dataloader_list = [dataModule.test_dataloader(), dataModule.train_dataloader()]
        prfx=['test_','train_']
    else:
        dataloader_list = [dataModule.test_dataloader()]
        prfx=['test_']

    
    #%% load model from checkpoint    
    if cfg['use_model'] == 'gan':
        model = TransGrowGANModel.load_from_checkpoint(cfg['ckpt_path_pred'])
    elif cfg['use_model'] == 'wgan':
        model = TransGrowWGANModel.load_from_checkpoint(cfg['ckpt_path_pred'])
    elif cfg['use_model'] == 'wgangp':
        model = TransGrowWGANGPModel.load_from_checkpoint(cfg['ckpt_path_pred'])
        
    # # set to eval mode
    model.eval()     
    # # sent model to device
    model.to(cfg['device'])
    
    
    #%% start predicting / scoring / plotting 
    for count, dataloader in enumerate(dataloader_list):
        
        
        #%% calculate overall test scores
        print('calculate scores and accumulate hook outputs...')
        plot_dir = os.path.join(cfg['pred_dir'], (prfx[count]+'scores_imgs'))
        if not os.path.exists(plot_dir):
            os.makedirs(plot_dir)
        img_dir_ip = os.path.join(cfg['pred_dir'], (prfx[count]+'scores_imgs'), 'pred_imgs', 'ip')
        if not os.path.exists(img_dir_ip):
            os.makedirs(img_dir_ip)
        img_dir_ep = os.path.join(cfg['pred_dir'], (prfx[count]+'scores_imgs'), 'pred_imgs', 'ep')
        if not os.path.exists(img_dir_ep):
            os.makedirs(img_dir_ep)
        
        # # times
        t_list = []
        t_in_list = []
        t_out_list = []
        min_dt_list = []
        # # scores
        score_l1_list = []
        score_msssim_list = []
        score_pla_list = []
        score_psnr_list = []
        # # is_extrapolation
        is_ep_list = []
        
        # # run prediciton
        for i_batch, batch in enumerate(dataloader):
            if cfg['target_pos'] > cfg['n_imgs'] or cfg['target_pos'] <=0 or cfg['target_pos'] is None:
                idx_target = random.randint(0,cfg['n_imgs_in'])
            else:
                idx_target = cfg['target_pos']-1
            idx_in = tuple(list(range(cfg['n_imgs']))[:idx_target]+list(range(cfg['n_imgs']))[idx_target+1:])
            with torch.no_grad():
                x = {'img_in': batch['seq_img'][:,idx_in,:], 
                      'timedelta_in': batch['seq_timedelta'][:,idx_in],
                      'timedelta_target': batch['seq_timedelta'][:,idx_target], 
                      'z': None}
                img_pred = model(x).cpu().detach()
                img_target = batch['seq_img'][:,idx_target,:]
                
                
                # # save times and minimum timediff
                t_list.append(batch['seq_timedelta'])
                t_in_list.append(batch['seq_timedelta'][:,idx_in])
                t_out_list.append(batch['seq_timedelta'][:,idx_target])
                min_dt_list.append(torch.min(torch.abs(batch['seq_timedelta'][:,idx_in]-torch.unsqueeze(batch['seq_timedelta'][:,idx_target],dim=1)),dim=1)[0])
                
                # # save scores
                for i in range(img_pred.shape[0]):
                    score_l1_list.append(loss_l1(img_pred[i,:], img_target[i,:]))
                    score_msssim_list.append(loss_msssim(torch.unsqueeze(img_pred[i,:],dim=0),torch.unsqueeze(img_target[i,:],dim=0)))
                    score_pla_list.append((np.abs(utils.pla_per_img(img_target[i,:])-utils.pla_per_img(img_pred[i,:]))))
                    # for PSNR-HVS-M: convert to YUV colorspace and to [0 1] and use only luma component (Y); see: https://pypi.org/project/psnr-hvsm/
                    img_pred_yuv = cv2.cvtColor(np.array(torch.permute((img_pred[i,:]),(1,2,0))), cv2.COLOR_RGB2YUV)
                    img_target_yuv = cv2.cvtColor(np.array(torch.permute((img_target[i,:]),(1,2,0))), cv2.COLOR_RGB2YUV)
                    score_psnr_list.append(psnr_hvsm(img_pred_yuv[:,:,0],img_target_yuv[:,:,0]))
                    if cfg['save_imgs']:
                        # save imgs
                        if idx_target == 0 or idx_target == cfg['n_imgs_in']:
                            cfg['toPIL'](cfg['deNorm'](img_pred[i,:])).save(os.path.join(img_dir_ep,str(i_batch)+'_'+str(i)+'.png'))
                        else:
                            cfg['toPIL'](cfg['deNorm'](img_pred[i,:])).save(os.path.join(img_dir_ip,str(i_batch)+'_'+str(i)+'.png'))
               
                # # save is_ep
                if idx_target == 0 or idx_target == cfg['n_imgs_in']:
                    is_ep_list.append(torch.ones(img_pred.shape[0]))
                else:
                    is_ep_list.append(torch.zeros(img_pred.shape[0]))
        
        
        # # save as arrays
        # # times
        t = (torch.cat(t_list))
        t = t.view(t.shape[0]*t.shape[1],-1).numpy() # flat sequence
        t_in = (torch.cat(t_in_list))
        t_in = t_in.view(t_in.shape[0]*t_in.shape[1],-1).numpy() # flat sequence
        t_out = (torch.cat(t_out_list)).numpy()
        min_dt = (torch.cat(min_dt_list)).numpy()
        # # scores
        score_l1 = np.array(score_l1_list)
        score_msssim = np.array(score_msssim_list)
        score_pla = np.array(score_pla_list)
        score_psnr = np.array(score_psnr_list)
        # # is extrapolation
        is_ep = (torch.cat(is_ep_list)).numpy()
        
        
        # # plot and save
        # # L1 vs. min_dt Interpolation
        fig, ax = plt.subplots(figsize=(cfg['figure_width'], cfg['figure_height']), dpi=cfg['plot_dpi'])
        im = ax.scatter(min_dt[is_ep==0], score_l1[is_ep==0], c=t_out[is_ep==0], cmap=plt.cm.plasma, edgecolor="darkslategray")
        im.set_clim(min(t_out), max(t_out))
        ax.set_xlabel(r"$\min(\Delta t)$")
        ax.set_ylabel(r"L1")
        fig.colorbar(im, ax=ax)
        plt.savefig(os.path.join(plot_dir,'min_dt_l1_ip.pdf'),dpi=cfg['plot_dpi'], bbox_inches='tight')
        plt.savefig(os.path.join(plot_dir,'min_dt_l1_ip.png'),dpi=cfg['plot_dpi'], bbox_inches='tight')
        plt.show()
        plt.close(fig)
        
        # # L1 vs. min_dt Extrapolation
        fig, ax = plt.subplots(figsize=(cfg['figure_width'], cfg['figure_height']), dpi=cfg['plot_dpi'])
        im = ax.scatter(min_dt[is_ep==1], score_l1[is_ep==1], c=t_out[is_ep==1], cmap=plt.cm.plasma, edgecolor="darkgray")
        im.set_clim(min(t_out), max(t_out))
        ax.set_xlabel(r"$\min(\Delta t)$")
        ax.set_ylabel(r"L1")
        fig.colorbar(im, ax=ax)
        plt.savefig(os.path.join(plot_dir,'min_dt_l1_ep.pdf'),dpi=cfg['plot_dpi'], bbox_inches='tight')
        plt.savefig(os.path.join(plot_dir,'min_dt_l1_ep.png'),dpi=cfg['plot_dpi'], bbox_inches='tight')
        plt.show()
        plt.close(fig)
        
        # # MSSSIM vs. min_dt Interpolation
        fig, ax = plt.subplots(figsize=(cfg['figure_width'], cfg['figure_height']), dpi=cfg['plot_dpi'])
        im = ax.scatter(min_dt[is_ep==0], score_msssim[is_ep==0], c=t_out[is_ep==0], cmap=plt.cm.plasma, edgecolor="darkslategray")
        im.set_clim(min(t_out), max(t_out))
        ax.set_xlabel(r"$\min(\Delta t)$")
        ax.set_ylabel(r"MS-SSIM")
        fig.colorbar(im, ax=ax)
        plt.savefig(os.path.join(plot_dir,'min_dt_msssim_ip.pdf'),dpi=cfg['plot_dpi'], bbox_inches='tight')
        plt.savefig(os.path.join(plot_dir,'min_dt_msssim_ip.png'),dpi=cfg['plot_dpi'], bbox_inches='tight')
        plt.show()
        plt.close(fig)
        
        # # MSSSIM vs. min_dt Extrapolation
        fig, ax = plt.subplots(figsize=(cfg['figure_width'], cfg['figure_height']), dpi=cfg['plot_dpi'])
        im = ax.scatter(min_dt[is_ep==1], score_msssim[is_ep==1], c=t_out[is_ep==1], cmap=plt.cm.plasma, edgecolor="darkgray")
        im.set_clim(min(t_out), max(t_out))
        ax.set_xlabel(r"$\min(\Delta t)$")
        ax.set_ylabel(r"MS-SSIM")
        fig.colorbar(im, ax=ax)
        plt.savefig(os.path.join(plot_dir,'min_dt_msssim_ep.pdf'),dpi=cfg['plot_dpi'], bbox_inches='tight')
        plt.savefig(os.path.join(plot_dir,'min_dt_msssim_ep.png'),dpi=cfg['plot_dpi'], bbox_inches='tight')
        plt.show()
        plt.close(fig)
        
        # # PSNR vs. min_dt Interpolation
        fig, ax = plt.subplots(figsize=(cfg['figure_width'], cfg['figure_height']), dpi=cfg['plot_dpi'])
        im = ax.scatter(min_dt[is_ep==0], score_psnr[is_ep==0], c=t_out[is_ep==0], cmap=plt.cm.plasma, edgecolor="darkslategray")
        im.set_clim(min(t_out), max(t_out))
        ax.set_xlabel(r"$\min(\Delta t)$")
        ax.set_ylabel(r"PSNR-HVS-M [db]")
        fig.colorbar(im, ax=ax)
        plt.savefig(os.path.join(plot_dir,'min_dt_psnr_ip.pdf'),dpi=cfg['plot_dpi'], bbox_inches='tight')
        plt.savefig(os.path.join(plot_dir,'min_dt_psnr_ip.png'),dpi=cfg['plot_dpi'], bbox_inches='tight')
        plt.show()
        plt.close(fig)
        
        # # PSNR vs. min_dt Extrapolation
        fig, ax = plt.subplots(figsize=(cfg['figure_width'], cfg['figure_height']), dpi=cfg['plot_dpi'])
        im = ax.scatter(min_dt[is_ep==1], score_psnr[is_ep==1], c=t_out[is_ep==1], cmap=plt.cm.plasma, edgecolor="darkgray")
        im.set_clim(min(t_out), max(t_out))
        ax.set_xlabel(r"$\min(\Delta t)$")
        ax.set_ylabel(r"PSNR-HVS-M [dB]")
        fig.colorbar(im, ax=ax)
        plt.savefig(os.path.join(plot_dir,'min_dt_psnr_ep.pdf'),dpi=cfg['plot_dpi'], bbox_inches='tight')
        plt.savefig(os.path.join(plot_dir,'min_dt_psnr_ep.png'),dpi=cfg['plot_dpi'], bbox_inches='tight')
        plt.show()
        plt.close(fig)
        
        scores = {'All: l1 mean': str(np.mean(score_l1)),
                  'All: l1 std': str(np.std(score_l1)),
                  'All: msssim mean': str(np.mean(score_msssim)),
                  'All: msssim std': str(np.std(score_msssim)),
                  'All: psnr mean': str(np.mean(score_psnr)),
                  'All: psnr std': str(np.std(score_psnr)),
                  'All: pla mean': str(np.mean(score_pla)),
                  'All: pla std': str(np.std(score_pla)),
                  'IP: l1 mean': str(np.mean(score_l1[is_ep==0])),
                  'IP: l1 std': str(np.std(score_l1[is_ep==0])),
                  'IP: msssim mean': str(np.mean(score_msssim[is_ep==0])),
                  'IP: msssim std': str(np.std(score_msssim[is_ep==0])),
                  'IP: psnr mean': str(np.mean(score_psnr[is_ep==0])),
                  'IP: psnr std': str(np.std(score_psnr[is_ep==0])),
                  'IP: pla mean': str(np.mean(score_pla[is_ep==0])),
                  'IP: pla std': str(np.std(score_pla[is_ep==0])),
                  'EP: l1 mean': str(np.mean(score_l1[is_ep==1])),
                  'EP: l1 std': str(np.std(score_l1[is_ep==1])),
                  'EP: msssim mean': str(np.mean(score_msssim[is_ep==1])),
                  'EP: msssim std': str(np.std(score_msssim[is_ep==1])),
                  'EP: psnr mean': str(np.mean(score_psnr[is_ep==1])),
                  'EP: psnr std': str(np.std(score_psnr[is_ep==1])),
                  'EP: pla mean': str(np.mean(score_pla[is_ep==1])),
                  'EP: pla std': str(np.std(score_pla[is_ep==1])),}
        
        with open(os.path.join(plot_dir,'scores.yaml'), 'w') as file:
            yaml.dump(scores, file)
        
        
        #%% generate target if specified with other ones as input
        print('generate target img...')
        max_plots = 3
        plot_dir = os.path.join(cfg['pred_dir'], (prfx[count]+'gen_target'))
        if not os.path.exists(plot_dir):
            os.makedirs(plot_dir)
        if cfg['target_pos'] > cfg['n_imgs'] or cfg['target_pos'] <=0 or cfg['target_pos'] is None:
            print('No target img specified.')
        else:
            idx_target = cfg['target_pos']-1 # target = specified target
            idx_in = tuple(list(range(cfg['n_imgs']))[:idx_target]+list(range(cfg['n_imgs']))[idx_target+1:])
            for i_batch, batch in enumerate(dataloader):
                if i_batch==max_plots:
                    break
                with torch.no_grad():
                    x = {'img_in': batch['seq_img'][:,idx_in,:], 
                         'timedelta_in': batch['seq_timedelta'][:,idx_in],
                         'timedelta_target': batch['seq_timedelta'][:,idx_target],
                         'z': None}
                    img_pred = model(x).cpu().detach()
                    
                    fig, axs = plt.subplots(1, cfg['n_imgs']+1)
                    for nf in range(cfg['n_imgs_in']):
                        axs[nf].imshow(cfg['toPIL'](cfg['deNorm'](np.squeeze(batch['seq_img'][2,idx_in[nf],:,:,:]))))
                        axs[nf].set_title('in:'+ str(batch['seq_timedelta'][2,idx_in[nf]].numpy()))
                    axs[cfg['n_imgs']-1].imshow(cfg['toPIL'](cfg['deNorm'](np.squeeze(batch['seq_img'][2,idx_target,:,:,:]))))
                    axs[cfg['n_imgs']-1].set_title('t:'+ str(batch['seq_timedelta'][2,idx_target].numpy()))    
                    axs[cfg['n_imgs']].imshow(cfg['toPIL'](cfg['deNorm'](img_pred[2,:])))
                    axs[cfg['n_imgs']].set_title('g:' + str(batch['seq_timedelta'][2,idx_target].numpy()))
                    [axi.set_axis_off() for axi in axs.ravel()]
                    plt.savefig(os.path.join(plot_dir,'pred_target_'+str(i_batch)), dpi=cfg['plot_dpi'], bbox_inches='tight')
                    plt.close(fig)
        
        
        #%% generate random img with other ones as input
        print('generate random img...')
        max_plots = 3
        plot_dir = os.path.join(cfg['pred_dir'], (prfx[count]+'gen_rand'))
        if not os.path.exists(plot_dir):
            os.makedirs(plot_dir)
        for i_batch, batch in enumerate(dataloader):
            if i_batch==max_plots:
                break   
            idx_target = random.randint(0,cfg['n_imgs_in'])
            idx_in = tuple(list(range(cfg['n_imgs']))[:idx_target]+list(range(cfg['n_imgs']))[idx_target+1:])
            with torch.no_grad():
                x = {'img_in': batch['seq_img'][:,idx_in,:], 
                      'timedelta_in': batch['seq_timedelta'][:,idx_in],
                      'timedelta_target': batch['seq_timedelta'][:,idx_target],
                      'z': None}
                img_pred = model(x).cpu().detach()
                
                fig, axs = plt.subplots(1, cfg['n_imgs']+1)
                for nf in range(cfg['n_imgs_in']):
                    axs[nf].imshow(cfg['toPIL'](cfg['deNorm'](np.squeeze(batch['seq_img'][0,idx_in[nf],:,:,:]))))
                    axs[nf].set_title('in:'+ str(batch['seq_timedelta'][0,idx_in[nf]].numpy()))
                axs[cfg['n_imgs']-1].imshow(cfg['toPIL'](cfg['deNorm'](np.squeeze(batch['seq_img'][0,idx_target,:,:,:]))))
                axs[cfg['n_imgs']-1].set_title('t:'+ str(batch['seq_timedelta'][0,idx_target].numpy()))
                axs[cfg['n_imgs']].imshow(cfg['toPIL'](cfg['deNorm'](img_pred[0,:])))
                axs[cfg['n_imgs']].set_title('g:' + str(batch['seq_timedelta'][0,idx_target].numpy()))
                [axi.set_axis_off() for axi in axs.ravel()]
                plt.savefig(os.path.join(plot_dir,'pred_rand_'+str(i_batch)), dpi=cfg['plot_dpi'], bbox_inches='tight')
                plt.close(fig)
                
                
        #%% generate first img with other ones as input
        print('generate first img...')
        max_plots = 3
        plot_dir = os.path.join(cfg['pred_dir'], (prfx[count]+'gen_first'))
        if not os.path.exists(plot_dir):
            os.makedirs(plot_dir)
        idx_target = 0 # target=first
        idx_in = tuple(list(range(cfg['n_imgs']))[:idx_target]+list(range(cfg['n_imgs']))[idx_target+1:])
        for i_batch, batch in enumerate(dataloader):
            if i_batch==max_plots:
                break
            with torch.no_grad():
                x = {'img_in': batch['seq_img'][:,idx_in,:], 
                     'timedelta_in': batch['seq_timedelta'][:,idx_in],
                     'timedelta_target': batch['seq_timedelta'][:,idx_target],
                     'z': None}
                img_pred = model(x).cpu().detach()
                
                fig, axs = plt.subplots(1, cfg['n_imgs']+1)
                for nf in range(cfg['n_imgs_in']):
                    axs[nf].imshow(cfg['toPIL'](cfg['deNorm'](np.squeeze(batch['seq_img'][2,idx_in[nf],:,:,:]))))
                    axs[nf].set_title('in:'+ str(batch['seq_timedelta'][2,idx_in[nf]].numpy()))
                axs[cfg['n_imgs']-1].imshow(cfg['toPIL'](cfg['deNorm'](np.squeeze(batch['seq_img'][2,idx_target,:,:,:]))))
                axs[cfg['n_imgs']-1].set_title('t:'+ str(batch['seq_timedelta'][2,idx_target].numpy()))    
                axs[cfg['n_imgs']].imshow(cfg['toPIL'](cfg['deNorm'](img_pred[2,:])))
                axs[cfg['n_imgs']].set_title('g:' + str(batch['seq_timedelta'][2,idx_target].numpy()))
                [axi.set_axis_off() for axi in axs.ravel()]
                plt.savefig(os.path.join(plot_dir,'pred_target_'+str(i_batch)), dpi=cfg['plot_dpi'], bbox_inches='tight')
                plt.close(fig)
                
                
        #%% generate last img with other ones as input
        print('generate last img...')
        max_plots = 3
        plot_dir = os.path.join(cfg['pred_dir'], (prfx[count]+'gen_last'))
        if not os.path.exists(plot_dir):
            os.makedirs(plot_dir)
        idx_target = cfg['n_imgs']-1 # target=last
        idx_in = tuple(list(range(cfg['n_imgs']))[:idx_target]+list(range(cfg['n_imgs']))[idx_target+1:])
        for i_batch, batch in enumerate(dataloader):
            if i_batch==max_plots:
                break
            with torch.no_grad():
                x = {'img_in': batch['seq_img'][:,idx_in,:], 
                     'timedelta_in': batch['seq_timedelta'][:,idx_in],
                     'timedelta_target': batch['seq_timedelta'][:,idx_target],
                     'z': None}
                img_pred = model(x).cpu().detach()
                
                fig, axs = plt.subplots(1, cfg['n_imgs']+1)
                for nf in range(cfg['n_imgs_in']):
                    axs[nf].imshow(cfg['toPIL'](cfg['deNorm'](np.squeeze(batch['seq_img'][2,idx_in[nf],:,:,:]))))
                    axs[nf].set_title('in:'+ str(batch['seq_timedelta'][2,idx_in[nf]].numpy()))
                axs[cfg['n_imgs']-1].imshow(cfg['toPIL'](cfg['deNorm'](np.squeeze(batch['seq_img'][2,idx_target,:,:,:]))))
                axs[cfg['n_imgs']-1].set_title('t:'+ str(batch['seq_timedelta'][2,idx_target].numpy()))    
                axs[cfg['n_imgs']].imshow(cfg['toPIL'](cfg['deNorm'](img_pred[2,:])))
                axs[cfg['n_imgs']].set_title('g:' + str(batch['seq_timedelta'][2,idx_target].numpy()))
                [axi.set_axis_off() for axi in axs.ravel()]
                plt.savefig(os.path.join(plot_dir,'pred_target_'+str(i_batch)), dpi=cfg['plot_dpi'], bbox_inches='tight')
                plt.close(fig)
                
            
        #%% generate images in input sequence range + extrapol, compute pla
        # # z fixed over time! 
        # # z0
        print('generate all imgs in input seq range, pla, pred-token emb; z0-z1...')
        max_plots = 1
        steps = 1
        extrapol = 72
        
        for i_batch, batch in enumerate(dataloader):
            if i_batch==max_plots:
                break
            plot_dir = os.path.join(cfg['pred_dir'], (prfx[count]+'short_dist_gen_'+str(i_batch)+'_z0'))
            if not os.path.exists(plot_dir):
                os.makedirs(plot_dir)
            idx_target = cfg['n_imgs_in']#random.randint(0,cfg['n_imgs_in'])
            idx_in = tuple(list(range(cfg['n_imgs']))[:idx_target]+list(range(cfg['n_imgs']))[idx_target+1:])
            
            # get in times and accordingly gap sizes
            in_times = batch['seq_timedelta'][0,idx_in]
            gap_sizes = torch.diff(batch['seq_timedelta'][0,idx_in])
            
            # since there is a bug, if there are two or more equal times in the input...
            if torch.unique(in_times).numel() == cfg['n_imgs_in']:
            
                ref_pla_list = []
                for nf in range(cfg['n_imgs_in']):
                    ref_pla_list.append(utils.pla_per_img(batch['seq_img'][0,idx_in[nf],:,:,:]))
                    cfg['toPIL'](cfg['deNorm'](np.squeeze(batch['seq_img'][0,idx_in[nf],:,:,:]))).save(os.path.join(plot_dir,str(batch['seq_timedelta'][0,idx_in[nf]].numpy())+'_in.png'))
                
                lb = max(0, batch['seq_timedelta'][0,idx_in[0]].item()-extrapol) # lower bound    
                ub = min(batch['seq_timedelta'][0,idx_in[-1]].item()+extrapol+1, cfg['pe_max_len']) # upper bound
                
                # # set z (fix it to one specific realisiation of N(0,1))
                # # different z per image in sequence
                # z = Variable(torch.tensor(np.random.normal(0, 1, (cfg['batch_size'],cfg['n_imgs'],cfg['dim_z']))).to(cfg['device']))
                # z = z.repeat(1, 1, int(cfg['dim_img']/cfg['dim_z']))
                # # same z per image in sequence
                z = Variable(torch.tensor(np.random.normal(0, 1, (cfg['batch_size'],1,cfg['dim_z']))).to(cfg['device']))
                z = z.repeat(1, cfg['n_imgs'], int(cfg['dim_img']/cfg['dim_z']))
               
                all_imgs = []
                pred_pla_list_z0 = []
                t = []
                t_in = []
                t_out = []
                for j in range(lb,ub,steps):
                    with torch.no_grad():
                        x = {'img_in': batch['seq_img'][:,idx_in,:], 
                              'timedelta_in': batch['seq_timedelta'][:,idx_in],
                              'timedelta_target': (torch.ones(cfg['batch_size'])*j).long(),
                              'z': z}
                        img_pred = model(x).cpu().detach()
                        all_imgs.append(cfg['toPIL'](cfg['deNorm'](img_pred[0,:])))
                        pred_pla_list_z0.append(utils.pla_per_img(img_pred[0,:]))
                        cfg['toPIL'](cfg['deNorm'](img_pred[0,:])).save(os.path.join(plot_dir,str(j)+'_pred.png'))
                        
                        # # save times
                        t.append(torch.cat((batch['seq_timedelta'][0,idx_in],torch.tensor(j).unsqueeze(dim=0))))
                        t_in.append(batch['seq_timedelta'][0,idx_in])
                        t_out.append(j)
                
                t_idx_in_out_z0 = [t_out.index(i) for i in t_in[0]] # list of idx of where t_in times are in t_out
                
                # # save video
                imageio.mimsave(os.path.join(plot_dir,'video.gif'), all_imgs)
    
                # # save t as arrays
                t = (torch.cat(t)).numpy()
                t_in = (torch.cat(t_in)).numpy()
                t_out_z0 = np.array(t_out)
                            
                # # PLA plot
                f, ax = plt.subplots(figsize=(cfg['figure_width'], cfg['figure_height']), dpi=cfg['plot_dpi'])
                # ref/pred
                ax.plot(t_out_z0, pred_pla_list_z0, '.', label="$z_0$: gen img", color='darkslategray')
                ax.plot(t_in[:cfg['n_imgs_in']], ref_pla_list, '.', label="in img", color='#ff336b')
                
                ax.legend(frameon=False)
                ax.set_ylabel("projected leaf area [px/img]")
                # ax.set_ylim(0, 0.8)
                ax.set_xlabel("time [h]")
                ax.grid(color='grey', linestyle='-', linewidth=0.25, alpha=0.5)
                ax.xaxis.grid(False)
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                
                plt.savefig(os.path.join(plot_dir,'pla_short_dist_gen.pdf'),dpi=cfg['plot_dpi'],bbox_inches='tight')
                plt.savefig(os.path.join(plot_dir,'pla_short_dist_gen.png'),dpi=cfg['plot_dpi'],bbox_inches='tight')
                f.tight_layout()
                f.savefig(os.path.join(plot_dir,'pla_short_dist_gen.pdf'))
                f.savefig(os.path.join(plot_dir,'pla_short_dist_gen.png'))
                #plt.show()
                plt.close(f)
                
                
    
                # # z1
                plot_dir = os.path.join(cfg['pred_dir'], (prfx[count]+'short_dist_gen_'+str(i_batch)+'_z1'))
                if not os.path.exists(plot_dir):
                    os.makedirs(plot_dir)
                idx_target = cfg['n_imgs_in']#random.randint(0,cfg['n_imgs_in'])
                idx_in = tuple(list(range(cfg['n_imgs']))[:idx_target]+list(range(cfg['n_imgs']))[idx_target+1:])
                
                ref_pla_list = []
                for nf in range(cfg['n_imgs_in']):
                    ref_pla_list.append(utils.pla_per_img(batch['seq_img'][0,idx_in[nf],:,:,:]))
                    cfg['toPIL'](cfg['deNorm'](np.squeeze(batch['seq_img'][0,idx_in[nf],:,:,:]))).save(os.path.join(plot_dir,str(batch['seq_timedelta'][0,idx_in[nf]].numpy())+'_in.png'))
                
                # lb = max(0, batch['seq_timedelta'][0,idx_in[0]]-extrapol) # lower bound    
                # ub = min(batch['seq_timedelta'][0,idx_in[-1]]+extrapol, cfg['pe_max_len']) # upper bound
                
                # # set z (fix it to one specific realisiation of N(0,1))
                # # different z per image in sequence
                # z = Variable(torch.tensor(np.random.normal(0, 1, (cfg['batch_size'],cfg['n_imgs'],cfg['dim_z']))).to(cfg['device']))
                # z = z.repeat(1, 1, int(cfg['dim_img']/cfg['dim_z']))
                # # same z per image in sequence
                z = Variable(torch.tensor(np.random.normal(0, 1, (cfg['batch_size'],1,cfg['dim_z']))).to(cfg['device']))
                z = z.repeat(1, cfg['n_imgs'], int(cfg['dim_img']/cfg['dim_z']))
                                
                all_imgs = []
                pred_pla_list_z1 = []
                t = []
                t_in = []
                t_out = []
                for j in range(lb,ub,steps):
                    with torch.no_grad():
                        x = {'img_in': batch['seq_img'][:,idx_in,:], 
                              'timedelta_in': batch['seq_timedelta'][:,idx_in],
                              'timedelta_target': (torch.ones(cfg['batch_size'])*j).long(),
                              'z': z}
                        img_pred = model(x).cpu().detach()
                        all_imgs.append(cfg['toPIL'](cfg['deNorm'](img_pred[0,:])))
                        pred_pla_list_z1.append(utils.pla_per_img(img_pred[0,:]))
                        cfg['toPIL'](cfg['deNorm'](img_pred[0,:])).save(os.path.join(plot_dir,str(j)+'_pred.png'))
                        
                        # # save times
                        t.append(torch.cat((batch['seq_timedelta'][0,idx_in],torch.tensor(j).unsqueeze(dim=0))))
                        t_in.append(batch['seq_timedelta'][0,idx_in])
                        t_out.append(j)
                
                t_idx_in_out_z1 = [t_out.index(i) for i in t_in[0]] # list of idx of where t_in times are in t_out
                
                # # save video
                imageio.mimsave(os.path.join(plot_dir,'video.gif'), all_imgs)
                
                # # save t as arrays
                t = (torch.cat(t)).numpy()
                t_in = (torch.cat(t_in)).numpy()
                t_out_z1 = np.array(t_out)
                
    
                # # PLA plot
                f, ax = plt.subplots(figsize=(cfg['figure_width'], cfg['figure_height']), dpi=cfg['plot_dpi'])
                # ref/pred
                ax.plot(t_out_z1, pred_pla_list_z1, 'v', label="$z_1$: gen img", color='darkgray')
                ax.plot(t_in[:cfg['n_imgs_in']], ref_pla_list, 'v', label="in img", color='#ff336b')
                
                ax.legend(frameon=False)
                ax.set_ylabel("projected leaf area [px/img]")
                # ax.set_ylim(0, 0.8)
                ax.set_xlabel("time [h]")
                ax.grid(color='grey', linestyle='-', linewidth=0.25, alpha=0.5)
                ax.xaxis.grid(False)
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                #plt.show()
                plt.close(f)
                
                plt.savefig(os.path.join(plot_dir,'pla_short_dist_gen.pdf'),dpi=cfg['plot_dpi'],bbox_inches='tight')
                plt.savefig(os.path.join(plot_dir,'pla_short_dist_gen.png'),dpi=cfg['plot_dpi'],bbox_inches='tight')
                f.tight_layout()
                f.savefig(os.path.join(plot_dir,'pla_short_dist_gen.pdf'))
                f.savefig(os.path.join(plot_dir,'pla_short_dist_gen.png'))
            
                
                # # make combined plots for z0-z1
                plot_dir = os.path.join(cfg['pred_dir'], (prfx[count]+'short_dist_gen_'+str(i_batch)+'_z0_z1'))
                if not os.path.exists(plot_dir):
                    os.makedirs(plot_dir)
                    
                # # PLA plot
                f, ax = plt.subplots(figsize=(cfg['figure_width'], cfg['figure_height']), dpi=cfg['plot_dpi'])
                # ref/pred
                ax.plot(t_out_z0, pred_pla_list_z0, '.', label="$z_0$: gen", color='darkslategray', markersize=3)
                ax.plot(t_out_z1, pred_pla_list_z1, 'v', label="$z_1$: gen", color='darkgray', markersize=3)
                ax.plot(t_in[:cfg['n_imgs_in']], ref_pla_list, 'D', label="in", color='#ff336b', markersize=3)
                
                ax.legend(frameon=False)
                ax.set_ylabel("projected leaf area [px/img]")
                # ax.set_ylim(0, 0.8)
                ax.set_xlabel("time [h]")
                ax.grid(color='grey', linestyle='-', linewidth=0.25, alpha=0.5)
                ax.xaxis.grid(False)
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                
                plt.savefig(os.path.join(plot_dir,'pla_short_dist_gen.pdf'),dpi=cfg['plot_dpi'],bbox_inches='tight')
                plt.savefig(os.path.join(plot_dir,'pla_short_dist_gen.png'),dpi=cfg['plot_dpi'],bbox_inches='tight')
                f.tight_layout()
                f.savefig(os.path.join(plot_dir,'pla_short_dist_gen.pdf'))
                f.savefig(os.path.join(plot_dir,'pla_short_dist_gen.png'))
                #plt.show()
                plt.close(f)
        

        #%% generate images over whole time from [0 steps cfg['pe_max_len']], compute pla
        # # random z
        print('generate images over whole time, pla, ...')
        max_plots = 1
        steps = 1
        for i_batch, batch in enumerate(dataloader):
            if i_batch==max_plots:
                break
            plot_dir = os.path.join(cfg['pred_dir'], (prfx[count]+'long_dist_gen_'+str(i_batch)+'_z'))
            if not os.path.exists(plot_dir):
                os.makedirs(plot_dir)
            idx_target = cfg['n_imgs_in']#random.randint(0,cfg['n_imgs_in'])
            idx_in = tuple(list(range(cfg['n_imgs']))[:idx_target]+list(range(cfg['n_imgs']))[idx_target+1:])
            
            ref_pla_list = []
            for nf in range(cfg['n_imgs_in']):
                ref_pla_list.append(utils.pla_per_img(batch['seq_img'][0,idx_in[nf],:,:,:]))
                cfg['toPIL'](cfg['deNorm'](np.squeeze(batch['seq_img'][0,idx_in[nf],:,:,:]))).save(os.path.join(plot_dir,str(batch['seq_timedelta'][0,idx_in[nf]].numpy())+'_in.png'))
            
            all_imgs = []
            pred_pla_list = []
            t = []
            t_in = []
            t_out = []
            for j in range(0,cfg['pe_max_len'],steps):
                with torch.no_grad():
                    x = {'img_in': batch['seq_img'][:,idx_in,:], 
                          'timedelta_in': batch['seq_timedelta'][:,idx_in],
                          'timedelta_target': (torch.ones(cfg['batch_size'])*j).long(),
                          'z': None}
                    img_pred = model(x).cpu().detach()
                    all_imgs.append(cfg['toPIL'](cfg['deNorm'](img_pred[0,:])))
                    pred_pla_list.append(utils.pla_per_img(img_pred[0,:]))
                    cfg['toPIL'](cfg['deNorm'](img_pred[0,:])).save(os.path.join(plot_dir,str(j)+'_pred.png'))
                    
                    # # save times
                    t.append(torch.cat((batch['seq_timedelta'][0,idx_in],torch.tensor(j).unsqueeze(dim=0))))
                    t_in.append(batch['seq_timedelta'][0,idx_in])
                    t_out.append(j)
            
            t_idx_in_out = [t_out.index(i) for i in t_in[0]] # list of idx of where t_in times are in t_out
            
            # # save video
            imageio.mimsave(os.path.join(plot_dir,'video.gif'), all_imgs)
            
            # # save t as arrays
            t = (torch.cat(t)).numpy()
            t_in = (torch.cat(t_in)).numpy()
            t_out = np.array(t_out)
            
            # # PLA plot
            f, ax = plt.subplots(figsize=(cfg['figure_width'], cfg['figure_height']), dpi=cfg['plot_dpi'])
            # ref/pred
            ax.plot(t_out, pred_pla_list, '.', label="gen img", color='#499f2d')
            ax.plot(t_in[:cfg['n_imgs_in']], ref_pla_list, '.', label="in img", color='#ff336b')
            
            ax.legend(frameon=False)
            ax.set_ylabel("projected leaf area [px/img]")
            # ax.set_ylim(0, 0.8)
            ax.set_xlabel("time [h]")
            ax.grid(color='grey', linestyle='-', linewidth=0.25, alpha=0.5)
            ax.xaxis.grid(False)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            
            plt.savefig(os.path.join(plot_dir,'pla_long_dist_gen.pdf'),dpi=cfg['plot_dpi'],bbox_inches='tight')
            plt.savefig(os.path.join(plot_dir,'pla_long_dist_gen.png'),dpi=cfg['plot_dpi'],bbox_inches='tight')
            f.tight_layout()
            f.savefig(os.path.join(plot_dir,'pla_long_dist_gen.pdf'))
            f.savefig(os.path.join(plot_dir,'pla_long_dist_gen.png'))
            #plt.show()
            plt.close(f)
        
        
        #%% generate images and stds in input sequence range + extrapol
        print('generate imgs in input seq range + extrapol and corresponding std...')
        max_plots = 1
        steps = 1
        extrapol = 72
        runs = 10

        for i_batch, batch in enumerate(dataloader):
            if i_batch==max_plots:
                break
            plot_dir = os.path.join(cfg['pred_dir'], (prfx[count]+'short_dist_gen_imgstd_'+str(i_batch)))
            if not os.path.exists(plot_dir):
                os.makedirs(plot_dir)
            idx_target = cfg['n_imgs_in']#random.randint(0,nframes_in)
            idx_in = tuple(list(range(cfg['n_imgs']))[:idx_target]+list(range(cfg['n_imgs']))[idx_target+1:])
            
            # get in times and accordingly gap sizes
            in_times = batch['seq_timedelta'][0,idx_in]
            gap_sizes = torch.diff(batch['seq_timedelta'][0,idx_in])
            
            # since there is a bug, if there are two or more equal times in the input...
            if torch.unique(in_times).numel() == cfg['n_imgs_in']:
                
                ref_pla_list = []
                for nf in range(cfg['n_imgs_in']):
                    ref_pla_list.append(utils.pla_per_img(batch['seq_img'][0,idx_in[nf],:,:,:]))
                    cfg['toPIL'](cfg['deNorm'](np.squeeze(batch['seq_img'][0,idx_in[nf],:,:,:]))).save(os.path.join(plot_dir,str(batch['seq_timedelta'][0,idx_in[nf]].numpy())+'_in.png'))
                
                lb = max(0, batch['seq_timedelta'][0,idx_in[0]].item()-extrapol) # lower bound    
                ub = min(batch['seq_timedelta'][0,idx_in[-1]].item()+extrapol+1, cfg['pe_max_len']) # upper bound
                
                # set z (if z=None, it will be generated randomly for every new generation)
                # use always the same z for ONLY first generation of each time point in order to get a consisent generation
                z = Variable(torch.tensor(np.random.normal(0, 1, (cfg['batch_size'],1,cfg['dim_z']))).to(cfg['device']))
                z = z.repeat(1, cfg['n_imgs'], int(cfg['dim_img']/cfg['dim_z']))
                
                for j in range(lb,ub,steps):
                    img_pred = torch.empty((runs,3,cfg['img_size'],cfg['img_size']))
                    for k in range(runs):
                        with torch.no_grad():
                            if k == 0:
                                x = {'img_in': batch['seq_img'][:,idx_in,:], 
                                      'timedelta_in': batch['seq_timedelta'][:,idx_in],
                                      'timedelta_target': (torch.ones(cfg['batch_size'])*j).long(),
                                      'z': z}
                            else:
                                x = {'img_in': batch['seq_img'][:,idx_in,:], 
                                      'timedelta_in': batch['seq_timedelta'][:,idx_in],
                                      'timedelta_target': (torch.ones(cfg['batch_size'])*j).long(),
                                      'z': None}
                            img_pred[k,:] = model(x).cpu().detach()[0,:]
                                    
                    cfg['toPIL'](cfg['deNorm'](img_pred[0,:])).save(os.path.join(plot_dir,str(j)+'_pred.png'))
                    
                    img_pred_mean = torch.mean(img_pred, axis=0)
                    img_pred_std = torch.mean(torch.std(img_pred, axis=0), axis=0)
                    
                    # # save mean
                    # trans(deNorm(img_pred_mean)).save(os.path.join(plot_dir,str(j)+'_mean.png'))
                    # save std
                    cmap = plt.cm.Blues # a colormap
                    # norm = plt.Normalize(vmin=img_pred_std.min(), vmax=img_pred_std.max())
                    norm = plt.Normalize(vmin=0, vmax=0.05)
                    plt.imsave(os.path.join(plot_dir,str(j)+'_std.png'), cmap(norm(img_pred_std)))