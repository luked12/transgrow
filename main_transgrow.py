"""
===============================================================================
TransGrow: Train TransGrow to generate time-variable image from image sequence with CNN and Transformer
===============================================================================
"""

import sys, os, io, warnings, time, logging
import torch
import torch.multiprocessing
import pytorch_lightning as pl
import torchvision
import numpy as np
import matplotlib.pyplot as plt
import random
import yaml

from configs.config_main_transgrow import cfg

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

if __name__ == '__main__':  
    
    #%% write cfg.yaml to exp_dir
    with io.open(os.path.join(cfg['exp_dir'], 'cfg_main.yaml'), 'w', encoding='utf8') as outfile:
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
    
    # write dataModule params
    with open(os.path.join(cfg['exp_dir'], 'hparams_data.yml'), 'w') as outfile:
        yaml.dump(dataModule.params, outfile, default_flow_style=False, allow_unicode=True)
    
    
    #%% visualize training sample
    # show x sample from train set (it is always the first image of the batch)
    max_plots = 3
    train_dataloader = dataModule.train_dataloader()
    for i_batch, batch in enumerate(train_dataloader):
        if i_batch==max_plots:
            break
        fig, axs = plt.subplots(1, cfg['n_imgs'])
        for nf in range(cfg['n_imgs']):
            axs[nf].imshow(cfg['toPIL'](cfg['deNorm'](np.squeeze(batch['seq_img'][1,nf,:,:,:]))))
            axs[nf].set_title(batch['seq_timedelta'][1,nf].numpy())
        for nf in range(cfg['n_imgs']):
            axs[nf].imshow(cfg['toPIL'](cfg['deNorm'](np.squeeze(batch['seq_img'][2,nf,:,:,:]))))
            axs[nf].set_title(batch['seq_timedelta'][2,nf].numpy())
        #plt.close(fig)
    
    
    #%% build a model
    if cfg['use_model'] == 'gan':
        model = TransGrowGANModel(dataModule.data_dims,cfg['g_enc_net'],cfg['g_enc_net_pretrained'],cfg['g_dec_net'],cfg['g_dec_net_pretrained'],cfg['pe_max_len'],cfg['pe_fusion_type'],cfg['dim_img'],cfg['dim_pe'],cfg['dim_z'],cfg['dim'],cfg['depth'],cfg['heads'],cfg['dropout_transformer'],cfg['dropout_emb'],cfg['prd_token_type'],cfg['d_net'],cfg['d_net_pretrained'],cfg['lr'],cfg['target_pos'],cfg['losses_w'],cfg['final_actvn'])
    elif cfg['use_model'] == 'wgan':
        model = TransGrowWGANModel(dataModule.data_dims,cfg['g_enc_net'],cfg['g_enc_net_pretrained'],cfg['g_dec_net'],cfg['g_dec_net_pretrained'],cfg['pe_max_len'],cfg['pe_fusion_type'],cfg['dim_img'],cfg['dim_pe'],cfg['dim_z'],cfg['dim'],cfg['depth'],cfg['heads'],cfg['dropout_transformer'],cfg['dropout_emb'],cfg['prd_token_type'],cfg['d_net'],cfg['d_net_pretrained'],cfg['lr'],cfg['target_pos'],cfg['losses_w'],cfg['final_actvn'])
    elif cfg['use_model'] == 'wgangp':
        model = TransGrowWGANGPModel(dataModule.data_dims,cfg['g_enc_net'],cfg['g_enc_net_pretrained'],cfg['g_dec_net'],cfg['g_dec_net_pretrained'],cfg['pe_max_len'],cfg['pe_fusion_type'],cfg['dim_img'],cfg['dim_pe'],cfg['dim_z'],cfg['dim'],cfg['depth'],cfg['heads'],cfg['dropout_transformer'],cfg['dropout_emb'],cfg['prd_token_type'],cfg['d_net'],cfg['d_net_pretrained'],cfg['lr'],cfg['target_pos'],cfg['losses_w'],cfg['final_actvn'])
    else:
        print('ERROR: FALSE MODEL SPECIFIED!')
    print(model.hparams)
    
    
    #%% training
    # # Build a trainer from train parameters, callbacks, and logger
    trainer = pl.Trainer(
        max_epochs=cfg['max_epochs'], 
        gpus=cfg['gpus'],
        callbacks=cfg['callbacks'],
        logger=[cfg['tb_logger']],
        precision=cfg['precision'],
        fast_dev_run=cfg['fast_dev_run'], 
        limit_train_batches=cfg['limit_train_batches'],
        limit_val_batches=cfg['limit_val_batches'],
        limit_test_batches=cfg['limit_test_batches'],
    )
        
    # # train
    start_time = time.time()
    trainer.fit(model, dataModule,ckpt_path=cfg['ckpt_path_resume'])
    print('Training finished. Elapsed Time:', str(round((time.time()-start_time)/60,2)), 'min')
    
    
    #%% test  
    if cfg['run_test']:
        trainer.test(verbose=False)
    
    
    #%% plotting
    if not cfg['run_plots']:
        sys.exit()
        
        
    #%% load model from best checkpoint if available otherwise last checkpoint is loaded automatically
    # # or uncomment last_model_path manually
    ckpt_path = trainer.checkpoint_callback.best_model_path
    # ckpt_path = trainer.checkpoint_callback.last_model_path
    print('ckpt_path: ', ckpt_path)
    
    if cfg['use_model'] == 'gan':
        model = TransGrowGANModel.load_from_checkpoint(ckpt_path)
    elif cfg['use_model'] == 'wgan':
        model = TransGrowWGANModel.load_from_checkpoint(ckpt_path)
    elif cfg['use_model'] == 'wgangp':
        model = TransGrowWGANGPModel.load_from_checkpoint(ckpt_path)
        
    # # set to eval mode
    model.eval()    
    # # sent model to device
    model.to(cfg['device'])
    
    
    #%% generate random test img with other ones as input
    max_plots = 25
    plot_dir = os.path.join(cfg['exp_dir'],'test_gen_rand')
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)
    for i_batch, batch in enumerate(dataModule.test_dataloader()):
        if i_batch==max_plots:
            break   
        idx_target = random.randint(0,cfg['n_imgs_in'])
        idx_in = tuple(list(range(cfg['n_imgs']))[:idx_target]+list(range(cfg['n_imgs']))[idx_target+1:])
        with torch.no_grad():
            x = {'img_in': batch['seq_img'][:,idx_in,:], 
                  'timedelta_in': batch['seq_timedelta'][:,idx_in],
                  'timedelta_target': batch['seq_timedelta'][:,idx_target]}
            img_pred = model(x).cpu().detach()
            
            fig, axs = plt.subplots(1, cfg['n_imgs']+1)
            for nf in range(cfg['n_imgs_in']):
                axs[nf].imshow(cfg['toPIL'](cfg['deNorm'](np.squeeze(batch['seq_img'][0,idx_in[nf],:,:,:]))))
                axs[nf].set_title('in:'+ str(batch['seq_timedelta'][0,idx_in[nf]].numpy()))
            axs[cfg['n_imgs']-1].imshow(cfg['toPIL'](cfg['deNorm'](np.squeeze(batch['seq_img'][0,idx_target,:,:,:]))))
            axs[cfg['n_imgs']-1].set_title('t:'+ str(batch['seq_timedelta'][0,idx_target].numpy()))
            axs[cfg['n_imgs']].imshow(cfg['toPIL'](cfg['deNorm'](img_pred[0,:])))
            axs[cfg['n_imgs']].set_title('p:' + str(batch['seq_timedelta'][0,idx_target].numpy()))
            [axi.set_axis_off() for axi in axs.ravel()]
            plt.savefig(os.path.join(plot_dir,'pred_rand_'+str(i_batch)), dpi=cfg['plot_dpi'], bbox_inches='tight')
            plt.close(fig)