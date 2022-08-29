import sys, os, warnings, time, logging
import yaml
import numpy as np
import math
import matplotlib
import torchvision
import torch
import pytorch_lightning as pl

from datetime import datetime
from utils import utils
from utils.custom_transforms import SinglePlantRotation

'''
Config for pred_transgrow.py as dictionary 'cfg'.
1. [REQUIRED] Specify exp_name and ckpt for which you want to run predictions
2. [AUTOMATIC] cfg_main.yaml is loaded from exp_name
3. [OPTIONAL] Update cfg as desired
4. [AUTOMATIC] Create pred_dir 
5. [OPTIONAL] Set font for figures

# *****************************************************************************
Config elements framed with # **** are are dependent on other elements
# *****************************************************************************
'''


#%% 1. [REQUIRED] Specify exp_name and ckpt for which you want to have predictions
'''
log_dir : str
    directory, where the experiments are saved
exp_name : str
    name of exp to be loaded
ckpt_type
    which saved model ckpt should be used for prediction?
    choose e.g. 'last' or 'best_epoch=123'
train_results : bool
    run also predictions for training set?
    Default: False (only predictions for test set are computed)
save_imgs : bool
    Save all generated images during the calculation of scores? (e.g. to calculate FID afterwards)
'''

log_dir = 'lightning_logs'
exp_name = '20220829_155222_abd_wgangp_tf_0_dim_512_pe_add_img_128_z_16'
ckpt_type = 'best_epoch=0'
train_results = False
save_imgs = False


#%% 2. [AUTOMATIC] cfg_main.yaml (stored after training) is loaded from exp_name
'''
Load cfg_main.yml, which was stored in training and add/update the following cfg elements:
add/update log_dir, exp_name, cklpt_type, train_results, save_imgs (all described above)
add/update device, ckpt_path_pred

device : int
    current cuda device
ckpt_path_pred : str
    whole path of the spefified ckpt to run predictions from
'''

# *****************************************************************************
# # Load cfg from training
cfg_path = os.path.join(log_dir, exp_name, 'cfg_main.yaml')
with open(cfg_path, 'r') as stream:
    cfg = yaml.load(stream, Loader=yaml.Loader)
# # add/update parameters specified above
cfg.update({'log_dir': log_dir}) 
cfg.update({'exp_name': exp_name}) 
cfg['ckpt_type'] = ckpt_type
cfg['train_results'] = train_results
cfg['save_imgs'] = save_imgs
cfg.update({'device': torch.cuda.current_device()}) 

# # Set ckpt path to run predictions from 
ckpt_dir = os.path.join(log_dir, exp_name, 'checkpoints')
ckpts_paths = utils.getListOfFiles(ckpt_dir)
matching = [s for s in ckpts_paths if ckpt_type in s]
cfg['ckpt_path_pred'] = matching[0]
# ***end***********************************************************************


#%% 3. [OPTIONAL] Update cfg as desired
'''
typically data params or transformations are varied, BUT NOT img_size

'''
cfg.update({'nworkers': 8}) 
cfg.update({'batch_size': 8}) 
cfg.update({'n_imgs_in': 3})

# *****************************************************************************
cfg.update({'n_imgs': cfg['n_imgs_in'] + cfg['n_imgs_out']})  
# ***end***********************************************************************

# cfg.update({'sample_type': 'semirand'}) 
# cfg.update({'rem_dup': False})
# cfg.update({'img_path_dist': 5}) 
# cfg.update({'img_path_skip': 4})
cfg.update({'sample_factor': 1.00})
cfg.update({'sample_range': None}) 
# cfg.update({'val_test_shuffle': True})
# cfg.update({'normalize': '01'}) 


#%% 4. [AUTOMATIC] Create pred_dir 
'''
pred_name : str
    Name of prediction folder
pred_dir : str
    directory of prediction folder in which all the prediction results are stored (in log_dir/exp_name)
'''

# *****************************************************************************
if cfg['sample_type']=='system':
    cfg['pred_name'] = datetime.now().strftime("%Y%m%d_%H%M%S")+'_pred_'+cfg['ckpt_type']+'_'+cfg['sample_type']+'_'+str(cfg['img_path_dist'])+'_'+str(cfg['img_path_skip'])+'_'+str(cfg['n_imgs_in'])
elif cfg['sample_type']=='semirand':
    cfg['pred_name'] = datetime.now().strftime("%Y%m%d_%H%M%S")+'_pred_'+cfg['ckpt_type']+'_'+cfg['sample_type']+'_'+str(cfg['sample_factor'])+'_'+str(cfg['sample_range'])+'_'+str(cfg['n_imgs_in'])
else:
    print('ERROR: Unknown sample_type')

cfg['pred_dir'] = os.path.join(os.getcwd(), cfg['log_dir'], cfg['exp_name'], cfg['pred_name'])

if not os.path.exists(cfg['pred_dir']):
    os.makedirs(cfg['pred_dir'])
# ***end***********************************************************************


#%% 5. [OPTIONAL] Set font for figures
matplotlib.rcParams.update(
    {
    'text.usetex': True,
    "font.family": "serif",
    "pgf.texsystem": "pdflatex",
    "pgf.rcfonts" : False,
    }
)