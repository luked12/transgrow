import sys, os
import numpy as np
import math
import matplotlib
import torchvision
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint

from datetime import datetime
from utils import utils
from utils.custom_transforms import SinglePlantRotation


#%% Create config dict 'cfg' ==================================================
'''
Config for main_transgrow.py as dictionary 'cfg'.
Find list and description of all elements in sections below.

# *****************************************************************************
Config elements framed with # **** are are dependent on other elements
# *****************************************************************************
'''

cfg = {}


#%% GENERAL PARAMS ============================================================
'''
log_dir : str
    Name of folder where to store experiments
    Created in the project folder at the level of main_transgrow.py
resume_train_exp_name : str
    Specify the exp_name folder, if you want to resume training (need to be in folder log_dir)
    Specify None, if you don't want to resume training from a previous ckpt
    Attention: If you resume training ...
    ... you may need to increase the max_epochs, because it starts counting at the epoch where it stopped at the old training
    ... please ensure that img_size and those model params (see below) affecting the architecture (saved in hparams.yaml) are identical with config in this file, otherwise weight missmatch will occur
    DEFAULT: None
device : int
    which cuda device
'''

cfg['log_dir'] = 'lightning_logs'
cfg['resume_train_exp_name'] = None

# ***start*********************************************************************
cfg['device'] = torch.cuda.current_device()
# ***end***********************************************************************


#%% DATA PARAMS ===============================================================
'''
data_name : str
    Name of dataset:
    'abd' = Arabidopsis thaliana
    'grf' = Growliflower
    'mix' = MixedCrop
    'dnt' = DENETHOR
img_dir : str
    Path to img directory
data_time : dict contating datetime objects
    time_start : datetime.datetime object (observation start of dataset)
    time_end : datetime.datetime object (observation end of dataset)
time_unit : string
    modeling time unit
    Choose from: 's'econd, 'm'inute, 'h'ours, 'd'ay or 'w'eek 
    The smaller the time unit, the larger the positional encoding vector will be
time_diff : datetime.timedelta
    Difference between time_start and time_end
factor_to_unit : int
    amount of seconds of chosen time_unit
pe_max_len : int
    Maximal length of positional encoding vector
    pe_max_len can also set manually to a LARGER value than the dataset specific pe_max_len. Cons: larger matrix multiplications -> so longer runtime
nworkers : int
    Number of workers, reasonable value depends on CPU kernels
    local typically 8, on GPU server 32 
img_ext : list of str
    Choose allowed img extension formats (e.g. 'png', 'jpg', 'tif')
img_size : int
    Processing img size
batch_size : int
    Batch size, maximum possible number mainly depend on GPU memory
    local typically 8, on GPU server 32
n_imgs_in : int
    Input sequence length
    For training, smaller sequences of equal length must be sampled from the potentially very long original sequences from the folders in img_dir to enable training.
    n_imgs specifies the input sequence length in training, i.e. the number of input images from which n_imgs_out images are generated.
n_imgs_out : int
    Output sequence length
    Currently only supported: n_imgs_out = 1
    This means that exactly one image can be generated at a time from an input sequence.
'n_imgs' : int
    n_imgs_in + n_imgs_out    
sample_type : str
    Choose how to sample smaller sequences for training from the original sequences
    'system'atic and 'semirand'om
    Normally: 'semirand' sampling for training and 'system' for inference
rem_dup : bool
    remove temporal duplicates in each sampled sequence? 
    Temporal duplicates can occur, if modeling time has a smaller sampling rate than observation rate (in parts of observation period)
    e.g. observation: 4 imgs/h, modeling time_unit='h' --> 4 imgs can have the same time in the sequence, which happens randomly in case of 'semirand' and often, when img_path_dist is <= 4 in case of 'system'
img_path_dist : int
    only used if sample_type = 'system': 
    Distance between img paths within one sequence (default: 1 = no distance between img paths) 
    The higher the img_path dist, the greater the temporal distance between images within a sequence.
    Attention: for abd dataset if time_unit='h', rem_dup=True, and img_path_dist=1 it can happen that several sequences are removed
    Default/Min = 1: take diretly consecutive image paths, Max = depending on dataset
img_path_skip : int
    only used if sample_type = 'system':
    increase img_path_skip to reduce the overlap and thus the number samples in the dataset
    Default/Min = 1: no imgs skipped, Max = depending on dataset
sample_factor : float
    only used if sample_type = 'semirand':
    The larger this factor, the more often the same image is sampled into different smaller sequences of length (n_imgs)
    Min = 0, default = 1, common = between 0.5 and 2, Max = unlimited (but training will take a while)
sample_range : int
    only used if sample_type = 'semirand':
    The larger this factor, the more often the same image is sampled into different smaller sequences of length n_imgs
    Min = n_imgs, Max/Default = None: no maximal range
val_test_shuffle : bool
    shuffle val and test set in DataLoader?
normalize : string
    '01' normalize to [0 1]
    '11' normalize to [-1 1]
    'standardizing_abd' data set specific standardizing with precomputed mean and variance
    Attention: May change also cfg['final_actvn']
'''

cfg['data_name'] = 'abd'

# ***start*********************************************************************
if cfg['data_name'] == 'abd':
    cfg['img_dir'] = 'data/arabidopsis_resized_256/' 
    cfg['data_time'] = {'time_start': datetime.strptime('2015-12-14--12-54-06', '%Y-%m-%d--%H-%M-%S'), 
                        'time_end': datetime.strptime('2016-01-18--09-05-40', '%Y-%m-%d--%H-%M-%S')}
    cfg['time_unit'] = 'h'
elif cfg['data_name'] == 'grf':
    cfg['img_dir'] = 'data/GrowliFlower/plantsort_256/'
    cfg['data_time'] = {'time_start': datetime.strptime('2021-06-16', '%Y-%m-%d'), 
                        'time_end': datetime.strptime('2021-09-08', '%Y-%m-%d')}
    cfg['time_unit'] = 'd'
elif cfg['data_name'] == 'mix':
    # cfg['img_dir'] = 'data/Mix_RGB_CKA_2020/plantsort_patch_128/'
    cfg['img_dir'] = 'data/Mix_RGB_CKA_2020/plantsort_patch_484/'
    cfg['data_time'] = {'time_start': datetime.strptime('2020-04-02', '%Y-%m-%d'), 
                        'time_end': datetime.strptime('2020-07-23', '%Y-%m-%d')}
    cfg['time_unit'] = 'd'
elif cfg['data_name'] == 'dnt':
    cfg['img_dir'] = 'data/DENETHOR/'
    cfg['data_time'] = {'time_start': datetime.strptime('2018-01-01', '%Y-%m-%d'), 
                        'time_end': datetime.strptime('2018-12-31', '%Y-%m-%d')}
    cfg['time_unit'] = 'd'
cfg['time_diff']  = cfg['data_time']['time_end']-cfg['data_time']['time_start']
cfg['factor_to_unit'] = utils.get_seconds_factor_to_time_unit(cfg['time_unit'])
cfg['pe_max_len'] = math.ceil(cfg['time_diff'].total_seconds()/cfg['factor_to_unit'])
# ***end***********************************************************************

cfg['nworkers'] = 8 
cfg['img_ext'] = ['png', 'jpg', 'tif']
cfg['img_size'] = 256
cfg['batch_size'] = 32
cfg['n_imgs_in'] = 3
cfg['n_imgs_out'] = 1 

# ***start*********************************************************************
cfg['n_imgs'] = cfg['n_imgs_in'] + cfg['n_imgs_out']
# ***end***********************************************************************

cfg['sample_type'] = 'semirand' 
cfg['rem_dup'] = False 
cfg['img_path_dist'] = 5
cfg['img_path_skip'] = 4
cfg['sample_factor'] = 0.5 
cfg['sample_range'] = None 
cfg['val_test_shuffle'] = True
cfg['normalize'] = '01'


#%% MODEL PARAMS ==============================================================
'''
use_model : str
    Which GAN model should be used? 'gan', 'wgan', 'wgangp'
    'gan': classic GAN (optimization: KL divergence), whole sequence goes to discriminator
    'wgan': WGAN weight clipping, whole sequence goes to discriminator
    'wgangp': WGAN-GP, whole sequence goes to discriminator
    DEFAULT = 'wgangp'
target_pos : int
    Which pos of the sequence is the target_pos in training?
    Choose = 0 for random (DEFAULT), =n_imgs for last pos, or other 0<target_pos<n_imgs for any specific position in the sequence
g_enc_net : str
    Genereator Encoder Network to produce img embeddings
    'res18': ResNet-18 (DEFAULT)
    'res50': ResNet-50
g_enc_net_pretrained : bool
    Use pretrained weights?
g_dec_net : str
    Generator Decoder Network to produce generate img from latent dimension
    'lightweight'
    'giraffe'
    'giraffe_base_dec'
g_dec_net_pretrained : bool
    Use prertrained weights?
pe_fusion_type : str
    How to fuse the positional encoding to the latent dim?
    'add' = addition (DEFAULT), 'cat' = concatenation
dim_img : int
    latent target/bottleneck dimension of img embedding
    Please consider: global average pooling in resnet backends already reduces the images to a ceratin pooling dimension res18=512, res50=2048
    DEFAULT: 512
dim_pe : int
    dimension of positonal encoding 
    For pe_fusion_type = 'add': dim_pe = dim_img
    for pe_fusion_type = 'cat': dim_pe can be different (typically reduced compared to dim_img)
dim_z : int
    dimension of stochasticity induced to the network
    Attention: dim_img must be divisible by this number
    DEFAULT = 16
dim : int
    dimension of transformer encoder
    dim=dim_img if pe_fusion_type=add else dim=dim_img+dim_pe 
heads : int
    Transformer Encoder Parameter
    Number of Attention heads
    -> has not that much influence, typically in range [4 16] 
    DEFAULT = 4
depth : int
    Transformer Encoder Parameter
    Number of consecutive transformer layers
    -> mainly influence the size of the transformer, typical in range [2 8] 
    DEFAULT = 3
dropout_transformer : float
    Transformer Encoder Parameter
    Dropout applied within Transformer Encoder
    range [0 1]
    DEFAULT = 0.1
dropout_emb : float
    Dropout applied before Transformer Encoder on Transformer input
    range [0 1]
    DEFAULT = 0.2
prd_token_type : str
    'pick': use only the tranformed prd_token to decode the img from
    'mean': use the mean transformer output to decode the img from
d_net : str
    Which Discriminator Network 'res18', 'res50', 'patchGAN', 'dcgan'
    Attention: Use 'dcgan' for WGAN since it has no Batchnormalization
    DEFAULT: 'dcgan'
d_net_pretrained : bool
    Use prertrained weights?
    Note: Only supported for d_net='res18' and d_net='res50'
'''

cfg['use_model'] = 'wgangp'
cfg['target_pos'] = 0 
cfg['g_enc_net'] = 'res18'
cfg['g_enc_net_pretrained'] = True
cfg['g_dec_net'] = 'lightweight'
cfg['g_dec_net_pretrained'] = False
cfg['pe_fusion_type'] = 'add' 
cfg['dim_img'] = 512
cfg['dim_pe'] = 512
cfg['dim_z'] = 16 

# ***start*********************************************************************
if cfg['pe_fusion_type'] == 'add':
    if cfg['dim_img']==cfg['dim_pe']:
        cfg['dim'] = cfg['dim_img']
    else:
        print('ERROR: dim_img must equal dim_pe for pe_fusion_type ->add<-')
        sys.exit()
elif cfg['pe_fusion_type'] == 'cat':
    cfg['dim'] = cfg['dim_img']+cfg['dim_pe']
else:
    print('Wrong pe_fusion_type:', cfg['pe_fusion_type'])
# ***end***********************************************************************

cfg['heads'] = 4 
cfg['depth'] = 3 
cfg['dropout_transformer'] = 0.1
cfg['dropout_emb'] = 0.2 
cfg['prd_token_type'] = 'pick'
cfg['d_net'] = 'dcgan' 
cfg['d_net_pretrained'] = False


#%% OPTIMIZATION AND TRAINING PARAMS ==========================================
'''
lr : float
    learning rate
    DEFAULT: 1e-4
losses_w : dict
    Losses (str) and corresponding weights (int) to be used for training
    'adv' = adversarial loss
    'l1' = l1 distance (reconstruction) loss
    'mse' = mean squared error (reconstruction) loss
    'dice' = dice loss
    'style' = style loss
    'msssim' = multi-scale structural similarity loss
    'percep' = perceptual latent space loss (using VGG net) 
    Attention: May check if activations are compatible beforehand
    Attention: 'percep' requires values in range [0 1] sigmoid activation
final_actvn : 'str'
    Coose from 'sigmoid', 'tanh', 'relu'
    Attention: normalization and losses may needs to be changes accordingly
    DEFAULT = 'sigmoid'
max_epochs : int
    Number of training epochs
    Attention: if you resume training this needs to be larger than before, otherwise it will do nothing
gpus : int
    Number of GPUs
    DEFAULT = 1
precision : int
    Precision mode, Choose 16 or 32
    DEFAULT = 32
fast_dev_run : bool
    only a trial trainings run?
    Default = False
limit_train_batches : float
    Limit train batches during training [0.0 1.0]
    DEFAULT: 1.0
    Attention: Set to 1.0 and NOT 1, if you want to include all train data, because otherwise only one training sample is taken
limit_val_batches : float
    Limit val batches during training [0.0 1.0]
limit_test_batches : float
    Limit test batches during training [0.0 1.0]
early_stop : bool
    Stop training, if val_loss does not decrease anymore
    DEFAULT: False, since sometimes tricky in GAN training
save_ckpts_last : bool
    Save last epoch?
    Default: True
save_ckpts_best : bool
    Save best epoch?
    This is measured by means of val_loss
    Default: True
exp_name : 'str'
    Name of experiment folder in log_dir 
    Automatically build from start time of the experiment and some essential parameters
exp_dir : 'str'
    Experiment directory
ckpt_path_resume : str
    Ckpt from which the training resumes if resume_train_exp_name is not None
    Otherwise None
callbacks : list of pytorch_lightning.callbacks
    EarlyStopping, LearningRateMonitor, ModelCheckpoint
    Are created partly based on previous configs and partly based on other defaults (e.g. patience at EarlyStopping) that could be changed.
tb_logger : pl.loggers
    Specify the logger, with which you want to log losses during training
    Default: tensorboard logger        
'''

cfg['lr'] = 1e-4
cfg['losses_w'] = {'weight_adv': 1,
                   'weight_l1': 1, 
                   'weight_msssim': 1, 
                   'weight_percep': 1,} 
cfg['final_actvn'] = 'sigmoid'
cfg['max_epochs'] = 100 
cfg['gpus'] = 1
cfg['precision'] = 32 
cfg['fast_dev_run'] = False
cfg['limit_train_batches'] = 1.00 
cfg['limit_val_batches'] = 1.00
cfg['limit_test_batches'] = 1.00
cfg['early_stop'] = False
cfg['save_ckpts_last'] = True
cfg['save_ckpts_best'] = True 

# ***start*********************************************************************
# # Exp Name
if cfg['resume_train_exp_name']:
    cfg['exp_name'] = cfg['resume_train_exp_name']
else:
    cfg['exp_name'] = datetime.now().strftime("%Y%m%d_%H%M%S")+'_'+cfg['data_name']+'_'+cfg['use_model']+'_tf_'+str(cfg['target_pos'])+'_dim_'+str(cfg['dim'])+'_pe_'+cfg['pe_fusion_type']+'_img_'+str(cfg['img_size'])+'_'+'z_'+str(cfg['dim_z'])
# # Exp directory
cfg['exp_dir'] = os.path.join(os.getcwd(), cfg['log_dir'], cfg['exp_name'])
if not os.path.exists(cfg['exp_dir']):
        os.makedirs(cfg['exp_dir'])
# # Ckpt_path_resume
if cfg['resume_train_exp_name']:
    cfg['ckpt_path_resume'] = cfg['exp_dir']+'/checkpoints/last.ckpt'
else:
    cfg['ckpt_path_resume'] = None

# # Callbacks
# # LearningRateMonitor
callbacks = []
callbacks = [
    LearningRateMonitor(log_momentum=True),
]
# # EarlyStopping
early_stop_callback = EarlyStopping(
    monitor="val_loss",
    min_delta=0.005,
    mode="min",
    patience=5,
    verbose=True,
)
if cfg['early_stop']:
    callbacks = [early_stop_callback] + callbacks   
# # ModelChecker
model_checker = ModelCheckpoint(
    monitor="val_loss",
    mode="min",
    save_last=cfg['save_ckpts_last'], 
    save_top_k=cfg['save_ckpts_best'], 
    filename="best_{epoch}",
)
if cfg['save_ckpts_best']:
    callbacks += [model_checker]
cfg['callbacks'] = callbacks

# # Logger
cfg['tb_logger'] = pl.loggers.TensorBoardLogger(save_dir=os.getcwd(),
                                          version=cfg['exp_name'],
                                          name=cfg['log_dir'])
# ***end***********************************************************************

#%% TESTING AND PLOTTING PARAMS ===============================================
'''
run_test : bool
    Compute losses for test data?
    Default: False
run_plots : bool
    run some plots after training for direct visualization?
    Default: True
figure_width / figure_height : float
    specify figure width and height for saving of plots in optimal shape
plot_dpi : int
    specify dpi for optimal plotting resolution of figures 
'''

cfg['run_test'] = False
cfg['run_plots'] = True

matplotlib.rcParams.update(
    {
    'text.usetex': True,
    "font.family": "serif",
    "pgf.texsystem": "pdflatex",
    "pgf.rcfonts" : False,
    }
)
# factor points to inch
pt_to_in = 1/72.27

# this needs to be set to the desired plot width in points
# for example can be determined using '\showthe\textwidth' or 
# '\the\columnwidth' in latex
# typical columnwidth for two-column paper: 252
# typical textwidth for two-column paper: 516
figure_width_in = 252

figure_width = figure_width_in * pt_to_in
cfg['figure_width'] = np.round(figure_width, 2)
cfg['figure_height'] = 2.5

cfg['plot_dpi'] = 200 # 100 = original; if > 100 view size >> true image size
   
    
#%% TRANSFORMATION/AUGMENTATION PARAMS ========================================
'''
norm_mean : np.array
    normalization mean (used in torchvision.transforms.Normalize)
norm_std : np.array
    normalization standard deviation (used in torchvision.transforms.Normalize)
bg_path : string
    path to folder with background imgs (needed for transformation SinglePlantRotation)
spec_aug_fg : bool
    do foreground augmentations? (needed for transformation SinglePlantRotation)
    Default = False
spec_aug_bg : bool
     do background augmentations? (needed for transformation SinglePlantRotation)        
     Default = False
transform_train : torchvision.transforms.transforms.Compose
    Object of training transformations
    Note: Transformation depend on dataset
transform_test : torchvision.transforms.transforms.Compose
    Object of val/test transformations
deNorm : utils.utils.DeNormalize
    Function to deNormalize the Normalized img in the end (used primarly in plotting)
toPIL : torchvision.transforms.ToPILImage
    Function to generate PIL img from tensor (used primarly in plotting)
'''

# ***start*********************************************************************
if cfg['normalize']=='01':
    # # transform [0 1] to [0 1] -> mean=0, std=1 -> do nothing
    cfg['norm_mean'] = np.zeros(3)
    cfg['norm_std'] = np.ones(3)    
elif cfg['normalize']=='11':
    # # shift [0 1] to [-1 1] (you could also do *2-1)
    cfg['norm_mean'] = np.asarray([0.5, 0.5, 0.5])
    cfg['norm_std'] = np.asarray([0.5, 0.5, 0.5])
elif cfg['normalize']=='standardizing_abd':
    # normalize from [0 1] to mean 0 +- std -> mean = true mean of dataset, std= true std of dataset
    # norm stats for Arabidopsis images
    cfg['norm_mean'] = np.asarray([0.32494438, 0.31354403, 0.26689884]) # arabidopsis stats
    cfg['norm_std'] = np.asarray([0.16464259, 0.16859856, 0.15636043]) # arabidopsis stats
else:
    print('ERROR: Wrong Normalization/Scaling Method specified.')
# ***end***********************************************************************

# # Trainings transformation
if cfg['data_name'] == 'abd':
    # params for SinglePlantRotations Augmentation
    cfg['bg_path'] = 'data/arabidopsis_resized_256/bg_cleaner/'
    cfg['spec_aug_fg'] = False
    cfg['spec_aug_bg'] = False
    
    transforms = [
        torchvision.transforms.ToTensor(), # ToTensor transforms to [0 1], if input is uint8 PIL image, otherwise there is no transformation
        torchvision.transforms.Normalize(cfg['norm_mean'], cfg['norm_std']),
        torchvision.transforms.Resize(size=(cfg['img_size'],cfg['img_size']), interpolation=torchvision.transforms.InterpolationMode.BILINEAR, antialias=True),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.RandomVerticalFlip(),
        # torchvision.transforms.RandomRotation(180),
        SinglePlantRotation(cfg['bg_path'], cfg['spec_aug_fg'], cfg['spec_aug_bg'], cfg['norm_mean'], cfg['norm_std'], cfg['img_size']),
        # torchvision.transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    ]
elif cfg['data_name'] == 'grf':
    transforms = [
        torchvision.transforms.ToTensor(), # ToTensor transforms to [0 1], if input is uint8 PIL image, otherwise there is no transformation
        torchvision.transforms.Normalize(cfg['norm_mean'], cfg['norm_std']),
        torchvision.transforms.Resize(size=(cfg['img_size'],cfg['img_size']), interpolation=torchvision.transforms.InterpolationMode.BILINEAR, antialias=True),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.RandomVerticalFlip(),
    ]
elif cfg['data_name'] == 'mix':
    transforms = [
        torchvision.transforms.ToTensor(), # ToTensor transforms to [0 1], if input is uint8 PIL image, otherwise there is no transformation
        torchvision.transforms.Normalize(cfg['norm_mean'], cfg['norm_std']),
        torchvision.transforms.Resize(size=(cfg['img_size'],cfg['img_size']), interpolation=torchvision.transforms.InterpolationMode.BILINEAR, antialias=True),
        # torchvision.transforms.RandomCrop(size=(cfg['img_size'],cfg['img_size'])),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.RandomVerticalFlip(),
        # torchvision.transforms.RandomRotation(180),
    ]
elif cfg['data_name'] == 'dnt':
    transforms = [
        torchvision.transforms.ToTensor(), # ToTensor transforms to [0 1], if input is uint8 PIL image, otherwise there is no transformation
        torchvision.transforms.Normalize(cfg['norm_mean'], cfg['norm_std']),
        torchvision.transforms.Resize(size=(cfg['img_size'],cfg['img_size']), interpolation=torchvision.transforms.InterpolationMode.BILINEAR, antialias=True),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.RandomVerticalFlip(),
    ]
    
cfg['transform_train'] = torchvision.transforms.Compose(transforms)

# # VAL / Test Transformations
if cfg['data_name'] == 'abd':
    transforms = [
        torchvision.transforms.ToTensor(), # ToTensor transforms to [0 1], if input is uint8 PIL image, otherwise there is no transformation
        torchvision.transforms.Normalize(cfg['norm_mean'], cfg['norm_std']),
        torchvision.transforms.Resize(size=(cfg['img_size'],cfg['img_size']), interpolation=torchvision.transforms.InterpolationMode.BILINEAR, antialias=True),
    ]
elif cfg['data_name'] == 'grf':
    transforms = [
        torchvision.transforms.ToTensor(), # ToTensor transforms to [0 1], if input is uint8 PIL image, otherwise there is no transformation
        torchvision.transforms.Normalize(cfg['norm_mean'], cfg['norm_std']),
        torchvision.transforms.Resize(size=(cfg['img_size'],cfg['img_size']), interpolation=torchvision.transforms.InterpolationMode.BILINEAR, antialias=True),
    ]
elif cfg['data_name'] == 'mix':
    transforms = [
        torchvision.transforms.ToTensor(), # ToTensor transforms to [0 1], if input is uint8 PIL image, otherwise there is no transformation
        torchvision.transforms.Normalize(cfg['norm_mean'], cfg['norm_std']),
        torchvision.transforms.Resize(size=(cfg['img_size'],cfg['img_size']), interpolation=torchvision.transforms.InterpolationMode.BILINEAR, antialias=True),
        # torchvision.transforms.RandomCrop(size=(cfg['img_size'],cfg['img_size'])),
    ]
elif cfg['data_name'] == 'dnt':
    transforms = [
        torchvision.transforms.ToTensor(), # ToTensor transforms to [0 1], if input is uint8 PIL image, otherwise there is no transformation
        torchvision.transforms.Normalize(cfg['norm_mean'], cfg['norm_std']),
        torchvision.transforms.Resize(size=(cfg['img_size'],cfg['img_size']), interpolation=torchvision.transforms.InterpolationMode.BILINEAR, antialias=True),
    ]

cfg['transform_test'] = torchvision.transforms.Compose(transforms)

# # denorrmalization
cfg['deNorm'] = utils.DeNormalize(cfg['norm_mean'], cfg['norm_std'])
cfg['toPIL'] = torchvision.transforms.ToPILImage()