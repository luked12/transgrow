'''
=============================================================================
PyTorch Dataset Class and helper functions to load sequence datasets (imgs+time)
=============================================================================
'''

import os
import math
import torch
import datetime
import numpy as np
import random
import re
from PIL import Image
from utils import utils


def has_file_allowed_extension(filename, ext):
    '''
    Checks if a file extension is an allowed extension.
    
    Args:
        filename : str
            Path to file.
        ext : list[str]
            List of allowed extensions.
        
    Returns:
        bool : True if the filename ends with an allowed extension
        
    ---------------------------------------------------------------------------
    code reference: https://github.com/pytorch/vision/blob/master/torchvision/datasets/folder.py
    '''
    
    filename_lower = filename.lower()
    return any(filename_lower.endswith(i) for i in ext)


def pil_loader(path):
    '''
    ---------------------------------------------------------------------------
    code reference: https://github.com/pytorch/vision/blob/master/torchvision/datasets/folder.py
    '''
    
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            if len(img.getbands())==1:
                return img.convert('L')
            else:
                return img.convert('RGB')


def accimage_loader(path):
    '''
    ---------------------------------------------------------------------------
    code reference: https://github.com/pytorch/vision/blob/master/torchvision/datasets/folder.py
    '''
    
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)

    
def default_image_loader():
    '''
    ---------------------------------------------------------------------------
    code reference: https://github.com/pytorch/vision/blob/master/torchvision/datasets/folder.py
    '''
    
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader
    else:
        return pil_loader


def find_seqs(seq_root):
    '''
    This function finds subdirectories containing the full length seqs in the root seq folder. 
    
    Args: 
        seq_root : str
            Path to root directory of seq folders.
            
    Returns: 
        seq_names : list
            List of seq names.
        seq_idx : dict
            Dict with items (seq_names, seq_idx)
        
    ---------------------------------------------------------------------------
    code adapted from: https://github.com/pytorch/vision/blob/master/torchvision/datasets/folder.py        
    '''
    
    seq_names = [d for d in os.listdir(seq_root) if os.path.isdir(os.path.join(seq_root, d))]
    seq_names.sort()
    seq_idx = {seq_names[i]: i for i in range(len(seq_names))}
    return seq_names, seq_idx


def seq_loader(img_paths):
    '''
    Function to load a sequence of imgs given a list of their paths.
    
    Args: 
        img_paths :  list[str]
            List of img paths of the sequence.
    
    Returns: 
        seq : list[Image]
            List of imgs.
    '''
 
    image_loader = default_image_loader()
    seq = []
    for img_path in img_paths:
        if os.path.exists(img_path):
            seq.append(image_loader(img_path))
        else:
            return seq
    return seq


def get_time_from_path(img_path, data_name):
    '''
    Function to get a time object from an img_path of a dataset
    You need the data_name, since in each dataset the time is indicated differently
    
    Args: 
        img_path : str
            Path of img in dataset
        data_name :  str
            name of dataset
    
    Returns: 
        time_obj :  datetime.datetime
            time object of image
    '''
    
    if data_name == 'abd':
        time_string = img_path[-32:-12]
        time_obj = datetime.datetime.strptime(time_string, '%Y-%m-%d--%H-%M-%S')
    elif data_name == 'grf':
        split_pos = [m.start() for m in re.finditer('/', img_path)]
        time_string = img_path[split_pos[-1]+1:split_pos[-1]+11]
        time_obj = datetime.datetime.strptime(time_string, '%Y_%m_%d')
    elif data_name == 'mix':
        split_pos = [m.start() for m in re.finditer('/', img_path)]
        time_string = img_path[split_pos[-1]+9:split_pos[-1]+19]
        time_obj = datetime.datetime.strptime(time_string, '%Y_%m_%d')
    elif data_name == 'dnt':
        time_string = img_path[-19:-9]
        time_obj = datetime.datetime.strptime(time_string, '%Y_%m_%d')
    else:
        print('Error: Wrong dataset_name:',data_name )
    return time_obj


def make_dataset_system(seq_root, seq_ext, n_imgs, data_name, time_unit, rem_dup=False, img_path_dist=1, img_path_skip=1):
    '''
    Function to sample smaller sequences of n_imgs imgs out of the total lenght sequences in seq_root in a systematic way.
    Can be used for inference very well (with rem_dup=False and img_path_dist=1 and img_path_skip=1)
    Img_Paths and corresponding times are saved.
    
    Args: 
        seq_root : str
            Path to root directory of seq folders.
        seq_ext : list[str]
            List of allowed extensions. 
        n_imgs :  int
            Number of imgs per seq sequence.
        data_name : str
            Name of dataset
        time_unit : str
            's', 'm', 'h', 'd', or 'w'
        rem_dup : bool
            remove duplicates?
        img_path_dist : int
            distance of consecutive images in img_path folder; the larger, the greater the time difference between the images within the sequence (default=1=directly consecutive paths used)
        img_path_skip : int
            increase to systematically skip img_paths and thus decrease the size of the dataset (default=1)
    
    Returns: 
        dataset_img_paths : (list (img_paths [str], seq_idx [int]))
            List of list of tuples (img_paths and seq_idx)
        dataset_times : list[datetime]
            List of times corresponding to img_paths
    '''
    
    # get seq folders and assign index to each seq
    seq_names, seq_idx = find_seqs(seq_root)
    
    dataset_img_paths = []
    dataset_times = []
    
    for i in range(len(seq_names)):   
        seq_dir = os.path.join(seq_root, seq_names[i])
        
        # get directories and names of files in seq folder
        for file_root, _, file_names in sorted(os.walk(seq_dir)):
            
            # number of imgs in seq
            nsample_files = len(file_names)
            
            seq_img_paths=[]
            seq_times=[]
            for file_name in sorted(file_names):
                # check whether file extension of file in seq folder is an allowed extension
                if has_file_allowed_extension(file_name, seq_ext):
                    file_path = os.path.join(file_root, file_name)            
                    # make a list of all img_paths in seq folder
                    img_path = (file_path, seq_idx[seq_names[i]])
                    time = get_time_from_path(file_path, data_name)
                    seq_img_paths.append(img_path)
                    seq_times.append(time)
            
            # seconds factor to time_unit
            factor_to_unit = utils.get_seconds_factor_to_time_unit(time_unit)
                    
            # pick systematic imgs along the time axis
            for j in range(0, (nsample_files-((n_imgs-1)*img_path_dist)), img_path_skip):
                sample_img_paths = []
                sample_times = []
                for k in range(0, n_imgs):
                    sample_img_paths.append(seq_img_paths[j+k*img_path_dist])
                    sample_times.append(seq_times[j+k*img_path_dist])
                # Ensure, that no times are in the sequence that fall into the same time_unit
                # Note: for abd with time_unit='h' ~10 % of samples are not appended
                if rem_dup:
                    timediffs = list(np.diff(sample_times))
                    check_timediffs= [1 for x in timediffs if x.total_seconds()>factor_to_unit]
                    if len(timediffs) == len(check_timediffs):
                        dataset_img_paths.append(sample_img_paths)
                        dataset_times.append(sample_times)
                else:
                    dataset_img_paths.append(sample_img_paths)
                    dataset_times.append(sample_times)
                
    return dataset_img_paths, dataset_times


def make_dataset_semirand(seq_root, seq_ext, n_imgs, data_name, time_unit, rem_dup=False, sample_factor=1, sample_range=None):
    '''
    Function to sample smaller sequences of n_imgs imgs out of the total lenght sequences in seq_root in a semirandom way.
    Img_Paths and corresponding times are saved.
    
    Args: 
        seq_root : str
            Path to root directory of seq folders.
        seq_ext : list[str]
            List of allowed extensions. 
        n_imgs :  int
            Number of imgs per seq sequence.
        data_name : str
            Name of dataset
        time_unit : str
            's', 'm', 'h', 'd', or 'w'
        rem_dup : bool
            remove duplicates?
        sample_factor : int
            the higher this factor the sequences are samples (in random different constellations) (default=1) 
        sample_range : int
            Max. img_paths range of one sequence, if None = not limited (default=None)
    
    Returns: 
        dataset_img_paths : (list (img_paths [str], seq_idx [int]))
            List of list of tuples (img_paths and seq_idx)
        dataset_times : list[datetime]
            List of times corresponding to img_paths
    '''
        
    # get seq folders and assign index to each seq
    seq_names, seq_idx = find_seqs(seq_root)
    
    dataset_img_paths = []
    dataset_times = []
    
    for i in range(len(seq_names)):   
        seq_dir = os.path.join(seq_root, seq_names[i])
        
        # get directories and names of files in seq folder
        for file_root, _, file_names in sorted(os.walk(seq_dir)):
            
            # number of imgs in seq
            nsample_files = len(file_names)
            
            seq_img_paths=[]
            seq_times=[]
            for file_name in sorted(file_names):
                # check whether file extension of file in seq folder is an allowed extension
                if has_file_allowed_extension(file_name, seq_ext):
                    file_path = os.path.join(file_root, file_name)            
                    # make a list of all img_paths in seq folder
                    img_path = (file_path, seq_idx[seq_names[i]])
                    time = get_time_from_path(file_path, data_name)
                    seq_img_paths.append(img_path)
                    seq_times.append(time)
            
            if sample_range is None or sample_range>=nsample_files:
                sample_range=nsample_files
            
            # seconds factor to time_unit
            factor_to_unit = utils.get_seconds_factor_to_time_unit(time_unit)
            
            # pick semirandom imgs along the time axis
            for j in range(0, int(np.round(nsample_files*sample_factor))):
                sample_img_paths = []
                sample_times = []
                rand_anchor = random.sample(range(nsample_files-(sample_range-1)),1)[0]
                rand_ind = random.sample(range(rand_anchor,rand_anchor+sample_range),n_imgs)
                rand_ind.sort()
                for k in range(0, n_imgs):
                    sample_img_paths.append(seq_img_paths[rand_ind[k]])
                    sample_times.append(seq_times[rand_ind[k]])
                # Ensure, that no times are in the sequence that fall into the same time_unit
                # Note: for abd with time_unit='h' ~10 % of samples are not appended
                if rem_dup:
                    timediffs = list(np.diff(sample_times))
                    check_timediffs= [1 for x in timediffs if x.total_seconds()>factor_to_unit]
                    if len(timediffs) == len(check_timediffs):
                        dataset_img_paths.append(sample_img_paths)
                        dataset_times.append(sample_times)
                else:
                    dataset_img_paths.append(sample_img_paths)
                    dataset_times.append(sample_times)
                    
    return dataset_img_paths, dataset_times


class SeqDataset(torch.utils.data.Dataset):
    '''
    Class to create a dataset of sequences (length n_imgs)   
    
    Data is assumed to be arranged in this way:
        seq_root/seq1/img1.ext
        seq_root/seq1/img2.ext
        ...
        seq_root/seq1/imgN.ext
        
        seq_root/seq2/img1.ext
        seq_root/seq2/img2.ext
        ...
        seq_root/seq2/imgM.ext
                        
    Args:   
        seq_root : str
            Path to root directory of seq folders.
        seq_ext : list[str]
            List of allowed extensions. 
        n_imgs :  int
            Number of imgs per seq sequence.
        data_name : str
            Name of dataset
        data_time :  {'time_start': [datetime], 'time_end': [datetime]} [dict]
            dict with start and endtime of the dataset
        time_unit : str
            's', 'm', 'h', 'd', or 'w'
        sample_type :  str
            'system' or 'semirand' to choose between the sampling method the dataset is created, semirand is recommended for training/testing, 'system' for inference
        rem_dup : bool
            remove duplicates?
        img_path_dist : int
            distance of consecutive images in img_path folder; the larger, the greater the time difference between the images within the sequence (default=1=directly consecutive paths used)
        img_path_skip : int
            increase to systematically skip img_paths and thus decrease the size of the dataset (default=1)
        sample_factor : int
            the higher this factor the sequences are samples (in random different constellations) (default=1) 
        sample_range : int
            Max. img_paths range of one sequence, if None = not limited (default=None)
        transform : [torchvision.transforms.Compose] 
            transform to apply to every img of the sequence 
    '''
    
    def __init__(self, seq_root, seq_ext, n_imgs, data_name, data_time, time_unit, sample_type, rem_dup=False, img_path_dist=1, img_path_skip=1, sample_factor=1, sample_range=None, transform=None):

        if sample_type == 'system':
            self.seq_dataset, self.seq_dataset_time = make_dataset_system(seq_root, seq_ext, n_imgs, data_name, time_unit, rem_dup, img_path_dist, img_path_skip)
        elif sample_type == 'semirand':
            self.seq_dataset, self.seq_dataset_time = make_dataset_semirand(seq_root, seq_ext, n_imgs, data_name, time_unit, rem_dup, sample_factor, sample_range)
        else:
            print('Wrong sample_type')
        
        # params
        self.root = seq_root
        self.ext = seq_ext
        self.n_imgs = n_imgs
        self.data_time = data_time
        self.time_unit = time_unit
        self.sample_type = sample_type
        self.rem_dup = rem_dup
        self.img_path_dist = img_path_dist
        self.img_path_skip = img_path_skip
        self.sample_factor = sample_factor
        self.transform = transform
        
        self.factor_to_unit = utils.get_seconds_factor_to_time_unit(time_unit)
        
        
    def __getitem__(self, idx):
        '''
        Args:   
            idx : int
                Index of dataset sample.
        
        Returns: 
            sample : dict
                Dict of img seq and corresponding times
                {'seq_img': [torch.float32], 'seq_timedelta': array[int]} [dict]
                seq_img: Dimension: [n_imgs,C,H,W], transformed sequence of images as torch tensors
                seq_timedelta: Dimension: [n_imgs,], timedelta of each img to data start point in unit time_unit)
        '''
        
        sample_img_paths = []
        sample_seq_idx = []        
        # get img_paths and sequence idx of dataset --> sample of paths and idx
        for img_path, seq_idx in self.seq_dataset[idx][:]:
            sample_img_paths.append(img_path)
            sample_seq_idx.append(seq_idx)            
        
        # load img path sample into a list of imgs (sequence) sample
        seq_sample = seq_loader(sample_img_paths)
        
        # set seed for identical transforms within the img sequence
        seed = np.random.randint(245985462) 
        # transform image
        seq_sample_transform = []
        if self.transform is not None:
            for img in seq_sample:
                random.seed(seed) # random number seed
                torch.manual_seed(seed) # torch number seed
                seq_sample_transform.append(self.transform(img))
            seq_sample = seq_sample_transform
            
        # make torch.FloatTensor from list of imgs in sequence
        seq_sample = torch.stack(seq_sample, 0)#.permute(1,0,2,3)
        
        # compute timedelta to dataset start point and convert to integeger of time_unit 
        time_sample = []
        for img_time in self.seq_dataset_time[idx]:
            timedelta_ToStart = img_time-self.data_time['time_start'] # timedelta object
            time_sample.append(round(timedelta_ToStart.total_seconds()/self.factor_to_unit)) # int
            
        sample = {'seq_img': seq_sample, 
                  'seq_timedelta': np.array(time_sample)}

        return sample

    
    def __len__(self):
        return len(self.seq_dataset)