"""
===============================================================================
PyTorch Lightning DataModule for Image Sequence Data
===============================================================================
"""

import torch
import pytorch_lightning as pl
from datasets.seq_dataset import SeqDataset

# SeqDataModule
class SeqDataModule(pl.LightningDataModule):
    def __init__(self, img_size, batch_size, n_workers, img_dir, img_ext, n_imgs, data_name, data_time, time_unit, sample_type, rem_dup, img_path_dist, img_path_skip, sample_factor, sample_range, transform_train=None, transform_test=None, val_test_shuffle=False):
        super().__init__()
        self.img_size=img_size
        self.batch_size=batch_size
        self.n_workers=n_workers
        self.img_dir=img_dir
        self.img_ext=img_ext
        self.n_imgs=n_imgs
        self.data_name=data_name
        self.data_time=data_time
        self.time_unit=time_unit
        self.sample_type=sample_type
        self.rem_dup=rem_dup
        self.img_path_dist=img_path_dist
        self.img_path_skip=img_path_skip
        self.sample_factor=sample_factor
        self.sample_range=sample_range
        self.transform_train=transform_train
        self.transform_test=transform_test
        self.val_test_shuffle=val_test_shuffle
        
        self.params = {'img_size': self.img_size,
                       'batch_size': self.batch_size,
                       'n_workers': self.n_workers,
                       'img_dir': self.img_dir,
                       'img_ext': self.img_ext,
                       'n_imgs': self.n_imgs,
                       'data_name': self.data_name,
                       'data_time': self.data_time,
                       'time_unit': self.time_unit,
                       'sample_type': self.sample_type,
                       'rem_dup': self.rem_dup,
                       'img_path_dist': self.img_path_dist,
                       'img_path_skip': self.img_path_skip,
                       'sample_factor': self.sample_factor,
                       'sample_range': self.sample_range,
                       'transform_train': self.transform_train,
                       'transform_test': self.transform_test,
                       'val_test_shuffle': self.val_test_shuffle
                    }

    def prepare_data(self):
        pass
    
    def setup(self, stage=None):
        # 'or stage is None' to run both (fit and test) if its not specified
        if stage == "fit" or stage is None:
            self.train_data = SeqDataset(self.img_dir+'train', self.img_ext, self.n_imgs, self.data_name, self.data_time, self.time_unit, self.sample_type, self.rem_dup, img_path_dist=self.img_path_dist, img_path_skip=self.img_path_skip, sample_factor=self.sample_factor, sample_range=self.sample_range, transform=self.transform_train)
            self.val_data = SeqDataset(self.img_dir+'val', self.img_ext, self.n_imgs, self.data_name, self.data_time, self.time_unit, self.sample_type, self.rem_dup, img_path_dist=self.img_path_dist, img_path_skip=self.img_path_skip, sample_factor=self.sample_factor, sample_range=self.sample_range, transform=self.transform_test)
            self.data_dims = self.train_data[0]['seq_img'].shape
        if stage == "test" or stage is None:
            self.test_data = SeqDataset(self.img_dir+'test', self.img_ext, self.n_imgs, self.data_name, self.data_time, self.time_unit, self.sample_type, self.rem_dup, img_path_dist=self.img_path_dist, img_path_skip=self.img_path_skip, sample_factor=self.sample_factor, sample_range=self.sample_range, transform=self.transform_test)
            self.data_dims = self.test_data[0]['seq_img'].shape

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_data, batch_size=self.batch_size, num_workers=self.n_workers, shuffle=True)
    
    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_data, batch_size=self.batch_size, num_workers=self.n_workers, shuffle=self.val_test_shuffle)
    
    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.test_data, batch_size=self.batch_size, num_workers=self.n_workers, shuffle=self.val_test_shuffle)