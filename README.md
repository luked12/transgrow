# TransGrow

Time dependent Image Generation of Plants from Incomplete Sequences with CNN-Transformer (GCPR 22)

## Requirments

A suitable conda environment can be installed from the provided package file environment.yml

    conda env create -f environment.yaml
    conda activate transgrow

## Configurations

This project is structured in such a way that all adjustable variables are stored in two configuration files as dictionary.
- Model training: ./configs/config_main_transgrow.py
- Model testing: ./configs/config_test_transgrow.py


## Data

#### General

In the following, we always use the term plant images to refer not only to datasets containing images with single plants, but also to field patches or satellite images that show a region of multiple plants per image.

The plant images must be stored as a multi-temporal time series in folders. This means that for each plant/region a folder exists that contains images of all available time points of this plant/region. 

The images should be of quadratic size and the file names must contain at least one date, or even date+time, if modeling unit is to be finer than daily. From this time information, the temporal positions are calculated. 
The number of images per time series is flexible, i.e. not every time point has to exist for every plant. 

All plants are divided into train, val, and test set.

    ./
    └── data
      ├── MixedCrop  
      | ├── train
      |   ├── plant_01
      |   | ├── plant_01_2022-09-10.png
      |   | ├── plant_01_2022-09-12.png
      |   | ├── plant_01_2022-09-16.png
      |   | └── ...
      |   ├── plant_02
      |   | ├── plant_02_2022-09-10.png
      |   | ├── plant_02_2022-09-15.png
      |   | └── ...
      | ├── val
      |   ├── plant_03
      |   | ├── plant_03_2022-09-12.png
      |   | ├── plant_03_2022-09-16.png
      |   | └── ...      
      | └── test
      |   ├── plant_04
      |   | ├── plant_04_2022-09-10.png
      |   | ├── plant_04_2022-09-12.png
      |   | └── ...
      ├── Arabidopsis
      └── ...

#### Use MixedCrop data set

- In ./configs/config_main_transgrow.py:

        cfg['data_name'] = 'mix'
        
- Download Data from PhenoRoam
- Patch Fields (provide code here)
- Sort Data (provide code here)


#### Use Arabidopsis data
- In ./configs/config_main_transgrow.py:

        cfg['data_name'] = 'abd'
        
- Download Data from xyz
- Sort Data (provide code here)


#### Use your own dataset


## Training
To train a TransGrow models with configurations from ./configs/config_main_transgrow.py use

    python ./main_transgrow.py


## Testing
To test a previously trained model with configurations from ./configs/config_test_transgrow.py use

    python ./test_transgrow.py
    



