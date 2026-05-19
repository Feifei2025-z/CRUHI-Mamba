# CRUHI-Mamba
Classification Model for Underwater Hyperspectral Images of Coral Reef Benthic
The Code for "CRUHI-Mamba: A Mamba-based underwater hyperspectral image classification framework for coral reef benthic mapping" Link:

# Installation
conda create -n CRUHIMamba python=3.9
conda activate CRUHIMamba
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia
pip install packaging==24.0
pip install triton==2.2.0
pip install mamba-ssm==1.2.0
pip install spectral
pip install scikit-learn==1.4.1.post1
pip install calflops
pip install xarray

# Data
The underwater hyperspectral benthic classification dataset for coral reefs was preprocessed prior to use, including hyperspectral band clipping and class reindexing of the ground truth labels. 
The preprocessed dataset supporting the findings of this study is available at https://doi.org/10.6084/m9.figshare.31130086. 
The original, unprocessed coral reef hyperspectral dataset is publicly available and can be accessed at https://doi.org/10.1594/PANGAEA.911300.

# Citation
Please kindly cite this paper if the code is useful and helpful for your research.
