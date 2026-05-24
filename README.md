# CRUHI-Mamba
The Code for "CRUHI-Mamba: A Mamba-based underwater hyperspectral image classification framework for coral reef benthic mapping" Link: https://doi.org/10.1080/15481603.2026.2679361

# Installation
## System Requirements

- **OS:** Linux (Ubuntu 20.04+ recommended)  
  - **Windows users:** This project requires [WSL2](https://learn.microsoft.com/en-us/windows/wsl/install) with NVIDIA GPU support. WSL1 is **not supported**.
- **GPU:** NVIDIA GPU with Compute Capability >= 7.0 (e.g., RTX 4090 D, RTX 3090, A100)
- **CUDA:** 11.7
- **Python:** 3.9

> **Note:** This project is developed and tested on **WSL2 + Ubuntu 20.04.1 LTS**.

```bash
# Create and activate environment
conda create -n CRUHIMamba python=3.9
conda activate CRUHIMamba

# Install PyTorch (CUDA 11.7)
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia

# Install other dependencies
pip install packaging==24.0
pip install triton==2.2.0
pip install ninja
pip install mamba-ssm==1.2.0
pip install einops
pip install spectral
pip install scikit-learn==1.4.1.post1
pip install xarray
pip install netCDF4
pip insatll tqdm
pip install calflops
```

# UHI Data
The underwater hyperspectral benthic classification dataset for coral reefs was preprocessed prior to use, including hyperspectral band clipping and class reindexing of the ground truth labels. 
The preprocessed dataset supporting the findings of this study is available at https://doi.org/10.6084/m9.figshare.31130086. 
The original, unprocessed coral reef hyperspectral dataset is publicly available and can be accessed at https://doi.org/10.1594/PANGAEA.911300.

## Data Structure
```text
data
├── cr_005/
│   ├── Ground_truth_005.nc 
│   └── transect_005.nc
├── cr_006/
│   ├── Ground_truth_006.nc
│   └── transect_006.nc
└── cr_019/
    ├── Ground_truth_019.nc 
    └── transect_019.nc
```

# Citation
Please kindly cite this paper if the code is useful and helpful for your research.

# Acknowledgement
Part of the model implementation is adapted from [MambaHSI](https://github.com/li-yapeng/MambaHSI). We sincerely thank the authors for their excellent work and open-source contribution.
