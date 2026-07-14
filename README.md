# Under Review

This repo is the PyTorch implementation of our paper:

**["Segmentation-Synthesis Co-Training for Semi-Supervised Domain Generalizable Medical Image Segmentation"](https://)** 


## Usage

### 0. Requirements
The code is developed using Python 3.8 with PyTorch 1.11.0, and CUDA 11.3.
All experiments in our paper were conducted on a single NVIDIA A40 GPU with 48GB memory.

Install the main packages:
```angular2html
pytorch == 1.11.0
torchvision == 0.12.0
cudatoolkit == 11.3.1
```

### 1. Data Preparation
#### 1.1. Download data
The original data can be downloaded in following links:
* MS-CMR Dataset - [Link](https://zmiclab.github.io/zxh/0/mscmrseg19/index.html)
* Fundus Benchmark - [Link](https://zenodo.org/records/8009107)

PS: Please cite the papers of original datasets when using the data in your publications


#### 1.2. Split Dataset
Following the list files (within the `data` folders) to split the datasets

### 2. Training
```angular2html
python train_rematch.py
```

### 3. Evaluation
```angular2html
python eval.py
```


## Citation
If you find this project useful, please consider citing:
```

```

## Contact
If you have any questions or suggestions, please feel free to contact me ([xxszqyy@gmail.com](xxszqyy@gmail.com)).