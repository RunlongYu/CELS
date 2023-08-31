# CELS
This repository is the official implementation for the KDD 2023 Paper "[Cognitive Evolutionary Search to Select Feature Interactions for Click-Through Rate Prediction](https://doi.org/10.1145/3580305.3599277 )".
For details, please refer to [Promotional Video](https://youtu.be/p3kE54lIWRw), [中文解读](https://mp.weixin.qq.com/s/IhcvJc8HQl_4srfz6jvsKQ)




## Requirements
* Install Python, Pytorch(>=1.8). We use Python 3.8, Pytorch 1.13.0.
* If you plan to use GPU computation, install CUDA.



## Before Start

Before you run the preprocess, plese run the `./data/mkdir.sh`

Then you will get the following three folder structures in the same level directory of the project:

```
criteo
├── bucket
├── feature_map
└── processed

avazu
└── processed

huawei
└── processed
```



## Dataset

We use three public real-world datasets (Avazu, Criteo, Huawei) in our experiments. You can download the datasets from the links below.

- **Criteo**: The raw dataset can be downloaded from https://www.kaggle.com/c/criteo-display-ad-challenge/data or https://www.kaggle.com/datasets/mrkmakr/criteo-dataset?resource=download. If you want to know how to preprocess the data, please refer to `./data/criteoPreprocess.py`
- **Avazu**: The raw dataset can be downloaded from https://www.kaggle.com/c/avazu-ctr-prediction/data. If you want to know how to preprocess the data, please refer to `./data/avazuPreprocess.py`
- **Huawei**: The raw dataset can be downloaded from https://www.kaggle.com/louischen7/2020-digix-advertisement-ctr-prediction . If you want to know how to preprocess the data, please refer to `./data/huaweiPreprocess.py`




## Example
If you have downloaded the source code, you can train CELS model.
```
$ cd main
$ python train.py --dataset=[dataset] --strategy=[strategy]  --gpu==[gpu_id] 
```

The options for the command parameter "strategy" are ['1,1',  '1+1',  'n,1',  'n+1'].

You can change the model parameters in `./config/configs.py`



## Visualization of Evolution Path

You can visualize the evolution path traced by ***gene maps*** of the model.

```
$ cd main
$ python plotUtils.py --dataset_strategy=[dataset_strategy]  --datetime=[datetime]
```



## Contact

Should you have any questions regarding our paper or codes, please don't hesitate to reach out via email at yrunl@mail.ustc.edu.cn or demon@mail.ustc.edu.cn.



## Acknowledgment 
Our code is developed based on [GitHub - shenweichen/DeepCTR-Torch: 【PyTorch】Easy-to-use,Modular and Extendible package of deep-learning based CTR models.](https://github.com/shenweichen/DeepCTR-Torch)







