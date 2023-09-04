# CELS

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/cognitive-evolutionary-search-to-select/click-through-rate-prediction-on-criteo)](https://paperswithcode.com/sota/click-through-rate-prediction-on-criteo?p=cognitive-evolutionary-search-to-select)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/cognitive-evolutionary-search-to-select/click-through-rate-prediction-on-avazu)](https://paperswithcode.com/sota/click-through-rate-prediction-on-avazu?p=cognitive-evolutionary-search-to-select)

This repository serves as the official implementation for the KDD 2023 Paper titled,  "[Cognitive Evolutionary Search to Select Feature Interactions for Click-Through Rate Prediction](https://doi.org/10.1145/3580305.3599277 )".
For a deeper understanding, kindly check out the [Promotional Video](https://youtu.be/p3kE54lIWRw), [Slides](https://www.researchgate.net/publication/373519329_Cognitive_Evolutionary_Search_to_Select_Feature_Interactions_for_Click-Through_Rate_Prediction ), [中文解读](https://mp.weixin.qq.com/s/IhcvJc8HQl_4srfz6jvsKQ).




## Requirements
* Ensure you have Python and Pytorch (version 1.8 or higher) installed. Our setup utilized Python 3.8 and Pytorch 1.13.0.
* Should you wish to leverage GPU processing, please install CUDA.



## Before Start

Before proceeding with the preprocessing, ensure you run the `./data/mkdir.sh`

Upon completion, you'll observe the following three directory structures created at the same level as the project:

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



## Datasets

We conducted our experiments using three publicly available real-world datasets: Avazu, Criteo, and Huawei. You can access and download these datasets from the links provided below.

- **Criteo**: The raw dataset can be downloaded from https://www.kaggle.com/c/criteo-display-ad-challenge/data or https://www.kaggle.com/datasets/mrkmakr/criteo-dataset?resource=download. If you want to know how to preprocess the data, please refer to `./data/criteoPreprocess.py`
- **Avazu**: The raw dataset can be downloaded from https://www.kaggle.com/c/avazu-ctr-prediction/data. If you want to know how to preprocess the data, please refer to `./data/avazuPreprocess.py`
- **Huawei**: The raw dataset can be downloaded from https://www.kaggle.com/louischen7/2020-digix-advertisement-ctr-prediction. If you want to know how to preprocess the data, please refer to `./data/huaweiPreprocess.py`




## Example
If you've acquired the source code, you can train the CELS model.

```
$ cd main
$ python train.py --dataset=[dataset] --strategy=[strategy]  --gpu==[gpu_id] 
```

The options for the command parameter "strategy" are ['1,1',  '1+1',  'n,1',  'n+1'].

You can change the model parameters in `./config/configs.py`



## Visualization of Evolution Path

You can visualize the evolutionary path depicted by ***gene maps*** of the model.

```
$ cd main
$ python plotUtils.py --dataset_strategy=[dataset_strategy]  --datetime=[datetime]
```



## Contact

Should you have any questions regarding our paper or codes, please don't hesitate to reach out via email at yrunl@mail.ustc.edu.cn or demon@mail.ustc.edu.cn.



## Acknowledgment 
Our code is developed based on [GitHub - shenweichen/DeepCTR-Torch: 【PyTorch】Easy-to-use,Modular and Extendible package of deep-learning based CTR models](https://github.com/shenweichen/DeepCTR-Torch).







