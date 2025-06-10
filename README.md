# Progressive Region Exchange Method
Progressive Region Exchange Method



# Requirements
This study is based on PyTorch 1.13.0, CUDA: 12.2 and python 3.8.17. All experiments in our paper were conducted on NVIDIA GeForce RTX 4090 GPU with an identical experimental setting.

# Usage
We provide `data_split` and `models` for ACDC, PROMISE12 and LA dataset and `code` for ACDC and LA dataset.

Data could be got at [ACDC](https://github.com/HiLab-git/SSL4MIS/tree/master/data/ACDC), [PROMISE12](https://promise12.grand-challenge.org/Download/) and [LA](https://github.com/yulequan/UA-MT/tree/master/data).

To train a model,
```
python ./code/PREM_ACDC_train.py  #for ACDC training
python ./code/PREM_LA_train.py  #for LA training
```

To test a model,
```
python ./code/test_ACDC.py  #for ACDC testing
python ./code/test_LA.py  #for LA testing
```
