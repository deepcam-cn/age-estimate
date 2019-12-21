#Age-Estimate

This repo is organized as follows:

```shell
├── 54532-0.jpg
├── checkpoints
│   └── coral_age_epoch_33.pth
├── config
│   └── config_coral.py
├── dataset
│   └── dataset_coral.py
├── models
│   ├── __pycache__
│   │   └── resnet_coral.cpython-36.pyc
│   └── resnet_coral.py
├── test.py
└── train_age.py
```

##Requirements

1.pytorch >=0.4

2.opencv >= 3.4

3.numpy ...

## Training

- CACD: http://bcsiriuschen.github.io/CARC/
- UTKFace: https://susanqq.github.io/UTKFace/
- AFAD: https://github.com/afad-dataset/tarball
- MORPH-2: https://www.faceaginggroup.com/morph/

Download dataset into detest dir,and then use face alignment with face detector and landmark like insight face.

## Benchmark

|       | AFAD | Morph II |
| ----- | ---- | -------- |
| SSR   |      | 3.16     |
| C3AE  |      | 2.75     |
| ORCNN | 3.66 | 3.27     |
| Ours  | 3.40 | 2.65     |

## References

https://github.com/derronqi/C3AE_Age_Estimation.git

https://github.com/Raschka-research-group/coral-cnn

https://github.com/shamangary/SSR-Net.git

