---
layout:     post
title:      mmsegementation 踩坑日记
subtitle:   mm 系列的艰难起步
date:       2020-10-13
author:     liyang
header-img: img/the-whale.png
catalog: true
tags:
    - Python
    - 编程
---

# A from-scratch setup script for mmsegmentation

```linux
conda create -n open-mmlab python=3.7 -y

conda activate open-mmlab

conda install pytorch=1.6.0 torchvision cudatoolkit=10.1 -c pytorch

pip install mmcv-full==latest+torch1.6.0+cu101 -f https://download.openmmlab.com/mmcv/dist/index.html

git clone https://github.com/open-mmlab/mmsegmentation.git

cd mmsegmentation

pip install -e .  # or "python setup.py develop"
```

**ImportError: libGL.so.1: cannot open shared object file: No such file or directory——docker容器内问题报错**

```
sudo apt update

sudo apt install libgl1-mesa-glx
```


就ok了

如果容器内没有sudo指令，可以:

```
apt-get update

apt-get install sudo
```

****