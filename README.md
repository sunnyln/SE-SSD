## SE-SSD: Self-Ensembling Single-Stage Object Detector From Point Cloud (CVPR 2021) [[Paper]](https://arxiv.org/abs/2104.09804)

An accurate and fast single-stage 3D object detection framework on KITTI dataset.

**Authors**: [Wu Zheng](https://github.com/Vegeta2020), Weiliang Tang, [Li Jiang](https://github.com/llijiang), Chi-Wing Fu.

original [README.md](./README_v0.md)

### **system env**
Ubuntu16.04 + CUDA10.2 + PyTorch1.6

注意：SE-SSD用到的apex库依赖pytorch1.6以上版本。


### **改源码**

1. pytorch版本问题

AT_CHECK相关的error， SE-SSD/det3d/core/iou3d应该是基于较老的pytorch版本，pytorch1.6不支持，需要将所有的AT_CHECK换成TORCH_CHECK。

2. 训练报错

训练时报错`TypeError: can't pickle _thread.RLock objects`，修改SE-SSD/det3d/torchie/apis/train_sessd.py：

```
from det3d.models import build_detector

# model_ema = copy.deepcopy(model)	# 这里报错 TypeError: can't pickle _thread.RLock objects	
model_ema = build_detector(cfg.model, train_cfg=cfg.train_cfg, test_cfg=cfg.test_cfg)
if distributed:
    model_ema = apex.parallel.convert_syncbn_model(model_ema)
    model_ema = DistributedDataParallel(
        model_ema.cuda(cfg.local_rank),
        device_ids=[cfg.local_rank],
        output_device=cfg.local_rank,
        # broadcast_buffers=False,
        find_unused_parameters=True,
    )
else:
    model_ema = model_ema.cuda()
```

3. dataset

SE-SSD/det3d/datasets/dataset_factory.py中，把nuscenes和lyft相关行注释掉：

```
from .kitti import KittiDataset
# from .nuscenes import NuScenesDataset
# from .lyft import LyftDataset

dataset_factory = {
    "KITTI": KittiDataset,
    # "NUSC": NuScenesDataset,
    # "LYFT": LyftDataset,
}
```

4. 确保numpy安装的版本低于1.18，否则会报错`TypeError: 'numpy.float64' object cannot be interpreted as an integer`：

```
pip install numpy==1.17.5 -i https://pypi.tuna.tsinghua.edu.cn/simple
```



### Installation

PS：多留意pip是否是当前conda环境下的pip，以及pip的版本。

```bash
# <1> 我系统里已经安装了cuda10.2，只需要重定向
$ cd /usr/local
$ sudo rm cuda
$ sudo ln -s cuda10.2 cuda

# <2> 搭建虚拟环境
$ conda create -n sessd python=3.6
$ conda activate sessd
$ conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/		#一般库的源
$ conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/pytorch/ 	#pytorch, torchvision的源
$ conda install pip				    # 保证环境隔离
$ pip install --upgrade pip		# 如果开了代理要先关掉
$ conda install pytorch==1.6.0 torchvision==0.7.0 cudatoolkit=10.2 			    #注意这里不加“-c pytorch”
$ PYTHONPATH=:/home/lina/anaconda3/envs/sessd/lib/python3.6/site-packages 	#我的bashrc里添加了多个路径给PYTHONPATH，这里删除其他环境影响

# <3> 安装SE-SSD
# <3.1> 安装依赖库
$ pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# <3.2> 安装spconv，实测SE-SSD仅支持spconv1.×版本
$ git clone https://github.com/traveller59/spconv.git --recursive -b v1.2.1
$ cd spconv
$ sudo apt-get install libboost-all-dev
$ cmake --version	# 确保cmake>= 3.13.2
$ python setup.py bdist_wheel
$ cd dist/
$ pip install spconv-1.2.1-cp36-cp36m-linux_x86_64.whl

# <3.3> 安装iou3d，先修改源码，把SE-SSD/det3d/core/iou3d下所有的AT_CHECK全部替换成TORCH_CHECK
$ conda install pyyaml
$ cd /home/lina/venti/SE-SSD/det3d/core/iou3d	# AT_CHECK error
$ python setup.py install

# <3.4> 安装ifp-sample
$ mkdir third_party
$ cd third_party
$ git clone https://github.com/jackd/ifp-sample.git
$ pip install Cython -i https://pypi.tuna.tsinghua.edu.cn/simple
$ pip install -e ifp-sample

# <3.4> 安装apex
$ git clone https://github.com/NVIDIA/apex.git	#apex是NIVIDIA用于混合精度训练的库，需要手动安装
$ cd apex
$ pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./

# <3.4> 安装det3d
$ cd ../..
$ python setup.py build develop
```

安装成功后的打印：

```bash
copying build/lib.linux-x86_64-3.6/det3d/ops/pointnet2/PN2.cpython-36m-x86_64-linux-gnu.so -> det3d/ops/pointnet2
copying build/lib.linux-x86_64-3.6/det3d/ops/rroi_align/RotateRoIAlign.cpython-36m-x86_64-linux-gnu.so -> det3d/ops/rroi_align
copying build/lib.linux-x86_64-3.6/det3d/ops/roipool3d/RoIPool3D.cpython-36m-x86_64-linux-gnu.so -> det3d/ops/roipool3d
copying build/lib.linux-x86_64-3.6/det3d/ops/iou3d/IoU3D.cpython-36m-x86_64-linux-gnu.so -> det3d/ops/iou3d
copying build/lib.linux-x86_64-3.6/det3d/ops/nms/nms.cpython-36m-x86_64-linux-gnu.so -> det3d/ops/nms
copying build/lib.linux-x86_64-3.6/det3d/ops/sigmoid_focal_loss/sigmoid_focal_loss_cuda.cpython-36m-x86_64-linux-gnu.so -> det3d/ops/sigmoid_focal_loss
copying build/lib.linux-x86_64-3.6/det3d/ops/syncbn/syncbn_gpu.cpython-36m-x86_64-linux-gnu.so -> det3d/ops/syncbn
Creating /home/lina/anaconda3/envs/sessd1/lib/python3.6/site-packages/det3d.egg-link (link to .)
Adding det3d 1.0rc0+b68068d to easy-install.pth file

Installed /home/lina/venti/SE-SSD
Processing dependencies for det3d==1.0rc0+b68068d
Searching for lyft-dataset-sdk==0.0.8
Best match: lyft-dataset-sdk 0.0.8
Adding lyft-dataset-sdk 0.0.8 to easy-install.pth file
...
...
...
Using /home/lina/anaconda3/envs/sessd1/lib/python3.6/site-packages
Searching for pycparser==2.21
Best match: pycparser 2.21
Adding pycparser 2.21 to easy-install.pth file

Using /home/lina/anaconda3/envs/sessd1/lib/python3.6/site-packages
Finished processing dependencies for det3d==1.0rc0+b68068d
```

train阶段需要安装的库：

```bash
$ pip install ipdb -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### TEST

```bash
$ cd SE-SSD/tools/
$ python create_data.py 
$ python test.py --checkpoint se-ssd-model.pth

Evaluation official_AP_11: car AP(Average Precision)@0.70, 0.70, 0.70:
bbox AP:98.72, 90.10, 89.57
bev  AP:90.61, 88.76, 88.18
3d   AP:90.21, 86.25, 79.22
aos  AP:98.67, 89.86, 89.16
car AP(Average Precision)@0.70, 0.50, 0.50:
bbox AP:98.72, 90.10, 89.57
bev  AP:98.76, 90.19, 89.77
3d   AP:98.73, 90.16, 89.72
aos  AP:98.67, 89.86, 89.16

Evaluation official_AP_40: car AP(Average Precision)@0.70, 0.70, 0.70:
bbox AP:99.57, 95.58, 93.16
bev  AP:96.70, 92.15, 89.74
3d   AP:93.75, 86.18, 83.50
aos  AP:99.52, 95.28, 92.69
car AP(Average Precision)@0.70, 0.50, 0.50:
bbox AP:99.57, 95.58, 93.16
bev  AP:99.60, 95.92, 93.42
3d   AP:99.59, 95.86, 93.36
aos  AP:99.52, 95.28, 92.69
```

### Train

```bash
$ cd SE-SSD/tools
$ python train.py  # Single GPU
```

Unfortunately, I cannot get a good trained model...

