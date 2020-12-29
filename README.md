
# RD3D: RGB-D Salient Object Detection via 3D Convolutional Neural Networks
This repo is the official implementation of ["RD3D: RGB-D Salient Object Detection via 3D Convolutional Neural Networks"] by [Qian Chen](https://github.com/PPOLYpubki), [Ze Liu](https://github.com/zeliu98), Yi Zhang and [Keren Fu](https://github.com/kerenfu).


## Main Results 

|Dataset | S<sub>α</sub>| F<sub>β</sub><sup>max</sup>|E<sub>Φ</sub><sup>max</sup>| MAE |
|:---:|:---:|:---:|:---:|:---:|
|NJU2K|0.916|0.914|0.947|0.036|
|NLPR|0.930|0.919|0.965|0.022|
|STERE|0.911|0.906|0.947|0.037|
|RGBD135|0.935|0.929|0.972|0.019|
|DUTLF-D|0.932|0.939|0.960|0.031|
|SIP|0.885|0.889|0.924|0.048|

All results can be found in: 
```
BaiDu: https://pan.baidu.com/s/132ChkfOY9hDQ4FO4ada2Mg, password: 3e96.
```


## Installation

### Requirements
- `Ubuntu 16.04`
- `python=3.6`
- `pytorch>=1.3`
- `torchvision` with  `pillow<7`
- `cuda>=10.1`
- others: `pip install termcolor opencv-python tensorboard`

### Datasets
Our training data can be download from:
```
BaiDu: https://pan.baidu.com/s/1uF6LxbH0RIcMFN71cEcGHQ, password: 5z48
```
or:
```
Google Drive: https://drive.google.com/open?id=1BpVabSlPH_GhozzRQYjxTOT_cS6xDUgf
```

## Usage

### Training
```bash
python train.py --data_dir <data directory> [--output_dir <output directory>]
```
### Evaluation
```bash
python test.py --data_dir <data directory> --model_path <model directory> --multi_load
```

We follow the RGB-D SOD benchmark setting from: http://dpfan.net/d3netbenchmark/

## Citation
```
@article{chen2021rd3d,
  title={RGB-D Salient Object Detection via 3D Convolutional Neural},
  author={Qian, Chen and Ze, Liu and Yi, Zhang and Keren, Fu},
  journal={AAAI},
  year={2021}
}
```


## License

The code is released under MIT License (see LICENSE file for details).