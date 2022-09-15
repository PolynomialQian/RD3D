# RGB-D Salient Object Detection via 3D Convolutional Neural Networks (AAAI 2021)

# 3-D Convolutional Neural Networks for RGB-D Salient Object Detection and Beyond (IEEE TNNLS)

## Preface
This repo contains the source code and prediction saliency maps of our **RD3D** and **RD3D+**. The latter is an extension of the former, which is lighter and more computationally efficient and accurate.

**RD3D**: ***RGB-D Salient Object Detection via 3D Convolutional Neural Networks*** ([PDF](https://ojs.aaai.org/index.php/AAAI/article/view/16191))

**RD3D+**: ***3-D Convolutional Neural Networks for RGB-D Salient Object Detection and Beyond*** ([PDF](https://ieeexplore.ieee.org/document/9889257))

## Update

:fire: **Update 2022/09/15** :fire: Our work of **RD3D+** is officially accepted and published in the IEEE Transactions on Neural Networks and Learning Systems now!

:fire: **Update 2021/09/10** :fire: The Pytorch implementation of **RD3D+** is now available! PDF is coming soon.

:fire: **Update 2020/12/29** :fire: The Pytorch implementation of **RD3D** is now available!

## Dataset

* Datasets in use: 

  * NJU2K (1,985 pairs)
  * NLPR (1,000 pairs)
  * STERE (1,000 pairs)
  * DES/RGBD135 (135 pairs)
  * SIP (929 pairs)
  * DUTLF-D (1,200 pairs)
  * RedWeb-S (1,000 pairs)

  More information and downloading links of the former six datasets can be found in [page](http://dpfan.net/d3netbenchmark/), and the RedWeb-S can be downloaded from this [project page](https://github.com/nnizhang/SMAC).

💡**Important tips**💡

  * 1485 paired RGB and depth images from NJU2K and 700 pairs from NLPR are used for training, while the remaining pairs are used for testing.
  * On the DUTLF-D, however, additional 800 pairs from it are used for training and the rest 400 pairs are used for testing.
  * In summary, our training set contains 2,185 pairs except when testing is conducted on DUTLF-D. More details can be found in the paper that will be released soon.


## Usage
### Repo clone

```bash
git clone https://github.com/PPOLYpubki/RD3D.git
cd RD3D
```

### Prerequisites
Required packages are listed below:

- `Ubuntu 16.04`
- `python=3.6`
- `pytorch>=1.6`
- `torchvision` with `pillow<7`
- `cuda>=10.1`
- others: `pip install termcolor opencv-python tensorboard`

### Inference

* Download the pre-trained weights and save them as `./model_path/RD3D.pth` and `./model_path/RD3D_plus.pth`.

  * RD3D [Baidu Cloud](https://pan.baidu.com/s/1CQLLcdfsGdOCqjd2iDGVNw), Fetch code: yoyj,[Google Drive](https://drive.google.com/file/d/121HSE8dkqEKEhTm3J2Qj-UYla-HOVQ2i/view?usp=sharing)
  * RD3D+ [Baidu Cloud](https://pan.baidu.com/s/17Sd1KYrWe2oD8u4i7kr6OA), Fetch code: 7d3g,[Google Drive](https://drive.google.com/file/d/1txqDEB9mOCwZcsZ1qhvjc61nS6RFvaW4/view?usp=sharing)
  
* Make sure your testing dataset be in `./data_path/` and run the following commands for inference:

  * On datasets except for the DUTLF-D.

  ```bash
  # RD3D
  python test.py --model RD3D --model_path ./model_path/RD3D.pth --data_path ./data_path/ --save_path ./save/all_results/ 
  # RD3D+
  python test.py --model RD3D_plus --model_path ./model_path/RD3D_plus.pth --data_path ./data_path/ --save_path ./save/all_results/
  ```

  * In particular, as what was mentioned in the [Important tips](#Dataset), we also provide pre-trained weights of RD3D ([Baidu Cloud](https://pan.baidu.com/s/1ioNJ78_7DsRFR2HY23Wmhg), Fetch code: enza),([Google Drive](https://drive.google.com/file/d/1v0ogTJL5DwqT5_bCUIbsS8Blh3QuKMEQ/view?usp=sharing)) and pre-trained weights of RD3D+ ([Baidu Cloud](https://pan.baidu.com/s/1iuhAMnRXo0Qa-aD5y7SN8w), Fetch code: 1lfc),([Google Drive](https://drive.google.com/file/d/1ZJRYwjj7Nx3nShyUccRDTrRn9jhE43gp/view?usp=sharing)) for the DUTLF-D case. Specifically, run the following command to test on the DUTLF-D:

  ```bash
  # RD3D
  python test.py --model RD3D --model_path ./pth/RD3D_DUTLF-D.pth --data_path ./data_path/ --save_path ./save/all_results/ 
  # RD3D+
  python test.py --model RD3D_plus --model_path ./pth/RD3D_plus_DUTLF-D.pth --data_path ./data_path/ --save_path ./save/all_results/ 
  ```

* All of our training processes are actually based on multiple GPUs. However, we have modified the key of some pre-trained weights, so please follow our command here for inference, otherwise there will be an error.

### Training

* By default, make sure the training datasets be in the folder `./data_path/`.
* Run the following command for training (Note that the `model_name` below can be either `RD3D` or `RD3D_plus`): 

    ```bash
    python train.py --model [model_name] --data_dir ./data_path/
    ```

* Note that for researchers training with multiple GPUs, remember to add `--multi_load`  to the inference command during testing. 

### Evaluation

We follow the authors of the [SINet](https://github.com/DengPingFan/SINet) to conduct evaluations on our testing results.
> We provide complete and fair one-key evaluation toolbox for benchmarking within a uniform standard. Please refer to this link for more information:
> Matlab version: https://github.com/DengPingFan/CODToolbox
> Python version: https://github.com/lartpang/PySODMetrics
### Result

* Qualitative performance

  Quantitative RGB-D SOD results in terms of S-measure (S<sub>α</sub>), maximum F-measure (F<sub>β</sub><sup>max</sup>), maximum E-measure (E<sub>Φ</sub><sup>max</sup>) and mean absolute error (MAE). [Seven datasets](#Dataset) are employed. For brevity, values in the table below are in the form of `RD3D|RD3D+`.

  | Dataset  | S<sub>α</sub> | F<sub>β</sub><sup>max</sup> | E<sub>Φ</sub><sup>max</sup> |     MAE      |
  | :------: | :-----------: | :-------------------------: | :-------------------------: | :----------: |
  |  NJU2K   | 0.916\|0.928  |        0.914\|0.928         |        0.947\|0.955         | 0.036\|0.033 |
  |   NLPR   | 0.930\|0.933  |        0.919\|0.921         |        0.965\|0.964         | 0.022\|0.022 |
  |  STERE   | 0.911\|0.914  |        0.906\|0.905         |        0.947\|0.946         | 0.037\|0.037 |
  | RGBD135  | 0.935\|0.950  |        0.929\|0.946         |        0.972\|0.982         | 0.019\|0.017 |
  | DUTLF-D  | 0.932\|0.936  |        0.939\|0.945         |        0.960\|0.964         | 0.031\|0.030 |
  |   SIP    | 0.885\|0.892  |        0.889\|0.900         |        0.924\|0.928         | 0.048\|0.046 |
  | ReDWeb-S | 0.700\|0.718  |        0.687\|0.697         |        0.780\|0.786         | 0.136\|0.130 |

* Downloading links of our result saliency maps:
  * RD3D:  [Baidu Cloud](https://pan.baidu.com/s/1OBCV4vDgjlpCsmRE76fxIg) (Fetch code: am16), [Google Drive](https://drive.google.com/file/d/14Kpdyh9EAFC4lphwGa3XNBr3_YXTtjV_/view?usp=sharing)
  * RD3D+: [Baidu Cloud](https://pan.baidu.com/s/10FuFy76JrP725i4q1-6R3A) (Fetch code: hwna), [Google Drive](https://drive.google.com/file/d/1mioF1YhZ78W6cZGAH_l-Ym-QLZtObOtJ/view?usp=sharing)

### Benchmark RGB-D SOD
The complete RGB-D SOD benchmark can be found in this [page](http://dpfan.net/d3netbenchmark/).


## Citation
Please cite our work if you find them useful:
```
@inproceedings{chen2021rgb,
	title={RGB-D Salient Object Detection via 3D Convolutional Neural Networks},
	author={Chen, Qian and Liu, Ze and Zhang, Yi and Fu, Keren and Zhao, Qijun and Du, Hongwei},
	booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
	volume={35},
	number={2},
	pages={1063--1071},
	year={2021}
    }
```