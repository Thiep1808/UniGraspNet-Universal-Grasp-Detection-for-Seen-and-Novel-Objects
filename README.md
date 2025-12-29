# UniGraspNet[[report](https://drive.google.com/file/d/1DtiOI9AhXIrL8J0N50SviAKJoy7OE4W-/view?fbclid=IwY2xjawO_Y5BleHRuA2FlbQIxMABicmlkETFKTk5JdXZibmFwd21SdHZGc3J0YwZhcHBfaWQQMjIyMDM5MTc4ODIwMDg5MgABHpSUioAudva2P-DdTBz-hh3hbGl7uWE8rWHiGqEJkeFFxTHtQv-RdOZgih2-_aem_33b8ibRpZ6E3DwiIITXNbA)]

**UniGraspNet: Universal grasp detection across known and new objects**<br>

## Introduction
The code is based on [Generalizing-Grasp](https://github.com/mahaoxiang822/Generalizing-Grasp)
### Note: The repo is still updating

## Environments
- Anaconda3
- Python == 3.7.9
- PyTorch >= 1.8.0
- Open3D >= 0.8

## Installation
Follow the installation of graspnet-baseline.

Get the code.
```bash
git clone https://github.com/Thiep1808/UniGraspNet-Universal-Grasp-Detection-for-Seen-and-Novel-Objects.git
cd UniGraspNet
```
Install packages via Pip.
```bash
pip install -r requirements.txt
```
Compile and install pointnet2 operators (code adapted from [votenet](https://github.com/facebookresearch/votenet)).
```bash
cd pointnet2
python setup.py install
```
Compile and install knn operator (code adapted from [pytorch_knn_cuda](https://github.com/chrischoy/pytorch_knn_cuda)).
```bash
cd knn
python setup.py install
```
Install graspnetAPI for evaluation.
```bash
git clone https://github.com/graspnet/graspnetAPI.git
cd graspnetAPI
pip install .
```


## Prepare Datasets
For GraspNet dataset, you can download from [GraspNet](https://graspnet.net)

#### Full scene data generation
You can generate fusion scene data by yourself by running:
```bash
cd scripts
python TSDFreconstruction_dataset.py
```
Or you can download the pre-generated data from [Google Drive](https://drive.google.com/file/d/12YODD0ZUu6XTudU1fZBhVtAmIpMZk8xQ/view?usp=sharing) and unzip it under the dataset root:

#### Object SDF generation
You can generate object SDF by running:
```bash
pip install mesh-to-sdf
python dataset/grid_sample.py
```

#### Tolerance Label Generation(Follow graspnet-baseline)
Tolerance labels are not included in the original dataset, and need additional generation. Make sure you have downloaded the orginal dataset from [GraspNet](https://graspnet.net/). The generation code is in [dataset/generate_tolerance_label.py](../Scale-Balanced-Grasp/dataset/generate_tolerance_label.py). You can simply generate tolerance label by running the script: (`--dataset_root` and `--num_workers` should be specified according to your settings)
```bash
cd dataset
sh command_generate_tolerance_label.sh
```

Or you can download the tolerance labels from [Google Drive](https://drive.google.com/file/d/1DcjGGhZIJsxd61719N0iWA7L6vNEK0ci/view?usp=sharing)/[Baidu Pan](https://pan.baidu.com/s/1HN29P-csHavJF-R_wec6SQ) and run:
```bash
mv tolerance.tar dataset/
cd dataset
tar -xvf tolerance.tar
```

## Train&Test

### Train with physical constrained regularization

```bash
sh command_train.sh
```

### Test
 - We offer our checkpoints for inference and evaluation, you can download from [Google Drive](https://drive.google.com/file/d/10FMQC98PLdkkM-c0m3fOxg6KwnIg3_y6/view?usp=sharing)
```bash
sh command_test.sh
```


#### Evaluation

```
python evaluate.py
```

