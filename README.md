# Recurrent U-Net for Resource Constraint Segmentation
[![license](https://img.shields.io/github/license/mashape/apistatus.svg)]

Iterative refinement recurrent U-Net for Semantic Segmentation in PyTorch
Here we provide the recurrent UNet implementation, as well as pipeline including training and validation. 

**IMPORTANT** Please download the datasets and modify the `config.json` accordingly. 

**If you find this useful in your research, please consider citing:**

```
@inproceedings{wang2019recurrent,
  title={Recurrent U-Net for resource-constrained segmentation},
  author={Wang, wei and Yu, Kaicheng and Hugonot, Joachim and Fua, Pascal and Salzmann, Mathieu},
  booktitle={Proceedings of the IEEE International Conference on Computer Vision},
  pages={2142--2151},
  year={2019}
}
```
 
#### Paper abstract
> State-of-the-art segmentation methods rely on very deep networks that are not always easy to train without very large 
training datasets and tend to be relatively slow to run on standard GPUs. In this paper, we introduce a novel recurrent 
U-Net architecture that preserves the compactness of the original U-Net~\cite{Ronneberger15}, while substantially 
increasing its performance to the point where it outperforms the state of the art on several benchmarks. 
We will demonstrate its effectiveness for several tasks, including hand segmentation, retina vessel segmentation, 
and road segmentation. We also introduce a large-scale dataset for hand segmentation.

#### Acknowledgement
This repository is built upon an online framework 
[pytorch-semseg](https://github.com/meetshah1995/pytorch-semseg).  

### Networks implemented

* [PSPNet](https://arxiv.org/abs/1612.01105) - With support for loading pretrained models w/o caffe dependency.
* [ICNet](https://arxiv.org/pdf/1704.08545.pdf) - With optional batchnorm and pretrained models.
* [FRRN](https://arxiv.org/abs/1611.08323) - Model A and B.
* [FCN](https://arxiv.org/abs/1411.4038) - All 1 (FCN32s), 2 (FCN16s) and 3 (FCN8s) stream variants.
* [U-Net](https://arxiv.org/abs/1505.04597) - With optional deconvolution and batchnorm.
* [Link-Net](https://codeac29.github.io/projects/linknet/) - With multiple resnet backends
* [Segnet](https://arxiv.org/abs/1511.00561) - With Unpooling using Maxpool indices.
* [Recurrent-UNet](https://arxiv.org/abs/1811.10914) - With support iterative refinement in semantic segmentation. 
* [RefineNet](https://arxiv.org/abs/1611.06612) - With online code found repository.

#### Upcoming 

* [E-Net](https://arxiv.org/abs/1606.02147)


### DataLoaders implemented

#### General segmentation datasets
* [CamVid](http://mi.eng.cam.ac.uk/research/projects/VideoRec/CamVid/)
* [Pascal VOC](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/segexamples/index.html)
* [ADE20K](http://groups.csail.mit.edu/vision/datasets/ADE20K/)
* [MIT Scene Parsing Benchmark](http://data.csail.mit.edu/places/ADEchallenge/ADEChallengeData2016.zip)
* [Cityscapes](https://www.cityscapes-dataset.com/)
* [NYUDv2](http://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html)
* [Sun-RGBD](http://rgbd.cs.princeton.edu/)
* [EPFL-Hand](TODO) If interested, contact [kaicheng.yu@epfl.ch](mailto:kaicheng.yu@epfl.ch) for access 

#### Resource constraint semantic segmentation

**Upcoming**
Support the download if necessary.
 
* [GTEA]() 
* [EYTH]() Youtube collected video sequences.
* [HOF]() Hand over face dataset.
* [EgoHand]() Egohand dataset.
* [KBH]() TO release soon.
* [Road]() Massachusatte Road segmentation dataset from satellite images.
* [DRIVE]() Retina vessel segmentation
 

### Requirements

* pytorch >=1.0.2
* torchvision >= 0.2.1
* TensorboardX 
* scipy
* tqdm
* pydensecrf

#### One-line installation

`pip install -r requirements.txt`

### Data

* Download data for desired dataset(s) from list of URLs [here](https://meetshah1995.github.io/semantic-segmentation/deep-learning/pytorch/visdom/2017/06/01/semantic-segmentation-over-the-years.html#sec_datasets).
* Extract the zip / tar and modify the path appropriately in `config.json`

### Usage

* Training dataset.

**To train the model :**

```
python train.py [-h] [--config [CONFIG]] 

--config                Configuration file to use
```

**To validate the model :**

```
usage: validate.py [-h] [--config [CONFIG]] [--model_path [MODEL_PATH]]
                       [--eval_flip] [--measure_time]

  --config              Config file to be used
  --model_path          Path to the saved model
  --eval_flip           Enable evaluation with flipped image | True by default
  --measure_time        Enable evaluation with time (fps) measurement | True
                        by default
```

**To test the model w.r.t. a dataset on custom images(s):**

```
python test.py [-h] [--model_path [MODEL_PATH]] [--dataset [DATASET]]
               [--dcrf [DCRF]] [--img_path [IMG_PATH]] [--out_path [OUT_PATH]]
 
  --model_path          Path to the saved model
  --dataset             Dataset to use ['pascal, camvid, ade20k etc']
  --dcrf                Enable DenseCRF based post-processing
  --img_path            Path of the input image
  --out_path            Path of the output segmap
```


