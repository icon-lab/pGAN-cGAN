# pGAN and cGAN

These techniques (pGAN and cGAN) are described in the [following](https://ieeexplore.ieee.org/abstract/document/8653423) paper:

Dar SUH, Yurt M, Karacan L, Erdem A, Erdem E, Çukur T. Image Synthesis in Multi-Contrast MRI with Conditional Generative Adversarial Networks. IEEE Transaction on Medical Imaging. 2019.


# Demo

The following commands train and test pGAN and cGAN models for T1 to T2 synthesis on images from the IXI dataset (registered dataset of training and testing subjects can be downloaded from [here](https://drive.google.com/open?id=1Vt-PVs7fHIX0m-hyZEx-Npabt78T89oE)). To run the code on other datasets, please create a file named 'data.mat' for training and testing samples and place them in their corresponding directories (datasets/yourdata/train, test). 'data.mat' should contain a variable named data_x for the source contrast and data_y for the target contrast. If you are creating the 'data.mat' file via Matlab please make sure that dimensions (1, 2, 3, 4) correspond to (neighbouring slices, number of samples, x-size, y-size). If you are saving the file via python then transpose the dimensions. Also, make sure that voxel intensity of each subject is normalized between 0-1.

## pGAN

### Training
python pGAN.py --dataroot datasets/IXI --name pGAN_run --which_direction BtoA --lambda_A 100 --batchSize 1 --output_nc 1 --input_nc 3 --gpu_ids 0 --niter 50 --niter_decay 50 --save_epoch_freq 25 --lambda_vgg 100 --checkpoints_dir checkpoints/ --training
 <br />
name - name of the experiment  <br />
which_direction - direction of synthesis. If it is set to 'AtoB' synthesis would be from data_x to data_y, and vice versa <br />
lambda_A - weighting of the pixel-wise loss function  <br />
input_nc, output_nc - number of neighbouring slices used. If you do not want to use the neighboring slices just set them to 1, the central slices would be selected.  <br />
niter, n_iter_decay - number of epochs with normal learning rate and number of epochs for which the learning leate is decayed to 0. Total number of epochs is equal to sum of them  <br />
save_epoch_freq -frequency of saving models <br />
lambda_vgg - weighting of the pixel-wise loss function 

### Testing
python pGAN.py --dataroot datasets/IXI --name pGAN_run --which_direction BtoA --phase test --output_nc 1 --input_nc 3 --how_many 1200 --results_dir results/ --checkpoints_dir checkpoints/

## cGAN

### Training
python cGAN.py --dataroot datasets/IXI --name cGAN_run --model cGAN --output_nc 1 --input_nc 1 --gpu_ids 0 --niter 50 --niter_decay 50 --save_epoch_freq 25 --lambda_A 100 --lambda_B 100 --checkpoints_dir checkpoints --dataset_mode unaligned_mat --training  <br />
lambda_A, lambda_B - weightings of the cycle loss functions for both directions
dataset_mode - if set to unaligned the indices of the source and target contrasts are shuffled (for unpaired training)
### Testing
python cGAN.py --dataroot datasets/IXI --name cGAN_run --phase test --output_nc 1 --input_nc 1 --how_many 1200 --results_dir results/ --checkpoints_dir checkpoints
## Prerequisites
Linux  <br />
Python 2.7  <br />
CPU or NVIDIA GPU + CUDA CuDNN  <br />
Pytorch [0.2.0](http://download.pytorch.org/whl/cu80/torch-0.2.0.post2-cp27-cp27mu-manylinux1_x86_64.whl) <br />
Other dependencies - visdom, dominate  

# Citation
You are encouraged to modify/distribute this code. However, please acknowledge this code and cite the paper appropriately.
```
@article{dar2019image,
  title={Image Synthesis in Multi-Contrast MRI with Conditional Generative Adversarial Networks},
  author={Dar, Salman UH and Yurt, Mahmut and Karacan, Levent and Erdem, Aykut and Erdem, Erkut and {\c{C}}ukur, Tolga},
  journal={IEEE Transaction on Medical Imaging},
  year={2019},
  publisher={IEEE}
}
```
For any questions, comments and contributions, please contact Salman Dar (salman[at]ee.bilkent.edu.tr) `<addr>`

(c) ICON Lab 2019


## Acknowledgments
This code is based on implementations by [pytorch-DCGAN](https://github.com/pytorch/examples/tree/master/dcgan) and [CycleGAN and pix2pix in PyTorch](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix).
