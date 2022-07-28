# Handheld Multi-Frame Neural Depth Refinement

This is the official code repository for the work: [The Implicit Values of A Good Hand Shake: Handheld Multi-Frame Neural Depth Refinement
](https://light.princeton.edu/publication/hndr/), presented at CVPR 2022.

If you use parts of this work, or otherwise take inspiration from it, please considering citing our paper:
```
@inproceedings{chugunov2022implicit,
  title={The Implicit Values of A Good Hand Shake: Handheld Multi-Frame Neural Depth Refinement},
  author={Chugunov, Ilya and Zhang, Yuxuan and Xia, Zhihao and Zhang, Xuaner and Chen, Jiawen and Heide, Felix},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={2852--2862},
  year={2022}
}
```

## Requirements:
- Developed using PyTorch 1.10.0 on Linux x64 machine
- Condensed package requirements are in `\requirements.txt`. Note that this contains the package versions at the time of publishing, if you update to, for example, a newer version of PyTorch you will need to watch out for changes in class/function calls

## Data:
- Download data from [this Google Drive link](https://drive.google.com/drive/folders/1M6W6meoNdi7XfTaJYILsLrm1bEv5i9qv?usp=sharing) and unpack into the `\data` folder
- Each folder corresponds to a scene [`castle`, `double`, `eagle`, `elephant`, `embrace`, `frog`, `ganesha`, `gourd`, `rocks`, `thinker`] and contains five files. 
    - `model.pt` is the frozen, trained MLP corresponding to the scene
    - `frame_bundle.npz` is the recorded bundle of data (images, depth, and poses)
    - `pose_bundle.npz` is a *much* smaller recorded bundle of data (poses only)
    - `reprojected_lidar.npy` is the merged LiDAR depth baseline as described in the paper
    - `snapshot.mp4` is a video of the recorded snapshot for visualization purposes

An explanation of the format and contents of the frame bundles (`frame_bundle.npz`) is given in an interactive format in `\0_data_format.ipynb`. We recommend you go through this jupyter notebook before you record your own bundles or otherwise manipulate the data.

## Project Structure:
```cpp
HNDR
  ├── checkpoints  
  │   └── // folder for network checkpoints
  ├── data  
  │   └── // folder for recorded bundle data
  ├── utils  
  │   ├── dataloader.py  // dataloader class for bundle data
  │   ├── neural_blocks.py  // MLP blocks and positional encoding
  │   └── utils.py  // miscellaneous helper functions (e.g. grid/patch sample)
  ├── 0_data_format.ipynb  // interactive tutorial for understanding bundle data
  ├── 1_reconstruction.ipynb  // interactive tutorial for depth reconstruction
  ├── model.py  // the learned implicit depth model
  │             // -> reproject points, query MLP for offsets, visualization
  ├── README.md  // a README in the README, how meta
  ├── requirements.txt  // frozen package requirements
  ├── train.py  // wrapper class for arg parsing and setting up training loop
  └── train.sh  // example script to run training
```
## Reconstruction:
The jupyter notebook `\1_reconstruction.ipynb` contains an interactive tutorial for depth reconstruction: loading a model, loading a bundle, generating depth.

## Training:
The script `\train.sh` demonstrates a basic call of `\train.py` to train a model on the `gourd` scene data. It contains the arguments
- `checkpoint_path` - path to save model and tensorboard checkpoints
- `device` - device for training [cpu, cuda]
- `bundle_path` - path to the bundle data

For other training arguments, see the argument parser section of `\train.py`.


Best of luck,  
Ilya
