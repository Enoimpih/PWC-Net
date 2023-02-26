# PWC-Net
Training and testing with our own dataset, correlation layer comes from sniklaus
## Instructions on training
In Pytorch/datasets/mpisintel.py, you can download MPI Sintel complete version through the given link.\
Followed the process explained in the paper, we first pretrain the model with flyingchairs dataset(paras of the pretrained model is located in ./pwc_net_chairs.pth.tar) and train the model with MPI Sintel dataset clean.\
For training in the pre-trained model, you can type
```python
python train.py /path/to/dataset --dataset mpi_sintel_clean --pretrained /path/to/pretrained/model

```
For more information about defined arguments, you can type
```python
python train.py -h
```
To visualize training results, you can type
```python
tensorboard --logdir=/path/to/checkpoints/
```
