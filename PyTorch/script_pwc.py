import sys
import cv2
import torch
import numpy as np
from math import ceil
import imageio.v3 as imageio
import matplotlib.pyplot as plt
import os
import flow_vis
import flow_visualization

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import models.PWCNet
from models import *

"""
Contact: Deqing Sun (deqings@nvidia.com); Zhile Ren (jrenzhile@gmail.com)
"""


def writeFlowFile(filename, uv):
    """
    According to the matlab code of Deqing Sun and c++ source code of Daniel Scharstein
    Contact: dqsun@cs.brown.edu
    Contact: schar@middlebury.edu
    """
    """
    function: store flow data in binary
    args:   filename: directory to store .flo
            uv: flows that are predicted from PWCNet
    TAG_STRING: according to Daniel Scharstein, floats are stroed in little endian order, so TAG_STRING is 
                a sanity check that floats are represented correctly
    """

    TAG_STRING = np.array(202021.25, dtype=np.float32)
    if uv.shape[2] != 2:
        sys.exit("writeFlowFile: flow must have two bands!");
    H = np.array(uv.shape[0], dtype=np.int32)  # horizontal
    W = np.array(uv.shape[1], dtype=np.int32)  # vertical
    with open(filename, 'wb') as f:
        f.write(TAG_STRING.tobytes())  # "PIEH" in ASCII,which in little endian happens to be the float 202021.25
        f.write(W.tobytes())  # width as an integer
        f.write(H.tobytes())  # height as an integer
        f.write(uv.tobytes())  # float values for u and v, W*H*2*4 bytes total


im1_fn = './data/TEST_FULL_no_facula_10-OPPOK9X_NONE_NONE_OUTDOOR_NONE_2.tif'
im2_fn = './data/TEST_FULL_no_facula_11-OPPOK9X_NONE_NONE_OUTDOOR_NONE_2.tif'
flow_fn = './tmp/TR_FULL_no_facula_10-11.flo'

# paras: directory to image pair and flow
if len(sys.argv) > 1:
    im1_fn = sys.argv[1]
if len(sys.argv) > 2:
    im2_fn = sys.argv[2]
if len(sys.argv) > 3:
    flow_fn = sys.argv[3]

pwc_model_fn = './pwc_net.pth.tar';  # path to the best model

im_all = [imageio.imread(img) for img in [im1_fn, im2_fn]]
im_all = [im[:, :, :3] for im in im_all]

# rescale the image size to be multiples of 64
divisor = 64.
H = im_all[0].shape[0]
W = im_all[0].shape[1]

H_ = int(ceil(H / divisor) * divisor)
W_ = int(ceil(W / divisor) * divisor)
for i in range(len(im_all)):
    im_all[i] = cv2.resize(im_all[i], (W_, H_))

for _i, _inputs in enumerate(im_all):
    im_all[_i] = im_all[_i][:, :, ::-1]
    im_all[_i] = 1.0 * im_all[_i] / 255.0

    im_all[_i] = np.transpose(im_all[_i], (2, 0, 1))
    im_all[_i] = torch.from_numpy(im_all[_i])
    im_all[_i] = im_all[_i].expand(1, im_all[_i].size()[0], im_all[_i].size()[1], im_all[_i].size()[2])
    im_all[_i] = im_all[_i].float()
with torch.no_grad():
    im_all = torch.autograd.Variable(torch.cat(im_all, 1).cuda())

net = models.PWCNet.pwc_dc_net(pwc_model_fn)
net = net.cuda()
net.eval()

flo = net(im_all)
flo = flo[0] * 20.0
flo = flo.cpu().data.numpy()

# scale the flow back to the input size 
flo = np.swapaxes(np.swapaxes(flo, 0, 1), 1, 2)
u_ = cv2.resize(flo[:, :, 0], (W, H))
v_ = cv2.resize(flo[:, :, 1], (W, H))
u_ *= W / float(W_)
v_ *= H / float(H_)
flo = np.dstack((u_, v_))

writeFlowFile(flow_fn, flo)  # store the generated flow in binary
flow=flow_visualization.load_flow_to_numpy(flow_fn)
img=flow_vis.flow_to_color(flow)
plt.imshow(img)
plt.show()