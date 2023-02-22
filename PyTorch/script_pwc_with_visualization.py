import sys
import cv2
import torch
import numpy as np
from math import ceil
import imageio.v3 as imageio
import numpy as np
from matplotlib.colors import hsv_to_rgb
import matplotlib.pyplot as plt
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import models.PWCNet
from models import *

"""
Contact: Deqing Sun (deqings@nvidia.com); Zhile Ren (jrenzhile@gmail.com)
"""


def writeFlowFile(filename, uv):  # flow visualization added on Feb 20, 2023
    """
    According to the matlab code of Deqing Sun and c++ source code of Daniel Scharstein
    Contact: dqsun@cs.brown.edu
    Contact: schar@middlebury.edu
    """
    TAG_STRING = np.array(202021.25, dtype=np.float32)
    if uv.shape[2] != 2:
        sys.exit("writeFlowFile: flow must have two bands!");
    H = np.array(uv.shape[0], dtype=np.int32)
    W = np.array(uv.shape[1], dtype=np.int32)
    with open(filename, 'wb') as f:
        f.write(TAG_STRING.tobytes())
        f.write(W.tobytes())
        f.write(H.tobytes())
        f.write(uv.tobytes())


def load_flow_to_numpy(path):
    with open(path, 'rb') as f:
        magic = np.fromfile(f, np.float32, count=1)
        assert (202021.25 == magic), 'Magic number incorrect. Invalid .flo file'
        h = np.fromfile(f, np.int32, count=1)[0]
        w = np.fromfile(f, np.int32, count=1)[0]
        data = np.fromfile(f, np.float32, count=2 * w * h)
    data2D = np.resize(data, (w, h, 2))
    return data2D


def load_flow_to_png(path):
    flow = load_flow_to_numpy(path)
    image = flow_to_image(flow)
    return image


def flow_to_image(flow, max_flow=256):
    if max_flow is not None:
        max_flow = max(max_flow, 1.)
    else:
        max_flow = np.max(flow)

    n = 8
    u, v = flow[:, :, 0], flow[:, :, 1]
    mag = np.sqrt(np.square(u) + np.square(v))
    angle = np.arctan2(v, u)
    im_h = np.mod(angle / (2 * np.pi) + 1, 1)
    im_s = np.clip(mag * n / max_flow, a_min=0, a_max=1)
    im_v = np.clip(n - im_s, a_min=0, a_max=1)
    im = hsv_to_rgb(np.stack([im_h, im_s, im_v], 2))
    return (im * 255).astype(np.uint8)


im1_fn = 'data/TEST_FULL_with_facula_0-HONOR50SE_NONE_NONE_INDOOR_LIGHT_3.tif';
im2_fn = 'data/TEST_FULL_with_facula_149-HONOR50SE_NONE_NONE_INDOOR_LIGHT_3.tif';
flow_fn = 'tmp/TR_FULL_with_facula_0-149.flo';

if len(sys.argv) > 1:
    im1_fn = sys.argv[1]
if len(sys.argv) > 2:
    im2_fn = sys.argv[2]
if len(sys.argv) > 3:
    flow_fn = sys.argv[3]

pwc_model_fn = './pwc_net.pth.tar';

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

writeFlowFile(flow_fn, flo)

if __name__ == '__main__':  # show the picture of optical flow in .png
    image = load_flow_to_png('tmp/TR_FULL_no_facula.flo')
    plt.imshow(image)
    plt.show()
