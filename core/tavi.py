import sys

sys.path.append("RAFT/core")
from core.raft import RAFT
from core.utils.flow_viz import flow_to_image
import torch
from argparse import Namespace
import numpy as np
import cv2
import imageio as iio
import matplotlib.pyplot as plt


device = "cuda" if torch.cuda.is_available() else "cpu"


def model_args(small=False):
    floargs = Namespace()
    floargs.small = small
    if floargs.small:
        floargs.model = "models/raft-small.pth"
    else:
        floargs.model = "models/raft-things.pth"
    floargs.mixed_precision = False
    return floargs


def load_model(small=False):
    args = model_args(small)
    print ('loading RAFT model')
    flowmodel = torch.nn.DataParallel(RAFT(args))
    flowmodel.load_state_dict(torch.load(args.model, map_location=torch.device("cpu")))
    flowmodel.to(device).eval()
    print (' RAFT model loaded')
    return flowmodel


def to_torch(im):
    return torch.from_numpy(im).float()[None].permute((0, 3, 1, 2)).to(device)


def preprocess_im(im):
    old_shape = im.shape[:2]
    new_shape = np.array(old_shape) // 8 * 8
    im = cv2.resize(im, dsize=tuple(new_shape)[::-1], interpolation=cv2.INTER_LINEAR)
    return to_torch(im)


def post_process(flow, to_shape):
    old_shape = np.array(flow.shape[:2])
    ratio = to_shape / old_shape
    flow = cv2.resize(flow, None, fx=ratio[1], fy=ratio[0])
    # ratio = ratio[None, None]
    # flow = flow * np.dstack([ratio, ratio])
    flow *= ratio
    return flow.astype(np.float32)


def compute_flow(model, im1, im2, iters=8):
    print('computing flow', flush=True)
    assert im1.shape == im2.shape
    with torch.no_grad():
        flow = model(im1, im2, iters=iters, test_mode=True)[1]
        flow = flow.squeeze().numpy().transpose((1, 2, 0))
    return flow


def compute_occlusions(flow):
    flow_backward = flow[..., -2:]
    flow = flow[..., :-2]
    assert flow.shape[-1] == 2 and flow_backward.shape[-1] == 2
    h, w = flow.shape[:2]
    x = np.linspace(0, 1, w, dtype=np.float32)
    y = np.linspace(0, 1, h, dtype=np.float32)
    coord = np.meshgrid(x, y, indexing='xy')
    interp = coord + flow.transpose((2, 0, 1))
    warped = cv2.remap(
        flow_backward,
        interp[0],
        interp[1],
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )
    cycle = warped + flow
    return np.linalg.norm(cycle, axis=-1)


# def compute_optical_flow(self):
#     self.load_optical_flow()
#     # self.optical_flow_vid.save()
#     for to_compute in self.frames_to_compute_of:
#         im1 = self.vid.get_data(to_compute)
#         last_frame_num = min(self.vid.get_length() - 1, to_compute + self.params.of_jump)
#         num_frames = last_frame_num - to_compute
#         im2 = self.vid.get_data(last_frame_num)
#         optical_flow_frame_rev = optical_flow.compute_flow(self.flowmodel, im2, im1)
#         optical_flow_frame_for = optical_flow.compute_flow(self.flowmodel, im1, im2)
#         for f in np.arange(to_compute, last_frame_num):
#             # print(to_compute, last_frame_num)
#             flow = np.concatenate([optical_flow_frame_rev, optical_flow_frame_for], axis=2)
#             flow /= num_frames
#             occlusions = self.compute_occlusions(flow)
#             self.optical_flow_vid.set_data(flow, occlusions, f)
#             # if f in self.frames_to_compute_of:
#             #     self.frames_to_compute_of.remove(f)
#     self.optical_flow_vid.save()

if __name__ == '__main__':
    dis_object = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    dis_object.setUseSpatialPropagation(False)

    model = load_model(small=False)
    p1 = '/Volumes/Datasets/insta_selfie_video/val/B_cYPBLHiNQ.00099_frames/0015.crop.jpg'
    p2 = '/Volumes/Datasets/insta_selfie_video/val/B_cYPBLHiNQ.00099_frames/0016.crop.jpg'
    im1 = iio.imread(p1)
    im2 = iio.imread(p2)

    h, w = im2.shape[:2]
    im1 = cv2.resize(im1, dsize=(w, h))
    # p = 50
    # im2 = cv2.resize(im1[p:-p, p:-p], (429, 429))

    # g1 = cv2.cvtColor(im1, cv2.COLOR_RGB2GRAY)
    # g2 = cv2.cvtColor(im2, cv2.COLOR_RGB2GRAY)
    #
    # cvflow = dis_object.calc(g1, g2, None)

    flow = compute_flow(model, preprocess_im(im1), preprocess_im(im2))
    rev_flow = compute_flow(model, preprocess_im(im2), preprocess_im(im1))
    flow = post_process(flow, np.array(im1.shape[:2]))
    rev_flow = post_process(rev_flow, np.array(im1.shape[:2]))
    im_flow = flow_to_image(flow)
    im_rev_flow = flow_to_image(rev_flow)

    mask = compute_occlusions(np.dstack([flow, rev_flow]))
    mask_normed = mask / mask.max()
    mask_normed = (mask_normed * 255).astype(np.uint8)

    l = np.vstack([im1, im_flow, np.dstack([mask_normed] * 3)])
    r = np.vstack([im2, im_rev_flow, np.zeros_like(im_rev_flow)])
    plt.imshow(np.hstack([l, r]))
    plt.show()

    # top = np.hstack([im1, t])
    # bot = np.hstack([im_flow, cv_im_flow])
    # out = np.vstack([top, bot])

    # plt.imshow(out)
    # plt.show()
