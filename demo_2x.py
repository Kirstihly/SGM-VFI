import cv2
import math
import sys
import torch
import numpy as np
import argparse
from imageio import mimsave

'''==========import from our code=========='''
sys.path.append('.')
import config as cfg
from Trainer_x4k import Model
from benchmark.utils.padder import InputPadder

parser = argparse.ArgumentParser()
parser.add_argument('--model', default='ours_small', type=str)
parser.add_argument('--exp_name', type=str, default='ours-1-2-points')
parser.add_argument('--num_key_points', default=0.5, type=float)
args = parser.parse_args()


'''==========Model setting=========='''
TTA = True
if args.model == 'ours_small':
    TTA = False
    cfg.MODEL_CONFIG['LOGNAME'] = 'ours_small'
    cfg.MODEL_CONFIG['MODEL_ARCH'] = cfg.init_model_config(
        F=16,
        depth=[2, 2, 2, 4],
        num_key_points=args.num_key_points
    )
else:
    cfg.MODEL_CONFIG['LOGNAME'] = 'ours'
    cfg.MODEL_CONFIG['MODEL_ARCH'] = cfg.init_model_config(
        F=32,
        depth=[2, 2, 2, 6],
        num_key_points=args.num_key_points
    )
model = Model(-1)
model.load_model(args.exp_name)
model.eval()
model.device()


print(f'=========================Start Generating=========================')

def genMid(I0, I2):

    I0_ = (torch.tensor(I0.transpose(2, 0, 1)).cuda() / 255.).unsqueeze(0)
    I2_ = (torch.tensor(I2.transpose(2, 0, 1)).cuda() / 255.).unsqueeze(0)

    padder = InputPadder(I0_.shape, divisor=32)
    I0_, I2_ = padder.pad(I0_, I2_)

    mid = (padder.unpad(model.inference(I0_, I2_, TTA=TTA, fast_TTA=TTA))[0].detach().cpu().numpy().transpose(1, 2, 0) * 255.0).astype(np.uint8)
    return mid

I0 = cv2.imread('figs/000694.png')
I2 = cv2.imread('figs/000696.png')

gif_imgs = [I0, I2]
for sub_num in [1, 1, 1, 1, 1]:
    gif_imgs_temp = [gif_imgs[0], ]
    for i, (img_start, img_end) in enumerate(zip(gif_imgs[:-1], gif_imgs[1:])):
        interp_imgs = [genMid(img_start, img_end)]
        gif_imgs_temp += interp_imgs
        gif_imgs_temp += [img_end, ]
    gif_imgs = gif_imgs_temp

# images = [I0[:, :, ::-1], mid[:, :, ::-1], I2[:, :, ::-1]]
# cv2.imwrite('figs/out_2x.jpg', mid)
# mimsave('figs/out_2x.gif', images, fps=3)
print('Interpolate 2 images to {} images'.format(len(gif_imgs)))
for i in range(len(gif_imgs)):
    cv2.imwrite('results/{:03d}.png'.format(i), gif_imgs[i])

print(f'=========================Done=========================')