from absl import logging
from absl import flags
from absl import app

import numpy as np
import cv2
import os

import utils
import color

project_root = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))


def main(_):
    image = cv2.imread(os.path.join(project_root, 'experiments/exp2/ComparedOne(do_not_click_me)/1-90.bmp_true.bmp'))
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    rgb = rgb.astype(np.float32) / 255
    yuv = color.rgb2ycbcr(rgb)
    h, w, _ = yuv.shape

    rain = utils.synthetic_rain((h, w), density=0.05, ksize=50, angle=186)

    yuv[:, :, 0] = yuv[:, :, 0] + rain
    rainy = color.ycbcr2rgb(yuv)

    cv2.imshow('asdf', cv2.cvtColor(rainy, cv2.COLOR_RGB2BGR))
    cv2.waitKey()


if __name__ == '__main__':
    app.run(main)
