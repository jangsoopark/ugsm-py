from absl import logging
from absl import flags
from absl import app

import time
import json
import os

from skimage import color
import numpy as np
import cv2

import utils
import ugsm

flags.DEFINE_string('image_path', default='experiments/exp3/obama.bmp', help='')
flags.DEFINE_string('config_path', default='experiments/config.json', help='')
FLAGS = flags.FLAGS

project_root = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))


def main(_):
    image = cv2.imread(os.path.join(project_root, FLAGS.image_path))
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # rgb = rgb.astype(np.float32)
    yuv = color.rgb2ycbcr(rgb) / 255

    with open(os.path.join(project_root, FLAGS.config_path), mode='r', encoding='utf-8') as f:
        params = json.load(f)

    start_time = time.time()
    s, i = ugsm.ugsm(yuv[:, :, 0], params)
    end_time = time.time()
    logging.info(f'Elapsed time: {end_time - start_time}')
    logging.info(i)

    de_rain = np.empty(image.shape, np.float32)

    de_rain[:, :, 0] = yuv[:, :, 0] - s.astype(np.float32)
    de_rain[:, :, 1] = yuv[:, :, 1]
    de_rain[:, :, 2] = yuv[:, :, 2]

    cv2.imshow('original', image)
    derain_rgb = color.ycbcr2rgb(de_rain * 255)
    cv2.imshow(f'de-rain iter={i}', cv2.cvtColor(derain_rgb, cv2.COLOR_RGB2BGR))
    cv2.imshow('s', s)
    cv2.waitKey()


if __name__ == '__main__':
    app.run(main)
