from absl import logging
from absl import flags
from absl import app

import utils
import cv2


def main(_):
    rain = utils.synthetic_rain((540, 960))
    cv2.imshow('asdf', rain)
    cv2.waitKey()


if __name__ == '__main__':
    app.run(main)
