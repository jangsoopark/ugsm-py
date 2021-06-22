import numpy as np


def rgb2ycbcr(img, only_y=True):
    # https://github.com/peteryuX/esrgan-tf2
    """Convert rgb to ycbcr
    only_y: only return Y channel
    Input:
        uint8, [0, 255]
        float, [0, 1]
    """
    in_img_type = img.dtype
    img.astype(np.float32)
    if in_img_type != np.uint8:
        img *= 255.
    img = img[:, :, ::-1]

    # convert
    if only_y:
        rlt = np.dot(img, [24.966, 128.553, 65.481]) / 255.0 + 16.0
    else:
        rlt = np.matmul(img, [
            [24.966, 112.0, -18.214],
            [128.553, -74.203, -93.786],
            [65.481, -37.797, 112.0]
        ]) / 255.0 + [16, 128, 128]

    if in_img_type == np.uint8:
        rlt = rlt.round()
    else:
        rlt /= 255.
    return rlt.astype(in_img_type)
