import numpy as np


# https://github.com/peteryuX/esrgan-tf2

def rgb2ycbcr(img, only_y=True):
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



def rgb2ycbcr(img, only_y=True):
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
        rlt = np.matmul(
            img, [[24.966, 112.0, -18.214],[128.553, -74.203, -93.786],[65.481, -37.797, 112.0]]) / 255.0 + [16, 128, 128]
    if in_img_type == np.uint8:
        rlt = rlt.round()
    else:
        rlt /= 255.
    return rlt.astype(in_img_type)


def bgr2ycbcr(bgr):
    bgr = bgr.astype(np.float32)
    im_ycrcb = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCR_CB)
    im_ycbcr = im_ycrcb[:, :, (0, 2, 1)].astype(np.float32)
    im_ycbcr[:, :, 0] = (im_ycbcr[:, :, 0] * (235 - 16) + 16) / 255.0  # to [16/255, 235/255]
    im_ycbcr[:, :, 1:] = (im_ycbcr[:, :, 1:] * (240 - 16) + 16) / 255.0  # to [16/255, 240/255]
    return im_ycbcr


def ycbcr2bgr(ycbcr):
    ycbcr = ycbcr.astype(np.float32)
    ycbcr[:, :, 0] = (ycbcr[:, :, 0] * 255.0 - 16) / (235 - 16)  # to [0, 1]
    ycbcr[:, :, 1:] = (ycbcr[:, :, 1:] * 255.0 - 16) / (240 - 16)  # to [0, 1]
    im_ycrcb = ycbcr[:, :, (0, 2, 1)].astype(np.float32)
    im_rgb = cv2.cvtColor(im_ycrcb, cv2.COLOR_YCR_CB2BGR)
    return im_rgb