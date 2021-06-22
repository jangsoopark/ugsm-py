import numpy as np

"""

# https://github.com/smoonka/SuperResolution/blob/master/ycbcr2rgb.m

[0.256788235294118         0.504129411764706        0.0979058823529412
-0.148223529411765        -0.290992156862745        0.43921568627451
 0.43921568627451         -0.367788235294118       -0.0714274509803921] * [R;G;B] + [16;128;128]
 
"""

ycbcr_from_rgb = np.array([[65.481, 128.553, 24.966],
                           [-37.797, -74.203, 112.0],
                           [112.0, -93.786, -18.214]])

rgb_from_ycbcr = np.linalg.inv(ycbcr_from_rgb)


def rgb2ycbcr(image):
    _type = image.dtype
    _image = image.astype(np.float32)
    if _type != np.uint8:
        _image = _image * 255

    _image = _image @ ycbcr_from_rgb.T / 255

    _image = _image + np.array([[[16, 128, 128]]])

    _image = np.round(_image)
    if _type != np.uint8:
        _image = _image / 255

    return _image.astype(_type)


def ycbcr2rgb(image):
    _type = image.dtype
    _image = image.astype(np.float32)
    if _type != np.uint8:
        _image = _image * 255

    _image = _image - np.array([[[16, 128, 128]]])
    _image = _image @ rgb_from_ycbcr.T * 255

    if _type != np.uint8:
        _image = _image / 255

    return _image.astype(_type)


if __name__ == '__main__':
    import cv2
    import os

    project_root = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))

    _image_ = cv2.imread(os.path.join(project_root, 'experiments/exp3/obama.bmp'))
    rgb = cv2.cvtColor(_image_, cv2.COLOR_BGR2RGB)

    # rgb = rgb.astype(np.float32) / 255
    yuv = rgb2ycbcr(rgb)
    _rgb = ycbcr2rgb(yuv)

    # _rgb = _rgb.astype(np.uint8)

    bgr = cv2.cvtColor(_rgb, cv2.COLOR_RGB2BGR)

    cv2.imshow('asdf', yuv[:, :, 0])
    cv2.imshow('bgr', bgr)

    cv2.waitKey()

