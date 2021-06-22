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

    _image = _image @ ycbcr_from_rgb.T

    if _type != np.uint8:
        _image = _image / 255

    return _image


if __name__ == '__main__':
    print(ycbcr_from_rgb)
    print(ycbcr_from_rgb.T)
