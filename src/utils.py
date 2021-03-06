from skimage import transform
from skimage import util
from scipy import signal
import numpy as np


def derivative_x(m):
    return np.append(np.diff(m, 1, axis=1), m[:, [0]] - m[:, [-1]], axis=1)


def derivative_y(m):
    return np.append(np.diff(m, 1, axis=0), m[[0], :] - m[[-1], :], axis=0)


def derivative_xy(x, y):
    _dxy = np.append(x[:, [-1]] - x[:, [0]], -np.diff(x, 1, axis=1), axis=1)
    _dxy += np.append(y[[-1], :] - y[[0], :], -np.diff(y, 1, axis=0), axis=0)
    return _dxy


def zero_pad(image, shape, position='corner'):
    """
    Extends image to a certain size with zeros
    Parameters
    ----------
    image: real 2d `numpy.ndarray`
        Input image
    shape: tuple of int
        Desired output shape of the image
    position : str, optional
        The position of the input image in the output one:
            * 'corner'
                top-left corner (default)
            * 'center'
                centered
    Returns
    -------
    padded_img: real `numpy.ndarray`
        The zero-padded image
    """
    shape = np.asarray(shape, dtype=int)
    imshape = np.asarray(image.shape, dtype=int)

    if np.alltrue(imshape == shape):
        return image

    if np.any(shape <= 0):
        raise ValueError("ZERO_PAD: null or negative shape given")

    dshape = shape - imshape
    if np.any(dshape < 0):
        raise ValueError("ZERO_PAD: target size smaller than source one")

    pad_img = np.zeros(shape, dtype=image.dtype)

    idx, idy = np.indices(imshape)

    if position == 'center':
        if np.any(dshape % 2 != 0):
            raise ValueError("ZERO_PAD: source and target shapes "
                             "have different parity.")
        offx, offy = dshape // 2
    else:
        offx, offy = (0, 0)

    pad_img[idx + offx, idy + offy] = image

    return pad_img


def psf2otf(psf, shape):
    """
    Convert point-spread function to optical transfer function.
    Compute the Fast Fourier Transform (FFT) of the point-spread
    function (PSF) array and creates the optical transfer function (OTF)
    array that is not influenced by the PSF off-centering.
    By default, the OTF array is the same size as the PSF array.
    To ensure that the OTF is not altered due to PSF off-centering, PSF2OTF
    post-pads the PSF array (down or to the right) with zeros to match
    dimensions specified in OUTSIZE, then circularly shifts the values of
    the PSF array up (or to the left) until the central pixel reaches (1,1)
    position.
    Parameters
    ----------
    psf : `numpy.ndarray`
        PSF array
    shape : int
        Output shape of the OTF array
    Returns
    -------
    otf : `numpy.ndarray`
        OTF array
    Notes
    -----
    Adapted from MATLAB psf2otf function
    """
    if np.all(psf == 0):
        return np.zeros_like(psf)

    in_shape = psf.shape
    # Pad the PSF to outsize
    psf = zero_pad(psf, shape, position='corner')

    # Circularly shift OTF so that the 'center' of the PSF is
    # [0,0] element of the array
    for axis, axis_size in enumerate(in_shape):
        psf = np.roll(psf, -int(axis_size / 2), axis=axis)

    # Compute the OTF
    otf = np.fft.fft2(psf)

    # Estimate the rough number of operations involved in the FFT
    # and discard the PSF imaginary part if within roundoff error
    # roundoff error  = machine epsilon = sys.float_info.epsilon
    # or np.finfo().eps
    n_ops = np.sum(psf.size * np.log2(psf.shape))
    otf = np.real_if_close(otf, tol=n_ops)

    return otf


def get_c(m):
    c = {
        'dxTdx': np.power(np.abs(psf2otf(np.array([[1, -1]]), m.shape)), 2),
        'dyTdy': np.power(np.abs(psf2otf(np.array([[1], [-1]]), m.shape)), 2)
    }
    return c


def synthetic_rain(shape, density=0.05, ksize=50, angle=190):
    rain = np.zeros(shape)
    rain = util.random_noise(rain, mode='s&p', amount=density)

    motion_blur_kernel = np.zeros((ksize, ksize))
    motion_blur_kernel[:, int((ksize - 1) / 2)] = np.ones(ksize)
    motion_blur_kernel = motion_blur_kernel / ksize

    motion_blur_kernel = transform.rotate(motion_blur_kernel, angle, resize=True)

    rain = signal.convolve2d(rain, motion_blur_kernel, mode='same')

    return rain
