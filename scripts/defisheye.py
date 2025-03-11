import os
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms as T
from numpy import ndarray, hypot, arctan, sin, tan, sqrt, arange, meshgrid, pi, zeros_like

class Defisheye:
    """
    Defisheye

    fov: fisheye field of view (aperture) in degrees
    pfov: perspective field of view (aperture) in degrees
    xcenter: x center of fisheye area
    ycenter: y center of fisheye area
    radius: radius of fisheye area
    angle: image rotation in degrees clockwise
    dtype: linear, equalarea, orthographic, stereographic
    format: circular, fullframe
    """
    def __init__(self, infile, **kwargs):
        vkwargs = {"fov": 180,
                   "pfov": 120,
                   "xcenter": None,
                   "ycenter": None,
                   "radius": None,
                   "angle": 0,
                   "dtype": "equalarea",
                   "format": "fullframe"
                   }
        self._start_att(vkwargs, kwargs)

        if type(infile) == str:
            _image = cv2.imread(infile)
        elif type(infile) == ndarray:
            _image = infile
        else:
            raise NameError("Image format not recognized")

        width = _image.shape[1]
        height = _image.shape[0]
        xcenter = width // 2
        ycenter = height // 2

        dim = min(width, height)
        x0 = xcenter - dim // 2
        xf = xcenter + dim // 2
        y0 = ycenter - dim // 2
        yf = ycenter + dim // 2

        self._image = _image[y0:yf, x0:xf, :]

        self._width = self._image.shape[1]
        self._height = self._image.shape[0]

        if self._xcenter is None:
            self._xcenter = (self._width - 1) // 2

        if self._ycenter is None:
            self._ycenter = (self._height - 1) // 2

    def _map(self, i, j, ofocinv, dim):

        xd = i - self._xcenter
        yd = j - self._ycenter

        rd = hypot(xd, yd)
        phiang = arctan(ofocinv * rd)

        if self._dtype == "linear":
            ifoc = dim * 180 / (self._fov * pi)
            rr = ifoc * phiang
            # rr = "rr={}*phiang;".format(ifoc)

        elif self._dtype == "equalarea":
            ifoc = dim / (2.0 * sin(self._fov * pi / 720))
            rr = ifoc * sin(phiang / 2)
            # rr = "rr={}*sin(phiang/2);".format(ifoc)

        elif self._dtype == "orthographic":
            ifoc = dim / (2.0 * sin(self._fov * pi / 360))
            rr = ifoc * sin(phiang)
            # rr="rr={}*sin(phiang);".format(ifoc)

        elif self._dtype == "stereographic":
            ifoc = dim / (2.0 * tan(self._fov * pi / 720))
            rr = ifoc * tan(phiang / 2)

        rdmask = rd != 0
        xs = xd.copy()
        ys = yd.copy()

        tmp = rr[rdmask] / rd[rdmask]

        xs[rdmask] = tmp * xd[rdmask] + self._xcenter
        ys[rdmask] = tmp * yd[rdmask] + self._ycenter

        xs[~rdmask] = 0
        ys[~rdmask] = 0

        xs = xs.astype(int)
        ys = ys.astype(int)
        return xs, ys

    def convert(self, circle=False):
        if self._format == "circular":
            dim = min(self._width, self._height)
        elif self._format == "fullframe":
            dim = sqrt(self._width ** 2.0 + self._height ** 2.0)

        if self._radius is not None:
            dim = 2 * self._radius

        ofoc = dim / (2 * tan(self._pfov * pi / 360))
        ofocinv = 1.0 / ofoc

        i = arange(self._width)
        j = arange(self._height)
        i, j = meshgrid(i, j)

        xs, ys, = self._map(i, j, ofocinv, dim)
        img = self._image.copy()

        img[i, j, :] = self._image[xs, ys, :]

        if circle is True:
            center = (self._width // 2, self._height // 2)
            radius = min(self._width // 2, self._height // 2)
            color = (255, 255, 255)
            thickness = -1
            alpha = cv2.circle(zeros_like(img),
                               center, radius, color, thickness)
            img[alpha == 0] = 0

        #cv2.imwrite(outfile, img)
        return img
    
    def _start_att(self, vkwargs, kwargs):
        """
        Starting atributes
        """
        pin = []

        for key, value in kwargs.items():
            if key not in vkwargs:
                raise NameError("Invalid key {}".format(key))
            else:
                pin.append(key)
                setattr(self, "_{}".format(key), value)

        pin = set(pin)
        rkeys = set(vkwargs.keys()) - pin
        for key in rkeys:
            setattr(self, "_{}".format(key), vkwargs[key])

def circular_mask(img):
    if len(img.shape) == 3:
        height, width, depth = img.shape
    else:
        height, width = img.shape
    circle_img = np.zeros((height, width), np.uint8)
    mask = cv2.circle(circle_img, (int(width / 2), int(height / 2)), int(width / 2), 1, thickness=-1)
    circular_masked = cv2.bitwise_and(img, img, mask=circle_img)
    return circular_masked

def rgb2gray(rgb):
    (R, G, B) = cv2.split(rgb)
    gray = 0.298*R + 0.587*G + 0.114*B
    gray8 = gray.astype(np.uint8)
    return gray8

def sun_mask(rgb, gray):
    th, W1 = cv2.threshold(gray, 250, 255, cv2.THRESH_BINARY); 
    W1 = cv2.erode(W1, None, iterations=6)
    W1 = cv2.dilate(W1, None, iterations=10)
    contours_1, hierarchy = cv2.findContours(W1, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)
    contours_1 = max(contours_1, key = cv2.contourArea)
    image_copy_1 = np.zeros(W1.shape)
    W1 = cv2.drawContours(image=image_copy_1, contours=contours_1, contourIdx=-1, color=(255, 255, 255), thickness=20, lineType=cv2.LINE_AA)
    (x, y), radius = cv2.minEnclosingCircle(contours_1)
    center = (int(x), int(y))
    radius = int(radius)
    sun_masked = cv2.circle(rgb, center, int(radius*3), (0, 0, 0), -1)
    return sun_masked

def rgb2rbr(rgb):
    (R, G, B) = cv2.split(rgb)
    ones = np.ones(rgb.shape[:2])
    B = ones + B
    rbr = (R/B)
    return rbr

def cloud_mask(rbr_img):
    retval, T1 = cv2.threshold(rbr_img, 0.87, 256, cv2.THRESH_BINARY)
    return T1 

def fusion(high,low):
    img_fn = [high, low]
    img_list = [cv2.imread(fn) for fn in img_fn]
    img_list = [i[60:710, 180:830] for i in img_list]
    img_list = [circular_mask(i) for i in img_list]
    img_rgb = [cv2.cvtColor(i, cv2.COLOR_RGB2BGR) for i in img_list]
    merge_mertens = cv2.createMergeMertens()
    res_mertens = merge_mertens.process(img_list)
    res_mertens_8bit = np.clip(res_mertens*255, 0, 255).astype('uint8'); 
    fusion = cv2.cvtColor(res_mertens_8bit, cv2.COLOR_RGB2BGR)
    return fusion

def cloudfind(img1,img2):
    rgb = fusion(img1,img2)
    gray = rgb2gray(rgb)
    sun = sun_mask(rgb, gray)
    rbr = rgb2rbr(sun)
    clouds = cloud_mask(rbr)
    clouds_m = circular_mask(clouds)
    return rgb, clouds_m


def plot(*images, titles=None):
    num_images = len(images)

    # Adjust the layout based on the number of images
    if num_images == 1:
        fig, ax = plt.subplots(1, num_images, figsize=(10, 10))
        axes = [ax]  # Make it a list so the enumeration works
    else:
        fig, axes = plt.subplots(1, num_images, figsize=(10, 10))

    fig.tight_layout()

    for i, image in enumerate(images):
        ax = axes[i]
        ax.imshow(image)
        ax.axis('off')
        if titles is not None and len(titles) > i:
            ax.set_title(titles[i])

    plt.show()

# Assuming the Defisheye class definition is available as provided earlier

def defisheye(image, image_type):
    """
    Apply defisheye correction to an image based on its type.

    :param image_path: Path to the image file.
    :param image_type: A string indicating the type/location of the image.
    :return: The defisheye-corrected image as a numpy array.
    """
    # Read and convert the image

    # Set default parameters for defisheye process
    dtype = 'orthographic'
    format = 'circular'
    fov = 180
    pfov = 120
    crop_size = None

    # Adjust parameters based on image type/location
    if image_type == 'W1':
        crop_size = 2200
        flip = True
    elif image_type == 'SIRTA':
        crop_size = 600
        flip = True
    elif image_type == 'LMU':
        crop_size = 850
        flip = False
    else:
        raise ValueError("Unknown image type/location")

    # Process the image with Defisheye
    defisheye_obj = Defisheye(image, dtype=dtype, format=format, fov=fov, pfov=pfov)
    undis_image = defisheye_obj.convert()

    # Convert to PIL for cropping and flipping
    undis_image_arr = Image.fromarray(undis_image)
    final_undis = T.CenterCrop(crop_size)(undis_image_arr)
    if flip:
        flipped_image = final_undis.transpose(Image.FLIP_LEFT_RIGHT)
    else:
        flipped_image = final_undis
    # Apply the circular mask and convert back to numpy array
    corrected_image = circular_mask(np.array(flipped_image))

    return corrected_image