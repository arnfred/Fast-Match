"""
Cache for storing data about an image used for matching said image later

Jonas Toft Arnfred, 2013-04-22
"""

####################################
#                                  #
#            Imports               #
#                                  #
####################################

from PIL import Image
from vipsCC.VImage import VImage
import cv2
import numpy

class scale :
    """ Class for scaling """

    def __init__(self, data) :
        # Open image
        if isinstance(data, basestring) :
            self.img = self.open(data)
        else :
            self.img = self.from_array(data)
        # Find height and width
        self.width, self.height = self.get_size()

    # Find resize width and height
    def resize(self, size=(200, 200)) :
        if self.width > self.height :
            width = size[0]
            height = int((width / float(self.width)) * self.height)
        else :
            height = size[1]
            width = int((height / float(self.height)) * self.width)
        return (width, height)


class scale_pil(scale) :
    def open(self, path) : return Image.open(path)
    def from_array(self, data) : return Image.fromarray(data)
    def get_size(self) : return self.img.size
    def scale(self, size = (200, 200)) :
        new_size = self.resize(size)
        self.img.thumbnail(new_size)
        return self.img

class scale_pil_antialias(scale_pil) :
    """ scale and antialias with PIL (fastest for rescaling of big image to thumbnail) """
    def scale(self, size = (200, 200)) :
        new_size = self.resize(size)
        self.img.thumbnail(map(lambda i : i*2, new_size))
        self.img.thumbnail(new_size, Image.ANTIALIAS)
        return self.img

class scale_vips(scale) :
    def open(self, path) : return VImage(path)
    def from_array(self, data) : return data
    def get_size(self) : return (self.img.Xsize(), self.img.Ysize())
    def scale(self, size = (200, 200)) :
        new_size = self.resize(size)
        ratio = float(new_size[0]) / self.width
        scaled = self.img.affinei_all("nearest", ratio, 0, 0, ratio, 0, 0)
        data = scaled.tobuffer()
        return numpy.reshape(numpy.array(data, dtype=numpy.uint8), (scaled.Xsize(), scaled.Ysize(), 3))


class scale_cv(scale) :
    """ scale with opencv (fastest for rescaling image slightly) """
    def open(self, path) : return cv2.imread(path)
    def from_array(self, data) : return data
    def get_size(self) : return self.img.shape[1::-1]
    def scale(self, size = (200, 200)) :
        new_size = self.resize(size)
        return cv2.resize(self.img, new_size, interpolation=cv2.INTER_AREA)


def get_thumbnail(path, size = (200, 200)) :
    """ Get thumbnail with PIL """
    return numpy.array(scale_pil_antialias(path).scale(size), dtype=numpy.uint8)

def get_size(path) :
    return scale_pil(path).get_size()

def open_pil(path) :
    return numpy.array(Image.open(path), dtype=numpy.uint8)

def open_vips(path) :
    img = VImage(path)
    return numpy.reshape(numpy.array(img.tobuffer(), dtype=numpy.uint8), (img.Xsize(), img.Ysize(), 3))

def open_cv(path) :
    return cv2.imread(path)

def open_img(path, size = None) :
    """ If we don't want a thumbnail but might want to rescale image slightly, opencv is faster """
    if size == None or size == -1 :
        return open_cv(path)
    else :
        return scale_cv(path).scale(size)
