from sklearn.neighbors.ball_tree import BallTree
import os
import numpy
import hashlib
from PIL import Image
import cv2


#########################################
#                                       #
#               Caching                 #
#                                       #
#########################################


class Cache :

    def __init__(self, path, options = {}) :
        """ Caches an image so it's ready for matching """
        # Get relevant options
        force_reload    = options.get("force_reload", False)
        max_size        = options.get("max_size", (4000, 4000))
        metric          = options.get("metric", "minkowski")
        thumb_size      = options.get("thumb_size", (200, 200))

        # check if the path exists
        if not force_reload and self.load(path) : return

        # save path
        self.path = path

        # Create thumbnail
        thumbnail = get_thumbnail(path, thumb_size)

        # Get thumbnail features
        self.thumbnail_keypoints, self.thumbnail_descriptors = get_features(thumbnail)
        self.thumbnail_positions = [k.pt for k in self.thumbnail_keypoints]

        # Get nearest neighbor within image (vector with touples of feature points and distances)
        self.descriptors, self.positions, self.distances = self_match(path, max_size)

        # build position_tree
        self.position_tree = BallTree(positions, metric = metric)



    def get(self, position, radius, options) :
        """ Retrieve all features within radius of position """

        # Get relevant options
        sort_results = options.get("sort_results", True)

        # Fetch all feature points within radius pixels of position
        indices, distances = self.position_tree.query_radius(position,
                                                             r = radius,
                                                             return_distance=True,
                                                             sort_results=sort_results)

        # Return all features and their positions
        return self.descriptors[indices], self.positions[indices], self.distances[indices]


    def save(self, dir = "data/image_data") :
        """ Exports cache to file """

        data_dict = {
            'thumbnail_positions' : self.thumbnail_positions,
            'thumbnail_descriptors' : self.thumbnail_descriptors,
            'descriptors' : self.descriptors,
            'positions' : self.positions,
            'distances' : self.distances
        }

        # Create unique identifier based on image path
        h = hashlib.new('ripemd160')
        h.update(self.path)
        data_path = h.hexdigest()

        # Save data
        numpy.savez("%s/%s" % (dir, data_path), data_dict)

        return data_path



    def load(self, path, dir = "data/image_data") :
        """ Loads file to Cache """

        # get hash for path
        h = hashlib.new('ripemd160')
        h.update(self.path)
        data_path = h.hexdigest()


        # Check if file exists
        full_path = "%s/%s" % (dir, data_path)

        if os.path.isfile(full_path) :
            data = numpy.load(full_path)
            self.thumbnail_positions = data.thumbnail_positions
            self.thumbnail_descriptors = data.thumbnail_descriptors
            self.descriptors = data.descriptors
            self.positions = data.positions
            self.distances = data.distances
            return data

        else :
            return False



#########################################
#                                       #
#                Images                 #
#                                       #
#########################################

class scale :
    """ Class for scaling """
    width = None
    height = None
    img = None

    def __init__(self, path) :
        self.img = self.open(path)
        self.width, self.height = self.get_size()

    def resize(self, size=(200, 200)) :
        if self.width > self.height :
            width = size[0]
            height = int((width / float(self.width)) * self.height)
        else :
            height = size[1]
            width = int((height / float(self.height)) * self.width)
        return (width, height)

class scale_pil_antialias(scale_pil) :
    """ scale and antialias with PIL (fastest for rescaling of big image to thumbnail) """
    def scale(self, size = (200, 200)) :
        new_size = self.resize(size)
        self.img.thumbnail(map(lambda i : i*2, new_size))
        self.img.thumbnail(new_size, Image.ANTIALIAS)
        return self.img


class scale_cv(scale) :
    """ scale with opencv (fastest for rescaling image slightly) """
    def open(self, path) : return cv2.imread(path)
    def get_size(self) : return self.img.shape[:2]
    def scale(self, size = (200, 200)) :
        new_size = self.resize(size)
        return cv2.resize(self.img, new_size, interpolation=cv2.INTER_AREA)


def get_thumbnail(path, size = (200, 200)) :
    """ Get thumbnail with PIL """
    return numpy.array(scale_pil_antialias(path).scale(size), dtype=numpy.uint8)


def open_img(path, size = None) :
    """ If we don't want a thumbnail but might want to rescale image slightly, opencv is faster """
    print(size)
    if size == None :
        return cv2.imread(path)
    else :
        return scale_cv(path).scale(size)


#########################################
#                                       #
#               Matching                #
#                                       #
#########################################


def get_features(data, feature_type = "SIFT") :
    # find the keypoints and descriptors with SIFT
    if feature_type == "SIFT" :
        return cv2.SIFT().detectAndCompute(data, None)


def self_match(path, max_size = (4000, 4000)) :
    """ Match an image with itself finding the closest neighbors within that image """
    # Open image
    img_data = open_img(path, max_size)
    # Get descriptors
    keypoints, descriptors = get_features(img_data)
    # Match
    matches = match(descriptors, descriptors, k=2)
    # Distances and positions
    distances = [r[1].distance for r in matches]
    positions = [k.pt for k in keypoints]
    return descriptors, positions, distances



def match(dt1, dt2, k = 1, options = {}) :
    """ Match two sets of descriptors """

    algorithm = options.get("algorithm", 1)
    trees = options.get("trees", 5)
    checks = options.get("checks", 100)

    # Construct approximate nearest neighbor tree
    # "algorithm"     : {
    #    "linear"    : 0,
    #    "kdtree"    : 1,
    #    "kmeans"    : 2,
    #    "composite" : 3,
    #    "kdtree_simple" : 4,
    #    "saved": 254,
    #    "autotuned" : 255,
    #    "default"   : 1
    #},
    index_params = dict(algorithm = algorithm, trees = trees)
    search_params = dict(checks = checks)
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    # Match features
    return flann.knnMatch(dt1, dt2, k = k)
