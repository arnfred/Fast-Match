# encoding: utf-8
# cython: profile=True
# filename: cache.pyx
"""
Cache for storing data about an image used for matching said image later

Jonas Toft Arnfred, 2013-04-22
"""

####################################
#                                  #
#            Imports               #
#                                  #
####################################

from sklearn.neighbors.ball_tree import BallTree
import pickle
import os
import numpy
import hashlib
import imaging
import matchutil
cimport numpy

#########################################
#                                       #
#              Grid Cache               #
#                                       #
#########################################

cdef class Grid_Cache :

    def __init__(self, numpy.ndarray[numpy.uint8_t, ndim=3] data, cell_size, caching_function = None, int margin = 25) :
        cdef int w, h
        # Get rows and cols
        self.width = data.shape[1]
        self.height = data.shape[0]
        self.cell_width = cell_size[0]
        self.cell_height = cell_size[1]
        w, h = self.width, self.height
        self.rows = int(w / cell_size[0]) + 1
        self.cols = int(h / cell_size[1]) + 1
        self.data = data
        self.fun = caching_function
        self.last = None
        self.margin = margin

        # Initialize grid
        self.grid = {n:{} for n in range(self.cols)}

    cpdef get(self, double x, double y) :
        # Check that pos is within bounds
        cdef int w, h, col, row

        w, h = self.width, self.height
        if x > w or y > h :
            raise Exception("(%i,%i) is outside data bounds of (%i,%i)" % (x,y, self.width, self.height))
        # What grid block are we looking for
        col, row = self.block(x, y)
        # has this grid cell been cached?
        return self.get_cell(col, row)


    cpdef offset(self, double x, double y) :
        cdef int col, row, x_min, y_min
        col, row = self.block(x, y)
        x_min = row * self.cell_width - self.margin
        y_min = col * self.cell_height - self.margin
        return (x_min, y_min)


    cpdef numpy.ndarray[numpy.int_t] get_neighbor(self, int col, int row, double pos_x, double pos_y) :
        """ Returns the position of the neighboring square in the image.
        The neighboring square is selected based on the border which the
        position is closest to in the current image """
        cdef int x, y, x_diff, y_diff, c_x, c_y
        cdef numpy.ndarray[numpy.int_t] n_pos, pos_error
        pos_error = numpy.ones(2, dtype=numpy.int) * -1
        # Find center of block of pos
        x, y = self.center(col, row)
        # Now find where we are relating to center
        x_diff = int(pos_x) - x # negative if left of center, positive otherwise
        y_diff = int(pos_y) - y # negative if above center, positive otherwise
        if y_diff < x_diff and y_diff < -1*x_diff :
            n_pos = self.center(col - 1, row) if col - 1 >= 0 else pos_error
        elif x_diff > y_diff :
            n_pos = self.center(col, row + 1) if row + 1 < self.rows else pos_error
        elif y_diff > -1*x_diff :
            n_pos = self.center(col + 1, row) if col + 1 < self.cols else pos_error
        else :
            n_pos = self.center(col, row - 1) if row - 1 >= 0 else pos_error
        return n_pos


    cpdef object block(self, double x, double y) :
        # What grid block are we looking for
        row = int(x / self.cell_width)
        col = int(y / self.cell_height)
        return col, row


    cdef object get_cell(self, int col, int row) :
        if not self.grid[col].get(row, False) :
            # If not, cache it and return
            self.last = self.cache(col, row)
        return self.grid[col][row]

    cdef bint is_cached(self, double x, double y) :
        cdef int col, row
        col, row = self.block(x, y)
        if self.grid[col].get(row, False) :
            return True
        else :
            return False

    cdef numpy.ndarray[numpy.int_t] center(self, int col, int row) :
        x = int((row + 0.5) * self.cell_width)
        y = int((col + 0.5) * self.cell_height)
        x_in = x if x < self.width - 1 else self.width - 1
        y_in = y if y < self.height - 1 else self.height - 1
        return numpy.array((x_in, y_in), dtype=numpy.int)


    cdef object cache(self, int col, int row) :
        cdef int x_min, x_max, y_min, y_max
        cdef numpy.ndarray[numpy.uint8_t, ndim=3] data_cell
        # Find area
        x_min = row * self.cell_width - (self.margin * (row > 0))
        x_max = x_min + self.cell_width + self.margin * 2 if row+1 < self.rows else self.width
        y_min = col * self.cell_height - (self.margin * (col > 0))
        y_max = y_min + self.cell_height + self.margin * 2 if col+1 < self.cols else self.height
        # Extract data, compute cache value and add it to grid
        data_cell = self.data[y_min:y_max, x_min:x_max,:]
        if self.fun == None :
            self.grid[col][row] = data_cell
        else :
            self.grid[col][row] = self.fun(data_cell)
        return ((x_min, x_max), (y_min, y_max))






#########################################
#                                       #
#             Metric Cache              #
#                                       #
#########################################

cdef class Metric_Cache :

    def __init__(self, char* path, options = {}) :
        """ Caches an image so it's ready for matching """
        # Get relevant options
        cdef int thumb_x, thumb_y
        cdef bint force_reload     = options.get("force_reload", False)
        cdef int max_size          = options.get("max_size", -1)
        cdef object metric         = options.get("metric", "minkowski")
        thumb_x, thumb_y           = options.get("thumb_size", (600, 600))
        # save path and init attributes
        self.path = path
        self.thumb = {}
        self.original = {}
        # check if the path exists
        if not force_reload and self.load() : return
        # Create thumbnail and image
        self.create_thumbnail(path, thumb_x, thumb_y)
        self.create_image(path, max_size, metric)
        self.save()


    def get(self, int x, int y, int radius, object options = {}) :
        """ Retrieve all features within radius of position """
        # Get relevant options and position tree
        cdef numpy.ndarray[object] indices
        cdef numpy.ndarray[numpy.long_t] idx
        cdef numpy.ndarray[object, ndim=1] distances
        cdef bint sort_results = options.get("sort_results", True)
        cdef object pos_tree = self.original["position_tree"]
        # Fetch all feature points within radius pixels of position
        indices, distances = pos_tree.query_radius(numpy.array((x,y)),
                                                   r = radius,
                                                   return_distance=True,
                                                   sort_results=sort_results)
        idx = indices[0]
        # Return all features and their positions
        return self.original["descriptors"][idx], self.original["positions"][idx], self.original["distances"][idx], idx


    def save(self, char* dir = "data/image_data") :
        """ Exports cache to file """
        # Create unique identifier based on image path
        cdef object h = hashlib.new('ripemd160')
        h.update(self.path)
        cdef object data_path = h.hexdigest()
        # Save data
        numpy.savez("%s/%s" % (dir, data_path),
                    descriptors = self.original["descriptors"],
                    positions = self.original["positions"],
                    distances = self.original["distances"],
                    position_tree = pickle.dumps(self.original["position_tree"]),
                    size = self.original["size"])
        numpy.savez("%s/%s_thumb" % (dir, data_path),
                    positions = self.thumb["positions"],
                    descriptors = self.thumb["descriptors"],
                    distances = self.thumb["distances"],
                    size = self.thumb["size"])
        return data_path


    cdef object load(self, char* dir = "data/image_data") :
        """ Loads file to Cache """
        # get hash for path
        cdef object data, data_thumb
        cdef object h = hashlib.new('ripemd160')
        h.update(self.path)
        cdef object data_path = h.hexdigest()
        # Check if file exists
        cdef object full_path_npz = "%s/%s.npz" % (dir, data_path)
        cdef object full_path_thumb = "%s/%s_thumb.npz" % (dir, data_path)
        # Check if path exists
        if os.path.isfile(full_path_npz) :
            data = numpy.load(full_path_npz)
            data_thumb = numpy.load(full_path_thumb)
        else :
            return False
        self.thumb["positions"] = data_thumb['positions']
        self.thumb["descriptors"] = data_thumb['descriptors']
        self.thumb["distances"] = data_thumb['distances']
        self.thumb["size"] = data_thumb['size']
        self.original["descriptors"] = data['descriptors']
        self.original["positions"] = data['positions']
        self.original["distances"] = data['distances']
        self.original["position_tree"] = pickle.loads(data['position_tree'])
        self.original["size"] = data['size']
        return data, data_thumb


    cdef create_thumbnail(self, char* path, int thumb_x, int thumb_y) :
        """ Get relevant data for thumbnail """
        cdef numpy.ndarray[numpy.uint8_t, ndim=3] thumbnail
        # Create thumbnail
        thumbnail = imaging.get_thumbnail(path, (thumb_x, thumb_y))
        # Get thumbnail features
        keypoints, descriptors = matchutil.get_features(thumbnail)
        # Get nearest neighbor within image (vector with touples of feature points and distances)
        matches = matchutil.bf_match(descriptors, descriptors, k = 2)
        # Distances to nearest neighbor and positions
        nn_distances = numpy.array([r[1].distance for r in matches])
        positions = numpy.array([k.pt for k in keypoints])
        # Collect data
        self.thumb = {
            "descriptors" : descriptors,
            "positions" : positions,
            "distances" : nn_distances,
            "size" : (thumbnail.shape[1], thumbnail.shape[0])
        }


    cdef create_image(self, char* path, int max_size, char* metric) :
        """ Match an image with itself finding the closest neighbors within that image """
        cdef numpy.ndarray[numpy.uint8_t, ndim=3] img_data
        # Open image
        img_data = imaging.open_img(path, max_size)
        # Get descriptors
        keypoints, descriptors = matchutil.get_features(img_data)
        # Match
        matches = matchutil.flann_match(descriptors, descriptors, k=2)
        # Distances and positions
        distances = numpy.array([r[1].distance for r in matches])
        positions = numpy.array([k.pt for k in keypoints])
        # build position_tree
        position_tree = BallTree(positions, metric = metric)
        # Collect data
        self.original = {
            "descriptors" : descriptors,
            "positions" : positions,
            "distances" : distances,
            "position_tree" : position_tree,
            "size" : (img_data.shape[1], img_data.shape[0])
        }
