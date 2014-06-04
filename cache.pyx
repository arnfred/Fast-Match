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
import cv2
cimport numpy
cimport cython

cdef class Size :
    def __init__(self, w, h) :
        self.w = w
        self.h = h

cdef Pos fromArray_fun(numpy.ndarray[numpy.double_t] pos) :
    return Pos(pos[0], pos[1])

cdef class Pos :
    def __init__(self, x, y) :
        self.x = x
        self.y = y

cdef class Pos_Pair :
    def __init__(self, q, t) :
        self.query = q
        self.target = t

cdef class Frame :
    def __init__(self, x_min, x_max, y_min, y_max) :
        self.x_min = x_min
        self.y_min = y_min
        self.x_max = x_max
        self.y_max = y_max

    def get_x_max(self) :
        return self.x_max
    def get_y_min(self) :
        return self.y_min
    def get_x_min(self) :
        return self.x_min
    def get_y_max(self) :
        return self.y_max

#########################################
#                                       #
#              cells Cache               #
#                                       #
#########################################

cdef class Grid_Cache :

    def __init__(self, numpy.ndarray[numpy.uint8_t, ndim=3] data, cell_size, int margin = 25, int factor = 3) :
        cdef int w, h
        # Get rows and cols
        self.img_size = Size(data.shape[1], data.shape[0])
        self.cell_size = Size(cell_size[0], cell_size[1])
        self.block_size = Size(cell_size[0]/factor, cell_size[1]/factor)
        self.data = data
        self.last_cell = None
        self.last_block = None
        self.margin = margin
        # Initialize cells
        self.cells = {n:{} for n in range(self.img_size.h / self.cell_size.h + 1)}
        self.blocks = {n:{} for n in range(self.img_size.h / self.block_size.h + 1)}


    #@cython.profile(False)
    cpdef get(self, Pos p) :
        cdef int w, h, cell_row, cell_col, block_row, block_col, x_min, x_max, y_min, y_max, m, x, y
        # Check that pos is within bounds
        w, h = self.img_size.w, self.img_size.h


        if p.x > w or p.y > h :
            raise Exception("(%i,%i) is outside data bounds of (%i,%i)" % (p.x,p.y, w, h))
        # Store what cell we are in for logging purposes
        cell_row, cell_col = self.cell(p)
        block_row, block_col = self.block(p)
        # Check if we already have the result
        if not self.blocks[block_row].get(block_col, False) :
            self.cache_block(cell_row, cell_col, block_row, block_col)
        return self.blocks[block_row][block_col]


    cdef cache_block(self, int cell_row, int cell_col, int block_row, int block_col) :
        cdef int x_min, x_max, y_min, y_max, m
        cdef numpy.ndarray pos_filter
        cdef numpy.ndarray[numpy.uint8_t, ndim=2] descriptors
        cdef numpy.ndarray[numpy.double_t, ndim=2] positions

        self.last_cell = self.frame(cell_row, cell_col, self.cell_size)
        self.last_block = self.frame(block_row, block_col, self.block_size)
        x_min = self.last_block.x_min
        x_max = self.last_block.x_max
        y_min = self.last_block.y_min
        y_max = self.last_block.y_max
        # Get keypoints and descriptors
        positions, descriptors = self.get_cell_data(cell_row, cell_col)
        # Check if we have any positions (This check is needed)
        if len(positions) == 0 or (len(positions) == 1 and len(positions[0]) == 0) :
            self.blocks[block_row][block_col] = positions.reshape((1,0)), descriptors.reshape((1,0))
            return
        # If we have, filter them and check again
        m = self.margin
        x = (x_min - m, x_max + m)
        y = (y_min - m, y_max + m)
        pos_filter = (positions > [x_min - m, y_min - m]).all(1) & (positions < [x_max + m, y_max + m]).all(1)

        if len(pos_filter) == 0 :
            self.blocks[block_row][block_col] = positions.reshape((1,0)), descriptors.reshape((1,0))
            return
        # Only return those in the correct block
        self.blocks[block_row][block_col] = numpy.array(positions[pos_filter], dtype=numpy.double), descriptors[pos_filter]


    cpdef object get_neighbor(self, int row, int col, Pos p) :
        """ Returns the position of the neighboring square in the image.
        The neighboring square is selected based on the border which the
        position is closest to in the current image
        +---+---+---+
        | 1 | 2 | 3 |
        +---+---+---+
        | 4 | 5 | 6 |
        +---+---+---+
        | 7 | 8 | 9 |
        +---+---+---+
        """
        cdef int x, y, x_diff, y_diff, c_x, c_y, w, h, square, rows, cols
        w, h = self.block_size.w, self.block_size.h
        x, y = w*col, h*row
        x_diff = int(p.x) - x
        y_diff = int(p.y) - y
        # Are we in square 1,2 or 3?
        if y_diff < h / 3 :
            if x_diff < w / 3 :
                square = 1
            elif x_diff < (w / 3) * 2 :
                square = 2
            else :
                square = 3
        # Are we in square 4,5 or 6?
        elif y_diff < (h / 3) * 2 :
            if x_diff < w / 3 :
                square = 4
            elif x_diff < (w / 3) * 2 :
                square = 5
            else :
                square = 6
        # Must be in square 7,8 or 9?
        else :
            if x_diff < w / 3 :
                square = 7
            elif x_diff < (w / 3) * 2 :
                square = 8
            else :
                square = 9

        neighbor_blocks = [self.square_to_block(b, row, col) for b in self.neighbors_square(square)]
        # Check that neighbors are within image frame
        cols = self.img_size.w / self.block_size.w + 1
        rows = self.img_size.h / self.block_size.h + 1
        neighbors_checked = [(row, col) for (row, col) in neighbor_blocks if (row > 0 and col > 0 and
                                                                              col < cols and row < rows)]
        # For each neighbor map to position
        neighbor_pos = [self.center(row, col) for row, col in neighbors_checked]
        return neighbor_pos

    cdef square_to_block(self, int n, int row, int col) :
        if n == 1 : return (row - 1, col - 1)
        if n == 2 : return (row - 1, col)
        if n == 3 : return (row - 1, col + 1)
        if n == 4 : return (row, col - 1)
        if n == 6 : return (row, col + 1)
        if n == 7 : return (row + 1, col - 1)
        if n == 8 : return (row + 1, col)
        if n == 9 : return (row + 1, col + 1)

    cdef neighbors_square(self, int n) :
        if n == 1 : return [4,2,1]
        if n == 2 : return [1,2,3]
        if n == 3 : return [2,3,6]
        if n == 4 : return [1,4,7]
        if n == 5 : return []
        if n == 6 : return [3,6,9]
        if n == 7 : return [4,7,8]
        if n == 8 : return [7,8,9]
        if n == 9 : return [6,8,9]

    #@cython.profile(False)
    cdef object block(self, Pos p) :
        return (int(p.y) / self.block_size.h, int(p.x) / self.block_size.w)

    #@cp.ython.profile(False)
    cdef object cell(self, Pos p) :
        return (int(p.y / self.cell_size.h), int(p.x / self.cell_size.w))


    cdef Frame frame(self, int row, int col, Size size) :
        cdef int x_min, x_max, y_min, y_max
        width, height = size.w, size.h
        x_min = col * width
        x_max = (col + 1) * width
        y_min = row * height
        y_max = (row + 1) * height
        return Frame(x_min, x_max, y_min, y_max)


    cdef object get_cell_data(self, int row, int col) :
        if not self.cells[row].get(col, False) :
            # If not, cache it and return
            self.cache(row, col)
        return self.cells[row][col]

    cdef Pos center(self, int row, int col) :
        cdef x, y, x_in, y_in
        x = int((col + 0.5) * self.block_size.w)
        y = int((row + 0.5) * self.block_size.h)
        x_in = x if x < self.img_size.w - 1 else self.img_size.w - 1
        y_in = y if y < self.img_size.h - 1 else self.img_size.h - 1
        return Pos(x_in, y_in)


    cdef object cache(self, int row, int col) :
        cdef int x_min, x_max, y_min, y_max, w, h, rows, cols
        cdef int x_min_margin, x_max_margin, y_min_margin, y_max_margin
        cdef numpy.ndarray[numpy.uint8_t, ndim=3] data_cell
        cdef numpy.ndarray positions
        w, h = self.cell_size.w, self.cell_size.h
        # Find area
        cols = int(self.img_size.w / self.cell_size.w) + 1
        rows = int(self.img_size.h / self.cell_size.h) + 1
        x_min = col * w
        x_max = x_min + w if col+1 < cols else self.img_size.w
        y_min = row * h
        y_max = y_min + h if row+1 < rows else self.img_size.h
        # Find margin
        x_min_margin = self.margin * (col > 0)
        x_max_margin = self.margin if self.img_size.w - x_max > self.margin else self.img_size.w - x_max
        y_min_margin = self.margin * (row > 0)
        y_max_margin = self.margin if self.img_size.h - y_max > self.margin else self.img_size.h - y_max
        # Extract data, compute cache value and add it to cells
        data_cell = self.data[y_min-y_min_margin:y_max + y_max_margin, x_min-x_min_margin:x_max+x_max_margin,:]
        # Filter keypoints so those that are in the margin aren't included
        keypoints, descriptors = matchutil.get_features(data_cell)
        if keypoints == None or len(keypoints) == 0 :
            self.cells[row][col] = numpy.empty((0,0)), numpy.empty((0,0), dtype=numpy.uint8)
            return

        positions = numpy.array([[k.pt[0] + x_min - x_min_margin, k.pt[1] + y_min - y_min_margin] for k in keypoints])
        self.cells[row][col] = positions, numpy.array(descriptors, dtype=numpy.uint8)







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


    cdef object get(self, int x, int y, int radius, object options = {}) :
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
        idx = numpy.array(indices[0], dtype=numpy.int)
        # Return all features and their positions
        return self.original["descriptors"][idx], self.original["positions"][idx], self.original["distances"][idx]


    cdef object save(self, char* dir = "data/image_data") :
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
            "descriptors" : numpy.array(descriptors, dtype=numpy.uint8),
            "positions" : positions,
            "distances" : nn_distances,
            "size" : (thumbnail.shape[1], thumbnail.shape[0])
        }


    cdef create_image(self, char* path, int max_size, char* metric) :
        """ Match an image with itself finding the closest neighbors within that image """
        cdef numpy.ndarray[numpy.uint8_t, ndim=3] img_data
        cdef numpy.ndarray[numpy.int_t, ndim=1] distances
        cdef numpy.ndarray[numpy.double_t, ndim=2] positions
        # Open image
        img_data = imaging.open_img(path, max_size)
        # Get descriptors
        keypoints, descriptors = matchutil.get_features(img_data)
        # Match
        matches = matchutil.flann_match(descriptors, descriptors, k=2)
        # Distances and positions
        distances = numpy.array([r[1].distance for r in matches], dtype=numpy.int)
        positions = numpy.array([k.pt for k in keypoints], dtype=numpy.double)
        # build position_tree
        position_tree = BallTree(positions, metric = metric)
        # Collect data
        self.original = {
            "descriptors" : numpy.array(descriptors, dtype=numpy.uint8),
            "positions" : positions,
            "distances" : distances,
            "position_tree" : position_tree,
            "size" : (img_data.shape[1], img_data.shape[0])
        }





#########################################
#                                       #
#            Feature Cache              #
#                                       #
#########################################

cdef class Feature_Cache :

    def __init__(self, char* path, options = {}) :
        """ Caches an image so it's ready for matching """
        # Get relevant options
        cdef bint force_reload     = options.get("force_reload", False)
        cdef int max_size          = options.get("max_size", -1)
        cdef object metric         = options.get("metric", "minkowski")
        # save path and init attributes
        self.path = path
        self.image = {}
        # check if the path exists
        if not force_reload and self.load() : return
        # Create thumbnail and image
        self.create_image(path, max_size, metric)
        self.save()


    cdef object get(self, numpy.ndarray[numpy.uint8_t, ndim=2] descriptors, int k) :
        """ Retrieve all features within radius of position """
        # Get relevant options and position tree
        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck = False)
        matches = bf.knnMatch(descriptors, self.image["descriptors"], k = k)
        cdef int index = k - 1
        return [m[index].distance for m in matches]


    cdef object save(self, char* dir = "data/image_data") :
        """ Exports cache to file """
        # Create unique identifier based on image path
        cdef object h = hashlib.new('ripemd160')
        h.update(self.path)
        cdef object data_path = h.hexdigest()
        # Save data
        numpy.savez("%s/%s_features" % (dir, data_path),
                    descriptors = self.image["descriptors"],
                    size = self.image["size"])
        return data_path


    cdef object load(self, char* dir = "data/image_data") :
        """ Loads file to Cache """
        # get hash for path
        cdef object data
        cdef object h = hashlib.new('ripemd160')
        h.update(self.path)
        cdef object data_path = h.hexdigest()
        # Check if file exists
        cdef object path_npz = "%s/%s_features.npz" % (dir, data_path)
        # Check if path exists
        if os.path.isfile(path_npz) :
            data = numpy.load(path_npz)
        else :
            return False
        self.image["descriptors"] = data['descriptors']
        self.image["size"] = data['size']
        return data


    cdef create_image(self, char* path, int max_size, char* metric) :
        """ Match an image with itself finding the closest neighbors within that image """
        cdef numpy.ndarray[numpy.uint8_t, ndim=3] img_data
        # Open image
        img_data = imaging.open_img(path, max_size)
        # Get descriptors
        keypoints, descriptors = matchutil.get_features(img_data)
        # Collect data
        self.original = {
            "descriptors" : numpy.array(descriptors, dtype=numpy.uint8),
            "size" : (img_data.shape[1], img_data.shape[0])
        }
