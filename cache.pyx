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

    def __init__(self, numpy.ndarray[numpy.uint8_t, ndim=3] data, cell_size, int margin = 25, int factor = 3) :
        cdef int w, h
        # Get rows and cols
        self.width = data.shape[1]
        self.height = data.shape[0]
        self.cell_width = cell_size[0]
        self.cell_height = cell_size[1]
        self.block_width = cell_size[0]/factor
        self.block_height = cell_size[1]/factor
        self.data = data
        self.last_cell = None
        self.last_block = None
        self.margin = margin
        # Initialize grid
        self.grid = {n:{} for n in range(self.height / self.cell_height + 1)}


    cpdef get(self, double x, double y) :
        # Check that pos is within bounds
        cdef int w, h, cell_row, cell_col, block_row, block_col, x_min, x_max, y_min, y_max, m
        cdef numpy.ndarray pos_filter, descriptors, positions # descriptors have a weird type which isn't numpy.float_t
        w, h = self.width, self.height
        if x > w or y > h :
            raise Exception("(%i,%i) is outside data bounds of (%i,%i)" % (x,y, self.width, self.height))
        # Store what cell we are in for logging purposes
        cell_row, cell_col = self.cell(x, y)
        block_row, block_col = self.block(x, y)
        self.last_cell = self.frame(cell_row, cell_col, self.cell_height, self.cell_width)
        self.last_block = self.frame(block_row, block_col, self.block_width, self.block_height)
        (x_min, x_max), (y_min, y_max) = self.last_block
        # Get keypoints and descriptors
        positions, descriptors = self.get_cell(cell_row, cell_col)
        if len(positions) == 0 :
            return positions, descriptors
        # Only return those in the correct block
        m = self.margin
        pos_filter = (positions > [x_min - m, y_min - m]).all(1) & (positions < [x_max + m, y_max + m]).all(1)
        return positions[pos_filter], descriptors[pos_filter]


    cpdef object get_neighbor(self, int row, int col, double pos_x, double pos_y) :
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
        w, h = self.block_width, self.block_height
        x, y = w*col, h*row
        x_diff = int(pos_x) - x
        y_diff = int(pos_y) - y
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

        # Now for each square, specify the appropriate neighbors
        neighbors_square = {
            1 : [4,2,1],
            2 : [1,2,3],
            3 : [2,3,6],
            4 : [1,4,7],
            5 : [],
            6 : [3,6,9],
            7 : [4,7,8],
            8 : [7,8,9],
            9 : [6,8,9]
        }
        square_to_block = {
            1 : (row - 1, col - 1),
            2 : (row - 1, col),
            3 : (row - 1, col + 1),
            4 : (row, col - 1),
            6 : (row, col + 1),
            7 : (row + 1, col - 1),
            8 : (row + 1, col),
            9 : (row + 1, col + 1)
        }

        neighbor_blocks = [square_to_block[b] for b in neighbors_square[square]]
        # Check that neighbors are within image frame
        cols = self.width / self.block_width + 1
        rows = self.height / self.block_height + 1
        neighbors_checked = [(row, col) for (row, col) in neighbor_blocks if (row > 0 and col > 0 and
                                                                        col < cols and row < rows)]
        # For each neighbor map to position
        neighbor_pos = [self.center(row, col) for row,col in neighbors_checked]
        #print("diff: (%i, %i) of block width of (%i, %i) which puts us in square %i\nPositions: %s" % (x_diff, y_diff, self.block_width, self.block_height, square, neighbor_pos))
        return neighbor_pos


    cpdef object block(self, double x, double y) :
        cdef int row, col
        # What grid block are we looking for
        row = int(y / self.block_height)
        col = int(x / self.block_width)
        return row, col

    cdef object cell(self, double x, double y) :
        cdef int row, col
        row = int(y / self.cell_height)
        col = int(x / self.cell_width)
        return row, col

    cdef object frame(self, int row, int col, int width, int height) :
        cdef int x_min, x_max, y_min, y_max
        x_min, x_max = col * width, (col + 1) * width
        y_min, y_max = row * height, (row + 1) * height
        return (x_min, x_max), (y_min, y_max)



    cdef object get_cell(self, int row, int col) :
        if not self.grid[row].get(col, False) :
            # If not, cache it and return
            self.cache(row, col)
        return self.grid[row][col]

    cdef numpy.ndarray[numpy.int_t] center(self, int row, int col) :
        cdef x, y, x_in, y_in
        x = int((col + 0.5) * self.block_width)
        y = int((row + 0.5) * self.block_height)
        x_in = x if x < self.width - 1 else self.width - 1
        y_in = y if y < self.height - 1 else self.height - 1
        return numpy.array((x_in, y_in), dtype=numpy.int)


    cdef object cache(self, int row, int col) :
        cdef int x_min, x_max, y_min, y_max, w, h, rows, cols
        cdef int x_min_margin, x_max_margin, y_min_margin, y_max_margin
        cdef numpy.ndarray[numpy.uint8_t, ndim=3] data_cell
        cdef numpy.ndarray positions
        w, h = self.cell_width, self.cell_height
        # Find area
        cols = int(self.width / self.cell_width) + 1
        rows = int(self.height / self.cell_height) + 1
        x_min = col * w
        x_max = x_min + w if col+1 < cols else self.width
        y_min = row * h
        y_max = y_min + h if row+1 < rows else self.height
        # Find margin
        x_min_margin = self.margin * (col > 0)
        x_max_margin = self.margin if self.width - x_max > self.margin else self.width - x_max
        y_min_margin = self.margin * (row > 0)
        y_max_margin = self.margin if self.height - y_max > self.margin else self.height - y_max
        # Extract data, compute cache value and add it to grid
        data_cell = self.data[y_min-y_min_margin:y_max + y_max_margin, x_min-x_min_margin:x_max+x_max_margin,:]
        # Filter keypoints so those that are in the margin aren't included
        keypoints, descriptors = matchutil.get_features(data_cell)
        if keypoints == None or descriptors == None :
            keypoints = []
            descriptors = []

        positions = numpy.array([[k.pt[0] + x_min - x_min_margin, k.pt[1] + y_min - y_min_margin] for k in keypoints])
        self.grid[row][col] = positions, numpy.array(descriptors, dtype=numpy.uint8)







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
            "descriptors" : numpy.array(descriptors, dtype=numpy.uint8),
            "positions" : positions,
            "distances" : distances,
            "position_tree" : position_tree,
            "size" : (img_data.shape[1], img_data.shape[0])
        }
