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



#########################################
#                                       #
#              Grid Cache               #
#                                       #
#########################################

class Grid_Cache :
    def __init__(self, data, cell_size, caching_function = None, margin = 25) :
        # Get rows and cols
        self.size = data.shape[1::-1]
        self.cell_size = cell_size
        w, h = self.size
        self.rows = int(w / cell_size[0]) + 1
        self.cols = int(h / cell_size[1]) + 1
        self.data = data
        self.fun = caching_function
        self.last = None
        self.margin = margin

        # Initialize grid
        self.grid = {n:{} for n in range(self.cols)}

    def get(self, pos) :
        # Check that pos is within bounds
        w, h = self.size
        if pos[0] > w or pos[1] > h :
            raise Exception("%s is outside data bounds of %s" % (pos, self.size))
        # What grid block are we looking for
        col, row = self.block(pos)
        # has this grid cell been cached?
        return self.get_cell(col, row)

    def get_cell(self, col, row) :
        if not self.grid[col].get(row, False) :
            # If not, cache it and return
            self.last = self.cache(col, row)
        return self.grid[col][row]

    def is_cached(self, pos) :
        col, row = self.block(pos)
        if self.grid[col].get(row, False) :
            return True
        else :
            return False

    def block(self, pos) :
        # What grid block are we looking for
        row = int(pos[0] / self.cell_size[0])
        col = int(pos[1] / self.cell_size[1])
        return col, row

    def offset(self, pos) :
        col, row = self.block(pos)
        x_min = row * self.cell_size[0] - self.margin
        y_min = col * self.cell_size[1] - self.margin
        return (x_min, y_min)


    def center(self, col, row) :
        x = int((row + 0.5) * self.cell_size[0])
        y = int((col + 0.5) * self.cell_size[1])
        return (x, y)


    def cache(self, col, row) :
        # Find area
        x_min = row * self.cell_size[0] - (self.margin * (row > 0))
        x_max = x_min + self.cell_size[0] + self.margin * 2 if row+1 < self.rows else self.size[0]
        y_min = col * self.cell_size[1] - (self.margin * (col > 0))
        y_max = y_min + self.cell_size[1] + self.margin * 2 if col+1 < self.cols else self.size[1]
        # Extract data, compute cache value and add it to grid
        data_cell = self.data[y_min:y_max, x_min:x_max,:]
        if self.fun == None :
            self.grid[col][row] = data_cell
        else :
            self.grid[col][row] = self.fun(data_cell)
        return ((x_min, x_max), (y_min, y_max))


    def get_neighbor(self, block, pos) :
        """ Returns the position of the neighboring square in the image.
        The neighboring square is selected based on the border which the
        position is closest to in the current image """
        # Get col and row indices of current block
        col, row = block
        # Find center of block of pos
        center = self.center(col, row)
        # Now find where we are relating to center
        x_diff = pos[0] - center[0] # negative if left of center, positive otherwise
        y_diff = pos[1] - center[1] # negative if above center, positive otherwise
        if y_diff < x_diff and y_diff < -1*x_diff :
            return self.center(col - 1, row) if col - 1 >= 0 else None
        elif x_diff > y_diff :
            return self.center(col, row + 1) if row + 1 < self.rows else None
        elif y_diff > -1*x_diff :
            return self.center(col + 1, row) if col + 1 < self.cols else None
        else :
            return self.center(col, row - 1) if row - 1 >= 0 else None




#########################################
#                                       #
#             Metric Cache              #
#                                       #
#########################################

class Metric_Cache :

    def __init__(self, path, options = {}) :
        """ Caches an image so it's ready for matching """
        # Get relevant options
        force_reload    = options.get("force_reload", False)
        max_size        = options.get("max_size", None)
        metric          = options.get("metric", "minkowski")
        thumb_size      = options.get("thumb_size", (800, 800))
        # save path and init attributes
        self.path = path
        self.thumb = {}
        self.original = {}
        # check if the path exists
        if not force_reload and self.load() : return
        # Create thumbnail and image
        self.create_thumbnail(path, thumb_size)
        self.create_image(path, max_size, metric)


    def get(self, position, radius, options = {}) :
        """ Retrieve all features within radius of position """
        # Get relevant options and position tree
        sort_results = options.get("sort_results", True)
        pos_tree = self.original["position_tree"]
        # Fetch all feature points within radius pixels of position
        indices, distances = pos_tree.query_radius(position,
                                                   r = radius,
                                                   return_distance=True,
                                                   sort_results=sort_results)
        idx = indices[0]
        # Return all features and their positions
        return self.original["descriptors"][idx], self.original["positions"][idx], self.original["distances"][idx]


    def save(self, dir = "data/image_data") :
        """ Exports cache to file """
        # Create unique identifier based on image path
        h = hashlib.new('ripemd160')
        h.update(self.path)
        data_path = h.hexdigest()
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


    def load(self, dir = "data/image_data") :
        """ Loads file to Cache """
        # get hash for path
        h = hashlib.new('ripemd160')
        h.update(self.path)
        data_path = h.hexdigest()
        # Check if file exists
        full_path_npz = "%s/%s.npz" % (dir, data_path)
        full_path_thumb = "%s/%s_thumb.npz" % (dir, data_path)
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


    def create_thumbnail(self, path, max_size) :
        """ Get relevant data for thumbnail """
        # Create thumbnail
        thumbnail = imaging.get_thumbnail(path, max_size)
        # Get thumbnail features
        keypoints, descriptors = matchutil.get_features(thumbnail)
        # Get nearest neighbor within image (vector with touples of feature points and distances)
        matches = matchutil.bf_match(descriptors, descriptors, k = 2)
        # Distances to nearest neighbor and positions
        nn_distances = [r[1].distance for r in matches]
        positions = [k.pt for k in keypoints]
        # Collect data
        self.thumb = {
            "descriptors" : descriptors,
            "positions" : positions,
            "distances" : nn_distances,
            "size" : thumbnail.shape
        }


    def create_image(self, path, max_size, metric) :
        """ Match an image with itself finding the closest neighbors within that image """
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
            "size" : img_data.shape
        }






# shouldn't be here
