"""
Python module for dealing with the 3D objects dataset published here:
http://www.vision.caltech.edu/pmoreels/Datasets/TurntableObjects/

Jonas Toft Arnfred, 2013-11-18
"""

####################################
#                                  #
#            Imports               #
#                                  #
####################################

import cv2
import numpy
import os
import colors
import fnmatch
import pylab
from itertools import chain
import matchutil
import imaging
from cache import Metric_Cache

####################################
#                                  #
#           Functions              #
#                                  #
####################################


def evaluate(match_fun, angles, object_type, thresholds, ground_truth_data = None, options = {}) :
    """ Returns number of correct and total matches of match_fun on object:
        match_fun : Function (Function that takes a list of paths and returns matches)
        angles : Int (Rotation in degrees. Must be divisible by 5)
        object_type : String (the object pictured on the turntable)
        thresholds : List[Float] (list of ratio thresholds)
        match_count : List[Int] (List of the amount of possible matches for each feature point)
        options : Dict (Set of parameters for matching etc)
    """

    # Get distance_threshold
    distance_threshold      = options.get("distance_threshold", 5)
    verbose                 = options.get("evaluate_verbose", False)
    thumb_strategy          = options.get("thumb_strategy", lambda n : n)

    # Get paths to the three images
    def get_path(i) : return {
            "A" : get_turntable_path(object_type, angles[0] + i*360, "Bottom"),
            "B" : get_turntable_path(object_type, angles[0] + i*360, "Top"),
            "C" : get_turntable_path(object_type, angles[1] + i*360, "Bottom")
        }

    if verbose :
        print("matching\n%s\n%s" %(get_path(0)["A"], get_path(0)["C"]))

    # Get paths
    def get_match_fun(i) :
        # Collect matches
        paths = get_path(i)
        query_path, target_path = paths["A"], paths["C"]
        query_cache, target_img = Metric_Cache(query_path), imaging.open_img(target_path)
        return match_fun(query_cache, target_img, options = options)

    # Get matches
    match_functions = map(get_match_fun, range(3))
    apply_threshold = lambda tau : list(chain(*(f(tau, thumb_strategy(tau)) for f in match_functions)))
    matches = [apply_threshold(tau) for tau in thresholds]

    # Get distances for the highest threshold (they include all other)
    uniques = { r : p for p, r in chain(*matches) }
    distances = match_distances(uniques, angles, object_type, distance_threshold)

    # For each set of matches count correct matches
    get_count = lambda ms, dt : sum([1 for p,r in ms if distances[r] < dt])
    correct = numpy.array([get_count(ms, distance_threshold) for ms in matches])
    total = numpy.array([len(ms) for ms in matches])
    accuracy = correct / numpy.array(total, dtype=numpy.float)

    return { "accuracy" : accuracy, "correct" : correct, "total" : total }



def evaluate_objects(match_fun, angles, object_types, thresholds, ground_truth_data = None, options = {}) :
    """ Returns number of correct and total matches of match_fun on objects:
        match_fun : Function (Function that takes a list of paths and returns matches)
        angles : (Int, Int) (two angles in degrees. Must be divisible by 5)
        object_types : List[String] (the objects pictured on the turntable)
        thresholds : List[Float] (list of ratio thresholds)
        options : Dict (Set of parameters for matching etc)
    """
    # Get correct and total matches for different thresholds at current angle_increment
    def get_results(object_type) :
        gt = None if ground_truth_data == None else ground_truth_data[object_type]
        return evaluate(match_fun, angles, object_type, thresholds, gt, options)


    results = [get_results(o) for o in object_types]
    correct =  { o : r["correct"] for r, o in zip(results, object_types) }
    total =  { o : r["total"] for r, o in zip(results, object_types) }

    return { "correct" : correct, "total" : total }



def get_turntable_path(object_type, angle, camera_position, turntable_dir = "../../Images/turntable/") :
    """ Returns path to image of image_set and object_type where:
        object_type : String (the object pictured on the turntable)
        angle : Int (Rotation in degrees. Must be divisible by 5)
        camera_position : String ("Top" or "Bottom")
        turntable_dir : String (path to folder with image sets)
    """
    image_set = get_image_set(object_type)
    directory = "%s/imageset_%i/%s/%s" % (turntable_dir, image_set, object_type, camera_position)
    file_names = os.listdir(directory)
    file_glob = "img_1-%03i_*_0.JPG" % angle
    image_paths = sorted(fnmatch.filter(file_names, file_glob))
    return "%s/%s" % (directory, image_paths[0])



def load_calibration_image(object_type, angle, camera_position, pattern_position, scale = None, turntable_dir = "../../Images/turntable/") :
    """ Loads a checkerboard pattern image taken from the same angle and rotation:
        angle : Int (Rotation in degrees. Must be divisible by 5)
        pattern_position : String (either: "flat", "angled" or "steep")
        scale : Float (the calibration images are double size, so put 2.0)
        turntable_dir : String (path to folder with image sets)
    """
    image_set = get_image_set(object_type)
    pattern_index = { "flat" : 0, "steep" : 1, "angled" : 2 }[pattern_position]
    image_index = (angle / 5) * 3 + 1 + pattern_index
    image_path = "%simageset_%i/%s/%s/calib%i.jpg" % (turntable_dir, image_set, "Calibration", camera_position, image_index)
    img = imaging.open_img(image_path)
    if scale != None :
        img_scaled = cv2.resize(img, (int(img.shape[1] / scale), int(img.shape[0] / scale)))
        return img_scaled
    else :
        return img



# Get calibration points
def get_calibration_points(object_type, angle, camera_position, pattern_position) :
    """ Returns the calibration points from an image with the checkerboard pattern """

    # Load img
    img = load_calibration_image(object_type, angle, camera_position, pattern_position)

    if pattern_position == "steep" :
        grid_size = (9, 13)
    else :
        grid_size = (13, 9)
    success, cv_points = cv2.findChessboardCorners(img, grid_size, flags = cv2.CALIB_CB_FILTER_QUADS)
    if not success :
        raise Exception("Can't get lock on chess pattern: Angle: %i, object: %s, camera position: %s, pattern_position: %s" % (angle, object_type, camera_position, pattern_position))
    points = numpy.array([[p[0][0], p[0][1]] for p in cv_points])
    return points# Get calibration points



def get_fundamental_matrix(object_type, angles, camera_position, scale = 1.0) :
    """ Returns fundamental matrix:
        angles : (Int, Int) (two angles in degrees. Must be divisible by 5)
        camera_position : String (either "Top" or "Bottom")
        scale : Float (scale of images compared to calibration images)
    """
    try :
        variables = (object_type, angles[0], angles[1], camera_position[0], camera_position[1], scale)
        mat_path = "data/fundamental_matrix/%s_%i_%i_%s_%s_%s" % variables
        F = numpy.load("%s.npz" % mat_path)["mat"]
    except IOError :
        # Fetch all images for the angle pair
        points1_flat = get_calibration_points(object_type, angles[0], camera_position[0], "flat")
        points2_flat = get_calibration_points(object_type, angles[1], camera_position[1], "flat")
        points1_angled = get_calibration_points(object_type, angles[0], camera_position[0], "angled")
        points2_angled = get_calibration_points(object_type, angles[1], camera_position[1], "angled")
        points1_steep = get_calibration_points(object_type, angles[0], camera_position[0], "steep")
        points2_steep = get_calibration_points(object_type, angles[1], camera_position[1], "steep")

        # Concatenate point sets
        points1 = numpy.concatenate((points1_flat, points1_angled, points1_steep)) / scale
        points2 = numpy.concatenate((points2_flat, points2_angled, points2_steep)) / scale

        # Find fundamental matrix based on points
        F, inliers = cv2.findFundamentalMat(points1, points2, method = cv2.FM_RANSAC)

        # Save fundamental matrix
        numpy.savez(mat_path, mat = F)
    return F



def epilines(img, points, lines, size = (12, 12)) :
    """ Draws a set of epilines and points on an image """
    # Generate figure
    pylab.imshow(img)

    # Get x values
    max_x = img.shape[1]
    max_y = img.shape[0]

    # Limit plot
    pylab.xlim(0,max_x)
    pylab.ylim(max_y-1,0)

    # plot lines
    for l, c in zip(lines, colors.get()) :
        # get line functions
        line_fun = lambda x : (-1 * l[0] * x - l[2]) / (float(l[1]))
        # plot line
        pylab.plot([0, max_x], [line_fun(0), line_fun(max_x)], color=c, marker='_')

    # plot points
    for p, c in zip(points, colors.get()) :
        # Plot feature match point
        pylab.plot(p[0], p[1], color=c, marker='o')



def match_distances(uniques, angles, object_type, check_threshold, scale = 1.0) :
    """ Find the distance of matches as measured against two intersection epipolar lines:
        matches : List[(Pos,Pos)] (List of corresponding coordinates in two images)
        angles : (Int, Int) (two angles in degrees. Must be divisible by 5)
        object_type : String (The type of 3d model we are looking at)
        check_threshold : Float (The threshold for a correct correspondence)
    """

    ratios, positions = uniques.keys(), uniques.values()

    # Get points
    path_B = get_turntable_path(object_type, angles[0], "Top")
    points = {
        "A" : numpy.array([p[0] for p in positions], dtype=numpy.float32),
        "B" : Metric_Cache(path_B).original["positions"],
        "C" : numpy.array([p[1] for p in positions], dtype=numpy.float32)
    }

    # Find fundamental matrices
    F = {
        "AC" : get_fundamental_matrix(object_type, (angles[0], angles[1]), ("Bottom", "Bottom"), scale = scale), # Reference view
        "AB" : get_fundamental_matrix(object_type, (angles[0], angles[0]), ("Bottom", "Top"), scale = scale), # Test view
        "BC" : get_fundamental_matrix(object_type, (angles[0], angles[1]), ("Top", "Bottom"), scale = scale) # Auxiliary view
    }

    # return distances
    return dict(calc_match_distances(points, ratios, F, check_threshold))



def calc_match_distances(points, ratios, F, check_threshold) :
    """ Helper function for match_distances()
        Check ground truth for a set of matches given fundamental matrices,
        As proposed in: "Evaluation of Features Detectors and Descriptors
        based on 3D objects by Pierre Moreels and Peitro Perona.
    """

    # Calculate distance between line and 2D-point
    def dist(line, point) :
        return numpy.abs(line.dot([point[0], point[1], 1]))

    # return epipolar line
    def get_lines(points, F_current) :
        if len(points) == 0 :
            return []
        else :
            return [l[0] for l in cv2.computeCorrespondEpilines(points.reshape(-1, 1, 2), 1, F_current)]

    # Find points that are on l_AB
    lines_AB = get_lines(points["A"], F["AB"])
    lines_AC = get_lines(points["A"], F["AC"])

    for p_A, p_C, l_AB, l_AC, r in zip(points["A"], points["C"], lines_AB, lines_AC, ratios) :

        # Is p_C on l_AC?
        AC_dist = dist(l_AC, p_C)
        min_dist = 999

        if AC_dist < check_threshold :
            # Collect all features in B that are on l_AB:
            points_B = numpy.array([p_B for p_B in points["B"] if dist(l_AB, p_B) < check_threshold], dtype=numpy.float32)

            if len(points_B) > 0 :
                lines_BC = get_lines(points_B, F["BC"])
                min_dist = numpy.min([dist(l_BC, p_C) for l_BC in lines_BC])

        yield (r, min_dist)


def get_image_set(object_type) :
    sets = {
        1 : ["Conch",  "FlowerLamp",        "Motorcycle",  "Rock", "Bannanas",  "Car",          "Desk",   "GrandfatherClock",  "Robot",       "TeddyBear", "Base",      "Car2",         "Dog",    "Horse", "Tricycle"],
        2 : ["Clock",  "EthernetHub",  "Hicama",  "Pepper"],
        3 : ["Dremel",  "JackStand",  "Sander",  "SlinkyMonster",  "SprayCan"],
        4 : ["FireExtinguisher",  "Frame",  "Hat", "StaplerRx"],
        5 : ["Carton",  "Clamp",  "EggPlant",  "Lamp",  "Mouse",  "Oil"],
        6 : ["Basket",  "Clipper",  "CupSticks",  "Filter",  "Mug",  "Shoe"],
        7 : ["Pops", "Speaker", "BoxingGlove",  "CollectorCup", "Utilities"],
        8 : ["Camera",  "DishSoap",  "Nesquik",  "PotatoChips",   "Sponge"],
        9 : ["Camera2",     "Cloth",     "FloppyBox", "CementBase",  "Dinosaur",  "PaperBin"],
        10 : ["Phone2",      "RollerBlade",  "Tripod",  "RiceCooker",  "Spoons",       "VolleyBall"],
        11 : ["Gelsole",   "MouthGuard",  "Razor",       "Toothpaste", "Abroller",  "DVD",          "Keyboard",  "PS2"],
        12 : ["Bush",  "DuckBank",  "Eggs",  "Frog",  "LightSaber"],
        13 : ["Coffee",  "LavaBase",  "SwanBank",  "ToolBox",  "Vegetta"],
        14 : ["BoxStuff",     "Standing", "BallSander", "StorageBin"],
        15 : ["Globe",  "Pineapple"]
    }
    for key,value in sets.iteritems() :
        if object_type in value :
            return key

    raise Exception("No object matching object_type of '%s'" % object_type)



def ground_truth(angles, object_type, options = {}) :
    """ Find the amount of total possible correspondences for all lightning conditions.
        angles : (Int, Int) (two angles in degrees. Must be divisible by 5)
        object_type : String (The type of 3d model we are looking at)
        return_matches : Boolean (set to True to return the correspondences found too)
        options : Dict (Set of parameters for matching etc)
    """
    verbose = options.get("evaluate_verbose", False)
    nb_correspondences = 0
    filter_features = []
    for i in range(3) :
        curr_matches = list(calc_ground_truth(angles, object_type, lightning_index = i, return_matches = False, options = options))
        # Create list of features that should be kept
        curr_ff = [i for i, m in enumerate(curr_matches) if len(m) == 0]
        nb_correspondences += len(curr_matches) - len(curr_ff)
        filter_features.append(curr_ff)

    if verbose :
        print("There are %i theoretically possible correspondences for object '%s' at angles (%i,%i)" % (nb_correspondences, object_type, angles[0], angles[1]))

    return { "nb_correspondences" : nb_correspondences, "filter_features" : filter_features }


# Load keypoints
def keypoints(object_type, angle, viewpoint, options = {}) :
    path = get_turntable_path(object_type, angle, viewpoint)
    img_data = imaging.open_img(path)
    points = [k.pt for k in matchutil.get_keypoints(img_data)]
    return numpy.array(points, dtype=numpy.float32)

# return epipolar line
def get_lines(points, F) :
    return [l[0] for l in cv2.computeCorrespondEpilines(points.reshape(-1, 1, 2), 1, F)]

# Calculate distance between line and 2D-point
def dist(line, point) :
    return numpy.abs(line.dot([point[0], point[1], 1]))



def calc_ground_truth(angles, object_type, lightning_index = 0, return_matches = False, options = {}) :
    """ Find the amount of total possible correspondences.
        For each feature in A, check if there is a feature in C such that the
        epipolar constraints for a correct match are fulfilled for any point in B:
        angles : (Int, Int) (two angles in degrees. Must be divisible by 5)
        object_type : String (The type of 3d model we are looking at)
        return_matches : Boolean (set to True to return the correspondences found too)
        options : Dict (Set of parameters for matching etc)
    """

# Get distance_threshold
    distance_threshold  = options.get("distance_threshold", 5)

    # Get paths to the three images
    keypoints_A = keypoints(object_type, angles[0]+360*lightning_index, "Bottom", options)
    keypoints_B = keypoints(object_type, angles[0]+360*lightning_index, "Top", options)
    keypoints_C = keypoints(object_type, angles[1]+360*lightning_index, "Bottom", options)

    # Find fundamental matrices
    F_AC = get_fundamental_matrix(object_type, (angles[0], angles[1]), ("Bottom", "Bottom"), scale = 2.0)
    F_AB = get_fundamental_matrix(object_type, (angles[0], angles[0]), ("Bottom", "Top"), scale = 2.0)
    F_BC = get_fundamental_matrix(object_type, (angles[0], angles[1]), ("Top", "Bottom"), scale = 2.0)

    # For every point in A find the corresponding lines in B and C
    lines_AB = get_lines(keypoints_A, F_AB)
    lines_AC = get_lines(keypoints_A, F_AC)

    # For every epiline in B and C corresponding to a point in A
    for i, (p_A, l_AB, l_AC) in enumerate(zip(keypoints_A, lines_AB, lines_AC)) :

        # Find all points on the line in B and C
        points_B = numpy.array([p_B for p_B in keypoints_B if dist(l_AB, p_B) < distance_threshold], dtype=numpy.float32)
        points_C = numpy.array([p_C for p_C in keypoints_C if dist(l_AC, p_C) < distance_threshold], dtype=numpy.float32)

        # For every point in B on l_AB, see if there is a point in C on l_AC that lies on the epipolar line of p_B in image C: l_BC
        if len(points_B) > 0 and len(points_C) > 0 :

            # Get distances from every point in C on line l_AC to every line in C corresponding to a point in B on line l_AB
            get_distances = lambda line : [(dist(line, p_C), p_C) for p_C in points_C]
            distances = list(chain([get_distances(l_BC) for l_BC in get_lines(points_B, F_BC)]))

            # Count how many potential matches there are
            yield [(p_A, p_C) for d, p_C in distances if d < distance_threshold]

        else :
            yield []


def get_weights(gt, exclude_keys = []) :
    """ Calculate a set of weights per object and index item (usually angle) """
    nb_corr_accu = { a : sum(map(lambda data : data["nb_correspondences"], gt_item.values())) for a, gt_item in gt.iteritems() if a not in exclude_keys }
    weights = { a : { o : 1.0 / (t["nb_correspondences"] / float(nb_corr_accu[a])) for o, t in gt_item.iteritems() } for a, gt_item in gt.iteritems() if a not in exclude_keys }
    weights_sum = { a : numpy.sum(ow.values()) for a, ow in weights.iteritems() }
    weights_normalize = { a : { o : w / weights_sum[a] for o,w in ow.iteritems() } for a,ow in weights.iteritems() }
    return weights_normalize


def accumulate(results, gt, weights = None, exclude_keys = []) :
    """ Take results from evaluate_objects and ground truth and compile a simple data structure
        containing correct, total and nb_correlations """

    # Get default weights
    if weights == None :
        weights = { a : { o : 1 for o in t.keys() } for a, t in gt.iteritems() }

    correct = {}
    total = {}
    nb_corr = {}
    for k, r_value in results.iteritems() :

        # Check that we are including this key
        if k not in exclude_keys :
            c_methods = {}
            t_methods = {}
            w_norm = weights[k]

            # Calculate sum of correct and total counts
            for method, r_method in r_value.iteritems() :
                r_weighted_correct = [numpy.array(r) * w_norm[o] for o, r in r_method["correct"].iteritems()]
                r_weighted_total = [numpy.array(r) * w_norm[o] for o, r in r_method["total"].iteritems()]
                c_methods[method] = numpy.sum(r_weighted_correct, axis = 0)
                t_methods[method] = numpy.sum(r_weighted_total, axis = 0)

            # Calculate sum of possible correspondences
            nb_corr[k] = int(sum(map(lambda (o,data) : data["nb_correspondences"]*w_norm[o], gt[k].iteritems())))

            # save accumulated counts
            correct[k] = c_methods
            total[k] = t_methods

    return correct, total, nb_corr
